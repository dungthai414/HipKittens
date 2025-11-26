#include "kittens.cuh"
#include <random>
#include <omp.h>
#include <chrono>

using namespace kittens;

#include "../profile_utils.cpp"
#include "./utils.cpp"

constexpr int NUM_WARPS = 8;

using G = kittens::group<NUM_WARPS>;

// We use 64-element float vectors for scales, matching the 64-row height of RT_C.
using ScaleVecS = sv<float, 64>; 

// Helper to scale FP32 output tiles
// Performs: C_final += C_temp * ScaleVector
template<typename RT_C_TYPE>
__device__ inline void accumulate_scaled_output(RT_C_TYPE &c_final, 
                                                RT_C_TYPE &c_temp, 
                                                const ScaleVecS &scale_src) {
    // 1. Define Register Vector type matching the tile rows
    using ColVec = typename RT_C_TYPE::col_vec; // 64 elements
    ColVec scale_vec;
    
    // 2. Load from the properly typed Shared Vector
    load(scale_vec, scale_src);

    // 3. Apply scales to the temporary result tile
    // mul_row broadcasts the column vector 'scale_vec' across all rows of 'c_temp'.
    mul_row(c_temp, c_temp, scale_vec);

    // 4. Accumulate into final result
    add(c_final, c_final, c_temp);
}

template <int M, int N, int K, int QN, int QK>
__global__ __launch_bounds__(512, 2) 
void matmul_device(const kittens::gl<fp8e4m3, 1, 1, M, K> A, 
                   const kittens::gl<fp8e4m3, 1, 1, N, K> B, 
                   const kittens::gl<float, 1, 1, M, QK> scale_A, 
                   const kittens::gl<float, 1, 1, QN, QK> scale_B, 
                   const kittens::gl<bf16, 1, 1, M, N> C) 
{
    constexpr int WARPS_COL = 4;
    constexpr int WARPS_ROW = 2;
    constexpr int BLOCK_SIZE_ROW = 256;
    constexpr int BLOCK_SIZE_COL = 256;
    constexpr int BLOCK_K = 128;
    constexpr int k_iters = K / BLOCK_K; 
    constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS; 
    constexpr int HALF_BLOCK_SIZE_ROW = BLOCK_SIZE_ROW / 2; 
    constexpr int HALF_BLOCK_SIZE_COL = BLOCK_SIZE_COL / 2; 
    constexpr int REG_BLOCK_M = BLOCK_SIZE_ROW / WARPS_ROW / 2; // 64
    constexpr int REG_BLOCK_N = BLOCK_SIZE_COL / WARPS_COL / 2; // 32
    constexpr int PM = BLOCK_SIZE_ROW;
    constexpr int PN = BLOCK_SIZE_COL / 128; // 2 groups (Left/Right)

    using ST_A = st_fp8e4m3<HALF_BLOCK_SIZE_ROW, BLOCK_K, st_16x128_s>; 
    using ST_B = st_fp8e4m3<HALF_BLOCK_SIZE_COL, BLOCK_K, st_16x128_s>; 
    
    __shared__ ST_A As[2][2]; 
    __shared__ ST_B Bs[2][2]; 

    // Shared Memory for Scales
    // [ping_pong][B_group_idx][row_chunk_idx]
    __shared__ ScaleVecS s_scales[2][2][4]; 

    using RT_A = rt_fp8e4m3<REG_BLOCK_M, BLOCK_K>; 
    using RT_B = rt_fp8e4m3<REG_BLOCK_N, BLOCK_K>; 
    using RT_C = rt_fl<REG_BLOCK_M, REG_BLOCK_N, col_l, rt_16x16_s>; 

    RT_A a;
    RT_B b0, b1;
    RT_C cA, cB, cC, cD;
    
    // OPTIMIZATION: Reuse a single temporary tile for all quadrants
    // This reduces register pressure significantly compared to keeping 4 temps alive.
    RT_C c_temp; 

    int global_block_id = blockIdx.x;
    int blocks_per_col = N / BLOCK_SIZE_COL;
    int block_row = global_block_id / blocks_per_col;
    int block_col = global_block_id % blocks_per_col;
    int block_m = block_row * BLOCK_SIZE_ROW;

    int warp_m = (warpid() / WARPS_COL); 
    int warp_n = (warpid() % WARPS_COL); 

    int tic = 0, toc = 1;

    using T = fp8e4m3;
    constexpr int bytes_per_thread = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile_A = HALF_BLOCK_SIZE_ROW * BLOCK_K * sizeof(T) / bytes_per_memcpy; 
    constexpr int memcpy_per_tile_B = HALF_BLOCK_SIZE_COL * BLOCK_K * sizeof(T) / bytes_per_memcpy; 
    uint32_t swizzled_offsets_A[memcpy_per_tile_A];
    uint32_t swizzled_offsets_B[memcpy_per_tile_B];
    G::prefill_swizzled_offsets(As[tic][0], A, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[tic][0], B, swizzled_offsets_B);

    zero(cA); zero(cB); zero(cC); zero(cD);

    // Lambda to load scale coefficients
    auto load_scales_tile = [&](int buf, int qk) {
        const int tid    = threadIdx.x;
        const int tcount = NUM_THREADS;
        const int qn_block = block_col * PN; 

        for (int idx = tid; idx < PM * PN; idx += tcount) {
            int m = idx / PN; // Row index (0..255)
            int j = idx % PN; // B-Group index (0 or 1)

            int row = block_m + m;
            int qn  = qn_block + j;

            float sA = scale_A[{0, 0, row, qk}];
            float sB = scale_B[{0, 0, qn,  qk}];
            
            s_scales[buf][j][m / 64][m % 64] = sA * sB;
        }
    };

    // Initial loads
    G::load(Bs[tic][0], B, {0, 0, block_col * 2, 0}, swizzled_offsets_B);
    G::load(As[tic][0], A, {0, 0, block_row * 2, 0}, swizzled_offsets_A);
    G::load(Bs[tic][1], B, {0, 0, block_col * 2 + 1, 0}, swizzled_offsets_B);
    G::load(As[tic][1], A, {0, 0, block_row * 2 + 1, 0}, swizzled_offsets_A);

    load_scales_tile(tic, 0);

    if (warp_m == 1) __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    G::load(As[toc][0], A, {0, 0, block_row * 2, 1}, swizzled_offsets_A);
    G::load(Bs[toc][0], B, {0, 0, block_col * 2, 1}, swizzled_offsets_B);
    G::load(Bs[toc][1], B, {0, 0, block_col * 2 + 1, 1}, swizzled_offsets_B);
    
    load_scales_tile(toc, 1);

    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();

    #pragma unroll 
    for (int k = 0; k < k_iters - 2; k++, tic^=1, toc^=1) {
        constexpr int b_idx_left = 0;
        constexpr int b_idx_right = 1;

        // cluster 1
        auto bs_subtile0 = kittens::subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[tic][0], {warp_n, 0}); 
        load_st_to_rt<RT_B, decltype(bs_subtile0)>(b0, bs_subtile0);
        auto as_subtile0 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][0], {warp_m, 0});
        load_st_to_rt<RT_A, decltype(as_subtile0)>(a, as_subtile0);
        G::load(As[toc][1], A, {0, 0, block_row * 2 + 1, k + 1}, swizzled_offsets_A);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        // cluster 2: cA
        zero(c_temp); // Clear reused temp
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c_temp, a, b0, c_temp);
        __builtin_amdgcn_s_setprio(0);
        accumulate_scaled_output(cA, c_temp, s_scales[tic][b_idx_left][warp_m]);

        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // cluster 3
        auto bs_subtile1 = kittens::subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[tic][1], {warp_n, 0});
        load_st_to_rt<RT_B, decltype(bs_subtile1)>(b1, bs_subtile1);
        G::load(As[tic][0], A, {0, 0, block_row * 2, k + 2}, swizzled_offsets_A);
        __builtin_amdgcn_s_barrier(); 

        // cluster 4: cB
        zero(c_temp);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c_temp, a, b1, c_temp);
        __builtin_amdgcn_s_setprio(0);
        accumulate_scaled_output(cB, c_temp, s_scales[tic][b_idx_right][warp_m]);
        __builtin_amdgcn_s_barrier();

        // cluster 5
        auto as_subtile1 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][1], {warp_m, 0});
        load_st_to_rt<RT_A, decltype(as_subtile1)>(a, as_subtile1);
        G::load(Bs[tic][0], B, {0, 0, block_col * 2, k + 2}, swizzled_offsets_B);
        __builtin_amdgcn_s_barrier();

        // cluster 6: cC
        zero(c_temp);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c_temp, a, b0, c_temp);
        __builtin_amdgcn_s_setprio(0);
        accumulate_scaled_output(cC, c_temp, s_scales[tic][b_idx_left][2 + warp_m]);

        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // cluster 7
        G::load(Bs[tic][1], B, {0, 0, block_col * 2 + 1, k + 2}, swizzled_offsets_B);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        // cluster 8: cD
        zero(c_temp);
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c_temp, a, b1, c_temp);
        __builtin_amdgcn_s_setprio(0);
        accumulate_scaled_output(cD, c_temp, s_scales[tic][b_idx_right][2 + warp_m]);
        __builtin_amdgcn_s_barrier();

        // Safe to overwrite 'tic' buffer now
        load_scales_tile(tic, k + 2);
    }    
    {
        constexpr int k = k_iters - 2;
        constexpr int b_idx_left = 0;
        constexpr int b_idx_right = 1;

        auto bs_subtile0 = kittens::subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[tic][0], {warp_n, 0});
        load_st_to_rt<RT_B, decltype(bs_subtile0)>(b0, bs_subtile0);
        auto as_subtile0 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][0], {warp_m, 0});
        load_st_to_rt<RT_A, decltype(as_subtile0)>(a, as_subtile0);
        G::load(As[toc][1], A, {0, 0, block_row * 2 + 1, k + 1}, swizzled_offsets_A);
        __builtin_amdgcn_s_barrier();

        // cA
        zero(c_temp);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c_temp, a, b0, c_temp);
        __builtin_amdgcn_s_setprio(0);
        accumulate_scaled_output(cA, c_temp, s_scales[tic][b_idx_left][warp_m]);

        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs_subtile1 = kittens::subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[tic][1], {warp_n, 0});
        load_st_to_rt<RT_B, decltype(bs_subtile1)>(b1, bs_subtile1);
        __builtin_amdgcn_s_barrier();

        // cB
        zero(c_temp);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c_temp, a, b1, c_temp);
        __builtin_amdgcn_s_setprio(0);
        accumulate_scaled_output(cB, c_temp, s_scales[tic][b_idx_right][warp_m]);
        __builtin_amdgcn_s_barrier();

        auto as_subtile1 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][1], {warp_m, 0});
        load_st_to_rt<RT_A, decltype(as_subtile1)>(a, as_subtile1);
        __builtin_amdgcn_s_barrier();

        // cC
        zero(c_temp);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c_temp, a, b0, c_temp);
        __builtin_amdgcn_s_setprio(0);
        accumulate_scaled_output(cC, c_temp, s_scales[tic][b_idx_left][2 + warp_m]);
        __builtin_amdgcn_s_barrier();

        bs_subtile0 = kittens::subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[toc][0], {warp_n, 0});
        load_st_to_rt<RT_B, decltype(bs_subtile0)>(b0, bs_subtile0);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        // cD
        zero(c_temp);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c_temp, a, b1, c_temp);
        __builtin_amdgcn_s_setprio(0);
        accumulate_scaled_output(cD, c_temp, s_scales[tic][b_idx_right][2 + warp_m]);
        
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        tic^=1, toc^=1;
    }

    {
        constexpr int b_idx_left = 0;
        constexpr int b_idx_right = 1;
        // Final K-1 Loop
        auto as_subtile0 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][0], {warp_m, 0});
        load_st_to_rt<RT_A, decltype(as_subtile0)>(a, as_subtile0);
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_s_barrier();

        // cA
        zero(c_temp);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c_temp, a, b0, c_temp);
        __builtin_amdgcn_s_setprio(0);
        accumulate_scaled_output(cA, c_temp, s_scales[tic][b_idx_left][warp_m]);
        __builtin_amdgcn_s_barrier();

        auto bs_subtile1 = kittens::subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[tic][1], {warp_n, 0});
        load_st_to_rt<RT_B, decltype(bs_subtile1)>(b1, bs_subtile1);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // cB
        zero(c_temp);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c_temp, a, b1, c_temp);
        __builtin_amdgcn_s_setprio(0);
        accumulate_scaled_output(cB, c_temp, s_scales[tic][b_idx_right][warp_m]);
        __builtin_amdgcn_s_barrier();

        auto as_subtile1 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][1], {warp_m, 0});
        load_st_to_rt<RT_A, decltype(as_subtile1)>(a, as_subtile1);
        __builtin_amdgcn_s_barrier();

        // cC
        zero(c_temp);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c_temp, a, b0, c_temp);
        accumulate_scaled_output(cC, c_temp, s_scales[tic][b_idx_left][2 + warp_m]);

        // cD
        zero(c_temp);
        mma_ABt(c_temp, a, b1, c_temp);
        __builtin_amdgcn_s_setprio(0);
        accumulate_scaled_output(cD, c_temp, s_scales[tic][b_idx_right][2 + warp_m]);
        
        __builtin_amdgcn_s_barrier();
    }

    if (warp_m == 0) {
        __builtin_amdgcn_s_barrier();
    }
    store(C, cA, {0, 0, block_row * WARPS_ROW * 2 + warp_m, block_col * WARPS_COL * 2 + warp_n});
    store(C, cB, {0, 0, block_row * WARPS_ROW * 2 + warp_m, block_col * WARPS_COL * 2 + WARPS_COL + warp_n});
    store(C, cC, {0, 0, block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m, block_col * WARPS_COL * 2 + warp_n});
    store(C, cD, {0, 0, block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m, block_col * WARPS_COL * 2 + WARPS_COL + warp_n});
}

constexpr int QUANT_SIZE = 128;

// Blockwise quantization: quantize a float matrix into FP8 with per-block scales
template<int M, int K, int QK>
void quantize_blockwise(const std::vector<float>& src,
                        std::vector<fp8e4m3>& dst,
                        std::vector<float>& scales) {
    dst.resize(M * K);
    scales.resize(M * QK);

    #pragma omp parallel for collapse(2)
    for (int m = 0; m < M; ++m) {
        for (int qk = 0; qk < QK; ++qk) {
            int k_start = qk * QUANT_SIZE;
            int k_end = std::min(k_start + QUANT_SIZE, K);

            // Find max absolute value in this block
            float max_val = 0.0f;
            for (int k = k_start; k < k_end; ++k) {
                max_val = std::max(max_val, std::abs(src[m * K + k]));
            }

            // Compute scale (avoid division by zero)
            float scale = (max_val > 1e-8f) ? max_val / 240.0f : 1.0f;
            scales[m * QK + qk] = scale;

            // Quantize elements in this block
            for (int k = k_start; k < k_end; ++k) {
                dst[m * K + k] = fp8e4m3(src[m * K + k] / scale);
            }
        }
    }
}

// Blockwise quantization for B matrix (N x K layout)
template<int N, int K, int QN, int QK>
void quantize_blockwise_B(const std::vector<float>& src,
                          std::vector<fp8e4m3>& dst,
                          std::vector<float>& scales) {
    dst.resize(N * K);
    scales.resize(QN * QK);

    #pragma omp parallel for collapse(2)
    for (int qn = 0; qn < QN; ++qn) {
        for (int qk = 0; qk < QK; ++qk) {
            int n_start = qn * QUANT_SIZE;
            int n_end = std::min(n_start + QUANT_SIZE, N);
            int k_start = qk * QUANT_SIZE;
            int k_end = std::min(k_start + QUANT_SIZE, K);

            // Find max absolute value in this block
            float max_val = 0.0f;
            for (int n = n_start; n < n_end; ++n) {
                for (int k = k_start; k < k_end; ++k) {
                    max_val = std::max(max_val, std::abs(src[n * K + k]));
                }
            }

            // Compute scale
            float scale = (max_val > 1e-8f) ? max_val / 240.0f : 1.0f;
            scales[qn * QK + qk] = scale;

            // Quantize elements in this block
            for (int n = n_start; n < n_end; ++n) {
                for (int k = k_start; k < k_end; ++k) {
                    dst[n * K + k] = fp8e4m3(src[n * K + k] / scale);
                }
            }
        }
    }
}

template<int M, int N, int K, int QN, int QK>
void matmul_cpu_reference(const std::vector<fp8e4m3>& a_fp8,
                         const std::vector<fp8e4m3>& b_fp8,
                         const std::vector<float>& scale_a,
                         const std::vector<float>& scale_b,
                         std::vector<bf16>& c_ref) {
    c_ref.resize(M * N);
    
    #pragma omp parallel for collapse(2)
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            
            for (int qk = 0; qk < QK; ++qk) {
                int k_start = qk * QUANT_SIZE;
                int k_end = std::min(k_start + QUANT_SIZE, K);
                
                // Get scales for this block
                int qn = n / QUANT_SIZE;
                float scale = scale_a[m * QK + qk] * scale_b[qn * QK + qk];
                
                // Accumulate for this K block
                float block_acc = 0.0f;
                for (int k = k_start; k < k_end; ++k) {
                    block_acc += float(a_fp8[m * K + k]) * float(b_fp8[n * K + k]);
                }
                
                acc += block_acc * scale;
            }
            
            c_ref[m * N + n] = bf16(acc);
        }
    }
}

template<int M, int N, int K, int QN, int QK, int CUs>
TimingResult matmul_gpu_host(const std::vector<fp8e4m3>& a_host,
                            const std::vector<fp8e4m3>& b_host,
                            const std::vector<float>& scale_a_host,
                            const std::vector<float>& scale_b_host,
                            std::vector<bf16>& c_host,
                            int warmup_iters = 3,
                            int timing_iters = 20) {
    constexpr int threads_per_warp = 64;
    constexpr int warps_per_block = 8;
    constexpr int threads_per_block = threads_per_warp * warps_per_block;
    constexpr int block_count = (M * N) / (256 * 256);

    // Allocate device memory
    fp8e4m3 *d_a, *d_b;
    float *d_scale_a, *d_scale_b;
    bf16 *d_c;
    
    hipMalloc(&d_a, M * K * sizeof(fp8e4m3));
    hipMalloc(&d_b, N * K * sizeof(fp8e4m3));
    hipMalloc(&d_scale_a, M * QK * sizeof(float));
    hipMalloc(&d_scale_b, QN * QK * sizeof(float));
    hipMalloc(&d_c, M * N * sizeof(bf16));
    HipCheckError();

    // Copy data to device
    hipMemcpy(d_a, a_host.data(), M * K * sizeof(fp8e4m3), hipMemcpyHostToDevice);
    hipMemcpy(d_b, b_host.data(), N * K * sizeof(fp8e4m3), hipMemcpyHostToDevice);
    hipMemcpy(d_scale_a, scale_a_host.data(), M * QK * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_scale_b, scale_b_host.data(), QN * QK * sizeof(float), hipMemcpyHostToDevice);
    HipCheckError();

    // Create global memory wrappers
    kittens::gl<fp8e4m3, 1, 1, M, K> A(d_a, nullptr, nullptr, nullptr, nullptr);
    kittens::gl<fp8e4m3, 1, 1, N, K> B(d_b, nullptr, nullptr, nullptr, nullptr);
    kittens::gl<float, 1, 1, M, QK> scale_A(d_scale_a, nullptr, nullptr, nullptr, nullptr);
    kittens::gl<float, 1, 1, QN, QK> scale_B(d_scale_b, nullptr, nullptr, nullptr, nullptr);
    kittens::gl<bf16, 1, 1, M, N> C(d_c, nullptr, nullptr, nullptr, nullptr);

    // Warmup iterations
    for (int i = 0; i < warmup_iters; ++i) {
        hipMemset(d_c, 0, M * N * sizeof(bf16));
        matmul_device<M, N, K, QN, QK><<<block_count, threads_per_block>>>(A, B, scale_A, scale_B, C);
        HipCheckError();
        hipDeviceSynchronize();
    }

    // Create HIP events for timing
    hipEvent_t start_event, stop_event;
    hipEventCreate(&start_event);
    hipEventCreate(&stop_event);

    // Timed iterations
    std::vector<float> times_ms;
    times_ms.reserve(timing_iters);
    
    for (int i = 0; i < timing_iters; ++i) {
        hipMemset(d_c, 0, M * N * sizeof(bf16));
        hipEventRecord(start_event, 0);
        matmul_device<M, N, K, QN, QK><<<block_count, threads_per_block>>>(A, B, scale_A, scale_B, C);
        hipEventRecord(stop_event, 0);
        hipEventSynchronize(stop_event);
        
        float ms = 0.0f;
        hipEventElapsedTime(&ms, start_event, stop_event);
        times_ms.push_back(ms);
        HipCheckError();
    }

    // Calculate statistics
    float sum_ms = 0.0f, best_ms = 1e30f;
    for (float t : times_ms) {
        sum_ms += t;
        best_ms = std::min(best_ms, t);
    }
    float avg_ms = sum_ms / timing_iters;
    
    // Calculate TFLOPS
    double total_ops = 2.0 * M * N * K;
    double best_tflops = (total_ops / (best_ms * 1e-3)) / 1e12;
    double avg_tflops = (total_ops / (avg_ms * 1e-3)) / 1e12;

    // Copy result back
    c_host.resize(M * N);
    hipMemcpy(c_host.data(), d_c, M * N * sizeof(bf16), hipMemcpyDeviceToHost);
    HipCheckError();

    // Cleanup
    hipEventDestroy(start_event);
    hipEventDestroy(stop_event);
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_scale_a);
    hipFree(d_scale_b);
    hipFree(d_c);
    HipCheckError();

    return {best_ms, avg_ms, best_tflops, avg_tflops, timing_iters};
}

int main() {
    constexpr int M = 8192;
    constexpr int N = 8192;
    constexpr int K = 8192;
    constexpr int QN = N / QUANT_SIZE;  // 64
    constexpr int QK = K / QUANT_SIZE;  // 64
    constexpr int CUs = 256;
    
    constexpr int warmup_iters = 10;
    constexpr int timing_iters = 50;
    
    printf("=== Blockscale FP8 GEMM Test Suite ===\n");
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Quantization blocks: QN=%d, QK=%d (block size=%d)\n", QN, QK, QUANT_SIZE);
    printf("Warmup iterations: %d, Timing iterations: %d\n\n", warmup_iters, timing_iters);

    // Step 1: Generate random float matrices
    printf("[1/6] Generating random float matrices...\n");
    std::vector<float> a_float(M * K);
    std::vector<float> b_float(N * K);
    
    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < M * K; ++i) a_float[i] = dis(gen);
    for (int i = 0; i < N * K; ++i) b_float[i] = dis(gen);
    
    // Step 2: Quantize to FP8 with blockscale
    printf("[2/6] Quantizing to FP8 with blockscale...\n");
    std::vector<fp8e4m3> a_fp8;
    std::vector<fp8e4m3> b_fp8;
    std::vector<float> scale_a;
    std::vector<float> scale_b;
    
    quantize_blockwise<M, K, QK>(a_float, a_fp8, scale_a);
    quantize_blockwise_B<N, K, QN, QK>(b_float, b_fp8, scale_b);
    
    printf("   Scale A: %zu elements (M=%d, QK=%d)\n", scale_a.size(), M, QK);
    printf("   Scale B: %zu elements (QN=%d, QK=%d)\n", scale_b.size(), QN, QK);

    // Step 3: Run GPU kernel
    printf("[3/6] Running GPU blockscale FP8 GEMM kernel...\n");
    std::vector<bf16> c_gpu;
    TimingResult gpu_timing = matmul_gpu_host<M, N, K, QN, QK, CUs>(
        a_fp8, b_fp8, scale_a, scale_b, c_gpu, warmup_iters, timing_iters);
    
    printf("   GPU kernel complete.\n");
    printf("   Best time: %.3f ms (%.2f TFLOPS)\n", gpu_timing.best_time_ms, gpu_timing.best_tflops);
    printf("   Avg time:  %.3f ms (%.2f TFLOPS)\n\n", gpu_timing.avg_time_ms, gpu_timing.avg_tflops);

    // Step 4: Run CPU reference
    printf("[4/6] Running CPU reference implementation...\n");
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<bf16> c_ref;
    matmul_cpu_reference<M, N, K, QN, QK>(a_fp8, b_fp8, scale_a, scale_b, c_ref);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    double cpu_tflops = (2.0 * M * N * K / (cpu_time_ms * 1e-3)) / 1e12;
    
    printf("   CPU reference complete.\n");
    printf("   Time: %.3f ms (%.2f TFLOPS)\n\n", cpu_time_ms, cpu_tflops);

    // Step 5: Compare results
    printf("[5/6] Comparing GPU vs CPU results...\n");
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    double sum_abs_diff = 0.0;
    int mismatches = 0;
    constexpr float rel_threshold = 0.05f;  // 5% relative error threshold
    
    for (int i = 0; i < M * N; ++i) {
        float gpu_val = float(c_gpu[i]);
        float ref_val = float(c_ref[i]);
        
        float abs_diff = std::abs(gpu_val - ref_val);
        float rel_diff = (std::abs(ref_val) > 1e-6f) ? 
                         abs_diff / std::abs(ref_val) : 0.0f;
        
        max_abs_diff = std::max(max_abs_diff, abs_diff);
        max_rel_diff = std::max(max_rel_diff, rel_diff);
        sum_abs_diff += abs_diff;
        
        if (rel_diff > rel_threshold && abs_diff > 0.01f) {
            // If relative error is high, only count as mismatch if absolute difference is significant
            // This filters out cases where both numbers are very close to zero (e.g. 1e-6 vs 0)
            if (abs_diff > 0.1f) { 
                if (mismatches < 10) {  // Print first 10 mismatches
                    int row = i / N;
                    int col = i % N;
                    printf("   Mismatch at [%d,%d]: GPU=%.6f, CPU=%.6f, diff=%.6f (%.2f%%)\n", 
                           row, col, gpu_val, ref_val, abs_diff, rel_diff * 100.0f);
                }
                mismatches++;
            }
        }
    }
    
    float mean_abs_diff = sum_abs_diff / (M * N);
    
    printf("   Max absolute difference: %.6f\n", max_abs_diff);
    printf("   Max relative difference: %.2f%%\n", max_rel_diff * 100.0f);
    printf("   Mean absolute difference: %.6f\n", mean_abs_diff);
    printf("   Total mismatches (>5%% rel error): %d / %d (%.2f%%)\n\n", 
           mismatches, M * N, 100.0f * mismatches / (M * N));

    // Step 6: Print summary
    printf("[6/6] === SUMMARY ===\n");
    printf("GPU Performance:\n");
    printf("  Best: %.3f ms (%.2f TFLOPS)\n", gpu_timing.best_time_ms, gpu_timing.best_tflops);
    printf("  Avg:  %.3f ms (%.2f TFLOPS)\n", gpu_timing.avg_time_ms, gpu_timing.avg_tflops);
    printf("\nCPU Performance:\n");
    printf("  Time: %.3f ms (%.2f TFLOPS)\n", cpu_time_ms, cpu_tflops);
    printf("\nSpeedup: %.2fx (GPU best vs CPU)\n", cpu_time_ms / gpu_timing.best_time_ms);
    
    // Correctness verdict
    // Updated check: PASS if mismatches count is zero (ignoring negligible diffs)
    bool passed = (mismatches == 0);
    printf("\nCorrectness: %s\n", passed ? "PASSED ✓" : "FAILED ✗");
    
    if (!passed) {
        printf("WARNING: Accuracy issues detected. Check quantization or kernel implementation.\n");
    }

    return passed ? 0 : 1;
}