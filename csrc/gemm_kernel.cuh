// Block-scaled FP8 GEMM kernel for sm_89, written with CuTe.
//
// One CTA computes a BM x BN tile of C over a range of 128-wide K tiles:
//   * global -> shared: 16-byte cp.async (zero-filled outside M/N/K) into a
//     Stages-deep ring of swizzled shared tiles;
//   * shared -> registers: ldmatrix.x4, conflict-free thanks to the swizzle;
//   * mma.sync m16n8k32 e4m3 x e4m3 with FP32 accumulation into a per-tile
//     partial accumulator;
//   * after each K tile (= one scale block) the partial is promoted on the
//     CUDA cores: acc += partial * (sa[m, kb] * sb[n / 128, kb]).
// With split-K, every CTA of a tile writes its FP32 partial tile to a
// workspace; the last CTA to arrive (an atomic counter decides who that is)
// sums all partials in split order, so the result is bitwise deterministic.
#pragma once

#include <cuda_bf16.h>

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include "blockfp8.h"

namespace blockfp8 {

using namespace cute;

struct KernelParams {
  GemmProblem p;
  int tiles_m, tiles_n;
  int k_tiles, k_tiles_per_split, splits;
  int group_m;  // rasterization: CTAs walk group_m tile rows before moving along N
  float* workspace;
  int* counters;
};

// StepPromote = false: MMAs of a K tile accumulate into a full-size FP32
//   partial, promoted once per tile (two accumulator sets in registers).
// StepPromote = true: every k32 MMA starts from zero and is promoted right
//   away (4x the FMAs, but no second accumulator set), which frees the
//   registers for a 64x64 warp tile and so a larger CTA tile.
// MinBlocks: CTAs per SM the register allocation must allow (__launch_bounds__).
template <int BM_, int BN_, int Stages_, int WarpsM_, int WarpsN_, int MinBlocks_ = 1,
          bool StepPromote_ = false>
struct Config {
  static constexpr int BM = BM_, BN = BN_, BK = 128, Stages = Stages_;
  static constexpr int WarpsM = WarpsM_, WarpsN = WarpsN_, MinBlocks = MinBlocks_;
  static constexpr bool StepPromote = StepPromote_;
  static constexpr int kThreads = 32 * WarpsM * WarpsN;
  using T = cutlass::float_e4m3_t;

  static_assert(BN <= 128 && 128 % BN == 0, "a CTA tile must lie inside one weight block row");
  static_assert(BM % (16 * WarpsM) == 0 && BN % (16 * WarpsN) == 0, "tile/warp mismatch");

  // Each warp owns 16 x 16 of every 16*WarpsM x 16*WarpsN step (two n8 atoms),
  // so one ldmatrix.x4 feeds the B operand of two MMAs.
  using TiledMma = decltype(make_tiled_mma(SM89_16x8x32_F32E4M3E4M3F32_TN{},
                                           Layout<Shape<Int<WarpsM>, Int<WarpsN>, _1>>{},
                                           Tile<Int<16 * WarpsM>, Int<16 * WarpsN>, _32>{}));

  // 128-byte rows; the swizzle XORs the 16-byte chunk index with the row index
  // (mod 8), so both the cp.async writes and the ldmatrix reads of 8 rows hit
  // 8 different chunk positions.
  using SmemAtom =
      decltype(composition(Swizzle<3, 4, 3>{}, Layout<Shape<_8, Int<BK>>, Stride<Int<BK>, _1>>{}));
  using SmemLayoutA = decltype(tile_to_shape(SmemAtom{}, Shape<Int<BM>, Int<BK>, Int<Stages>>{}));
  using SmemLayoutB = decltype(tile_to_shape(SmemAtom{}, Shape<Int<BN>, Int<BK>, Int<Stages>>{}));

  static constexpr int kChunks = BK / 16;  // 16-byte chunks per tile row
  static constexpr int kCopyRows = kThreads / kChunks;
  static_assert(BM % kCopyRows == 0 && BN % kCopyRows == 0, "copy layout does not tile");
  using G2S = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>, T>{},
      Layout<Shape<Int<kCopyRows>, Int<kChunks>>, Stride<Int<kChunks>, _1>>{},
      Layout<Shape<_1, _16>>{}));
  using S2R = Copy_Atom<SM75_U32x4_LDSM_N, T>;

  static constexpr int kSmemBytes =
      int((cosize_v<SmemLayoutA> + cosize_v<SmemLayoutB>) * sizeof(T));
};

template <class Cfg, bool kSplitK>
__global__ void __launch_bounds__(Cfg::kThreads, Cfg::MinBlocks) gemm_kernel(KernelParams kp) {
  using T = typename Cfg::T;
  constexpr int BM = Cfg::BM, BN = Cfg::BN, BK = Cfg::BK, kStages = Cfg::Stages;
  const GemmProblem& p = kp.p;

  // ---- which tile and which K range
  const int tile = blockIdx.x;
  const int group_span = kp.group_m * kp.tiles_n;
  const int first_m = (tile / group_span) * kp.group_m;
  const int group_rows = min(kp.tiles_m - first_m, kp.group_m);
  const int tm = first_m + (tile % group_span) % group_rows;
  const int tn = (tile % group_span) / group_rows;
  const int split = blockIdx.y;
  const int kt_begin = split * kp.k_tiles_per_split;
  const int kt_end = min(kp.k_tiles, kt_begin + kp.k_tiles_per_split);
  const int m0 = tm * BM, n0 = tn * BN;
  const int m_rem = p.M - m0, n_rem = p.N - n0;

  Tensor mA = make_tensor(make_gmem_ptr(static_cast<const T*>(p.a)), make_shape(p.M, p.K),
                          make_stride(p.lda, _1{}));
  Tensor mB = make_tensor(make_gmem_ptr(static_cast<const T*>(p.b)), make_shape(p.N, p.K),
                          make_stride(p.ldb, _1{}));
  Tensor gA = local_tile(mA, make_shape(Int<BM>{}, Int<BK>{}), make_coord(tm, _));  // (BM,BK,k)
  Tensor gB = local_tile(mB, make_shape(Int<BN>{}, Int<BK>{}), make_coord(tn, _));  // (BN,BK,k)

  extern __shared__ __align__(128) uint8_t smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);
  Tensor sA = make_tensor(make_smem_ptr(smem), typename Cfg::SmemLayoutA{});  // (BM,BK,S)
  Tensor sB = make_tensor(make_smem_ptr(smem + cosize_v<typename Cfg::SmemLayoutA>),
                          typename Cfg::SmemLayoutB{});  // (BN,BK,S)

  // ---- global -> shared, predicated per 16-byte chunk
  typename Cfg::G2S g2s;
  auto thr_g2s = g2s.get_slice(threadIdx.x);
  Tensor tAgA = thr_g2s.partition_S(gA);  // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_g2s.partition_D(sA);  // (CPY,CPY_M,CPY_K,S)
  Tensor tBgB = thr_g2s.partition_S(gB);
  Tensor tBsB = thr_g2s.partition_D(sB);
  Tensor tAcA = thr_g2s.partition_S(make_identity_tensor(make_shape(Int<BM>{}, Int<BK>{})));
  Tensor tBcB = thr_g2s.partition_S(make_identity_tensor(make_shape(Int<BN>{}, Int<BK>{})));

  auto load_tile = [&](int kt, int stage) {
    const int k_rem = p.K - kt * BK;
    CUTE_UNROLL
    for (int i = 0; i < size<1>(tAsA); ++i) {
      CUTE_UNROLL
      for (int j = 0; j < size<2>(tAsA); ++j) {
        bool pred = get<0>(tAcA(0, i, j)) < m_rem && get<1>(tAcA(0, i, j)) < k_rem;
        copy(g2s.with(pred), tAgA(_, i, j, kt), tAsA(_, i, j, stage));
      }
    }
    CUTE_UNROLL
    for (int i = 0; i < size<1>(tBsB); ++i) {
      CUTE_UNROLL
      for (int j = 0; j < size<2>(tBsB); ++j) {
        bool pred = get<0>(tBcB(0, i, j)) < n_rem && get<1>(tBcB(0, i, j)) < k_rem;
        copy(g2s.with(pred), tBgB(_, i, j, kt), tBsB(_, i, j, stage));
      }
    }
  };

  // ---- MMA partitioning
  typename Cfg::TiledMma mma;
  auto thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));  // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));  // (MMA,MMA_N,MMA_K)
  Tensor acc = partition_fragment_C(mma, Shape<Int<BM>, Int<BN>>{});  // (MMA,MMA_M,MMA_N) fp32
  Tensor part = make_fragment_like(acc);
  Tensor zero = make_fragment_like(acc(_, 0, 0));  // C operand of step-promoted MMAs
  clear(zero);
  Tensor zeros_c = make_fragment_like(acc);  // C operand of a K tile's first MMAs
  clear(zeros_c);
  Tensor tCcC = thr_mma.partition_C(make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{})));
  clear(acc);
  clear(part);

  auto s2r_a = make_tiled_copy_A(typename Cfg::S2R{}, mma);
  auto thr_s2r_a = s2r_a.get_slice(threadIdx.x);
  Tensor tXsA = thr_s2r_a.partition_S(sA);  // (CPY,MMA_M,MMA_K,S)
  Tensor tXrA = thr_s2r_a.retile_D(tCrA);
  auto s2r_b = make_tiled_copy_B(typename Cfg::S2R{}, mma);
  auto thr_s2r_b = s2r_b.get_slice(threadIdx.x);
  Tensor tXsB = thr_s2r_b.partition_S(sB);  // (CPY,MMA_N,MMA_K,S)
  Tensor tXrB = thr_s2r_b.retile_D(tCrB);

  constexpr int kVals = decltype(size<0>(acc))::value;
  constexpr int kMmaM = decltype(size<1>(acc))::value;
  constexpr int kMmaN = decltype(size<2>(acc))::value;
  constexpr int kBlocks = decltype(size<2>(tCrA))::value;  // k32 steps per K tile

  // Per-row activation-scale offsets of this thread's accumulator rows (the
  // rows do not depend on n; values v and v + 1 share a row), -1 past M.
  static_assert(kVals == 4, "m16n8 accumulator: (2 columns) x (2 row halves)");
  int sa_off[2][kMmaM];
  CUTE_UNROLL
  for (int m = 0; m < kMmaM; ++m) {
    CUTE_UNROLL
    for (int h = 0; h < 2; ++h) {
      int r = get<0>(tCcC(2 * h, m, 0));
      sa_off[h][m] = r < m_rem ? int((m0 + r) * p.sa_m) : -1;
    }
  }
  const float* sb_col = p.sb + (n0 / 128) * p.sb_n;
  auto load_scales = [&](int kt, float (&s)[2][kMmaM]) {
    const float sb = __ldg(sb_col + kt * p.sb_k);
    const float* sa = p.sa + kt * p.sa_k;
    CUTE_UNROLL
    for (int m = 0; m < kMmaM; ++m) {
      CUTE_UNROLL
      for (int h = 0; h < 2; ++h) s[h][m] = sa_off[h][m] >= 0 ? __ldg(sa + sa_off[h][m]) * sb : 0.f;
    }
  };

  // ---- prologue: fill Stages - 1 shared stages
  int kt_load = kt_begin;
  CUTE_UNROLL
  for (int s = 0; s < kStages - 1; ++s) {
    if (kt_load < kt_end) load_tile(kt_load, s);
    cp_async_fence();
    ++kt_load;
  }
  int smem_read = 0, smem_write = kStages - 1;
  Tensor tXsA_p = tXsA(_, _, _, smem_read);
  Tensor tXsB_p = tXsB(_, _, _, smem_read);
  cp_async_wait<kStages - 2>();
  __syncthreads();
  copy(typename Cfg::S2R{}, tXsA_p(_, _, Int<0>{}), tXrA(_, _, Int<0>{}));
  copy(typename Cfg::S2R{}, tXsB_p(_, _, Int<0>{}), tXrB(_, _, Int<0>{}));

  // ---- main loop: one iteration per 128-wide K tile (one scale block).
  // Scales are loaded one tile ahead so their latency is never exposed.
  float scale[2][kMmaM], scale_next[2][kMmaM];
  if (kt_begin < kt_end) load_scales(kt_begin, scale_next);
  CUTE_NO_UNROLL
  for (int kt = kt_begin; kt < kt_end; ++kt) {
    CUTE_UNROLL
    for (int m = 0; m < kMmaM; ++m) scale[0][m] = scale_next[0][m], scale[1][m] = scale_next[1][m];
    if (kt + 1 < kt_end) load_scales(kt + 1, scale_next);

    CUTE_UNROLL
    for (int kb = 0; kb < kBlocks; ++kb) {
      if (kb == kBlocks - 1) {
        tXsA_p = tXsA(_, _, _, smem_read);
        tXsB_p = tXsB(_, _, _, smem_read);
        cp_async_wait<kStages - 2>();
        __syncthreads();
      }
      const int kb_next = (kb + 1) % kBlocks;
      copy(typename Cfg::S2R{}, tXsA_p(_, _, kb_next), tXrA(_, _, kb_next));
      copy(typename Cfg::S2R{}, tXsB_p(_, _, kb_next), tXrB(_, _, kb_next));
      if (kb == 0) {
        if (kt_load < kt_end) load_tile(kt_load, smem_write);
        cp_async_fence();
        ++kt_load;
        smem_write = smem_read;
        smem_read = smem_read == kStages - 1 ? 0 : smem_read + 1;
      }
      if constexpr (!Cfg::StepPromote) {
        // The first k32 step starts from zero (C = 0) instead of clearing part.
        if (kb == 0)
          gemm(mma, part, tCrA(_, _, kb), tCrB(_, _, kb), zeros_c);
        else
          gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), part);
      } else {
        CUTE_UNROLL
        for (int n = 0; n < kMmaN; ++n) {
          CUTE_UNROLL
          for (int m = 0; m < kMmaM; ++m) {
            Tensor step = make_fragment_like(zero);
            gemm(mma, step, tCrA(_, m, kb), tCrB(_, n, kb), zero);
            CUTE_UNROLL
            for (int v = 0; v < kVals; ++v)
              acc(v, m, n) = fmaf(step(v), scale[v / 2][m], acc(v, m, n));
          }
        }
      }
    }

    if constexpr (!Cfg::StepPromote) {
      // Promote: one FMA per accumulator per scale block.
      CUTE_UNROLL
      for (int n = 0; n < kMmaN; ++n) {
        CUTE_UNROLL
        for (int m = 0; m < kMmaM; ++m) {
          CUTE_UNROLL
          for (int v = 0; v < kVals; ++v) {
            acc(v, m, n) = fmaf(part(v, m, n), scale[v / 2][m], acc(v, m, n));
          }
        }
      }
    }
  }

  // ---- epilogue. Values v and v + 1 of the m16n8 accumulator are adjacent columns.
  static_assert(kVals % 2 == 0, "unexpected accumulator layout");
  auto c_ptr = [&](int r, int c) {
    return static_cast<__nv_bfloat16*>(p.c) + int64_t(m0 + r) * p.ldc + (n0 + c);
  };

  if constexpr (!kSplitK) {
    CUTE_UNROLL
    for (int n = 0; n < kMmaN; ++n)
      CUTE_UNROLL
      for (int m = 0; m < kMmaM; ++m)
        CUTE_UNROLL
        for (int v = 0; v < kVals; v += 2) {
          int r = get<0>(tCcC(v, m, n)), c = get<1>(tCcC(v, m, n));
          if (r < m_rem && c < n_rem)
            *reinterpret_cast<__nv_bfloat162*>(c_ptr(r, c)) =
                __floats2bfloat162_rn(acc(v, m, n), acc(v + 1, m, n));
        }
  } else {
    const int tiles = kp.tiles_m * kp.tiles_n;
    auto ws = [&](int s, int r, int c) {
      return kp.workspace + (int64_t(s) * tiles + tile) * (BM * BN) + r * BN + c;
    };
    CUTE_UNROLL
    for (int n = 0; n < kMmaN; ++n)
      CUTE_UNROLL
      for (int m = 0; m < kMmaM; ++m)
        CUTE_UNROLL
        for (int v = 0; v < kVals; v += 2) {
          int r = get<0>(tCcC(v, m, n)), c = get<1>(tCcC(v, m, n));
          if (r < m_rem)
            __stcg(reinterpret_cast<float2*>(ws(split, r, c)),
                   make_float2(acc(v, m, n), acc(v + 1, m, n)));
        }
    __threadfence();
    __syncthreads();
    __shared__ int is_last;
    if (threadIdx.x == 0) is_last = atomicAdd(kp.counters + tile, 1) == kp.splits - 1;
    __syncthreads();
    if (!is_last) return;
    __threadfence();

    CUTE_UNROLL
    for (int n = 0; n < kMmaN; ++n)
      CUTE_UNROLL
      for (int m = 0; m < kMmaM; ++m)
        CUTE_UNROLL
        for (int v = 0; v < kVals; v += 2) {
          int r = get<0>(tCcC(v, m, n)), c = get<1>(tCcC(v, m, n));
          if (r >= m_rem || c >= n_rem) continue;
          // Fixed order 0, 1, ..., splits - 1, whichever CTA arrived last.
          float x = 0.f, y = 0.f;
          for (int s = 0; s < kp.splits; ++s) {
            float2 t = s == split ? make_float2(acc(v, m, n), acc(v + 1, m, n))
                                  : __ldcg(reinterpret_cast<const float2*>(ws(s, r, c)));
            x += t.x;
            y += t.y;
          }
          *reinterpret_cast<__nv_bfloat162*>(c_ptr(r, c)) = __floats2bfloat162_rn(x, y);
        }
    if (threadIdx.x == 0) kp.counters[tile] = 0;  // ready for the next launch
  }
}

}  // namespace blockfp8
