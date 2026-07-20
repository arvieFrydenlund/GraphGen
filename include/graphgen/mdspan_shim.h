// Mdspan namespace shim: prefer std::mdspan (C++23), fall back to the
// vendored Kokkos reference implementation.
//
// Rationale. C++23 ships std::mdspan (P0009 + friends) and every major
// toolchain we care about supports it in libc++ 18+ / libstdc++ 14+ /
// MSVC 19.40+. Codebase-wide we want to use the standard version by
// default; the Kokkos header stays vendored under third_party/mdspan/
// as a portability fallback for older toolchains and to insulate the
// build from occasional gaps in a specific libc++ release.
//
// Usage. Include this header instead of either <mdspan> or
// <mdspan/mdspan.hpp>. All types live under `graphgen::md`:
//
//     graphgen::md::mdspan<int, graphgen::md::dextents<int, 2>> m(...);
//     auto sub = graphgen::md::submdspan(m, std::pair{0, n}, ...);
//     int v = m[i, j];   // C++23 multi-arg subscript
//
// Element access is m[i, j] on both backends, so callers write
// portable code without knowing which implementation is active.

#ifndef GRAPHGEN_MDSPAN_SHIM_H
#define GRAPHGEN_MDSPAN_SHIM_H

#include <version>

#if defined(__cpp_lib_mdspan) && __cpp_lib_mdspan >= 202207L
    #include <mdspan>
    namespace graphgen {
    // Route "md" at the std implementation. std::mdspan,
    // std::dextents, std::layout_stride, std::submdspan, etc. all
    // become graphgen::md::* -- consumer code never mentions std or
    // Kokkos directly.
    namespace md = std;
    }  // namespace graphgen
#else
    #include <mdspan/mdspan.hpp>
    namespace graphgen {
    // Fallback: vendored Kokkos reference impl. Same public API shape
    // as std::mdspan (P0009); alias makes it a drop-in.
    namespace md = Kokkos;
    }  // namespace graphgen
#endif

#endif  // GRAPHGEN_MDSPAN_SHIM_H
