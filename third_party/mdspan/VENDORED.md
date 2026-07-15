# Vendored Kokkos mdspan

Source: https://github.com/kokkos/mdspan (`stable` branch)
Vendored: 2026-07-15
License: BSD-3-Clause (see `LICENSE`)

Header-only. Use as `#include <mdspan/mdspan.hpp>` after adding
`third_party/mdspan/include` to your include path (CMakeLists.txt does this
via `target_include_directories(generator PRIVATE ...)`).

To refresh:
    curl -sSL -o /tmp/mdspan.tar.gz \
        https://github.com/kokkos/mdspan/archive/refs/heads/stable.tar.gz
    tar -xzf /tmp/mdspan.tar.gz -C /tmp
    rm -rf third_party/mdspan
    mv /tmp/mdspan-stable third_party/mdspan
    cd third_party/mdspan && rm -rf benchmarks tests examples comp_bench \
        compilation_tests scripts cmake CMakeLists.txt make_single_header.py \
        .github .gitignore .clang-tidy
