// Runtime assertion helpers.
//
// GG_CHECK(cond, msg) fires in every build. It throws std::runtime_error
// carrying the source location and a human-readable message when `cond`
// evaluates false. Use it to guard API surfaces, decode input, and defend
// against programmer error that should never appear in a shipped run.
//
// GG_ASSERT(cond, msg) is the debug-only variant. It compiles down to a
// no-op when NDEBUG is defined. Reserve it for hot-path checks whose per-
// iteration cost would otherwise be unacceptable.

#ifndef GRAPHGEN_CHECK_H
#define GRAPHGEN_CHECK_H

#include <sstream>
#include <stdexcept>

// Graph Generation (GG) CHECK / ASSERT helpers. See the file-level comment for usage guidance.
#define GG_CHECK(cond, msg)                                                    \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::ostringstream _gg_oss;                                         \
            _gg_oss << "GG_CHECK failed at " << __FILE__ << ":" << __LINE__     \
                    << ": " << #cond << " -- " << msg;                          \
            throw std::runtime_error(_gg_oss.str());                            \
        }                                                                       \
    } while (0)

#ifdef NDEBUG
#define GG_ASSERT(cond, msg) ((void)0)
#else
#define GG_ASSERT(cond, msg) GG_CHECK(cond, msg)
#endif

#endif  // GRAPHGEN_CHECK_H
