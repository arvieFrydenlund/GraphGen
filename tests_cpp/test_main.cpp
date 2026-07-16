// doctest test-runner entry point. Keeps the main definition in a single
// translation unit so per-test files can `#include "doctest.h"` without
// each one dragging in its own main().

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
