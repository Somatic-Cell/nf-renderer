// Minimal preprocessor reproduction only; not a Windows/OptiX SDK test.
#include <algorithm>
#include <limits>
#ifndef NOMINMAX
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif
int main() {
#ifdef TEST_PARENTHESIZED
    const unsigned limit = (std::numeric_limits<unsigned>::max)();
    const unsigned larger = (std::max)(1u, 2u);
#else
    const unsigned limit = std::numeric_limits<unsigned>::max();
    const unsigned larger = std::max(1u, 2u);
#endif
    return limit > larger ? 0 : 1;
}
