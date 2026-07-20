// Shared BFS-with-frontier-snapshot helper.
//
// Both section_plan.cpp (length computation for WithQueue scratchpad
// style) and tokenizer.cpp (actual queue-token emission) need the
// exact same FIFO-BFS trace + frontier-at-each-pop. Extracted into a
// helper so the two callers can't disagree.
//
// The BFS uses a strict FIFO queue with CSR-order tie breaking. Its
// visit order is identical to the ping-pong BFS in scratchpad_sampler
// (both are BFS with the same tie-breaking rule), but the queue state
// at each pop is only well-defined under FIFO.

#ifndef GRAPHGEN_BFS_HELPERS_H
#define GRAPHGEN_BFS_HELPERS_H

#include <vector>

namespace graphgen {

class CsrGraph;

// One pop of the FIFO BFS: which vertex we're about to dequeue, and
// the queue's contents right before that dequeue (i.e. the frontier
// the popped vertex was chosen from). `frontier[0]` is always the
// vertex being popped.
struct BFSPop {
    int              vertex;
    std::vector<int> frontier;
};

// FIFO BFS from `start`. Returns one BFSPop per popped vertex in the
// order they were popped. Visits every vertex reachable from `start`.
// Neighbour order at each expansion matches the CSR neighbour list.
std::vector<BFSPop> bfs_fifo_pops(const CsrGraph& g, int start);

}  // namespace graphgen

#endif  // GRAPHGEN_BFS_HELPERS_H
