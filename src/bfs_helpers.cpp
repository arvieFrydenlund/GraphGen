#include "graphgen/bfs_helpers.h"

#include <cstddef>
#include <deque>
#include <vector>

#include "graphgen/csr_graph.h"

namespace graphgen {

std::vector<BFSPop> bfs_fifo_pops(const CsrGraph& g, int start) {
    const int n = g.num_vertices();
    std::vector<BFSPop> pops;
    pops.reserve(static_cast<std::size_t>(n));

    std::vector<char> visited(static_cast<std::size_t>(n), 0);
    std::deque<int>   queue;

    visited[static_cast<std::size_t>(start)] = 1;
    queue.push_back(start);

    while (!queue.empty()) {
        // Snapshot the frontier BEFORE popping, so the front vertex
        // appears as element 0 of the recorded queue. This matches
        // the tokenized layout `{ q_0 q_1 ... q_k } v [ nbrs ]`
        // where q_0 == v.
        BFSPop pop;
        pop.frontier.assign(queue.begin(), queue.end());
        pop.vertex = queue.front();
        queue.pop_front();

        for (int nbr : g.neighbours(pop.vertex)) {
            if (!visited[static_cast<std::size_t>(nbr)]) {
                visited[static_cast<std::size_t>(nbr)] = 1;
                queue.push_back(nbr);
            }
        }
        pops.push_back(std::move(pop));
    }
    return pops;
}

}  // namespace graphgen
