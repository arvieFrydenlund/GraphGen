// Tokenizer -- the compute-phase stage. Writes real token ids into
// caller-owned row buffers, using the sizes fixed by Layout in the
// precompute phase.
//
// Named `tokenize_into_row` (rather than `.run(...)` like the other
// stages) to flag its different shape: it doesn't produce a fresh data
// class, it *writes* into memory the caller allocated. Taking raw row
// pointers rather than a BatchOutputArrays lets pure-C++ tests
// exercise this code without a Python interpreter.
//
// Internally the tokenizer walks the same SectionPlan that Layout used
// (see section_plan.h). Layout advanced a counter over the plan; the
// tokenizer replays the walk and writes tokens at each position. That
// shared walk is why they can't drift.

#ifndef GRAPHGEN_TOKENIZER_H
#define GRAPHGEN_TOKENIZER_H

#include <cstdint>

#include "graphgen/section_plan.h"

namespace graphgen {

struct GeneratorConfig;
struct Layout;
class SampledGraph;
struct Scratchpad;
struct Task;
struct WorkerSharedContext;

class Tokenizer {
public:
    // Convenience form: builds the SectionPlan internally, then
    // delegates. Preferred for tests where the plan isn't in hand.
    //
    // `targets_row` points at the (max_gen_len, max_labels) block for
    // this row, pre-filled with TOK_PAD. `max_labels` is the batch's
    // label-smoothing stride; the tokenizer writes at most that many
    // alternatives per position.
    void tokenize_into_row(int32_t* src_tokens_row,
                           int32_t* targets_row,
                           int      max_labels,
                           int32_t* positions_row,
                           const SampledGraph& sg,
                           const Task& task,
                           const Scratchpad& scratchpad,
                           const Layout& layout,
                           const WorkerSharedContext& ctx,
                           const GeneratorConfig& cfg);

    // Plan-explicit form: for callers (Worker) that already
    // precomputed the plan alongside the layout and want to avoid
    // rebuilding it. `plan` must be the same one used to derive
    // `layout`; this is asserted by the seq_len check at the end.
    void tokenize_into_row(int32_t* src_tokens_row,
                           int32_t* targets_row,
                           int      max_labels,
                           int32_t* positions_row,
                           const SectionPlan& plan,
                           const SampledGraph& sg,
                           const Task& task,
                           const Scratchpad& scratchpad,
                           const Layout& layout,
                           const WorkerSharedContext& ctx,
                           const GeneratorConfig& cfg);
};

}  // namespace graphgen

#endif  // GRAPHGEN_TOKENIZER_H
