// SectionPlan -- an ordered, mode-independent description of one
// training example's sequential structure.
//
// The generator has two orthogonal axes:
//
//   (1) Sequential axis:  which sections exist, in what order, and how
//                         many sequence positions each takes. This
//                         depends on the task, scratchpad, and a few
//                         config flags (no_graph, num_thinking_tokens,
//                         ...).  Lives HERE.
//
//   (2) Annotation axis:  for each sequence position, how many content
//                         columns and positional columns exist and what
//                         fills them. This depends on TokenizationMode.
//                         Lives in tokenization_mode.h + the tokenizer.
//
// A SectionPlan is a std::vector<Section> produced by make_section_plan.
// It carries only *semantic counts* -- number of edges, path length,
// visit_len, scratchpad content length -- not sequence positions. The
// position cost per section is decided later by num_positions() using
// the TokenizationMode. Both Layout::from and Tokenizer::tokenize_into_row
// consume the same plan; that keeps them from drifting.
//
// Adding a new task kind: add cases to append_query_sections /
// append_target_sections. Adding a new tokenization mode: add a
// case to num_positions and the mode-dispatched writers in the
// tokenizer. Adding a new scratchpad kind or style: add cases to
// append_scratchpad_sections. No cross-product helpers.

#ifndef GRAPHGEN_SECTION_PLAN_H
#define GRAPHGEN_SECTION_PLAN_H

#include <vector>

#include "graphgen/tokenization_mode.h"

namespace graphgen {

class SampledGraph;
struct Task;
struct Scratchpad;
struct GeneratorConfig;

// Names of every semantic block that can appear in a training example.
// Content sections carry a semantic count; marker sections are single-
// token boundaries and ignore count. Every content section is bracketed
// by a matching Start/End marker pair (Query{Start,End},
// Scratchpad{Start,End}, Task{Start,End}). BOS/EOS are the sole outer
// sentinels, unpaired with any content section.
enum class SectionKind {
    // Outer sentinels.
    BOS,
    EOS,
    // Query section brackets.
    QueryStart,
    QueryEnd,
    // Thinking section brackets. When present, `num_thinking_tokens`
    // repeated TOK_THINK tokens go between them, sitting at the start
    // of the generation region (before scratchpad if present, before
    // target otherwise).
    ThinkStart,
    ThinkEnd,
    // Scratchpad section brackets.
    ScratchpadStart,
    ScratchpadEnd,
    // Target/task section brackets. TaskStart marks the start of the
    // target itself, NOT the start of generation as a whole -- the
    // Python side handles "start generating at the first of
    // ThinkStart / ScratchpadStart / TaskStart that exists".
    TaskStart,
    TaskEnd,
    // Content sections -- count is a semantic unit count (edges,
    // vertices, thinking tokens, scratchpad tokens, ...) that
    // num_positions() turns into sequence positions given the current
    // TokenizationMode.
    GraphNodes,   // count = number of vertex tokens (all n vertices)
    GraphEdges,   // count = number of edges
    Query,        // count = number of query tokens (vertices)
    Thinking,     // count = number of TOK_THINK tokens (all identical)
    Scratchpad,   // count = number of scratchpad content tokens
    Target,       // count = number of target tokens (path vertices, visit order, ...)
    KhopsPrefix,  // count = length of the pre-query khops prefix (raw vocab tokens)
};

// One entry in the ordered plan. `count` is only meaningful for
// content sections; markers set it to 0. `Section` is deliberately
// tiny so plans can be built freely on the stack per item.
struct Section {
    SectionKind kind;
    int         count = 0;
};

using SectionPlan = std::vector<Section>;

// Compose the plan from the semantic ingredients. Pure sequential-axis
// logic: no mode, no token ids, no columns.
SectionPlan make_section_plan(const SampledGraph& sg,
                              const Task& task,
                              const Scratchpad& scratchpad,
                              const GeneratorConfig& cfg);

// How many sequence positions does this section occupy under the given
// mode? Markers are always 1. Content sections are `count` positions
// except for GraphEdges under SEAN, where each edge takes 3 positions
// (u, v, EDGE).
int num_positions(const Section& s, TokenizationMode mode);

// Number of content columns per sequence position (struct_dim). SEAN
// is 1, STAN is 3.
int struct_dim_for(TokenizationMode mode);

}  // namespace graphgen

#endif  // GRAPHGEN_SECTION_PLAN_H
