// Task -- ground truth for one training example, produced by TaskSampler.
//
// Bundles the enum (which task kind this is) with the flat data class
// that carries the sample. Downstream stages read the fields the
// current `kind` populates and ignore the rest.
//
// A Task has two sides:
//   * query   -- the model's *input*: what we're asking it. E.g. for
//                shortest_path, the (start, end) pair.
//   * target  -- the model's *output*: the correct answer(s). E.g. for
//                shortest_path, the sequence of vertices making up the
//                path, plus per-step alternatives so the loss can spread
//                mass across equally-valid answers.
//
// Every field is std::optional. A given TaskKind populates the subset
// it cares about; the tokenizer reads based on task.kind. Overlap
// across kinds is the norm (multiple tasks emit a `path`, multiple use
// `start_node`), so a flat struct with optionals wins over a variant
// or an inheritance hierarchy. Upgrade to std::variant if a task ever
// shows up with a completely disjoint field set.

#ifndef GRAPHGEN_TASK_H
#define GRAPHGEN_TASK_H

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace graphgen {

// Which task the current sample is for. Only ShortestPath is wired up
// at this stage; the rest are enum entries so the TaskSampler switch
// can be exhaustive and per-kind stubs throw.
enum class TaskKind {
    None,
    ShortestPath,
    BFS,
    Center,
    Centroid,
    Khops,
    KhopsGen,
};

std::string_view to_string(TaskKind t);

// Throws std::invalid_argument on unknown names.
TaskKind task_kind_from_string(std::string_view s);

struct Task {
    TaskKind kind = TaskKind::None;

    // ---- Query (source side / model input) ------------------------------
    struct Query {
        // Path-oriented tasks (shortest_path, bfs): start and (for SP)
        // end endpoints of the walk.
        std::optional<int> start_node;
        std::optional<int> end_node;

        // Set-oriented tasks (center, centroid): the input query is
        // an unordered SET of vertices Q, and the target asks for the
        // vertex/vertices with minimum aggregate distance to Q. Left
        // unset for path-oriented tasks.
        std::optional<std::vector<int>> vertex_set;

        // K-hops tasks (khops_gen for now): random sequence of raw
        // vocab token ids that the model reads BEFORE the query
        // marker. The last element is the "cursor" the model is asked
        // to backtrack from; the k-hops ground truth is derived by
        // walking k steps back through the prefix. Emitted as the
        // `KhopsPrefix` content section; the query section emits just
        // the cursor. Tokens are raw vocab ids (already sampled from
        // the node-vocab range), NOT internal vertex indices.
        std::optional<std::vector<int>> prefix;

        // K-hops (per-position variant): the sampled k value. The
        // tokenizer looks up the "D<k>" marker in ctx.token_dict and
        // emits it as the sole query token. Requires the token
        // dictionary to include extras of the form "D1", "D2", ...
        // covering the max_khops range.
        std::optional<int> khops_k;
    } query;

    // ---- Target (target side / model output) ----------------------------
    struct Target {
        // Sequence of vertex ids from start to end. Length is path
        // length + 1 (endpoints included).
        std::optional<std::vector<int>> path;

        // For each step i in [0, path->size() - 1), the set of vertices
        // that are *equally correct* at position i+1 -- i.e. every
        // vertex y such that some start->end shortest path passes
        // through path[i] -> y. This is what the loss uses to spread
        // label mass across equally-valid shortest paths; without it,
        // the model gets penalised for producing a different but
        // equally-shortest answer than the one we happened to sample.
        //
        // The chosen path[i+1] is always one element of
        // valid_next_hops[i], not excluded from it.
        std::optional<std::vector<std::vector<int>>> valid_next_hops;

        // Set-oriented tasks (center, centroid): unordered set of
        // vertices tied at the minimum aggregate distance to the
        // query set. Multiple entries when several vertices tie.
        std::optional<std::vector<int>> vertex_set;

        // K-hops (per-position variant): for each seq position i in
        // [0, |target.path|), the ordered list of "chosen + valid
        // alternate" labels that populate the targets tensor row at
        // that position. Entry [0] is what the model should emit
        // (typically hops[k-1][i], the k-th back-hop target); further
        // entries carry intermediate-hop labels when
        // cfg.intermediate_labels is set. An EMPTY inner vector marks
        // "this seq position is masked" -- fill_targets writes PAD in
        // all label slots. Semantics -- including the deduplicating
        // walk over intermediate hops and the mask_to_size /
        // mask_to_vocab_size rules -- are baked in at sample time so
        // the tokenizer stays task-agnostic.
        std::optional<std::vector<std::vector<int>>> per_position_labels;
    } target;
};

}  // namespace graphgen

#endif  // GRAPHGEN_TASK_H
