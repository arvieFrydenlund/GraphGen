// Which tokenization scheme the pipeline emits for one training example.
// Every enum entry is a *named combination* of choices: how many content
// columns per sequence position, how many positional-embedding columns,
// which columns hold role invariants vs raw positional ids, and how
// edges are laid out on the sequence axis. Consumers dispatch on the
// enum rather than composing orthogonal knobs so invalid combinations
// are unrepresentable.
//
//   Sean  -- one content column per sequence position; edges laid out
//            as `u v EDGE` (3 sequence positions per edge). Output
//            tensor is 2D `[batch, seq_len]`. Default.
//
//   Stan  -- edges packed as a single sequence position with content
//            columns holding `(u, v, EDGE)` across struct_dim=3. Other
//            sequence positions pad the extra columns. Output tensor
//            is 3D `[batch, seq_len, struct_dim]`.
//
// Future variants (extra column layouts, alternative invariant/pos
// mixes, weighted-edge columns, ...) land as new enum entries -- one
// name per fully-specified scheme -- rather than as additional knobs.

#ifndef GRAPHGEN_TOKENIZATION_MODE_H
#define GRAPHGEN_TOKENIZATION_MODE_H

#include <string>
#include <string_view>

namespace graphgen {

enum class TokenizationMode {
    Sean,
    Stan,
};

// Canonical lowercase form used on GeneratorConfig.tokenization_mode
// and in error messages.
std::string_view to_string(TokenizationMode m);

// Throws std::invalid_argument on unknown names.
TokenizationMode tokenization_mode_from_string(std::string_view s);

}  // namespace graphgen

#endif  // GRAPHGEN_TOKENIZATION_MODE_H
