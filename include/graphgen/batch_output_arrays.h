// Batch of tokenised graph tasks packaged as owned numpy arrays.
//
// A generate_batch call materialises one BatchOutputArrays, populates it
// row-by-row from the per-item sampled/tokenised results, and returns it to
// Python as a dict. Concrete numpy members will be added once the tokenizer
// is wired in.

#ifndef GRAPHGEN_BATCH_OUTPUT_ARRAYS_H
#define GRAPHGEN_BATCH_OUTPUT_ARRAYS_H

namespace graphgen {

struct BatchOutputArrays {
    BatchOutputArrays() = default;
};

}  // namespace graphgen

#endif  // GRAPHGEN_BATCH_OUTPUT_ARRAYS_H
