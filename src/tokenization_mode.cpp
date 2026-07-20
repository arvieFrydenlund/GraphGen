#include "graphgen/tokenization_mode.h"

#include <stdexcept>
#include <string>

namespace graphgen {

std::string_view to_string(TokenizationMode m) {
    switch (m) {
        case TokenizationMode::Sean: return "sean";
        case TokenizationMode::Stan: return "stan";
    }
    return "<invalid>";
}

TokenizationMode tokenization_mode_from_string(std::string_view s) {
    if (s == "sean") return TokenizationMode::Sean;
    if (s == "stan") return TokenizationMode::Stan;
    throw std::invalid_argument("unknown tokenization_mode='" + std::string(s) + "'");
}

}  // namespace graphgen
