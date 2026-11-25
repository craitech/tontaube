#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace rv {

// Written English (UTF-8) to the words a narrator would say. Thread-safe; no state between calls.
std::string verbalize(std::string_view text);

// A piece of the spoken text and the input bytes it came from: copied text byte for byte (copied, both lengths
// equal), a reading the whole span it replaced, an inserted space an empty span.
struct Piece {
    uint32_t out_begin, out_end, in_begin, in_end;
    bool copied;
};
// verbalize(), and the map from the spoken text back to the input: pieces in output order that cover every byte.
std::string verbalize(std::string_view text, std::vector<Piece>& map);

}  // namespace rv
