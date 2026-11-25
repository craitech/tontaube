// Text front end of TontaubeV1.1 Flash as one small library: the rule verbalizer (include/, src/) and the chunker
// (chunker.h). Built by flash/gpu/frontend.py on first use.
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "chunker.h"
#include "rv/verbalizer.h"

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

extern "C" {

// text (UTF-8, n bytes) -> chunks as code points, written one after another into out (capacity cap); lens[i] = length
// of chunk i (at most max_chunks). times[0] = seconds in the verbalizer, times[1] = in the chunker. Returns the number
// of chunks, or -1 if the buffers are too small.
// first_max > 0: the first chunk is cut again to at most first_max characters (faster first audio for long texts),
// the text after it is chunked as usual.
int flash_prepare(const char *text, int64_t n, int verbalize, int first_max, int32_t *out, int64_t cap, int32_t *lens,
                   int max_chunks, double *times) {
    const double t0 = now_s();
    const std::string spoken = verbalize ? rv::verbalize(std::string_view(text, (size_t)n)) : std::string(text, (size_t)n);
    const double t1 = now_s();
    const chunker::S all = chunker::from_utf8(spoken);
    std::vector<chunker::S> cs = chunker::chunks(all);
    if (first_max > 0 && !cs.empty() && (int64_t)cs[0].size() > first_max) {
        const chunker::S clean = chunker::sanitize(all);
        const std::vector<chunker::S> head = chunker::chunks(clean, first_max, std::min<int64_t>(50, first_max / 2));
        if (head.size() > 1 && clean.compare(0, head[0].size(), head[0]) == 0) {
            cs = chunker::chunks(clean.substr(head[0].size()));
            cs.insert(cs.begin(), head[0]);
        }
    }
    times[0] = t1 - t0;
    times[1] = now_s() - t1;
    if ((int)cs.size() > max_chunks) return -1;
    int64_t o = 0;
    for (size_t i = 0; i < cs.size(); i++) {
        if (o + (int64_t)cs[i].size() > cap) return -1;
        for (char32_t c : cs[i]) out[o++] = (int32_t)c;
        lens[i] = (int32_t)cs[i].size();
    }
    return (int)cs.size();
}

// The verbalizer's output alone (for display); returns the number of bytes written (without a terminator).
int64_t flash_verbalize(const char *text, int64_t n, char *out, int64_t cap) {
    const std::string spoken = rv::verbalize(std::string_view(text, (size_t)n));
    if ((int64_t)spoken.size() > cap) return -1;
    memcpy(out, spoken.data(), spoken.size());
    return (int64_t)spoken.size();
}

}  // extern "C"
