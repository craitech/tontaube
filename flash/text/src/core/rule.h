#pragma once
// Everything a rule module needs. A rule never edits the text: it proposes candidates (a byte span of the
// original text, the words for it, a score), and the engine keeps the best non-overlapping ones.
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/doc.h"
#include "core/readers.h"
#include "core/utf8.h"

namespace rv {

struct RuleDef;

struct Candidate {
    uint32_t start, end;  // byte span of the (prepared) text; start == end inserts spoken at that position
    int score;
    const RuleDef* rule;
    std::string spoken;   // "" deletes the span; the original text keeps (protects) it
    bool keep_case = false;  // no capital at a sentence start (identifiers, addresses, flags)
};

class Out {
public:
    Out(const Doc& doc, std::vector<Candidate>& sink) : doc_(doc), sink_(sink) {}
    // Propose spoken text for bytes [start, end), or insert it at start when start == end. Aborts on an
    // out-of-range span or an empty insertion (a rule bug).
    // keep_case: the reading keeps its first letter as written at a sentence start.
    void add(uint32_t start, uint32_t end, std::string spoken, int score, bool keep_case = false);
    // Propose spoken text for tokens first..last (inclusive).
    void tokens(size_t first, size_t last, std::string spoken, int score, bool keep_case = false) {
        add(doc_[first].start, doc_[last].end, std::move(spoken), score, keep_case);
    }
    // Protect tokens first..last: they stay as written and no lower-scored reading can take them.
    void keep(size_t first, size_t last, int score) {
        add(doc_[first].start, doc_[last].end, std::string(doc_.slice(doc_[first].start, doc_[last].end)), score);
    }
    const RuleDef* rule = nullptr;

private:
    const Doc& doc_;
    std::vector<Candidate>& sink_;
};

using RuleFn = void (*)(const Doc&, Out&);

struct RuleDef {
    std::string name;  // "module.function"
    std::string module;
    RuleFn fn;
    int order;         // position in rules(): module order, then order of definition
};

int register_rule(const char* module, const char* name, RuleFn fn);
const std::vector<const RuleDef*>& rules();  // every registered rule, in order

// Defines and registers a rule:  RV_RULE("money", amount) { for (...) out.tokens(i, j, words, 50); }
#define RV_RULE(MODULE, NAME)                                                                   \
    static void NAME(const ::rv::Doc& doc, ::rv::Out& out);                                     \
    [[maybe_unused]] static const int NAME##_registered = ::rv::register_rule(MODULE, #NAME, NAME); \
    static void NAME([[maybe_unused]] const ::rv::Doc& doc, [[maybe_unused]] ::rv::Out& out)

// A fixed set of words, looked up by binary search. Build it once as a static const; pass lower-case words
// and look up with doc.low(i) for case-insensitive matching.
class WordSet {
public:
    WordSet(std::initializer_list<std::string_view> words) : words_(words) {
        std::sort(words_.begin(), words_.end());
    }
    bool contains(std::string_view w) const { return std::binary_search(words_.begin(), words_.end(), w); }
    const std::vector<std::string_view>& words() const { return words_; }

private:
    std::vector<std::string_view> words_;
};

// A fixed map from written form to spoken form; find() returns nullptr if absent.
class PhraseMap {
public:
    PhraseMap(std::initializer_list<std::pair<std::string_view, std::string_view>> items) : items_(items) {
        std::sort(items_.begin(), items_.end());
    }
    const std::string_view* find(std::string_view key) const {
        auto it = std::lower_bound(items_.begin(), items_.end(), key,
                                   [](const auto& a, std::string_view k) { return a.first < k; });
        return it != items_.end() && it->first == key ? &it->second : nullptr;
    }
    const std::vector<std::pair<std::string_view, std::string_view>>& items() const { return items_; }

private:
    std::vector<std::pair<std::string_view, std::string_view>> items_;
};

}  // namespace rv
