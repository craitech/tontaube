#include <cstdio>
#include <cstdlib>
#include <memory>

#include "core/rule.h"

namespace rv {

namespace {

// The last tie-break between equal candidates: module order, then order of definition in the module.
const char* const MODULE_ORDER[] = {"core",         "numbers_core",    "numbers_context", "money",
                                    "datetime",     "units_symbols",   "roman_citations", "abbreviations",
                                    "letters_products", "electronic",  "structure",       "math",
                                    "chars"};

std::vector<std::unique_ptr<RuleDef>>& registered() {
    static std::vector<std::unique_ptr<RuleDef>> defs;
    return defs;
}

int module_rank(const std::string& m) {
    int n = static_cast<int>(sizeof(MODULE_ORDER) / sizeof(MODULE_ORDER[0]));
    for (int k = 0; k < n; ++k)
        if (m == MODULE_ORDER[k]) return k;
    return n;
}

}  // namespace

int register_rule(const char* module, const char* name, RuleFn fn) {
    auto& defs = registered();
    std::string full = std::string(module) + "." + name;
    for (const auto& d : defs)
        if (d->name == full) {
            std::fprintf(stderr, "rule %s is registered twice\n", full.c_str());
            std::abort();
        }
    defs.push_back(std::make_unique<RuleDef>(RuleDef{full, module, fn, static_cast<int>(defs.size())}));
    return static_cast<int>(defs.size());
}

const std::vector<const RuleDef*>& rules() {
    static const std::vector<const RuleDef*> sorted = [] {
        std::vector<const RuleDef*> v;
        for (const auto& d : registered()) v.push_back(d.get());
        std::stable_sort(v.begin(), v.end(), [](const RuleDef* a, const RuleDef* b) {
            return module_rank(a->module) < module_rank(b->module);
        });
        for (size_t k = 0; k < v.size(); ++k) const_cast<RuleDef*>(v[k])->order = static_cast<int>(k);
        return v;
    }();
    return sorted;
}

void Out::add(uint32_t start, uint32_t end, std::string spoken, int score, bool keep_case) {
    bool insertion = start == end && !spoken.empty();
    if (!((start < end || insertion) && end <= doc_.text.size())) {
        std::fprintf(stderr, "rule %s proposed the bad span %u:%u (text has %zu bytes)\n",
                     rule ? rule->name.c_str() : "?", start, end, doc_.text.size());
        std::abort();
    }
    sink_.push_back(Candidate{start, end, score, rule, std::move(spoken), keep_case});
}

}  // namespace rv
