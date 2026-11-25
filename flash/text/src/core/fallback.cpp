// The fallback reading every other rule outranks: a standalone number in full.
#include "core/rule.h"

namespace {

// A number counts as standalone unless it sticks to a letter, a digit, or a '.' / ',' before it, or to a letter,
// a digit, or ".5" / ",5" after it ("k8s", "v2.0", "3rd", "1,23,456" stay for other rules or as written).
RV_RULE("core", cardinal) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i)) continue;
        rv::NumberSpan n = rv::parse_number(doc, i);
        size_t next = n.last + 1;
        bool stuck_before = doc.joined(i) && (doc[i - 1].kind != rv::Kind::Punct || doc[i - 1].cp == '.' ||
                                              doc[i - 1].cp == ',');
        bool stuck_after = false;
        if (next < doc.size() && doc.joined(next)) {
            const rv::Token& t = doc[next];
            stuck_after = t.kind != rv::Kind::Punct ||
                          ((t.cp == '.' || t.cp == ',') && doc.number(next + 1) && doc.joined(next + 1));
        }
        if (!stuck_before && !stuck_after) {
            std::string words = rv::read_number(n);
            if (!words.empty()) out.tokens(i, n.last, words, 10);
        }
        i = n.last;
    }
}

}  // namespace
