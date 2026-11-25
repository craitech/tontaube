// Rule of the module "math": a formula the page extractor marked, U+FDD0 TeX U+FDD1, read in words. The
// tokenizer makes the whole formula one token (no other rule sees its source); the reading covers both markers
// and outranks everything (score 100), so the markers never reach the output. Texts without the markers' lead
// byte pay one memchr.
#include <cstring>

#include "core/rule.h"
#include "rules/math/math.h"

namespace {

std::string reading(std::string_view tex) {
    if (tex.find_first_not_of(" \t\r") == std::string_view::npos) return "";
    std::string words = mth::speak(mth::parse(tex));
    if (words.empty()) return "a formula";  // nothing readable: a neutral phrase, never raw TeX
    // the sentence mark that ends the source ("x = 1," in a display) ends the reading
    size_t e = tex.find_last_not_of(" \t\r");
    char m = tex[e];
    if ((m == '.' || m == ',' || m == ';' || m == ':') && !(e > 0 && tex[e - 1] == '\\')) {
        while (!words.empty() && (words.back() == '.' || words.back() == ',' || words.back() == ';' ||
                                  words.back() == ':'))
            words.pop_back();
        words += m;
    }
    return words;
}

RV_RULE("math", formula) {
    std::string_view t = doc.text;
    if (!std::memchr(t.data(), '\xEF', t.size())) return;
    for (size_t i = 0; i < doc.size(); ++i) {
        const rv::Token& tok = doc[i];
        if (tok.cp == 0xFDD0) {
            std::string_view body = doc.str(i).substr(3);
            if (body.size() >= 3 && body.substr(body.size() - 3) == "\xEF\xB7\x91") body.remove_suffix(3);
            std::string words = reading(body);
            uint32_t end = tok.end;
            // an empty formula at the start of a line takes the space after it along
            if (words.empty() && (i == 0 || tok.newline_before) && i + 1 < doc.size() && !doc[i + 1].newline_before)
                end = doc[i + 1].start;
            out.add(tok.start, end, words, 100);
        } else if (tok.cp == 0xFDD1) {  // a closing marker without an opening one
            out.tokens(i, i, "", 100);
        }
    }
}

}  // namespace
