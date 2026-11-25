#pragma once
#include <string>
#include <string_view>
#include <vector>

#include "core/rule.h"
#include "rv/verbalizer.h"

namespace rv {

struct Run {
    std::string text;            // the input after prepare(); every span refers to it
    std::vector<Candidate> candidates;
    std::vector<int> beaten_by;  // per candidate: index of the winner that beat it, -1 if it applies
    std::string spoken;
};

// Removes invisible characters and turns the spacing entities (&nbsp; &thinsp; &ensp; &emsp; &shy; &zwj; &zwnj;
// and their numeric forms) into the space or nothing they stand for. Other entities are left to the rules.
// src (optional): for each output byte the input byte it came from, then text.size().
std::string prepare(std::string_view text, std::vector<uint32_t>* src = nullptr);
// Runs the given rules on text: prepare, propose, resolve, assemble. map (optional): as verbalize() gives it.
Run run(std::string_view text, const std::vector<const RuleDef*>& active, std::vector<Piece>* map = nullptr);
// The rules of the listed modules ("money,datetime"); "core" is always included. Empty list: every rule.
std::vector<const RuleDef*> select_modules(std::string_view modules_csv);

// Where candidates overlap, the higher score wins, then the longer span, the earlier start, the rule order.
// The loser drops out whole. Candidates that do not overlap all apply. Insertions (start == end) are decided
// after all replacements: one per position, and none strictly inside a winning replacement.
void resolve(Run& r);
// The text with every winner's span replaced. A replacement gets a capital where the output so far ends a
// sentence, unless it is marked keep_case, or it is a fragment (a whole input of one or two words without closing
// punctuation, or alone on its line) whose original does not start with a capital. It gets a space where it would
// run into a letter or digit. A deletion between two spaces drops one.
// map (optional): the pieces of the result, with spans of text.
std::string assemble(std::string_view text, const Run& r, std::vector<Piece>* map = nullptr);
// One line per candidate: "+" applied, "-" lost (with the winner that beat it).
std::string describe(const Run& r);

}  // namespace rv
