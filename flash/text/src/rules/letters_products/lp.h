#pragma once
// Shared pieces of the module "letters_products": capital-letter conventions, product names, model numbers,
// versions and standards. Tables live in tables.cpp and english.cpp, readers in common.cpp.
#include <string>
#include <string_view>

#include "core/rule.h"

namespace lp {

using rv::Doc;

// Tables (tables.cpp, english.cpp).
const std::string_view* respelling(std::string_view word);   // exact written form -> spoken ("NASA" -> "Nasa")
const std::string_view* lower_initialism(std::string_view w); // lower case initialism -> capitals ("nda" -> "NDA")
bool keep_caps(std::string_view upper);       // spelled out although it looks pronounceable ("POST", "NIST")
bool emphasis(std::string_view upper);        // short emphasis words read in lower case ("NOT", "YES")
bool log_level(std::string_view upper);       // "ERROR", "WARN": one capital, lower case when shouted
bool english(std::string_view lower);         // common English word
const rv::WordSet& english_words();           // the table english() looks up
bool name(std::string_view lower);            // four or five letters, written with one capital ("lima", "june")
bool unit_word(std::string_view w);           // unit symbols, exact case ("kWh", "GHz", "GB")
bool version_cue(std::string_view lower);     // words before a version number ("python", "release")
bool not_model_cue(std::string_view lower);   // words before a number that make it something else ("chapter")
bool count_noun(std::string_view lower);      // nouns counted by 2.9B / 1.5M / 12k ("parameters", "users")
bool model_noun(std::string_view lower);      // nouns after a model size ("model", "variant")
bool sentence_opener(std::string_view w);     // words that start a sentence after "U.K." ("It", "The")
bool shorthand(std::string_view w);           // written-only business shorthand left to other rules ("YoY")

// Readers (common.cpp).
bool pronounceable(std::string_view lower);   // letters form English-like syllables ("nasa" yes, "gpu" no)
bool english_letters(std::string_view lower); // every letter triple, word edges included, occurs in english() words
std::string two_words(const std::string& lower);  // two English words run together ("buynow" -> "buy now"), or ""
// An all-capital word no table lists, outside a shouted line: said as a word when it is a name(), or has six or
// seven letters with English syllables and English letter triples, or eight or more with English syllables. Other
// ones are spelled, as initialisms and codes are.
bool word_like(std::string_view upper);
std::string upper(std::string_view w);
std::string lower(std::string_view w);
std::string title(std::string_view w);        // "NASA" -> "Nasa"
std::string model_number(std::string_view digits);    // 7 / 52 / seven thirty-seven / fifty ninety / twelve seven hundred
std::string standard_number(std::string_view digits); // ISO: nine thousand one, eighty-six oh one, thirteen four eighty-five
std::string version_part(std::string_view digits);    // "12" twelve, "04" zero four
// A dotted version "3.12.1" read part by part: three point twelve point one. "" if not digits and dots.
std::string version_words(std::string_view written);

// Spoken form of one all-capital word by the capital-letter conventions, in a context that is not shouted:
// the respelling, the word with one capital, or the word unchanged.
std::string caps_word(std::string_view w);
// Spoken form of a word inside a name or code ("Qwen", "TontaubeV", "GPT", "int", "rc"): mixed case and
// capitals by the conventions; lower case letters that are not a word become capitals.
std::string name_word(std::string_view w);
// Mixed-case name split by its capitals ("GitLab" -> "Gitlab", "OpenAI" -> "Open AI"); "" if not handled.
std::string mixed_word(std::string_view w);

// Token helpers.
bool code_context(const Doc& d, size_t i);    // the word is part of an identifier, path, file name or handle
bool is_upper_word(std::string_view w);       // letters only, all capitals

}  // namespace lp
