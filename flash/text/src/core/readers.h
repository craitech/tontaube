#pragma once
#include <cstdint>
#include <string>
#include <string_view>

#include "core/doc.h"

// Number words shared by every module. Rules build their spoken text from these; they never call each other.
namespace rv {

std::string cardinal(uint64_t n);                     // 1180 -> one thousand one hundred eighty
std::string cardinal_digits(std::string_view digits); // any length up to 24 digits, no separators; "" if invalid
std::string ordinal(uint64_t n);                      // 23 -> twenty-third
std::string ordinal_of(std::string cardinal_words);   // "twenty-three" -> "twenty-third"
std::string digits(std::string_view s);               // "4502917" -> four five zero two nine one seven
std::string pairs(uint32_t n);                        // 100..9999 in pairs: 1962 nineteen sixty-two, 115 one fifteen,
                                                      // 2005 two thousand five, 1900 nineteen hundred
// "27,040,500" / "3.14" / "0.05" -> words (decimals digit by digit). "" for anything else: leading zeros,
// bad grouping, more than 24 digits.
std::string read_number(std::string_view written);
std::string read_number(const NumberSpan& n);         // same, from a parsed span; "" if invalid
int roman_value(std::string_view s);                  // canonical numeral in one case (XIV, xiv) -> 14; 0 if none
std::string capital(std::string s);                   // first ASCII letter upper case
// A one-letter variable (Latin or Greek) as the speech model says it best: capitals for Latin letters ('f' ->
// "F", 'X' -> "X"), "emm" and "ess" for m and s, Greek by name; "" for anything else.
std::string variable_name(uint32_t cp);

}  // namespace rv
