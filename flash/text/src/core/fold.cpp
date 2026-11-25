// Character folds shared by the modules: mathematical alphanumerics to plain letters, and Latin letters a
// speech model does not know to letters it knows.
#include "core/utf8.h"

namespace rv {

namespace {

// Greek of the mathematical alphabets, in their order: capitals with the theta symbol, nabla, small letters,
// partial, then the symbol forms of epsilon, theta, kappa, phi, rho, pi.
const uint16_t MATH_GREEK[58] = {
    0x391, 0x392, 0x393, 0x394, 0x395, 0x396, 0x397, 0x398, 0x399, 0x39A, 0x39B, 0x39C, 0x39D, 0x39E, 0x39F,
    0x3A0, 0x3A1, 0x3F4, 0x3A3, 0x3A4, 0x3A5, 0x3A6, 0x3A7, 0x3A8, 0x3A9, 0x2207, 0x3B1, 0x3B2, 0x3B3, 0x3B4,
    0x3B5, 0x3B6, 0x3B7, 0x3B8, 0x3B9, 0x3BA, 0x3BB, 0x3BC, 0x3BD, 0x3BE, 0x3BF, 0x3C0, 0x3C1, 0x3C2, 0x3C3,
    0x3C4, 0x3C5, 0x3C6, 0x3C7, 0x3C8, 0x3C9, 0x2202, 0x3F5, 0x3D1, 0x3F0, 0x3D5, 0x3F1, 0x3D6};

// Spellings of U+00C0-U+024F and U+1E00-U+1EFF from their Unicode decompositions: the base letter, or the
// accented letter the model knows when there is one (ő -> ö); "" where nothing changes.
const char* const LATIN_A[400] = {
    "", "", "", "A", "A", "A", "", "", "E", "", "", "E", "I", "I", "I", "I",  // U+00C0
    "D", "N", "O", "O", "O", "O", "", "", "O", "U", "U", "U", "", "Y", "Th", "ss",  // U+00D0
    "", "", "", "", "", "a", "", "", "", "", "", "", "", "", "", "",  // U+00E0
    "d", "", "", "", "", "", "", "", "o", "", "", "", "", "y", "th", "",  // U+00F0
    "A", "a", "A", "a", "A", "a", "C", "c", "C", "c", "C", "c", "C", "c", "D", "d",  // U+0100
    "D", "d", "E", "e", "E", "e", "E", "e", "E", "e", "E", "e", "G", "g", "G", "g",  // U+0110
    "G", "g", "G", "g", "H", "h", "H", "h", "I", "i", "I", "i", "I", "i", "I", "i",  // U+0120
    "I", "i", "IJ", "ij", "J", "j", "K", "k", "k", "L", "l", "L", "l", "L", "l", "L",  // U+0130
    "l", "L", "l", "N", "n", "N", "n", "N", "n", "'n", "Ng", "ng", "O", "o", "O", "o",  // U+0140
    "\xc3\x96", "\xc3\xb6", "", "", "R", "r", "R", "r", "R", "r", "S", "s", "S", "s", "S", "s",  // U+0150
    "S", "s", "T", "t", "T", "t", "T", "t", "U", "u", "U", "u", "U", "u", "U", "u",  // U+0160
    "\xc3\x9c", "\xc3\xbc", "U", "u", "W", "w", "Y", "y", "Y", "Z", "z", "Z", "z", "Z", "z", "s",  // U+0170
    "b", "B", "", "", "", "", "", "C", "c", "", "D", "", "", "", "E", "",  // U+0180
    "", "", "f", "G", "", "", "", "I", "K", "k", "l", "l", "", "N", "n", "O",  // U+0190
    "O", "o", "", "", "P", "p", "", "", "", "", "", "t", "T", "t", "T", "U",  // U+01A0
    "u", "", "V", "Y", "y", "Z", "z", "", "", "", "", "", "", "", "", "",  // U+01B0
    "", "", "", "", "DZ", "Dz", "dz", "LJ", "Lj", "lj", "NJ", "Nj", "nj", "A", "a", "I",  // U+01C0
    "i", "O", "o", "U", "u", "\xc3\x9c", "\xc3\xbc", "\xc3\x9c", "\xc3\xbc", "\xc3\x9c", "\xc3\xbc", "\xc3\x9c", "\xc3\xbc", "e", "A", "\xc3\xa4",  // U+01D0
    "A", "a", "\xc3\x86", "\xc3\xa6", "", "", "G", "g", "K", "k", "O", "o", "O", "o", "", "",  // U+01E0
    "j", "DZ", "Dz", "dz", "G", "g", "", "", "N", "n", "A", "a", "\xc3\x86", "\xc3\xa6", "O", "o",  // U+01F0
    "A", "a", "A", "a", "E", "e", "E", "e", "I", "i", "I", "i", "O", "o", "O", "o",  // U+0200
    "R", "r", "R", "r", "U", "u", "U", "u", "S", "s", "T", "t", "", "", "H", "h",  // U+0210
    "", "d", "", "", "Z", "z", "A", "a", "E", "e", "\xc3\x96", "\xc3\xb6", "O", "\xc3\xb5", "O", "o",  // U+0220
    "O", "o", "Y", "y", "l", "n", "t", "j", "db", "qp", "A", "C", "c", "L", "T", "s",  // U+0230
    "z", "", "", "B", "U", "", "E", "e", "J", "j", "", "", "R", "r", "Y", "y",  // U+0240
};
const char* const LATIN_B[256] = {
    "A", "a", "B", "b", "B", "b", "B", "b", "\xc3\x87", "\xc3\xa7", "D", "d", "D", "d", "D", "d",  // U+1E00
    "D", "d", "D", "d", "E", "e", "E", "e", "E", "e", "E", "e", "E", "e", "F", "f",  // U+1E10
    "G", "g", "H", "h", "H", "h", "H", "h", "H", "h", "H", "h", "I", "i", "I", "\xc3\xaf",  // U+1E20
    "K", "k", "K", "k", "K", "k", "L", "l", "L", "l", "L", "l", "L", "l", "M", "m",  // U+1E30
    "M", "m", "M", "m", "N", "n", "N", "n", "N", "n", "N", "n", "O", "\xc3\xb5", "O", "\xc3\xb5",  // U+1E40
    "O", "o", "O", "o", "P", "p", "P", "p", "R", "r", "R", "r", "R", "r", "R", "r",  // U+1E50
    "S", "s", "S", "s", "S", "s", "S", "s", "S", "s", "T", "t", "T", "t", "T", "t",  // U+1E60
    "T", "t", "U", "u", "U", "u", "U", "u", "U", "u", "U", "u", "V", "v", "V", "v",  // U+1E70
    "W", "w", "W", "w", "W", "w", "W", "w", "W", "w", "X", "x", "X", "x", "Y", "y",  // U+1E80
    "Z", "z", "Z", "z", "Z", "z", "h", "t", "w", "y", "a", "s", "s", "s", "SS", "d",  // U+1E90
    "A", "a", "A", "a", "\xc3\x82", "\xc3\xa2", "\xc3\x82", "\xc3\xa2", "\xc3\x82", "\xc3\xa2", "\xc3\x82", "\xc3\xa2", "A", "a", "A", "a",  // U+1EA0
    "A", "a", "A", "a", "A", "a", "A", "a", "E", "e", "E", "e", "E", "e", "\xc3\x8a", "\xc3\xaa",  // U+1EB0
    "\xc3\x8a", "\xc3\xaa", "\xc3\x8a", "\xc3\xaa", "\xc3\x8a", "\xc3\xaa", "E", "e", "I", "i", "I", "i", "O", "o", "O", "o",  // U+1EC0
    "O", "\xc3\xb4", "O", "\xc3\xb4", "O", "\xc3\xb4", "O", "\xc3\xb4", "O", "o", "O", "o", "O", "o", "O", "o",  // U+1ED0
    "O", "o", "O", "o", "U", "u", "U", "u", "U", "u", "U", "u", "U", "u", "U", "u",  // U+1EE0
    "U", "u", "Y", "y", "Y", "y", "Y", "y", "Y", "y", "LL", "ll", "V", "v", "Y", "y",  // U+1EF0
};

}  // namespace

uint32_t plain_math_alnum(uint32_t cp) {
    if (cp < 0x210A) return 0;
    if (cp >= 0x1D400 && cp <= 0x1D6A3) {
        uint32_t k = (cp - 0x1D400) % 52;
        return k < 26 ? 'A' + k : 'a' + (k - 26);
    }
    if (cp == 0x1D6A4) return 'i';
    if (cp == 0x1D6A5) return 'j';
    if (cp >= 0x1D6A8 && cp <= 0x1D7C9) return MATH_GREEK[(cp - 0x1D6A8) % 58];
    if (cp >= 0x1D7CE && cp <= 0x1D7FF) return '0' + (cp - 0x1D7CE) % 10;
    switch (cp) {
        case 0x210E: return 'h';
        case 0x210A: return 'g';
        case 0x212F: return 'e';
        case 0x2134: return 'o';
        case 0x212C: return 'B';
        case 0x2130: return 'E';
        case 0x2131: return 'F';
        case 0x210B: return 'H';
        case 0x2110: return 'I';
        case 0x2112: return 'L';
        case 0x2133: return 'M';
        case 0x211B: return 'R';
        case 0x212D: return 'C';
        case 0x210C: return 'H';
        case 0x2128: return 'Z';
        default: return 0;
    }
}

const char* latin_fold(uint32_t cp) {
    const char* s = nullptr;
    if (cp >= 0xC0 && cp < 0x250) s = LATIN_A[cp - 0xC0];
    else if (cp >= 0x1E00 && cp < 0x1F00) s = LATIN_B[cp - 0x1E00];
    return s && *s ? s : nullptr;
}

}  // namespace rv
