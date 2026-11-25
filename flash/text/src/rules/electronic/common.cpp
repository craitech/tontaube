#include "rules/electronic/common.h"

#include <vector>

#include "core/utf8.h"

namespace rv::el {

namespace {

const char* const ONES[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};

// Generic endings of domains (country endings are below).
const WordSet GENERIC_TLD{
    "academy", "africa", "agency", "ai", "app", "art", "asia", "auto", "bank", "beauty",
    "beer", "berlin", "bike", "bio", "biz", "blog", "boutique", "business", "cafe", "capital",
    "care", "cars", "cash", "center", "charity", "chat", "church", "city", "clinic", "cloud",
    "club", "co", "coffee", "college", "com", "community", "company", "consulting", "coop", "corp",
    "dental", "design", "dev", "digital", "doctor", "dog", "earth", "eco", "edu", "education",
    "email", "energy", "events", "example", "expert", "family", "farm", "fashion", "film", "finance",
    "fit", "fitness", "foundation", "fun", "fund", "gallery", "game", "games", "garden", "global",
    "golf", "gov", "green", "group", "guru", "health", "help", "holiday", "homes", "hospital",
    "host", "hotel", "house", "inc", "info", "institute", "insurance", "int", "internal", "io",
    "jobs", "kids", "lab", "land", "law", "legal", "life", "link", "live", "llc",
    "local", "lol", "london", "ltd", "market", "media", "mil", "mobi", "money", "movie",
    "museum", "music", "name", "net", "network", "news", "ngo", "ninja", "nyc", "online",
    "org", "page", "paris", "pet", "photo", "photography", "pizza", "place", "plus", "press",
    "pro", "realty", "rentals", "restaurant", "rocks", "run", "sale", "salon", "school", "science",
    "services", "shop", "site", "social", "software", "solar", "solutions", "space", "store", "stream",
    "studio", "style", "support", "systems", "tax", "team", "tech", "test", "tickets", "today",
    "tokyo", "tools", "tours", "travel", "university", "vacations", "ventures", "vet", "video", "website",
    "wiki", "wine", "works", "world", "xyz", "yoga", "zone"};

// Country endings; the ones that are file extensions first (main.py, notes.md, weights.pt) are left out.
const WordSet CCTLD{
    "ac", "ad", "ae", "af", "ag", "ai", "al", "am", "ao", "ar", "as", "at", "au", "aw", "az", "ba", "bb", "bd",
    "be", "bf", "bg", "bh", "bi", "bj", "bm", "bn", "bo", "br", "bs", "bt", "bw", "by", "bz", "ca", "cd", "cf",
    "cg", "ch", "ci", "ck", "cl", "cm", "cn", "co", "cr", "cu", "cv", "cx", "cy", "cz", "de", "dj", "dk", "dm",
    "do", "dz", "ec", "ee", "eg", "es", "et", "eu", "fi", "fj", "fm", "fo", "fr", "ga", "gd", "ge", "gg", "gh",
    "gi", "gl", "gm", "gr", "gt", "gy", "hk", "hn", "hr", "ht", "hu", "id", "ie", "il", "im", "in", "io", "iq",
    "ir", "is", "it", "je", "jm", "jo", "jp", "ke", "kg", "kh", "ki", "kr", "kw", "ky", "kz", "la", "lb", "lc",
    "li", "lk", "lr", "lt", "lu", "lv", "ly", "ma", "mc", "me", "mg", "mn", "mo", "mt", "mu", "mv", "mw", "mx",
    "my", "mz", "na", "ne", "ng", "ni", "nl", "no", "np", "nu", "nz", "om", "pa", "pe", "pg", "ph", "pk", "pl",
    "pr", "ps", "pw", "qa", "re", "ro", "ru", "rw", "sa", "sc", "se", "sg", "si", "sk", "sn", "so", "st", "su",
    "sv", "sx", "sz", "tc", "th", "tj", "tk", "tl", "tm", "tn", "to", "tr", "tt", "tv", "tw", "tz", "ua", "ug",
    "uk", "us", "uy", "uz", "vc", "ve", "vg", "vn", "vu", "ws", "za", "zm", "zw"};

const WordSet LABEL_UPPER{"bbc", "cnn", "fbi", "ibm", "nhs", "usa", "www"};

// File extensions and how they are said: spelled ones in capitals, words with one capital.
const PhraseMap EXT{
    {"3gp", "three GP"}, {"7z", "seven Z"}, {"aab", "AAB"},
    {"aac", "AAC"}, {"accdb", "ACCDB"}, {"adoc", "Adoc"},
    {"aiff", "AIFF"}, {"amr", "AMR"}, {"apk", "APK"},
    {"appimage", "Appimage"}, {"arrow", "Arrow"}, {"arw", "ARW"},
    {"asc", "ASC"}, {"asm", "ASM"}, {"aspx", "ASPX"},
    {"astro", "Astro"}, {"avi", "AVI"}, {"avif", "Avif"},
    {"avro", "Avro"}, {"azw3", "AZW three"}, {"bak", "BAK"},
    {"bam", "BAM"}, {"bas", "BAS"}, {"bat", "BAT"},
    {"bib", "Bib"}, {"bin", "Bin"}, {"blend", "Blend"},
    {"bmp", "BMP"}, {"bz2", "BZ two"}, {"bzl", "BZL"},
    {"c", "C"}, {"cabal", "Cabal"}, {"cbr", "CBR"},
    {"cbz", "CBZ"}, {"cc", "CC"}, {"cer", "CER"},
    {"cfg", "CFG"}, {"cif", "CIF"}, {"cjs", "CJS"},
    {"ckpt", "CKPT"}, {"clj", "CLJ"}, {"cljs", "CLJS"},
    {"cmake", "Cmake"}, {"cmd", "CMD"}, {"conf", "Conf"},
    {"cpp", "CPP"}, {"cr2", "CR two"}, {"crt", "CRT"},
    {"cs", "CS"}, {"csproj", "CS Proj"}, {"csr", "CSR"},
    {"css", "CSS"}, {"csv", "CSV"}, {"cu", "CU"},
    {"cuh", "CUH"}, {"cxx", "CXX"}, {"dart", "Dart"},
    {"dat", "DAT"}, {"db", "DB"}, {"db3", "DB three"},
    {"dbf", "DBF"}, {"dcd", "DCD"}, {"dcm", "DCM"},
    {"dds", "DDS"}, {"deb", "Deb"}, {"diff", "Diff"},
    {"djvu", "DJVU"}, {"dll", "DLL"}, {"dmg", "DMG"},
    {"dng", "DNG"}, {"doc", "DOC"}, {"dockerfile", "Dockerfile"},
    {"docm", "DOCM"}, {"docx", "DOCX"}, {"dotx", "DOTX"},
    {"dta", "DTA"}, {"duckdb", "Duckdb"}, {"dvi", "DVI"},
    {"dwg", "DWG"}, {"dxf", "DXF"}, {"dylib", "Dylib"},
    {"edf", "EDF"}, {"edr", "EDR"}, {"ejs", "EJS"},
    {"elf", "ELF"}, {"elm", "Elm"}, {"emf", "EMF"},
    {"eml", "EML"}, {"env", "ENV"}, {"eps", "EPS"},
    {"epub", "E Pub"}, {"erb", "ERB"}, {"exe", "EXE"},
    {"exr", "EXR"}, {"f90", "F ninety"}, {"fasta", "Fasta"},
    {"fastq", "Fastq"}, {"fbx", "FBX"}, {"feather", "Feather"},
    {"flac", "Flac"}, {"flatpak", "Flatpak"}, {"flv", "FLV"},
    {"fq", "FQ"}, {"fsx", "FSX"}, {"gcode", "G Code"},
    {"geojson", "Geojson"}, {"gff", "GFF"}, {"gguf", "GGUF"},
    {"gif", "GIF"}, {"glb", "GLB"}, {"gltf", "GLTF"},
    {"go", "Go"}, {"gpg", "GPG"}, {"gpkg", "GPKG"},
    {"gpx", "GPX"}, {"gql", "GQL"}, {"gradle", "Gradle"},
    {"graphql", "Graphql"}, {"gro", "GRO"}, {"groovy", "Groovy"},
    {"gtf", "GTF"}, {"gz", "GZ"}, {"h", "H"},
    {"h5", "H five"}, {"h5ad", "H five AD"}, {"hbs", "HBS"},
    {"hcl", "HCL"}, {"hdf5", "HDF five"}, {"heic", "HEIC"},
    {"heif", "HEIF"}, {"hh", "HH"}, {"hpp", "HPP"},
    {"htm", "HTM"}, {"html", "HTML"}, {"hxx", "HXX"},
    {"icns", "ICNS"}, {"ico", "ICO"}, {"ics", "ICS"},
    {"idx", "IDX"}, {"img", "IMG"}, {"indd", "INDD"},
    {"ini", "INI"}, {"inl", "INL"}, {"ipa", "IPA"},
    {"ipynb", "IPYNB"}, {"iso", "ISO"}, {"itp", "ITP"},
    {"j2", "J two"}, {"jar", "Jar"}, {"java", "Java"},
    {"jfif", "JFIF"}, {"jinja", "Jinja"}, {"jl", "JL"},
    {"joblib", "Joblib"}, {"jp2", "JP two"}, {"jpeg", "JPEG"},
    {"jpg", "JPG"}, {"js", "JS"}, {"json", "Json"},
    {"jsonl", "Jsonl"}, {"jsp", "JSP"}, {"jsx", "JSX"},
    {"jxl", "JXL"}, {"key", "Key"}, {"kml", "KML"},
    {"kmz", "KMZ"}, {"kt", "KT"}, {"kts", "KTS"},
    {"lck", "LCK"}, {"lisp", "Lisp"}, {"lmdb", "LMDB"},
    {"lnk", "LNK"}, {"lock", "Lock"}, {"log", "Log"},
    {"lua", "Lua"}, {"lz4", "LZ four"}, {"lzma", "LZMA"},
    {"m2ts", "M two TS"}, {"m3u", "M three U"}, {"m3u8", "M three U eight"},
    {"m4a", "M four A"}, {"m4v", "M four V"}, {"map", "Map"},
    {"mat", "MAT"}, {"mbox", "Mbox"}, {"md", "MD"},
    {"md5", "MD five"}, {"mdb", "MDB"}, {"mdp", "MDP"},
    {"mdx", "MDX"}, {"mgz", "MGZ"}, {"midi", "Midi"},
    {"mjs", "MJS"}, {"mk", "MK"}, {"mkv", "MKV"},
    {"mobi", "Mobi"}, {"mol", "MOL"}, {"mol2", "MOL two"},
    {"mov", "MOV"}, {"mp3", "MP three"}, {"mp4", "MP four"},
    {"mpg", "MPG"}, {"msi", "MSI"}, {"mts", "MTS"},
    {"mustache", "Mustache"}, {"nc", "NC"}, {"ndjson", "ND Json"},
    {"nef", "NEF"}, {"new", "New"}, {"nii", "NII"},
    {"nim", "Nim"}, {"ninja", "Ninja"}, {"nix", "Nix"},
    {"npy", "NPY"}, {"npz", "NPZ"}, {"nupkg", "NUPKG"},
    {"odg", "ODG"}, {"odp", "ODP"}, {"ods", "ODS"},
    {"odt", "ODT"}, {"ogg", "OGG"}, {"ogv", "OGV"},
    {"old", "Old"}, {"onnx", "Onnx"}, {"opus", "Opus"},
    {"orc", "ORC"}, {"orig", "Orig"}, {"otf", "OTF"},
    {"ova", "OVA"}, {"ovf", "OVF"}, {"p12", "P twelve"},
    {"parq", "Parq"}, {"parquet", "Parquet"}, {"part", "Part"},
    {"patch", "Patch"}, {"pb", "PB"}, {"pdb", "PDB"},
    {"pdf", "PDF"}, {"pem", "PEM"}, {"pfx", "PFX"},
    {"php", "PHP"}, {"pickle", "Pickle"}, {"pkl", "PKL"},
    {"plist", "Plist"}, {"ply", "PLY"}, {"png", "PNG"},
    {"ppm", "PPM"}, {"ppt", "PPT"}, {"pptm", "PPTM"},
    {"pptx", "PPTX"}, {"proto", "Proto"}, {"ps1", "PS one"},
    {"ps1xml", "PS one XML"}, {"psd", "PSD"}, {"psf", "PSF"},
    {"psm1", "PSM one"}, {"pt", "PT"}, {"pth", "PTH"}, {"pub", "Pub"},
    {"pug", "Pug"}, {"purs", "PURS"}, {"py", "PY"},
    {"pyc", "PYC"}, {"pyd", "PYD"}, {"pyi", "PYI"},
    {"pyx", "PYX"}, {"qcow2", "QCOW two"}, {"qmd", "QMD"},
    {"rar", "RAR"}, {"raw", "Raw"}, {"rb", "RB"},
    {"rds", "RDS"}, {"resx", "RESX"}, {"rkt", "RKT"},
    {"rmd", "RMD"}, {"rpm", "RPM"}, {"rs", "RS"},
    {"rst", "RST"}, {"rtf", "RTF"}, {"safetensors", "Safetensors"},
    {"sass", "Sass"}, {"sav", "SAV"}, {"sbt", "SBT"},
    {"sbv", "SBV"}, {"scala", "Scala"}, {"scm", "SCM"},
    {"scss", "SCSS"}, {"sdf", "SDF"}, {"sh", "SH"},
    {"sha256", "SHA two fifty-six"}, {"shp", "SHP"}, {"sln", "SLN"},
    {"sql", "SQL"}, {"sqlite", "Sqlite"}, {"sqlite3", "Sqlite three"},
    {"srt", "SRT"}, {"ssa", "SSA"}, {"stl", "STL"},
    {"stp", "STP"}, {"sty", "STY"}, {"svelte", "Svelte"},
    {"svg", "SVG"}, {"swift", "Swift"}, {"swp", "SWP"},
    {"tar", "Tar"}, {"temp", "Temp"}, {"tex", "Tex"},
    {"tfrecord", "TF Record"}, {"tfvars", "TF Vars"}, {"tga", "TGA"},
    {"tgz", "TGZ"}, {"tif", "TIF"}, {"tiff", "TIFF"},
    {"tmp", "TMP"}, {"tmpl", "Tmpl"}, {"toml", "Toml"},
    {"tpl", "TPL"}, {"tpr", "TPR"}, {"trr", "TRR"},
    {"ts", "TS"}, {"tsv", "TSV"}, {"tsx", "TSX"},
    {"ttf", "TTF"}, {"twig", "Twig"}, {"txt", "TXT"},
    {"vba", "VBA"}, {"vbs", "VBS"}, {"vcf", "VCF"},
    {"vdi", "VDI"}, {"vhd", "VHD"}, {"vhdx", "VHDX"},
    {"vmdk", "VMDK"}, {"vob", "VOB"}, {"vsdx", "VSDX"},
    {"vsix", "VSIX"}, {"vtt", "VTT"}, {"vue", "Vue"},
    {"wasm", "Wasm"}, {"wav", "WAV"}, {"webm", "Webm"},
    {"webp", "WEBP"}, {"whl", "WHL"}, {"wma", "WMA"},
    {"wmf", "WMF"}, {"wmv", "WMV"}, {"woff", "Woff"},
    {"wsdl", "WSDL"}, {"xcf", "XCF"}, {"xcodeproj", "Xcode Proj"},
    {"xib", "XIB"}, {"xliff", "Xliff"}, {"xls", "XLS"},
    {"xlsm", "XLSM"}, {"xlsx", "XLSX"}, {"xml", "XML"},
    {"xps", "XPS"}, {"xsd", "XSD"}, {"xsl", "XSL"},
    {"xtc", "XTC"}, {"xz", "XZ"}, {"yaml", "Yaml"},
    {"yml", "YML"}, {"zarr", "Zarr"}, {"zig", "Zig"},
    {"zip", "Zip"}, {"zsh", "ZSH"}, {"zst", "ZST"}};

// Upper-case file names that are words: README.md -> Readme dot MD.
const PhraseMap FILE_WORDS{{"AUTHORS", "Authors"}, {"CHANGELOG", "Changelog"}, {"CHANGES", "Changes"},
                           {"CONTRIBUTING", "Contributing"}, {"COPYING", "Copying"}, {"INSTALL", "Install"},
                           {"LICENCE", "Licence"}, {"LICENSE", "License"}, {"NOTICE", "Notice"}, {"README", "Readme"},
                           {"SECURITY", "Security"}};

// Lower-case words inside identifiers that are said as letters.
const WordSet LOWER_UP{"bf", "cpu", "fp", "gpu", "id", "tls", "ttl"};
// Upper-case words of constants that stay initialisms (others become lower-case words).
const WordSet UPPER_KEEP{"AI",  "API", "ASR", "AWS", "CD",  "CDN", "CI",  "CPU", "CSS",  "CSV",  "DB",
                         "DNS", "ERR", "GB",  "GCP", "GPU", "HF",  "HTML", "HTTP", "HTTPS", "ID", "IO",
                         "IP",  "JS",  "JWT", "KB",  "LLM", "MB",  "ML",  "MS",  "NLP", "OMP", "OS",
                         "PDF", "SDK", "SMTP", "SQL", "SSH", "SSL", "STT", "TB",  "TLS", "TTL", "TTS",
                         "UI",  "URI", "URL", "UTC", "UUID", "UX", "VM",  "WX",  "XML"};
// Short words of handles and repository names that are not initialisms.
const WordSet COMMON2{"am", "an", "as", "at", "be", "by", "co", "do", "go", "he", "hi", "if", "in", "is",
                      "it", "me", "my", "no", "of", "oh", "ok", "on", "or", "so", "to", "up", "us", "we"};
// Longer lower-case words of names that are said as letters (@lumen_gpu, lumen-tts-small).
const WordSet NAME_UP{"asr", "cpu", "gpu", "llm", "nlp", "sdk", "stt", "tts"};
// Lower-case parts of dotted names in code that are said as letters.
const WordSet CODE_UP{"db", "df", "dns", "io", "nn", "os", "tf", "ui"};

bool ascii_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }

// Splits s into pieces: lower-case runs (with one leading capital), capital runs (an acronym; a capital
// before lower-case letters starts the next piece), digit runs. Other bytes separate pieces.
void humps(std::string_view s, std::vector<std::string_view>& out) {
    size_t i = 0, n = s.size();
    while (i < n) {
        char c = s[i];
        size_t j = i;
        if (dg(c)) {
            while (j < n && dg(s[j])) ++j;
        } else if (lo(c)) {
            while (j < n && lo(s[j])) ++j;
        } else if (up(c)) {
            while (j < n && up(s[j])) ++j;
            if (j < n && lo(s[j])) {
                if (j - i > 1) --j;  // "HTTPResponse": HTTP | Response
                while (j < n && lo(s[j])) ++j;
            }
        } else {
            ++i;
            continue;
        }
        out.push_back(s.substr(i, j - i));
        i = j;
    }
}

}  // namespace

std::string lower_s(std::string_view s) {
    std::string r(s);
    for (char& c : r) c = lower_c(c);
    return r;
}

std::string upper_s(std::string_view s) {
    std::string r(s);
    for (char& c : r) c = upper_c(c);
    return r;
}

bool all_digits(std::string_view s) {
    if (s.empty()) return false;
    for (char c : s)
        if (!dg(c)) return false;
    return true;
}

bool all_lower(std::string_view s) {
    if (s.empty()) return false;
    for (char c : s)
        if (!lo(c)) return false;
    return true;
}

bool all_upper(std::string_view s) {
    if (s.empty()) return false;
    for (char c : s)
        if (!up(c)) return false;
    return true;
}

bool space_before(const Doc& doc, uint32_t b) {
    if (b == 0) return true;
    char c = doc.text[b - 1];
    if (ascii_space(c)) return true;
    if (static_cast<unsigned char>(c) < 0x80) return false;
    // no-break and other Unicode spaces
    size_t k = b - 1;
    while (k > 0 && (static_cast<unsigned char>(doc.text[k]) & 0xC0) == 0x80) --k;
    size_t m = k;
    return is_space(decode(doc.text, m));
}

bool left_edge(const Doc& doc, uint32_t b) {
    if (space_before(doc, b)) return true;
    char c = doc.text[b - 1];
    switch (c) {
        case '(': case '[': case '{': case '"': case '\'': case '`': case '<': case '=': case ',': case '*':
        case '|': case ';': case '>':
            return true;
        default:
            break;
    }
    if (static_cast<unsigned char>(c) >= 0x80) {
        size_t k = b - 1;
        while (k > 0 && (static_cast<unsigned char>(doc.text[k]) & 0xC0) == 0x80) --k;
        size_t m = k;
        uint32_t cp = decode(doc.text, m);
        return cp == 0x201C || cp == 0x2018 || cp == 0xAB || cp == 0x2014 || cp == 0x2013 || cp == 0x200B ||
               cp == 0xFEFF;
    }
    return false;
}

bool right_edge(const Doc& doc, uint32_t e) {
    if (e >= doc.text.size()) return true;
    char c = doc.text[e];
    if (idc(c)) return false;
    if (static_cast<unsigned char>(c) >= 0x80) {
        size_t m = e;
        return !is_letter(decode(doc.text, m));
    }
    return true;
}

uint32_t trim_tail(std::string_view text, uint32_t s, uint32_t e) {
    while (e > s) {
        char c = text[e - 1];
        if (c == '.' || c == ',' || c == ';' || c == ':' || c == '!' || c == '?' || c == '\'' || c == '"' ||
            c == '*' || c == '`') {
            --e;
            continue;
        }
        if (c == ')' || c == ']' || c == '}') {
            char open = c == ')' ? '(' : c == ']' ? '[' : '{';
            int depth = 0;
            for (uint32_t k = s; k < e; ++k) {
                if (text[k] == open) ++depth;
                else if (text[k] == c) --depth;
            }
            if (depth < 0) {
                --e;
                continue;
            }
        }
        break;
    }
    return e;
}

uint32_t url_run(std::string_view text, uint32_t b) {
    uint32_t e = b;
    while (e < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[e]);
        if (c <= ' ' || c >= 0x7F || c == '"' || c == '<' || c == '>' || c == '`' || c == '|' || c == '\\' ||
            c == '^')
            break;
        ++e;
    }
    return e;
}

size_t tok_at(const Doc& doc, uint32_t b) {
    size_t i = doc.token_at(b);
    return i < doc.size() && doc[i].start == b ? i : doc.size();
}

bool is_tld(std::string_view label) {
    if (!all_lower(label)) return false;
    return label.size() == 2 ? CCTLD.contains(label) : GENERIC_TLD.contains(label);
}

std::string tld_word(std::string_view label) {
    std::string l = lower_s(label);
    if (l.size() == 2 && all_lower(l)) return l == "co" || l == "me" ? l : upper_s(l);
    if (l == "edu") return "EDU";
    return std::string(label);
}

std::string label_word(std::string_view label) {
    std::string l = lower_s(label);
    if (LABEL_UPPER.contains(l)) return upper_s(l);
    bool digit = false;
    for (char c : label) digit |= dg(c);
    return digit ? mixed_words(label) : std::string(label);
}

std::string mixed_words(std::string_view s) {
    std::string out;
    size_t i = 0;
    bool join = false;
    while (i < s.size()) {
        size_t j = i;
        if (dg(s[i])) {
            while (j < s.size() && dg(s[j])) ++j;
            append_word(out, name_num(s.substr(i, j - i)));
        } else if (al(s[i])) {
            while (j < s.size() && al(s[j])) ++j;
            std::string word(s.substr(i, j - i));  // letters stay as written (labels: slash backup slash q two)
            if (j - i == 1 && (s[i] == 'v' || s[i] == 'V') && j < s.size() && dg(s[j])) word = "V";  // v3: a version
            if (join) out += word;
            else append_word(out, word);
        } else {
            ++j;
        }
        // a hyphen between two words stays (lab-west); next to a number it is a space
        join = j + 1 < s.size() && s[j] == '-' && al(s[j - 1]) && al(s[j + 1]);
        if (join) out += '-';
        i = j;
    }
    return out;
}

bool is_ext(std::string_view ext) {
    if (ext.empty() || ext.size() > 11) return false;
    // one case throughout: json, JSON (not Json, which is a word)
    bool l = false, u = false;
    for (char c : ext) {
        if (lo(c)) l = true;
        else if (up(c)) u = true;
        else if (!dg(c)) return false;
    }
    if (l && u) return false;
    char buf[12];
    for (size_t k = 0; k < ext.size(); ++k) buf[k] = lower_c(ext[k]);
    return EXT.find(std::string_view(buf, ext.size())) != nullptr;
}

std::string ext_word(std::string_view ext) {
    std::string l = lower_s(ext);
    if (const std::string_view* w = EXT.find(l)) return std::string(*w);
    std::string out;
    std::vector<std::string_view> ps;
    humps(l, ps);
    for (std::string_view p : ps) {
        if (all_digits(p)) append_word(out, num_words(p));
        else append_word(out, p.size() <= 3 ? upper_s(p) : capital(std::string(p)));
    }
    return out;
}

void append_word(std::string& out, std::string_view w) {
    if (w.empty()) return;
    if (!out.empty()) out += ' ';
    out += w;
}

void add_lc(const Doc&, Out& out, uint32_t b, uint32_t e, std::string spoken, int score) {
    if (b < e) out.add(b, e, std::move(spoken), score, /*keep_case=*/true);
}

std::string name_num(std::string_view d) {
    if (d.size() == 4 && (d.substr(0, 2) == "19" || d.substr(0, 2) == "20"))
        return pairs(static_cast<uint32_t>(std::stoi(std::string(d))));
    return num_words(d);
}

std::string num_words(std::string_view d) {
    if (d.size() > 1 && d[0] == '0') return digits(d);
    return cardinal_digits(d);
}

std::string ident_words(std::string_view s, Words mode) {
    std::vector<std::string_view> ps;
    ps.reserve(8);
    humps(s, ps);
    std::string out;
    for (size_t k = 0; k < ps.size(); ++k) {
        std::string_view p = ps[k];
        if (all_digits(p)) {
            append_word(out, num_words(p));
            continue;
        }
        std::string l = lower_s(p);
        if (p.size() == 1) {
            // a single letter: "transform_A" A, "top_p" p, "v2" V two
            bool vnum = l == "v" && k + 1 < ps.size() && all_digits(ps[k + 1]) && p.data() + 1 == ps[k + 1].data();
            append_word(out, vnum ? std::string("V") : std::string(p));
            continue;
        }
        if (l == "id") {
            append_word(out, "ID");
            continue;
        }
        if (all_upper(p)) {
            if (UPPER_KEEP.contains(p)) append_word(out, p);
            else append_word(out, l);
            continue;
        }
        if (mode == Words::Handle && l.size() == 2 && !COMMON2.contains(l)) {
            append_word(out, upper_s(l));
            continue;
        }
        if ((mode == Words::Code || mode == Words::Upper) && LOWER_UP.contains(l)) {
            append_word(out, upper_s(l));
            continue;
        }
        if (mode == Words::Placeholder && UPPER_KEEP.contains(upper_s(l))) {
            append_word(out, upper_s(l));
            continue;
        }
        if (mode == Words::Placeholder && up(p[0]) && all_lower(p.substr(1))) {  // <Report_ID_73B>
            append_word(out, p);
            continue;
        }
        append_word(out, l);
    }
    return out;
}

std::string name_words(std::string_view s, bool up2) {
    std::string out;
    std::vector<std::string_view> ps;
    humps(s, ps);
    for (std::string_view p : ps) {
        if (all_digits(p)) append_word(out, name_num(p));
        else if (up2 && all_lower(p) && ((p.size() == 2 && !COMMON2.contains(p)) || NAME_UP.contains(p)))
            append_word(out, upper_s(p));
        else append_word(out, p);
    }
    return out;
}

std::string code_part(std::string_view s) {
    if (s.empty()) return "";
    std::string l = lower_s(s);
    if (all_lower(s) && CODE_UP.contains(s)) return upper_s(s);
    bool has_us = s.find('_') != std::string_view::npos;
    bool has_up = false, has_lo = false, has_dg = false;
    for (char c : s) {
        if (up(c)) has_up = true;
        else if (lo(c)) has_lo = true;
        else if (dg(c)) has_dg = true;
    }
    if (has_us || has_dg || (lo(s[0]) && has_up)) return ident_words(s, Words::Code);
    if (up(s[0]) && has_lo) {
        // PascalCase: one word with one capital, a trailing acronym kept apart (BytesIO -> Bytes IO)
        size_t caps = 0;
        for (char c : s.substr(1))
            if (up(c)) ++caps;
        if (caps == 0) return std::string(s);
        size_t t = s.size();
        while (t > 0 && up(s[t - 1])) --t;
        if (t < s.size() - 1) return capital(lower_s(s.substr(0, t))) + " " + std::string(s.substr(t));
        return capital(l);
    }
    return std::string(s);
}

// Letters as written, digit runs as numbers: vault2 -> vault two, x86 -> x eighty-six.
void alnum_words(std::string& out, std::string_view w) { append_word(out, mixed_words(w)); }

std::string path_part(std::string_view s, bool api, bool hyphen) {
    if (s.size() >= 3 && s.front() == '{' && s.back() == '}')
        return ident_words(s.substr(1, s.size() - 2), Words::Placeholder);
    if (s.size() >= 2 && (s[0] == 'v' || s[0] == 'V') && all_digits(s.substr(1))) return "V " + num_words(s.substr(1));
    size_t dot = s.rfind('.');
    if (dot != std::string_view::npos && dot > 0 && is_ext(s.substr(dot + 1))) return file_words(s);
    if (dot != std::string_view::npos && s.find_first_not_of("0123456789.") == std::string_view::npos && dg(s[0]) &&
        dg(s.back()) && s.find("..") == std::string_view::npos) {
        // a numbered identifier: /abs/2401.01234 -> two four zero one dot zero one two three four
        std::string out;
        size_t i = 0;
        while (i < s.size()) {
            size_t j = s.find('.', i);
            if (j == std::string_view::npos) j = s.size();
            append_word(out, s.size() >= 8 ? digits(s.substr(i, j - i)) : num_words(s.substr(i, j - i)));
            if (j < s.size()) append_word(out, "dot");
            i = j + 1;
        }
        return out;
    }
    std::string out;
    size_t i = 0;
    bool join = false;  // the last word ended at a hyphen that stays
    while (i <= s.size()) {
        size_t j = i;
        while (j < s.size() && an(s[j])) ++j;
        std::string_view w = s.substr(i, j - i);
        if (!w.empty()) {
            std::string part;
            if (all_digits(w)) part = w.size() >= 5 ? digits(w) : name_num(w);  // /record/123456: an ID
            else if (w == "faq" || (api && w == "tts")) part = upper_s(w);
            else if (api && all_lower(w) && w.size() <= 4 && EXT.find(w) && up(EXT.find(w)->front()) &&
                     up(EXT.find(w)->back()))
                part = std::string(*EXT.find(w));  // a spelled extension: /exports/csv -> CSV
            else alnum_words(part, w);
            if (join) out += part;
            else append_word(out, part);
        }
        join = false;
        if (j < s.size() && s[j] == '.') append_word(out, "dot");
        // a hyphen between two words stays in a file system path: ~/voices/bank-two
        if (hyphen && j + 1 < s.size() && s[j] == '-' && !w.empty() && !all_digits(w) && al(s[j + 1])) {
            out += '-';
            join = true;
        }
        i = j + 1;
    }
    return out;
}

std::string file_words(std::string_view s) {
    // the extensions: the longest run of known extensions at the end, at most two
    size_t cut = s.size();
    int n = 0;
    while (n < 2) {
        size_t dot = s.rfind('.', cut - 1);
        if (dot == std::string_view::npos || dot == 0 || !is_ext(s.substr(dot + 1, cut - dot - 1))) break;
        cut = dot;
        ++n;
        if (cut == 0) break;
    }
    std::string_view name = s.substr(0, cut);
    std::string out;
    // name: parts split at '_', '-', '.'; "v2.0" is a version; a hyphen between two words stays (codec-notes)
    size_t i = 0;
    bool join = false;
    while (i < name.size()) {
        size_t j = i;
        while (j < name.size() && name[j] != '_' && name[j] != '-' && name[j] != '.') ++j;
        std::string_view w = name.substr(i, j - i);
        size_t mark = out.size();
        if ((w == "v" || w == "V" || (w.size() > 1 && (w[0] == 'v' || w[0] == 'V') && all_digits(w.substr(1)))) &&
            j < name.size() && name[j] == '.' && j + 1 < name.size() && dg(name[j + 1]) && w.size() > 1) {
            // v2.0 -> version two point zero
            std::string v = "version " + num_words(w.substr(1));
            while (j < name.size() && name[j] == '.' && j + 1 < name.size() && dg(name[j + 1])) {
                size_t k = j + 1;
                while (k < name.size() && dg(name[k])) ++k;
                v += " point " + num_words(name.substr(j + 1, k - j - 1));
                j = k;
            }
            append_word(out, v);
        } else if (all_digits(w) && j < name.size() && name[j] == '.' && j + 1 < name.size() && dg(name[j + 1])) {
            // a version: setup-1.4.2.exe -> setup one point four point two dot EXE
            std::string v = num_words(w);
            while (j < name.size() && name[j] == '.' && j + 1 < name.size() && dg(name[j + 1])) {
                size_t k = j + 1;
                while (k < name.size() && dg(name[k])) ++k;
                v += " point " + num_words(name.substr(j + 1, k - j - 1));
                j = k;
            }
            append_word(out, v);
        } else if (!w.empty()) {
            bool code = false, hump = false;
            for (size_t k = 0; k < w.size(); ++k) {
                if (dg(w[k]) || (k > 0 && up(w[k]))) code = true;
                if (k > 0 && up(w[k]) && lo(w[k - 1])) hump = true;
            }
            const std::string_view* cap = all_upper(w) ? FILE_WORDS.find(w) : nullptr;
            if (all_digits(w)) append_word(out, name_num(w));
            else if (code && hump) append_word(out, ident_words(w, Words::Code));
            else if (code && !all_upper(w)) append_word(out, mixed_words(w));
            else if (cap) append_word(out, *cap);
            else append_word(out, w);
        }
        if (join && out.size() > mark && out[mark] == ' ') out.erase(mark, 1);
        join = false;
        if (j < name.size() && name[j] == '.') append_word(out, "dot");
        if (j + 1 < name.size() && name[j] == '-' && !w.empty() && !all_digits(w) && !dg(w.back()) && al(name[j + 1])) {
            size_t k = j + 1;
            while (k < name.size() && al(name[k])) ++k;
            if (k == name.size() || name[k] == '.' || name[k] == '-' || name[k] == '_') {
                out += '-';
                join = true;
            }
        }
        i = j + 1;
    }
    for (size_t k = cut; k < s.size();) {
        size_t e = s.find('.', k + 1);
        if (e == std::string_view::npos) e = s.size();
        append_word(out, "dot");
        append_word(out, ext_word(s.substr(k + 1, e - k - 1)));
        k = e;
    }
    return out;
}

std::string port_words(std::string_view d) {
    if (d.empty() || d.size() > 5) return "";
    if (d.size() <= 2) return num_words(d);
    if (d.size() == 3) return digits(d);
    if (d.size() == 5 || d[0] == '0') return num_words(d);
    unsigned hi = static_cast<unsigned>((d[0] - '0') * 10 + (d[1] - '0'));
    unsigned lo2 = static_cast<unsigned>((d[2] - '0') * 10 + (d[3] - '0'));
    if (d.substr(1) == "000") return cardinal(static_cast<uint64_t>(hi / 10)) + " thousand";
    if (lo2 == 0) return cardinal(hi) + " hundred";
    if (lo2 < 10) return cardinal(hi) + " oh " + ONES[lo2];
    return cardinal(hi) + " " + cardinal(lo2);
}

std::string ip_part(std::string_view d) {
    if (d.size() == 3 && d[0] != '0') return pairs(static_cast<uint32_t>(std::stoi(std::string(d))));
    return cardinal(static_cast<uint64_t>(std::stoi(std::string(d))));
}

std::string hex_words(std::string_view s) {
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
        if (dg(s[i])) {
            append_word(out, ONES[s[i] - '0']);
            ++i;
        } else if (al(s[i])) {
            size_t j = i;
            while (j < s.size() && al(s[j])) ++j;
            append_word(out, upper_s(s.substr(i, j - i)));
            i = j;
        } else {
            ++i;
        }
    }
    return out;
}

uint32_t parse_ipv4(std::string_view text, uint32_t b, uint32_t limit) {
    uint32_t p = b;
    for (int part = 0; part < 4; ++part) {
        if (part > 0) {
            if (p >= limit || text[p] != '.') return 0;
            ++p;
        }
        uint32_t q = p;
        int v = 0;
        while (q < limit && dg(text[q]) && q - p < 4) v = v * 10 + (text[q++] - '0');
        if (q == p || q - p > 3 || v > 255) return 0;
        p = q;
    }
    if (p < limit && (dg(text[p]) || al(text[p]))) return 0;
    if (p + 1 < limit && text[p] == '.' && dg(text[p + 1])) return 0;
    return p;
}

Host parse_host(std::string_view text, uint32_t b, uint32_t limit, bool strict) {
    Host h;
    if (uint32_t ip = parse_ipv4(text, b, limit)) {
        h.end = ip;
        h.labels = 4;
        h.ip = true;
        return h;
    }
    uint32_t p = b;
    int labels = 0;
    bool first_alpha = false;
    while (p < limit) {
        uint32_t q = p;
        bool alpha = false;
        while (q < limit && (an(text[q]) || text[q] == '-')) {
            if (al(text[q])) alpha = true;
            ++q;
        }
        if (q == p || text[p] == '-' || text[q - 1] == '-') break;
        std::string_view label = text.substr(p, q - p);
        if (labels == 0) first_alpha = alpha;
        ++labels;
        if (labels >= 2 && first_alpha) {
            bool ok = strict ? is_tld(label) : (label.size() >= 2 && alpha && (all_lower(label) || all_upper(label)));
            if (ok) {
                h.end = q;
                h.labels = labels;
            }
        }
        if (q < limit && text[q] == '.' && q + 1 < limit && an(text[q + 1])) p = q + 1;
        else break;
    }
    return h;
}

std::string host_words(std::string_view host) {
    std::string out;
    if (parse_ipv4(host, 0, static_cast<uint32_t>(host.size())) == host.size()) {
        size_t i = 0;
        while (i < host.size()) {
            size_t j = host.find('.', i);
            if (j == std::string_view::npos) j = host.size();
            append_word(out, ip_part(host.substr(i, j - i)));
            if (j < host.size()) append_word(out, "dot");
            i = j + 1;
        }
        return out;
    }
    size_t i = 0;
    bool first = true;
    while (i < host.size()) {
        size_t j = host.find('.', i);
        bool last = j == std::string_view::npos;
        if (last) j = host.size();
        std::string_view label = host.substr(i, j - i);
        if (first && lower_s(label) == "www") append_word(out, "WWW");
        else if (last && !first) append_word(out, tld_word(label));
        else append_word(out, label_word(label));
        if (!last) append_word(out, "dot");
        first = false;
        i = j + 1;
    }
    return out;
}

}  // namespace rv::el
