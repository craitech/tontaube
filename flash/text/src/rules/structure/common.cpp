#include "rules/structure/common.h"

namespace st {

std::vector<Line> split_lines(const Doc& d) {
    std::vector<Line> lines;
    size_t n = d.size(), count = 1;
    for (size_t i = 1; i < n; ++i) count += d[i].newline_before;
    lines.reserve(count);
    for (size_t i = 0; i < n;) {
        size_t j = i + 1;
        while (j < n && !d[j].newline_before) ++j;
        Line l{i, j, static_cast<uint32_t>(d.text.size()), 0};
        size_t nl = d.text.find('\n', d[j - 1].end);
        if (nl != std::string_view::npos && (j == n || nl < d[j].start)) l.nl = static_cast<uint32_t>(nl);
        if (i > 0) {
            int count = 0;
            for (uint32_t b = d[i - 1].end; b < d[i].start; ++b) count += d.text[b] == '\n';
            l.blank_before = count > 0 ? count - 1 : 0;
        }
        lines.push_back(l);
        i = j;
    }
    return lines;
}

std::string gap_newlines(const Doc& d, size_t i) {
    std::string s;
    for (uint32_t b = gap_start(d, i), e = gap_end(d, i); b < e; ++b)
        if (d.text[b] == '\n') s += '\n';
    return s;
}

bool attaches_left(const Doc& d, size_t i) {
    if (!is_punct(d, i)) return false;
    switch (d[i].cp) {
        case ',': case '.': case ';': case ':': case '!': case '?': case ')': case ']': case '}':
        case 0x201D: case 0x2019: case 0x2026:
            return true;
        default:
            return false;
    }
}

Span deletion(const Doc& d, size_t first, size_t last) {
    bool has_prev = first > 0 && !d[first].newline_before;
    bool has_next = last + 1 < d.size() && !d[last + 1].newline_before;
    uint32_t s = d[first].start, e = d[last].end;
    if (!has_prev) return {s, has_next ? d[last + 1].start : e};
    if (!has_next || attaches_left(d, last + 1)) return {d[first - 1].end, e};
    if (d[first].space_before && d[last + 1].space_before) return {s, d[last + 1].start};
    return {s, e};
}

namespace {

const rv::WordSet HTML_TAGS{
    "a", "abbr", "answer", "article", "aside", "assistant", "b", "big", "blockquote", "body", "br", "caption",
    "center", "cite", "code", "context", "dd", "del", "details", "div", "dl", "document", "dt", "em", "figcaption",
    "figure", "font", "footer", "function_call", "h1", "h2", "h3", "h4", "h5", "h6", "head", "header", "hr", "html",
    "i", "img", "ins", "kbd", "label", "li", "main", "mark", "nav", "ol", "output", "p", "pre", "q", "query",
    "reasoning", "response", "result", "s", "samp", "section", "small", "span", "strike", "strong", "sub",
    "summary", "sup", "system", "table", "tbody", "td", "th", "thead", "think", "thinking", "time", "title",
    "tool", "tool_call", "tool_response", "tool_result", "tr", "tt", "u", "ul", "user", "var", "wbr"};
const rv::WordSet BLOCK_TAGS{"answer", "article", "blockquote", "br", "dd", "div", "dt", "figcaption",
                             "figure", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "ol",
                             "p", "section", "table", "td", "th", "think", "thinking", "tr", "ul", "reasoning"};

bool same_line_joined(const Doc& d, size_t i) { return i < d.size() && d.joined(i) && !d[i].newline_before; }

}  // namespace

size_t tag_at(const Doc& d, size_t i, bool* block) {
    size_t n = d.size();
    if (block) *block = false;
    if (p(d, i, '[')) {  // [INST] [/INST]
        size_t k = i + 1;
        if (p(d, k, '/') && same_line_joined(d, k)) ++k;
        if (same_line_joined(d, k) && d.is(k, "inst") && d[k].casing == rv::UPPER && p(d, k + 1, ']') &&
            d.joined(k + 1)) {
            if (block) *block = true;
            return k + 1;
        }
        return 0;
    }
    if (!p(d, i, '<') || !same_line_joined(d, i + 1)) return 0;
    if (p(d, i + 1, '|')) {  // <|im_end|>
        for (size_t e = i + 2; e + 1 < n && e < i + 12 && !d[e].newline_before; ++e)
            if (p(d, e, '|') && p(d, e + 1, '>') && d.joined(e + 1)) {
                if (block) *block = true;
                return e + 1;
            }
        return 0;
    }
    size_t k = i + 1;
    if (p(d, k, '/') && same_line_joined(d, k + 1)) ++k;
    if (!d.word(k) || !(d[k].casing == rv::LOWER || (d[k].casing == rv::UPPER && d.str(k).size() == 1))) return 0;
    std::string name;
    size_t after = k;
    while (after < n && (after == k || same_line_joined(d, after)) &&
           (d.word(after) || d.number(after) || (after > k && (p(d, after, '_') || p(d, after, '-'))))) {
        name += lower(d.str(after));
        ++after;
    }
    if (!HTML_TAGS.contains(name)) {  // other tag-like tokens (<scratchpad>, </notes>) only in the bare form
        if (!p(d, after, '>') || !d.joined(after)) return 0;
        // <your-api-key>, <user_name>: a placeholder, read as words (electronic); <end_of_turn> is a token
        if (name.find('-') != std::string::npos) return 0;
        if (name.find('_') != std::string::npos && name.rfind("end_", 0) != 0 && name.rfind("start_", 0) != 0 &&
            name.rfind("im_", 0) != 0)
            return 0;
        if (block) *block = true;
        return after;
    }
    size_t e = after;
    while (e < n && !d[e].newline_before && !p(d, e, '>') && !p(d, e, '<') && e < after + 60) ++e;
    if (e >= n || !p(d, e, '>') || d[e].newline_before) return 0;
    if (e != after && !d[after].space_before && !p(d, after, '/')) return 0;
    if (block) *block = BLOCK_TAGS.contains(name);
    return e;
}

bool initialism(std::string_view w) {
    static const rv::WordSet INITIALISMS{
        "ai",  "api", "apis", "asr", "aws", "b2b", "ceo", "cfo", "ci", "cli", "coo", "cpu", "crm", "csv", "cto",
        "dns", "eu",  "ev",  "faq", "fda", "fy",  "gcp", "gdpr", "gps", "gpu", "gpus", "hdmi", "hr", "html",
        "http", "https", "id", "ide", "ieee", "iot", "ip", "iso", "kpi", "kpis", "lan", "led", "llm", "llms",
        "lte", "ml", "nda", "nhs", "nlp", "ocr", "os", "pc", "pdf", "pr", "qa", "qc", "ram", "rf", "roi", "sdk",
        "sla", "sms", "sql", "ssd", "tcp", "tts", "ui", "uk", "un", "url", "usa", "usb", "ux", "vpn", "wan", "nato",
        "ssh", "tls", "ssl", "rtx", "gpt", "sso", "erp", "etl", "llc", "ltd", "plc", "nyc", "dc", "cd", "tv",
        "nyse", "nasdaq", "ansi", "ascii", "jpeg", "mpeg", "uefi", "bios", "cdn", "db", "dbs", "gw", "jwt",
        "mfa", "otp", "pos", "rpc", "sftp", "smtp", "ftp", "vm", "vms", "k8s", "ttl", "udp", "uat", "seo"};
    return INITIALISMS.contains(w);
}

std::string lower(std::string_view s) {
    std::string out(s);
    for (char& c : out)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
    return out;
}

std::string cap_cardinal(uint64_t n) { return rv::capital(rv::cardinal(n)); }

}  // namespace st
