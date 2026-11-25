// Web addresses, domains, IP addresses and ports, e-mail addresses, handles, hashtags, Reddit names,
// repository names, HTTP status codes.
#include "rules/electronic/common.h"

namespace {

using namespace rv::el;
using rv::Doc;
using rv::Kind;

const rv::WordSet VERSION_CUES{"build", "firmware", "patch", "python", "release", "rev", "revision", "v",
                               "ver", "version", "versions"};

// A ".Capital" inside [b, e) that starts a new sentence (a missing space), unless it is an extension.
uint32_t cut_missing_space(std::string_view t, uint32_t b, uint32_t e) {
    for (uint32_t k = b; k + 1 < e; ++k) {
        if (t[k] != '.' || !up(t[k + 1])) continue;
        uint32_t q = k + 1;
        while (q < e && an(t[q])) ++q;
        if (q - k - 1 >= 2 && all_upper(t.substr(k + 1, q - k - 1)) && is_ext(t.substr(k + 1, q - k - 1))) continue;
        return k;
    }
    return e;
}

// An address in angle brackets (<https://acme.org>, <jane@acme.com>): the brackets go with it.
void autolink(std::string_view t, uint32_t& b, uint32_t& e) {
    uint32_t s = b;
    if (s > 0 && t[s - 1] == '<' && e < t.size() && t[e] == '>') {
        b = s - 1;
        ++e;
    }
}

// Web addresses: scheme left out, www. as WWW dot, a host by its labels, a port, at most two path parts.
void url_at(const Doc& doc, rv::Out& out, size_t i) {
    const std::string_view t = doc.text;
    uint32_t b = doc[i].start;
    if (!left_edge(doc, b)) return;
    uint32_t run = url_run(t, b);
    uint32_t p = b;
    bool scheme = false;
    if (doc.lower_at(p, "https://")) p += 8, scheme = true;
    else if (doc.lower_at(p, "http://")) p += 7, scheme = true;
    else if (doc.lower_at(p, "ftp://")) p += 6, scheme = true;
    bool www = doc.lower_at(p, "www.");
    Host h;
    bool localhost = doc.lower_at(p, "localhost") && (p + 9 >= run || !idc(t[p + 9]));
    if (localhost) h.end = p + 9;
    else h = parse_host(t, p, run, !scheme && !www);
    bool caps = false;  // EXAMPLE.COM: an address in capitals is read in lower case
    if (!h.end && !localhost && p < run && up(t[p])) {
        uint32_t q = p;
        while (q < run && (up(t[q]) || dg(t[q]) || t[q] == '.' || t[q] == '-')) ++q;
        while (q > p && (t[q - 1] == '.' || t[q - 1] == '-')) --q;
        std::string low = lower_s(t.substr(p, q - p));
        Host g = parse_host(low, 0, static_cast<uint32_t>(low.size()), true);
        bool dotnet = low.size() > 4 && low.compare(low.size() - 4, 4, ".net") == 0 && g.labels == 2;  // ASP.NET
        if (g.end && !g.ip && g.labels >= 2 && (q >= run || !al(t[q])) && !(dotnet && !scheme)) {
            h = g;
            h.end = p + g.end;
            caps = true;
        }
    }
    if (!h.end) return;
    uint32_t e = h.end;
    if (e < t.size() && (t[e] == '-' || t[e] == '_' || t[e] == '@')) return;
    // a dotted name in code goes on after the ending (astropy.io.fits) or has a capital after the first part
    // (tf.keras.Model.fit)
    if (!scheme && !www && e + 1 < run && t[e] == '.' && (lo(t[e + 1]) || t[e + 1] == '_')) return;
    // (a name in capitals throughout, "St.Mary.org", is an address; "tf.keras.Model" is code)
    if (!scheme && !www && !caps && !up(t[p])) {
        uint32_t first_dot = p;
        while (first_dot < e && t[first_dot] != '.') ++first_dot;
        for (uint32_t k = first_dot; k < e; ++k)
            if (up(t[k])) return;
    }
    if (h.ip && !scheme) {
        size_t pw = doc.prev_word(i);
        if (pw != Doc::npos && VERSION_CUES.contains(doc.low(pw))) return;
    }
    uint32_t end = trim_tail(t, b, run);
    if (end > e + 2 && t[end - 2] == '\'' && t[end - 1] == 's') end -= 2;  // github.com/acme's
    end = cut_missing_space(t, e, end);
    if (end < e) end = e;
    std::string spoken;
    std::string host(t.substr(www ? p + 4 : p, e - (www ? p + 4 : p)));
    bool lower_letter = false;
    for (char c : host) lower_letter |= lo(c);
    if (caps || !lower_letter) host = lower_s(host);
    if (www) {
        spoken = "WWW dot";
        append_word(spoken, host_words(host));
    } else if (localhost) {
        spoken = "localhost";
    } else {
        spoken = host_words(host);
    }
    uint32_t q = e;
    bool port = false;
    if (q < end && t[q] == ':') {
        uint32_t r = q + 1;
        while (r < end && dg(t[r])) ++r;
        if (r > q + 1 && r - q - 1 <= 5 && (r >= end || !an(t[r]))) {
            append_word(spoken, "port");
            append_word(spoken, port_words(t.substr(q + 1, r - q - 1)));
            q = r;
            port = true;
        }
    }
    if (localhost && !port && !scheme) return;
    // a possessive stays as written: Amazon.com's revenue -> Amazon dot com's revenue
    if (q + 1 < end && t[q] == '\'' && t[q + 1] == 's' && (q + 2 >= end || !an(t[q + 2]))) end = q;
    if (q < end && t[q] != '/' && t[q] != '?' && t[q] != '#') {
        // something else sticks to the host: not an address
        if (!(t[q] == ':' && scheme)) return;
    }
    if (q < end && t[q] == '/') {
        uint32_t pe = q;
        while (pe < end && t[pe] != '?' && t[pe] != '#') ++pe;
        // path segments
        std::string_view segs[3];
        int n = 0;
        uint32_t s = q + 1;
        while (s <= pe) {
            uint32_t z = s;
            while (z < pe && t[z] != '/') ++z;
            if (z > s) {
                if (n < 3) segs[n] = t.substr(s, z - s);
                ++n;
            }
            s = z + 1;
        }
        // on a code host the first two parts name the repository: github.com/owner/repo/issues
        std::string lh = lower_s(host);
        bool code_host = lh == "github.com" || lh == "gitlab.com" || lh == "bitbucket.org" || lh == "huggingface.co" ||
                         lh == "codeberg.org";
        if (n == 3 && code_host) n = 2;
        if (n <= 2)
            for (int k = 0; k < n; ++k) {
                append_word(spoken, "slash");
                append_word(spoken, path_part(segs[k]));
            }
    }
    int score = scheme || www ? 86 : 84;
    autolink(t, b, end);
    add_lc(doc, out, b, end, spoken, score);
}

RV_RULE("electronic", url) {
    auto start = [&](size_t k) {
        if (k == 0 || doc[k - 1].kind == Kind::Punct || !doc.joined(k) || !glued(doc, k)) return;
        // a first label with hyphens: agri-carbon.example.org, smithsons-labs.org
        size_t s = k - 1;
        while (s >= 1 && doc.joined(s)) {
            if (doc[s - 1].kind != Kind::Punct) --s;  // web01, 1password
            else if (s >= 2 && doc.punct(s - 1, '-') && doc.joined(s - 1) && doc[s - 2].kind != Kind::Punct) s -= 2;
            else break;
        }
        url_at(doc, out, s);
    };
    each_mark(doc, '.', start);
    each_mark(doc, ':', start);
}

std::string local_words(std::string_view s);

// Schemes other than the web's are read: s3://bucket/path -> S three colon slash slash bucket slash path.
const rv::PhraseMap SCHEMES{{"amqp", "AMQP"},  {"file", "file"},   {"ftp", "FTP"},     {"git", "git"},
                            {"gs", "GS"},      {"hdfs", "HDFS"},   {"jdbc", "JDBC"},   {"ldap", "LDAP"},
                            {"mongodb", "mongodb"}, {"mqtt", "MQTT"}, {"mysql", "mysql"}, {"nfs", "NFS"},
                            {"postgres", "postgres"}, {"postgresql", "postgresql"}, {"redis", "redis"},
                            {"rtmp", "RTMP"},  {"rtsp", "RTSP"},   {"s3", "S three"},  {"sftp", "SFTP"},
                            {"smb", "SMB"},    {"ssh", "SSH"},     {"svn", "SVN"},     {"tcp", "TCP"},
                            {"udp", "UDP"},    {"ws", "WS"},       {"wss", "WSS"}};

void scheme_url_at(const Doc& doc, rv::Out& out, size_t k) {
    const std::string_view t = doc.text;
    uint32_t c = doc[k].start;
    if (!doc.text_at(c, "://") || c == 0) return;
    uint32_t b = c;
    while (b > 0 && (lo(t[b - 1]) || up(t[b - 1]) || dg(t[b - 1]))) --b;
    if (b == c || !al(t[b]) || !left_edge(doc, b)) return;
    std::string name = lower_s(t.substr(b, c - b));
    const std::string_view* said = SCHEMES.find(name);
    if (!said || name == "ftp") return;  // ftp:// is read like a web address
    uint32_t run = url_run(t, c + 3);
    uint32_t end = trim_tail(t, c + 3, run);
    std::string spoken = std::string(*said) + " colon slash slash";
    uint32_t p = c + 3;
    uint32_t stop = p;
    while (stop < end && t[stop] != '?' && t[stop] != '#') ++stop;
    if (p < stop && t[p] != '/') {
        // user (and password) before '@'
        uint32_t slash = p;
        while (slash < stop && t[slash] != '/') ++slash;
        uint32_t at = p;
        while (at < slash && t[at] != '@') ++at;
        if (at < slash) {
            uint32_t u = p;
            while (u < at && t[u] != ':') ++u;
            append_word(spoken, local_words(t.substr(p, u - p)));
            if (u + 1 < at) {
                append_word(spoken, "colon");
                append_word(spoken, local_words(t.substr(u + 1, at - u - 1)));
            }
            append_word(spoken, "at");
            p = at + 1;
        }
        uint32_t h = p;
        while (h < slash && t[h] != ':') ++h;
        std::string_view host = t.substr(p, h - p);
        if (host.empty()) return;
        append_word(spoken, host.find('.') != std::string_view::npos ? host_words(host) : label_word(host));
        p = h;
        if (p < slash && t[p] == ':') {
            uint32_t r = p + 1;
            while (r < slash && dg(t[r])) ++r;
            if (r == p + 1 || r - p - 1 > 5) return;
            append_word(spoken, "port");
            append_word(spoken, port_words(t.substr(p + 1, r - p - 1)));
            p = r;
        }
        if (p < slash) return;
    }
    // the path, every part
    int parts = 0;
    while (p < stop) {
        uint32_t q = p + 1;
        while (q < stop && t[q] != '/') ++q;
        if (q > p + 1) {
            append_word(spoken, "slash");
            append_word(spoken, path_part(t.substr(p + 1, q - p - 1), false, true));
            if (++parts > 6) return;
        } else if (parts == 0 && q < stop) {
            append_word(spoken, "slash");  // file:///home
        }
        p = q;
    }
    add_lc(doc, out, b, end, spoken, 87);
}

// A range of two IP addresses: the dash between them is "to".
RV_RULE("electronic", ip_range) {
    const std::string_view t = doc.text;
    auto at = [&](size_t i) {
        if (i == 0 || i + 1 >= doc.size() || !doc.joined(i) || !doc.joined(i + 1) || !doc.number(i - 1) ||
            !doc.number(i + 1))
            return;
        uint32_t a = doc[i].start;
        // the address before the dash
        uint32_t s = a;
        while (s > 0 && (dg(t[s - 1]) || t[s - 1] == '.')) --s;
        if (parse_ipv4(t, s, a) != a) return;
        if (!parse_ipv4(t, doc[i].end, static_cast<uint32_t>(t.size()))) return;
        out.tokens(i, i, "to", 84);
    };
    each_mark(doc, '-', at);
    each_mark(doc, '\xE2', [&](size_t i) {
        if (doc.punct(i, 0x2013) || doc.punct(i, 0x2014)) at(i);
    });
}

// A port after a colon on its own: ":8200" -> "port eighty-two hundred".
RV_RULE("electronic", bare_port) {
    const std::string_view t = doc.text;
    each_mark(doc, ':', [&](size_t i) {
        if (!space_before(doc, doc[i].start) || !doc.number(i + 1) || !doc.joined(i + 1)) return;
        std::string_view d = doc.str(i + 1);
        if (d.size() < 2 || d.size() > 5) return;
        uint32_t e = doc[i + 1].end;
        if (!right_edge(doc, e)) return;
        if (e < t.size() && (t[e] == ':' || (t[e] == '.' && e + 1 < t.size() && dg(t[e + 1])))) return;
        out.tokens(i, i + 1, "port " + port_words(d), 75);
    });
}

const char* const HOST_PARTS[] = {"backend", "broker", "cache", "docker", "frontend", "gateway", "host", "kafka",
                                  "mongo", "mysql", "nginx", "node", "postgres", "proxy", "redis", "server", "staging",
                                  "worker"};
const char* const HOST_ENDS[] = {"api", "app", "box", "db", "dev", "prod", "srv", "svc", "vm", "web"};

// A machine name without a domain: dbhost, devbox, redis, web01, api-server.
bool host_name(std::string_view h) {
    if (h.size() < 2 || h.size() > 24 || !lo(h[0])) return false;
    for (char c : h)
        if (!(lo(c) || dg(c) || c == '-')) return false;
    for (const char* x : HOST_PARTS)
        if (h.find(x) != std::string_view::npos) return true;
    size_t n = h.size();
    while (n > 0 && (dg(h[n - 1]) || h[n - 1] == '-')) --n;  // web01, node-2
    std::string_view core = h.substr(0, n);
    for (const char* x : HOST_ENDS) {
        std::string_view e(x);
        if (core.size() >= e.size() && (core.substr(0, e.size()) == e || core.substr(core.size() - e.size()) == e))
            return true;
    }
    return false;
}

// A machine name with a port: devbox:5555 -> devbox port fifty-five fifty-five.
void host_port_at(const Doc& doc, rv::Out& out, size_t k) {
    const std::string_view t = doc.text;
    if (k == 0 || !doc.joined(k) || !doc.number(k + 1) || !doc.joined(k + 1) || doc[k - 1].kind == Kind::Punct) return;
    uint32_t c = doc[k].start, b = c;
    while (b > 0 && (an(t[b - 1]) || t[b - 1] == '-')) --b;
    if (!left_edge(doc, b) || !host_name(t.substr(b, c - b)) || doc.lower_at(b, "localhost")) return;
    std::string_view d = doc.str(k + 1);
    uint32_t e = doc[k + 1].end;
    if (d.size() < 2 || d.size() > 5 || d[0] == '0' || !right_edge(doc, e)) return;
    if (e < t.size() && (t[e] == ':' || t[e] == '-' || (t[e] == '.' && e + 1 < t.size() && dg(t[e + 1])))) return;
    add_lc(doc, out, b, e, label_word(t.substr(b, c - b)) + " port " + port_words(d), 84);
}

// Everything that starts at a colon: other schemes (s3://), machine names with a port (devbox:5555), scope
// operators (std::vector), IPv6 addresses.
RV_RULE("electronic", colon_marks) {
    uint32_t last6 = 0;
    each_mark(doc, ':', [&](size_t k) {
        scheme_url_at(doc, out, k);
        host_port_at(doc, out, k);
        scope_operator_at(doc, out, k);
        ipv6_at(doc, out, k, last6);
    });
}

std::string local_words(std::string_view s) {
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = i;
        while (j < s.size() && an(s[j])) ++j;
        if (j > i) {
            std::string_view w = s.substr(i, j - i);
            if (all_digits(w)) append_word(out, num_words(w));
            else if (w.size() == 1) append_word(out, upper_s(w));
            else append_word(out, mixed_words(w));
        }
        if (j < s.size()) {
            char c = s[j];
            append_word(out, c == '.' ? "dot" : c == '_' ? "underscore" : c == '-' ? "dash" : c == '+' ? "plus" : "");
        }
        i = j + 1;
    }
    return out;
}

// E-mail addresses: user@domain.com -> user at domain dot com.
RV_RULE("electronic", email) {
    const std::string_view t = doc.text;
    each_mark(doc, '@', [&](size_t i) {
        if (i == 0 || i + 1 >= doc.size() || !doc.joined(i) || !doc.joined(i + 1)) return;
        uint32_t at = doc[i].start;
        uint32_t s = at;
        while (s > 0 && (an(t[s - 1]) || t[s - 1] == '.' || t[s - 1] == '_' || t[s - 1] == '-' || t[s - 1] == '+' ||
                         t[s - 1] == '%'))
            --s;
        while (s < at && !an(t[s])) ++s;
        uint32_t lead = s;  // "mailto:" is left out
        if (s >= 7 && doc.lower_at(s - 7, "mailto:")) lead = s - 7;
        if (s == at || !left_edge(doc, lead)) return;
        uint32_t run = url_run(t, at + 1);
        Host h = parse_host(t, at + 1, run, true);
        // an address in capitals: SALES@ACME.COM -> sales at acme dot com
        bool caps = false;
        if (!h.end && at + 1 < run && up(t[at + 1])) {
            uint32_t q = at + 1;
            while (q < run && (up(t[q]) || dg(t[q]) || t[q] == '.' || t[q] == '-')) ++q;
            std::string low = lower_s(t.substr(at + 1, q - at - 1));
            Host g = parse_host(low, 0, static_cast<uint32_t>(low.size()), true);
            if (g.end && !g.ip && at + 1 + g.end == q) {
                h = g;
                h.end = q;
                caps = true;
            }
        }
        if (!h.end) return;
        uint32_t e = h.end;
        if (!right_edge(doc, e) || (e + 1 < t.size() && t[e] == '.' && lo(t[e + 1]))) return;
        std::string_view local = t.substr(s, at - s);
        std::string spoken = local_words(caps ? std::string_view(lower_s(local)) : local);
        append_word(spoken, "at");
        append_word(spoken, host_words(caps ? std::string_view(lower_s(t.substr(at + 1, e - at - 1)))
                                            : t.substr(at + 1, e - at - 1)));
        // a git remote: git@github.com:acme/widget.git -> ... colon acme slash widget dot git
        if (e + 1 < t.size() && t[e] == ':' && al(t[e + 1])) {
            uint32_t q = e + 1;
            while (q < t.size() && (an(t[q]) || t[q] == '/' || t[q] == '.' || t[q] == '-' || t[q] == '_')) ++q;
            q = trim_tail(t, e + 1, q);
            std::string path;
            if (q > e + 1 && t.substr(e + 1, q - e - 1).find('/') != std::string_view::npos) {
                std::string_view p = t.substr(e + 1, q - e - 1);
                size_t i0 = 0;
                bool first = true;
                while (i0 < p.size()) {
                    size_t j0 = p.find('/', i0);
                    if (j0 == std::string_view::npos) j0 = p.size();
                    if (j0 > i0) {
                        if (!first) append_word(path, "slash");
                        append_word(path, path_part(p.substr(i0, j0 - i0), false, true));
                        first = false;
                    }
                    i0 = j0 + 1;
                }
                append_word(spoken, "colon");
                append_word(spoken, path);
                e = q;
            }
        }
        autolink(t, lead, e);
        add_lc(doc, out, lead, e, spoken, 86);
    });
}

const rv::WordSet DETERMINERS{"a", "an", "her", "his", "its", "my", "our", "that", "the", "their", "this", "your"};

// The "at" or "dot" of a spelled-out address at token i: "(at)", "[at]", "{at}" or the word in capitals ("AT").
// Returns the last token, or Doc::npos.
size_t spelled_mark(const Doc& doc, size_t i, std::string_view word) {
    if (i >= doc.size()) return Doc::npos;
    if (doc.punct(i, '(') || doc.punct(i, '[') || doc.punct(i, '{')) {
        uint32_t close = doc.punct(i, '(') ? ')' : doc.punct(i, '[') ? ']' : '}';
        if (doc.is(i + 1, word) && doc.joined(i + 1) && doc.punct(i + 2, close) && doc.joined(i + 2)) return i + 2;
        return Doc::npos;
    }
    if (doc.word(i) && doc[i].casing == rv::UPPER && doc.is(i, word)) return i;
    return Doc::npos;
}

// Addresses spelled out against spam: jane (at) example (dot) com, alice AT example DOT com -> jane at example
// dot com.
RV_RULE("electronic", spelled_email) {
    // only where "(at)", "[at]", "{at}" or "AT " is written
    const std::string_view t = doc.text;
    bool any = false;
    for (char close : {')', ']', '}'})
        for (size_t p = t.find(close); p != std::string_view::npos && !any; p = t.find(close, p + 1))
            any = p >= 2 && (t[p - 2] | 0x20) == 'a' && (t[p - 1] | 0x20) == 't';
    for (size_t p = t.find('T'); p != std::string_view::npos && !any; p = t.find('T', p + 1))
        any = p >= 1 && t[p - 1] == 'A' && p + 1 < t.size() && t[p + 1] == ' ';
    if (!any) return;
    for (size_t i = 1; i + 4 < doc.size(); ++i) {
        size_t a = spelled_mark(doc, i, "at");
        if (a == Doc::npos || !doc[i].space_before || !doc.word(i - 1) || !doc[a + 1].space_before) continue;
        // the name before: one word or a dotted name
        size_t first = i - 1;
        while (first >= 2 && doc.joined(first) && (doc.punct(first - 1, '.') || doc.punct(first - 1, '_')) &&
               doc.joined(first - 1) && doc[first - 2].kind != Kind::Punct)
            first -= 2;
        if (!left_edge(doc, doc[first].start)) continue;
        std::string spoken = local_words(doc.slice(doc[first].start, doc[i - 1].end));
        append_word(spoken, "at");
        size_t k = a + 1, last = Doc::npos;
        int labels = 0;
        bool tld = false;
        bool prose = false;
        while (k < doc.size() && (doc.word(k) || doc.number(k)) && labels < 6) {
            std::string_view label = doc.str(k);
            size_t d = k + 1 < doc.size() && doc[k + 1].space_before ? spelled_mark(doc, k + 1, "dot") : Doc::npos;
            bool end = d == Doc::npos || d + 1 >= doc.size() || !doc[d + 1].space_before;
            ++labels;
            last = k;
            tld = labels >= 2 && is_tld(label);
            append_word(spoken, end && tld ? tld_word(label) : label_word(label));
            prose |= DETERMINERS.contains(doc.low(k));
            if (end) break;
            append_word(spoken, "dot");
            k = d + 1;
        }
        if (!tld || prose || last == Doc::npos || !right_edge(doc, doc[last].end)) continue;
        add_lc(doc, out, doc[first].start, doc[last].end, spoken, 86);
        i = last;
    }
}

// Social handles: @tontaube_ai -> at tontaube AI; @museum.org -> at museum dot org.
RV_RULE("electronic", handle) {
    const std::string_view t = doc.text;
    each_mark(doc, '@', [&](size_t i) {
        if (i + 1 >= doc.size() || !left_edge(doc, doc[i].start) || !doc.joined(i + 1)) return;
        if (doc[i + 1].kind == Kind::Punct) return;
        uint32_t s = doc[i + 1].start, e = s;
        while (e < t.size() && idc(t[e])) ++e;
        if (!al(t[s]) && t[s] != '_') return;
        Host h = parse_host(t, s, url_run(t, s), true);
        // "researchers at @vantage_sec" says "at" once
        size_t pw = doc.prev_word(i);
        bool said = pw != Doc::npos && pw + 1 == i && doc.is(pw, "at") && doc[i].space_before;
        std::string spoken = said ? "" : "at";
        if (h.end && !h.ip && right_edge(doc, h.end)) {
            append_word(spoken, host_words(t.substr(s, h.end - s)));
            e = h.end;
        } else {
            if (!right_edge(doc, e)) return;
            append_word(spoken, name_words(t.substr(s, e - s), true));
            // a package scope: @angular/core -> at angular slash core
            if (e + 1 < t.size() && t[e] == '/' && al(t[e + 1])) {
                uint32_t q = e + 1;
                while (q < t.size() && (an(t[q]) || t[q] == '-' || t[q] == '_')) ++q;
                if (right_edge(doc, q) && !(q < t.size() && (t[q] == '/' || t[q] == '.') && q + 1 < t.size() &&
                                            an(t[q + 1]))) {
                    append_word(spoken, "slash");
                    append_word(spoken, name_words(t.substr(e + 1, q - e - 1), true));
                    e = q;
                }
            }
            // a dotted name: @sarah.smith -> at sarah dot smith
            while (e + 1 < t.size() && t[e] == '.' && lo(t[e + 1])) {
                uint32_t q = e + 1;
                while (q < t.size() && idc(t[q])) ++q;
                if (!right_edge(doc, q) || (q + 1 < t.size() && t[q] == '.' && an(t[q + 1]))) break;
                append_word(spoken, "dot");
                append_word(spoken, name_words(t.substr(e + 1, q - e - 1), true));
                e = q;
            }
        }
        add_lc(doc, out, doc[i].start, e, spoken, 80);
    });
}

const rv::WordSet PREPROCESSOR{"define", "elif", "endif", "ifdef", "ifndef", "import", "include", "pragma", "undef"};

// Hashtags: #opensource -> hashtag opensource, #TechTrends -> hashtag Tech Trends.
RV_RULE("electronic", hashtag) {
    const std::string_view t = doc.text;
    each_mark(doc, '#', [&](size_t i) {
        if (!left_edge(doc, doc[i].start) || i + 1 >= doc.size() || !doc.joined(i + 1)) return;
        // #100DaysOfCode: digits then letters (a bare #12 is a number)
        if (!doc.word(i + 1) && !(doc.number(i + 1) && doc.word(i + 2) && doc.joined(i + 2) && doc.str(i + 2).size() >= 3))
            return;
        if (doc[i].newline_before || i == 0) {
            // a heading mark at the start of a line is markup, unless more of the line is tags or the tag is a
            // lower-case or CamelCase word followed by more text ("#throwback to 2019")
            size_t k = i + 2;
            bool tags = k < doc.size() && !doc[k].newline_before && doc.punct(k, '#');
            bool tag_word = doc.word(i + 1) && (doc[i + 1].casing == rv::LOWER || doc[i + 1].casing == rv::MIXED) &&
                            k < doc.size() && !doc[k].newline_before;
            if (!tags && !tag_word) return;
        }
        uint32_t s = doc[i + 1].start, e = s;
        while (e < t.size() && idc(t[e])) ++e;
        if (!right_edge(doc, e)) return;
        if (PREPROCESSOR.contains(t.substr(s, e - s))) {  // #include <stdio.h> is code, not a tag
            add_lc(doc, out, doc[i].start, e, "hash " + std::string(t.substr(s, e - s)), 80);
            return;
        }
        add_lc(doc, out, doc[i].start, e, "hashtag " + name_words(t.substr(s, e - s), false), 80);
    });
}

// Reddit names: r/TrueCrime -> R slash True Crime, u/HistoryBuff99 -> U slash History Buff ninety-nine.
RV_RULE("electronic", reddit) {
    const std::string_view t = doc.text;
    each_mark(doc, '/', [&](size_t k) {
        if (k == 0 || k + 1 >= doc.size()) return;
        size_t i = k - 1;
        if (!(doc.is(i, "r") || doc.is(i, "u")) || !left_edge(doc, doc[i].start)) return;
        if (!doc.joined(k) || !doc.joined(i + 2) || !doc.word(i + 2)) return;
        uint32_t s = doc[i + 2].start, e = s;
        while (e < t.size() && idc(t[e])) ++e;
        std::string tail;
        if (e + 1 < t.size() && (t[e] == '\'' || doc.text_at(e, "\xE2\x80\x99"))) {
            uint32_t k = t[e] == '\'' ? e + 1 : e + 3;
            if (k < t.size() && t[k] == 's' && right_edge(doc, k + 1)) {
                tail = std::string(t.substr(e, k + 1 - e));
                e = k + 1;
            }
        }
        if (!right_edge(doc, e)) return;
        uint32_t ne = tail.empty() ? e : e - static_cast<uint32_t>(tail.size());
        std::string spoken = upper_s(doc.str(i)) + " slash " + name_words(t.substr(s, ne - s), false) + tail;
        out.add(doc[i].start, e, spoken, 80);
    });
}

const rv::WordSet REPO_CUES{"checkpoint", "clone", "cloned", "fork", "forked", "install", "installed", "model",
                            "namespace", "package", "pin", "pinned", "pull", "pulled", "repo", "repos",
                            "repository", "slug"};
const rv::WordSet REPO_AFTER{"checkpoint", "codebase", "fork", "model", "package", "project", "repo",
                             "repositories", "repository"};
const char* const NAME_ENDINGS[] = {"bench", "core", "forge", "hub", "kit", "kits", "lab", "labs", "ops",
                                    "pipe", "sdk", "stack", "sync", "tech", "works"};

// A compound name with a typical ending: backupkit, kilnops, craitech (the ending alone is not one).
bool name_ending(std::string_view s) {
    std::string l = lower_s(s);
    for (const char* x : NAME_ENDINGS) {
        std::string_view e(x);
        if (l.size() >= e.size() + 3 && l.compare(l.size() - e.size(), e.size(), e) == 0) return true;
    }
    return false;
}
const rv::WordSet REPO_PLACE{"as", "at", "from", "in", "is", "of", "on", "through", "under", "via"};
const rv::WordSet REPO_DOC{"branch", "checkpoint", "checkpoints", "clone", "commit", "dataset", "datasets", "fork",
                           "forks", "github", "gitlab", "huggingface", "maintainers", "manifest", "mirror", "model",
                           "models", "package", "packages", "pull", "repo", "repos", "repositories", "repository",
                           "tokenizer", "weights"};

const rv::WordSet STRONG_PLACE{"at", "from", "in", "into", "through", "under", "via"};
const rv::WordSet PUSH_VERBS{"merged", "mirrored", "moved", "publish", "published", "push", "pushed", "pushes",
                             "pushing", "upload", "uploaded", "uploads"};
const char* const PAIR_SUFFIXES[] = {"able", "al", "ary", "ed", "ent", "ful", "ic", "ing", "ion", "ive", "less", "ory", "ous"};

// Two words that form an "or" pair: the same ending (approved/rejected, optional/mandatory) or the same start
// (lossy/lossless, in-scope/out-of-scope).
bool word_pair(std::string_view a, std::string_view c) {
    std::string x = lower_s(a), y = lower_s(c);
    size_t k = 0;
    while (k < x.size() && k < y.size() && x[k] == y[k]) ++k;
    if (k >= 3) return true;
    auto ends = [](const std::string& w, std::string_view e) {
        return w.size() > e.size() + 2 && w.compare(w.size() - e.size(), e.size(), e) == 0;
    };
    for (const char* e : PAIR_SUFFIXES)
        if (ends(x, e) && ends(y, e)) return true;
    if (x.size() >= 4 && y.size() >= 4 && x.compare(x.size() - 3, 3, y, y.size() - 3, 3) == 0) return true;
    return false;
}

bool repo_context(const Doc& doc) {
    for (size_t i = 0; i < doc.size(); ++i)
        if (doc.word(i) && doc[i].end - doc[i].start <= 12 && REPO_DOC.contains(doc.low(i))) return true;
    return false;
}

// Words of hyphenated compounds that pair with "or" (full-time/part-time, opt-in/opt-out).
const rv::WordSet COMPOUND{"append", "full", "go", "house", "in", "non", "only", "opt", "out", "part", "premise",
                           "read", "scope", "time", "write"};
// Pairs that are words with "or" or a fixed technical pair, never names.
const rv::WordSet PAIR_WORDS{"and", "client", "decoder", "download", "encoder", "export", "false", "her", "his",
                             "import", "input", "key", "no", "off", "on", "or", "output", "publish", "read",
                             "receive", "request", "response", "send", "server", "start", "stop", "subscribe",
                             "true", "upload", "value", "write", "yes"};

bool repo_part(std::string_view s, bool& signal) {
    if (s.empty() || !al(s[0]) || !an(s.back())) return false;
    bool digit = false, alpha = false;
    for (size_t k = 0; k < s.size(); ++k) {
        char c = s[k];
        if (dg(c)) digit = true;
        else if (al(c)) alpha = true;
        else if (c == '-' || c == '_') {
            if (k + 1 < s.size() && (s[k + 1] == '-' || s[k + 1] == '_')) return false;
            signal = true;
        } else {
            return false;
        }
        if (k >= 2 && up(c) && lo(s[k - 1]) && lo(s[k - 2])) signal = true;  // TontaubeAI
    }
    if (digit && alpha) signal = true;
    // hyphenated compounds of the "or" kind (opt-in, full-time)
    if (s.find('-') != std::string_view::npos) {
        size_t i = 0;
        while (i < s.size()) {
            size_t j = i;
            while (j < s.size() && al(s[j])) ++j;
            if (j > i && COMPOUND.contains(lower_s(s.substr(i, j - i)))) return false;
            i = j + 1;
        }
    }
    return alpha;
}

// Repository and model names: acme/atlas-viewer -> acme slash atlas viewer.
RV_RULE("electronic", repo) {
    const std::string_view t = doc.text;
    each_mark(doc, '/', [&](size_t k) {
        uint32_t m = doc[k].start, s = m;
        if (k == 0 || !doc.joined(k)) return;
        while (s > 0 && (an(t[s - 1]) || t[s - 1] == '-' || t[s - 1] == '_')) --s;
        size_t i = tok_at(doc, s);
        if (s == m || i >= doc.size() || !doc.word(i) || !left_edge(doc, s)) return;
        if (i > 0 && doc.number(i - 1)) return;
        uint32_t r = m + 1, e = r;
        while (e < t.size() && (an(t[e]) || t[e] == '-' || t[e] == '_')) ++e;
        if (e == r || !right_edge(doc, e)) return;
        if (e < t.size() && (t[e] == '/' || (t[e] == '.' && e + 1 < t.size() && an(t[e + 1])))) return;
        std::string_view a = t.substr(s, m - s), c = t.substr(r, e - r);
        if (a.size() == 1 || c.size() == 1) return;
        bool signal = false;
        if (!repo_part(a, signal) || !repo_part(c, signal)) return;
        if (PAIR_WORDS.contains(lower_s(a)) || PAIR_WORDS.contains(lower_s(c))) return;
        if ((a == "origin" || a == "upstream") && !word_pair(a, c)) signal = true;  // a git branch: origin/main
        if (!signal) {
            size_t pw = doc.prev_word(i);
            size_t j = doc.token_at(e);
            size_t nw = j < doc.size() && doc[j].kind != Kind::Punct ? j : doc.next_word(j);
            bool strong = (pw != Doc::npos && REPO_CUES.contains(doc.low(pw))) ||
                          (nw != Doc::npos && nw < doc.size() && REPO_AFTER.contains(doc.low(nw)));
            // name shapes: compounds with a typical ending (backupkit, kilnops) or Owner/name
            bool shape = name_ending(a) || name_ending(c) ||
                         (up(a[0]) && all_lower(a.substr(1)) && all_lower(c) && !doc.sentence_start(i));
            bool long_parts = a.size() >= 4 && c.size() >= 4;
            // in a text about code, models or data: two lower-case names after "at", "under", "from" ...
            bool place = pw != Doc::npos && REPO_PLACE.contains(doc.low(pw)) && a.size() >= 5 && c.size() >= 5 &&
                         all_lower(c) && (all_lower(a) || (up(a[0]) && all_lower(a.substr(1)))) && repo_context(doc);
            // after "at", "from", "through", "under" a pair of names is a repository nearly always (in pairs of
            // words the second one contrasts with the first: pass/fail, lossy/lossless, approved/rejected)
            bool where = pw != Doc::npos && pw + 1 == i && STRONG_PLACE.contains(doc.low(pw)) && a.size() >= 3 &&
                         c.size() >= 3 && !word_pair(a, c) &&
                         (signal || (all_lower(c) && (all_lower(a) || (up(a[0]) && all_lower(a.substr(1)))) &&
                                     a.size() + c.size() >= 12 && std::max(a.size(), c.size()) >= 7));
            // "the repository is nimbusnet/edge", "pushed the model to harbor/pinetext"
            if (!where && pw != Doc::npos && pw + 1 == i && a.size() >= 3 && c.size() >= 3 && !word_pair(a, c)) {
                size_t p2 = doc.prev_word(pw);
                if (doc.is(pw, "is") || doc.is(pw, "was") || doc.is(pw, "are"))
                    where = p2 != Doc::npos && (REPO_CUES.contains(doc.low(p2)) || REPO_AFTER.contains(doc.low(p2)) ||
                                                doc.is(p2, "identifier") || doc.is(p2, "identifiers"));
                else if (doc.is(pw, "to") || doc.is(pw, "into"))
                    for (int back = 0; back < 4 && p2 != Doc::npos && !where; ++back, p2 = doc.prev_word(p2))
                        where = PUSH_VERBS.contains(doc.low(p2));
            }
            if (!strong && !(long_parts && shape) && !place && !where) return;
        }
        add_lc(doc, out, s, e, name_words(a, true) + " slash " + name_words(c, true), 70);
    });
}

const rv::WordSet HTTP_BEFORE{"answered", "answers", "code", "codes", "emits", "emitted", "error", "gave", "get", "gets",
                              "gives", "got", "hit", "hits", "http", "logged", "raised", "raises", "reported",
                              "reporting", "reports", "respond", "responded", "responds", "return", "returned",
                              "returning", "returns", "saw", "sees", "send", "sends", "sent", "served", "status",
                              "throw", "throws", "yielded", "yields"};
const rv::WordSet HTTP_AFTER{"accepted", "bad", "case", "cases", "code", "conflict", "created", "error", "errors",
                             "forbidden", "found", "gateway", "handler", "handlers", "internal", "moved", "ok", "page",
                             "pages", "path", "paths", "redirect", "redirects", "reply", "response", "retries",
                             "retry", "spike", "spikes", "status", "timeout", "timeouts", "unauthorized",
                             "unavailable"};
// Verbs before a code that is not in common use (449, 466): the code is what a server sends.
const rv::WordSet SEND_BEFORE{"answered", "answers", "http", "respond", "responded", "responds", "return", "returned",
                              "returning", "returns", "sends", "sent", "served", "status", "throw", "throws"};

const rv::WordSet WITH_VERBS{"answered", "answers", "ended", "ends", "fail", "failed", "fails", "rejected",
                             "rejects", "replied", "replies", "responded", "responds", "returned"};

bool status_code(const Doc& doc, size_t i) {
    if (!doc.number(i) || doc.str(i).size() != 3) return false;
    char c = doc.str(i)[0];
    if (c < '1' || c > '5') return false;
    uint32_t s = doc[i].start, e = doc[i].end;
    if (!space_before(doc, s) && !(s > 0 && (doc.text[s - 1] == '(' || doc.text[s - 1] == '`'))) return false;
    if (!right_edge(doc, e)) return false;
    if (e < doc.text.size() && (doc.text[e] == '.' || doc.text[e] == ',') && e + 1 < doc.text.size() &&
        dg(doc.text[e + 1]))
        return false;
    // a group of a number written with spaces: 2 480 000
    if (i > 0 && doc.number(i - 1) && doc[i - 1].end + 1 == s && doc.str(i - 1).size() <= 3) return false;
    if (doc.number(i + 1) && doc[i + 1].start == e + 1 && doc.str(i + 1).size() == 3) return false;
    return true;
}

// Codes in use: the 1xx-3xx that servers send, 400-451 and the 4xx of proxies and clouds (460-499), 500-511,
// 520-530, 598-599.
bool known_code(unsigned n) {
    switch (n) {
        case 100: case 101: case 102: case 103: case 200: case 201: case 202: case 203: case 204: case 205:
        case 206: case 207: case 208: case 226: case 444: case 449: case 598: case 599:
            return true;
        default:
            return (n >= 300 && n <= 308) || (n >= 400 && n <= 431) || (n >= 451 && n <= 499) ||
                   (n >= 500 && n <= 511) || (n >= 520 && n <= 530);
    }
}

const rv::WordSet HTTP_DOC{"api", "apis", "backend", "cdn", "crawler", "endpoint", "endpoints", "gateway", "http",
                           "https", "nginx", "proxy", "server", "servers", "upstream", "webhook", "webhooks"};
// Words that make a text about HTTP only two at a time ("client" and "request").
const rv::WordSet HTTP_WEAK{"client", "clients", "origin", "portal", "request", "requests", "retries", "retry",
                            "timeout", "timeouts"};
const rv::WordSet AFTER_CODE{"after", "again", "also", "and", "appeared", "appears", "are", "as", "at", "back",
                             "because", "before", "by", "came", "comes", "during", "for", "from", "happened",
                             "happens", "if", "in", "indicates", "instead", "into", "is", "later", "means",
                             "occurred", "occurs", "on", "once", "only", "or", "persisted", "points", "should",
                             "showed", "shows", "signals", "suggests", "then", "to", "under", "unless", "until",
                             "usually", "was", "were", "when", "whenever", "while", "with", "within"};
const rv::WordSet CODE_NOUNS{"codes", "rejections", "replies", "responses", "statuses"};
const rv::WordSet NOT_PLURAL{"across", "always", "appears", "as", "causes", "comes", "has", "indicates", "is",
                             "its", "less", "means", "occurs", "plus", "requires", "returns", "shows", "signals",
                             "suggests", "this", "thus", "triggers", "unless", "was"};
const rv::WordSet UNIT_AFTER{"c", "cal", "cm", "dpi", "f", "fps", "ft", "g", "gb", "h", "hz", "k", "kb", "kcal",
                             "kg", "khz", "km", "kw", "kwh", "l", "lb", "lbs", "m", "mb", "mi", "min", "ml", "mm",
                             "mph", "ms", "oz", "px", "rpm", "sec", "tb", "v", "w"};
const rv::WordSet LIST_HEADS{"error", "http", "response", "return", "status"};

unsigned code_value(const Doc& doc, size_t i) { return static_cast<unsigned>(std::stoi(std::string(doc.str(i)))); }

// A word that makes a text about HTTP: "API", "server", "REST", "health check"; not the scheme of a web address.
// weak: "client", "request" (counted by the caller).
bool http_word(const Doc& doc, size_t i, int& weak) {
    if (doc[i].kind != Kind::Word || doc[i].end - doc[i].start < 3 || doc[i].end - doc[i].start > 9) return false;
    if (doc.is(i, "health") || doc.is(i, "synthetic"))
        return doc.is(i + 1, "check") || doc.is(i + 1, "checks") || doc.is(i + 1, "probe") || doc.is(i + 1, "probes");
    if (doc.str(i) == "REST") return true;
    if (HTTP_WEAK.contains(doc.low(i))) return ++weak >= 2;
    if (!HTTP_DOC.contains(doc.low(i))) return false;
    return !(doc.punct(i + 1, ':') && doc.joined(i + 1) && doc.punct(i + 2, '/'));
}

// HTTP status codes: HTTP 404 -> HTTP four oh four, a 429 error, returns 500, 200 OK, status: 503, and the codes
// listed right after one of them; in a text about HTTP or an API, any code in use.
RV_RULE("electronic", http_status) {
    // three-digit numbers 100-599 that stand alone
    size_t cand[32];
    size_t n = 0;
    for (size_t i = 0; i < doc.size() && n < 32; ++i)
        if (doc[i].kind == Kind::Number && doc[i].end - doc[i].start == 3 && status_code(doc, i)) cand[n++] = i;
    if (n == 0) return;
    bool done[32] = {};
    bool ctx = false;
    for (size_t c = 0; c < n; ++c) {
        size_t i = cand[c];
        if (done[c]) continue;
        bool known = known_code(code_value(doc, i));
        size_t pw = doc.prev_word(i);
        if (!known && !(pw != Doc::npos && (pw + 1 == i || pw + 2 == i))) continue;
        size_t nw = doc.next_word(i);
        bool after = known && nw == i + 1 && doc[nw].space_before && HTTP_AFTER.contains(doc.low(nw));
        bool before = pw != Doc::npos && pw + 1 == i && HTTP_BEFORE.contains(doc.low(pw));
        if (!before && pw != Doc::npos && pw + 1 == i && doc.is(pw, "with")) {  // failed with 413
            size_t v = doc.prev_word(pw);
            before = v != Doc::npos && v + 1 == pw && WITH_VERBS.contains(doc.low(v));
        }
        if (!before && pw != Doc::npos && pw + 1 == i && doc.is(pw, "of")) {  // a status of 202
            size_t v = doc.prev_word(pw);
            before = v != Doc::npos && v + 1 == pw && (doc.is(v, "status") || doc.is(v, "code"));
        }
        if (!before && pw != Doc::npos && pw + 2 == i && doc.punct(i - 1, ':'))  // Status: 503
            before = doc.is(pw, "status") || doc.is(pw, "code") || doc.is(pw, "http");
        if (!before && !after && known && pw != Doc::npos && pw + 1 == i && (doc.is(pw, "a") || doc.is(pw, "an"))) {
            // "retry a 504 after a delay": a 4xx or 5xx code that is not a round number, then a pause or a
            // function word (never "a 401(k)")
            unsigned v = code_value(doc, i);
            bool tail = !(nw == i + 1 && doc.word(nw)) || AFTER_CODE.contains(doc.low(nw));
            before = v >= 400 && v % 100 != 0 && tail && !doc.punct(i + 1, '(') && !doc.punct(i + 1, '%') &&
                     !(doc.word(i + 1) && doc.joined(i + 1));
        }
        if (before && !after && !doc.is(pw, "http") && nw == i + 1 && doc.word(nw)) {
            // "logged 503 requests" counts something; "saw 503 responses" are codes
            std::string_view w = doc.low(nw);
            if (UNIT_AFTER.contains(w) ||
                (w.size() > 2 && w.back() == 's' && !NOT_PLURAL.contains(w) && !CODE_NOUNS.contains(w)))
                before = false;
        }
        if (before && !known) {
            // a code not in common use only after a verb of sending, followed by punctuation or a function word
            bool send = SEND_BEFORE.contains(doc.low(pw)) || doc.punct(i - 1, ':') || doc.is(pw, "of");
            bool tail = !(nw == i + 1 && doc.word(nw)) || AFTER_CODE.contains(doc.low(nw)) ||
                        HTTP_AFTER.contains(doc.low(nw));
            before = send && tail;
        }
        if (!before && !after) continue;
        ctx = true;
        out.tokens(i, i, rv::pairs(code_value(doc, i)), 70);
        done[c] = true;
        // a list right after it: 404, 429 and 500
        size_t k = i + 1, d = c + 1;
        while (d < n) {
            size_t m = k;
            if (doc.punct(m, ',') || doc.punct(m, '/')) ++m;
            if (doc.is(m, "and") || doc.is(m, "or")) ++m;
            if (m == k || cand[d] != m) break;
            out.tokens(m, m, rv::pairs(code_value(doc, m)), 70);
            done[d++] = true;
            k = m + 1;
        }
    }
    bool rest = false;
    for (size_t c = 0; c < n; ++c) rest |= !done[c] && known_code(code_value(doc, cand[c]));
    if (!rest) return;
    // "Status codes seen in the log, in order: 418, 502, 451": a list after the words "status codes"
    bool code_list = false;
    for (size_t i = 0; i + 1 < doc.size() && !code_list; ++i)
        code_list = doc[i].kind == Kind::Word && doc[i].end - doc[i].start >= 4 && doc[i].end - doc[i].start <= 8 &&
                    (doc.is(i + 1, "codes") || doc.is(i + 1, "code")) && LIST_HEADS.contains(doc.low(i));
    int weak = 0;
    ctx |= code_list;
    for (size_t i = 0; i < doc.size() && !ctx; ++i) ctx = http_word(doc, i, weak);
    if (!ctx) return;
    for (size_t c = 0; c < n; ++c) {
        size_t i = cand[c];
        if (done[c] || !known_code(code_value(doc, i))) continue;
        if (code_list && i > 0 && (doc.punct(i - 1, ':') || doc.punct(i - 1, ',') || doc.is(i - 1, "and")) &&
            !(i + 1 < doc.size() && doc.word(i + 1) && !doc.is(i + 1, "and"))) {
            out.tokens(i, i, rv::pairs(code_value(doc, i)), 60);
            continue;
        }
        // a code in running text is followed by punctuation, a function word or a status phrase (206 Partial
        // Content), or comes after "a" (a 422 appeared); never a count of something
        size_t pw = doc.prev_word(i);
        bool article = pw != Doc::npos && pw + 1 == i && (doc.is(pw, "a") || doc.is(pw, "an") || doc.is(pw, "any"));
        if (i + 1 < doc.size() && doc.word(i + 1) && !article && !AFTER_CODE.contains(doc.low(i + 1)) &&
            !HTTP_AFTER.contains(doc.low(i + 1)) && !CODE_NOUNS.contains(doc.low(i + 1)) &&
            !(doc[i + 1].casing == rv::TITLE && !doc.sentence_start(i + 1)))
            continue;
        if (i + 1 < doc.size() && (doc.punct(i + 1, ']') || doc.punct(i + 1, '%'))) continue;
        if (doc.word(i + 1) && UNIT_AFTER.contains(doc.low(i + 1))) continue;  // a 480 ms ceiling
        if (i > 0 && (doc.punct(i - 1, '[') || doc.punct(i - 1, ':') || doc.punct(i - 1, '#') ||
                      doc.punct(i - 1, '$')))
            continue;
        if (pw != Doc::npos && pw + 2 == i && doc.punct(i - 1, ',') && doc.number(pw)) continue;  // [101, 102]
        // a range of counts: "rose from 412 to 655"
        if ((doc.is(i + 1, "to") || doc.is(i + 1, "and")) && doc.number(i + 2) &&
            !(c + 1 < n && cand[c + 1] == i + 2 && known_code(code_value(doc, i + 2))))
            continue;
        if (pw != Doc::npos && pw + 1 == i && (doc.is(pw, "from") || doc.is(pw, "between")) && doc.is(i + 1, "to"))
            continue;
        out.tokens(i, i, rv::pairs(code_value(doc, i)), 60);
    }
}

}  // namespace
