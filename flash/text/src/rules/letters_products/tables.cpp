// Fixed tables of the module "letters_products".
#include "rules/letters_products/lp.h"

namespace lp {

// Written form -> spoken form, exact case: acronyms said as a word, names with inner capitals, letters plus a
// word. Regular CamelCase names need no entry (mixed_word() splits them).
const std::string_view* respelling(std::string_view w) {
    static const rv::PhraseMap MAP{
        {"ACME", "Acme"}, {"AIDS", "Aids"}, {"AMOLED", "Amoled"}, {"ASCII", "Ascii"}, {"AVIF", "Avif"}, {"AVRO", "Avro"}, {"BERT", "Bert"}, {"BIO", "Bio"}, {"BIOS", "Bios"}, {"BLEU", "Bleu"},
        {"BOGO", "Bogo"}, {"BSON", "Bson"}, {"CALEA", "Calea"}, {"CAPEX", "Capex"}, {"CLIP", "Clip"}, {"CAPTCHA", "Captcha"},
        {"COBIT", "Cobit"}, {"COBOL", "Cobol"}, {"COGS", "Cogs"}, {"COP", "Cop"}, {"COPPA", "Coppa"},
        {"COSO", "Coso"}, {"COVID", "Covid"}, {"CUDA", "Cuda"}, {"CapEx", "Capex"}, {"ChatGPT", "Chat GPT"},
        {"CoV", "Cov"}, {"DORA", "Dora"}, {"DRAM", "D Ram"}, {"DataLoader", "Dataloader"},
        {"DeBERTa", "Deberta"}, {"DeepMind", "Deepmind"}, {"DistilBERT", "Distilbert"}, {"EBIT", "Ebit"}, {"EBITA", "Ebita"}, {"EBITDA", "Ebitda"},
        {"EBITDAR", "Ebitdar"}, {"ELO", "Elo"}, {"EPUB", "E Pub"}, {"EPUBs", "E Pubs"}, {"ERISA", "Erisa"},
        {"ESOP", "Esop"}, {"FAISS", "Faiss"}, {"FATCA", "Fatca"}, {"FEMA", "Fema"}, {"FERPA", "Ferpa"},
        {"FHIR", "Fhir"}, {"FIFA", "Fifa"}, {"FIFO", "Fifo"}, {"FINRA", "Finra"}, {"FIPS", "Fips"},
        {"FISMA", "Fisma"}, {"FLAC", "Flac"}, {"FLOPS", "Flops"}, {"FORTRAN", "Fortran"},
        {"FastAPI", "Fast API"}, {"FedRAMP", "Fedramp"}, {"FinOps", "Finops"}, {"FinTech", "Fintech"},
        {"FlashAttention", "Flashattention"}, {"GAAP", "Gaap"}, {"GitHub", "Github"}, {"GitLab", "Gitlab"},
        {"GIF", "Gif"}, {"GIFs", "Gifs"}, {"GmbH", "GMBH"}, {"HECVAT", "Hecvat"}, {"HEPA", "Hepa"}, {"HIPAA", "Hipaa"}, {"HITECH", "Hitech"},
        {"HITRUST", "Hitrust"}, {"IaaS", "Iaas"}, {"ITAR", "Itar"}, {"IoT", "IOT"}, {"JPEG", "J Peg"},
        {"JSON", "Json"}, {"JavaScript", "Javascript"}, {"KOSPI", "Kospi"}, {"LASER", "Laser"},
        {"LDAP", "Ldap"}, {"LIDAR", "Lidar"}, {"LIMS", "Lims"}, {"LangChain", "Langchain"},
        {"LLaMA", "Llama"}, {"LaTeX", "Latex"}, {"LiDAR", "Lidar"}, {"LoRaWAN", "Lorawan"}, {"LinkedIn", "Linkedin"}, {"LoRA", "Lora"}, {"MEMS", "Mems"},
        {"MERS", "Mers"}, {"MIDI", "Midi"}, {"MIPS", "Mips"}, {"MLOps", "ML Ops"}, {"MODIS", "Modis"},
        {"MOOC", "Mooc"}, {"MOSFET", "Mosfet"}, {"MPEG", "M Peg"}, {"MacBook", "Macbook"},
        {"NACHA", "Nacha"}, {"NAFTA", "Nafta"}, {"NASA", "Nasa"}, {"NDJSON", "ND Json"}, {"NASCAR", "Nascar"},
        {"NASDAQ", "Nasdaq"}, {"NATO", "Nato"}, {"NERC", "Nerc"}, {"NOAA", "Noaa"}, {"NOPAT", "Nopat"},
        {"NOPLAT", "Noplat"}, {"NUMA", "Numa"}, {"NVIDIA", "Nvidia"}, {"NVMe", "NVM E"}, {"NumPy", "Numpy"},
        {"OLED", "Oled"}, {"ONNX", "Onnx"}, {"OPEC", "Opec"}, {"OPEX", "Opex"}, {"OPFOR", "Opfor"},
        {"OSHA", "Osha"}, {"OpEx", "Opex"}, {"OpenAI", "Open AI"}, {"OpenAPI", "Open API"},
        {"OpenCV", "Open CV"}, {"OpenMP", "Open MP"}, {"OpenSSH", "Open SSH"}, {"OpenSSL", "Open SSL"},
        {"PESTLE", "Pestle"}, {"PIPEDA", "Pipeda"}, {"PaaS", "Paas"}, {"PARQUET", "Parquet"},
        {"PostgreSQL", "Postgre SQL"}, {"PyTorch", "Pytorch"}, {"RADAR", "Radar"}, {"RAID", "Raid"},
        {"RAM", "Ram"}, {"ROAS", "Roas"}, {"ROCE", "Roce"}, {"ROUGE", "Rouge"}, {"ResNet", "Resnet"}, {"RoBERTa", "Roberta"},
        {"SARS", "Sars"}, {"SATA", "Sata"}, {"SCADA", "Scada"}, {"SCSI", "Scsi"}, {"SDRAM", "S Dram"}, {"SCUBA", "Scuba"},
        {"SIMD", "Simd"}, {"SOAP", "Soap"}, {"SONAR", "Sonar"}, {"SOX", "Sox"},
        {"SQLite", "Sqlite"}, {"SQUID", "Squid"}, {"SRAM", "S Ram"}, {"STOXX", "Stoxx"}, {"SWOT", "Swot"},
        {"SaaS", "Saas"}, {"SoundCloud", "Soundcloud"}, {"TISAX", "Tisax"}, {"TOML", "Toml"},
        {"TensorFlow", "Tensorflow"}, {"TypeScript", "Typescript"}, {"UNESCO", "Unesco"},
        {"UNICEF", "Unicef"}, {"UNIX", "Unix"}, {"VRAM", "V Ram"}, {"VRAMs", "V Rams"}, {"WaveNet", "Wavenet"},
        {"WebRTC", "Web RTC"}, {"WiFi", "Wifi"}, {"XBox", "Xbox"}, {"XRAY", "Xray"}, {"YAML", "Yaml"},
        {"YOLO", "Yolo"}, {"YouTube", "Youtube"}, {"arXiv", "Arxiv"}, {"chromeOS", "Chrome OS"},
        {"dGPU", "D GPU"}, {"eBay", "Ebay"}, {"eCommerce", "Ecommerce"}, {"eDRAM", "E Dram"}, {"eBook", "E Book"}, {"eMMC", "E MMC"}, {"eSATA", "E Sata"}, {"eSIM", "E Sim"},
        {"fireOS", "Fire OS"}, {"iOS", "I OS"}, {"iPad", "Ipad"}, {"iPadOS", "Ipad OS"}, {"iPhone", "Iphone"},
        {"iSCSI", "I Scsi"}, {"mRNA", "M RNA"}, {"mSATA", "M Sata"}, {"macOS", "Mac OS"}, {"pH", "PH"},
        {"tvOS", "TV OS"}, {"vCPU", "V CPU"}, {"vCPUs", "V CPUs"}, {"vGPU", "V GPU"}, {"vLLM", "V LLM"},
        {"vPro", "V Pro"}, {"vSAN", "V San"}, {"vTPM", "V TPM"}, {"visionOS", "Vision OS"},
        {"watchOS", "Watch OS"}, {"wearOS", "Wear OS"}, {"webOS", "Web OS"}, {"zRAM", "Z Ram"},
        // acronyms, names and spelling-alphabet words said as a word that lp::word_like() does not catch
        {"AJAX", "Ajax"}, {"ALFA", "Alfa"}, {"AMTRAK", "Amtrak"}, {"ANZAC", "Anzac"}, {"ASEAN", "Asean"},
        {"ASICS", "Asics"}, {"AWOL", "Awol"}, {"CERN", "Cern"}, {"DARPA", "Darpa"}, {"DEFCON", "Defcon"},
        {"ECMO", "Ecmo"}, {"EPCOT", "Epcot"}, {"FAFSA", "Fafsa"}, {"FIAT", "Fiat"}, {"FLOTUS", "Flotus"},
        {"FOMO", "Fomo"}, {"FOXTROT", "Foxtrot"}, {"FUBAR", "Fubar"}, {"GEICO", "Geico"}, {"GLUE", "Glue"},
        {"GOLF", "Golf"}, {"GRUB", "Grub"}, {"HAMAS", "Hamas"}, {"HAZMAT", "Hazmat"}, {"IKEA", "Ikea"},
        {"INDIA", "India"}, {"ISIS", "Isis"}, {"JULIETT", "Juliett"}, {"KILO", "Kilo"}, {"LEGO", "Lego"},
        {"LIFO", "Lifo"}, {"MIKE", "Mike"}, {"MIMO", "Mimo"}, {"NAND", "Nand"}, {"NIMBY", "Nimby"},
        {"NOMAD", "Nomad"}, {"NORAD", "Norad"}, {"OSCAR", "Oscar"}, {"PAPA", "Papa"}, {"PEST", "Pest"},
        {"POSIX", "Posix"}, {"POTUS", "Potus"}, {"QANTAS", "Qantas"}, {"SAMBA", "Samba"}, {"SCOTUS", "Scotus"},
        {"SETI", "Seti"}, {"SNAFU", "Snafu"}, {"SWAT", "Swat"}, {"TANGO", "Tango"}, {"TASER", "Taser"},
        {"TESOL", "Tesol"}, {"TOEFL", "Toefl"}, {"UEFA", "Uefa"}, {"WYSIWYG", "Wysiwyg"}, {"YANKEE", "Yankee"},
        {"ZULU", "Zulu"},
    };
    return MAP.find(w);
}

// Initialisms written in lower case that are spelled out, and chat shorthand said as letters.
const std::string_view* lower_initialism(std::string_view w) {
    static const rv::PhraseMap MAP{
        {"api", "API"}, {"apis", "APIs"}, {"asap", "ASAP"}, {"asr", "ASR"}, {"atm", "ATM"}, {"atms", "ATMs"},
        {"aws", "AWS"}, {"bbc", "BBC"}, {"btw", "BTW"}, {"ceo", "CEO"}, {"ceos", "CEOs"},
        {"cfo", "CFO"}, {"cia", "CIA"}, {"cli", "CLI"}, {"cnn", "CNN"}, {"cpu", "CPU"}, {"cpus", "CPUs"},
        {"crm", "CRM"}, {"css", "CSS"}, {"csv", "CSV"}, {"cto", "CTO"}, {"diy", "DIY"}, {"dm", "DM"},
        {"dms", "DMs"}, {"dna", "DNA"}, {"dns", "DNS"}, {"dvd", "DVD"}, {"dvds", "DVDs"}, {"faq", "FAQ"},
        {"faqs", "FAQs"}, {"fbi", "FBI"}, {"fps", "FPS"}, {"fyi", "FYI"}, {"gdp", "GDP"}, {"gps", "GPS"},
        {"gpu", "GPU"}, {"gpus", "GPUs"}, {"gtx", "GTX"}, {"hdd", "HDD"}, {"hdmi", "HDMI"}, {"hiv", "HIV"},
        {"html", "HTML"}, {"ios", "I OS"}, {"ipados", "Ipad OS"},
        {"json", "Json"}, {"kpi", "KPI"}, {"macos", "Mac OS"},
        {"kpis", "KPIs"}, {"llm", "LLM"}, {"llms", "LLMs"}, {"lol", "LOL"}, {"nda", "NDA"}, {"ndas", "NDAs"},
        {"nba", "NBA"}, {"nfl", "NFL"}, {"nhl", "NHL"}, {"nlp", "NLP"}, {"npu", "NPU"}, {"nyc", "NYC"},
        {"ocr", "OCR"}, {"omg", "OMG"}, {"pc", "PC"}, {"pcs", "PCs"}, {"pdf", "PDF"}, {"pdfs", "PDFs"},
        {"roi", "ROI"}, {"rsvp", "RSVP"}, {"rtf", "RTF"}, {"rtx", "RTX"}, {"sdk", "SDK"}, {"sdks", "SDKs"}, {"seo", "SEO"},
        {"sftp", "SFTP"}, {"suv", "SUV"}, {"suvs", "SUVs"},
        {"sla", "SLA"}, {"slas", "SLAs"}, {"sms", "SMS"}, {"sop", "SOP"}, {"sops", "SOPs"}, {"sql", "SQL"},
        {"ssd", "SSD"}, {"ssds", "SSDs"}, {"ssh", "SSH"}, {"tcp", "TCP"}, {"tpu", "TPU"}, {"tpus", "TPUs"},
        {"tts", "TTS"}, {"tv", "TV"}, {"tvs", "TVs"}, {"udp", "UDP"}, {"ui", "UI"}, {"uk", "UK"},
        {"url", "URL"}, {"urls", "URLs"}, {"usa", "USA"}, {"usb", "USB"},
        {"ux", "UX"}, {"vpn", "VPN"}, {"vram", "V Ram"}, {"wer", "WER"}, {"xml", "XML"}, {"yaml", "Yaml"},
    };
    // quick reject on the first two letters before the search
    struct Pairs {
        uint8_t bits[26 * 26 / 8 + 1] = {};
        explicit Pairs(const rv::PhraseMap& m) {
            for (const auto& it : m.items()) {
                unsigned k = static_cast<unsigned>(it.first[0] - 'a') * 26 + static_cast<unsigned>(it.first[1] - 'a');
                bits[k / 8] = static_cast<uint8_t>(bits[k / 8] | (1u << (k % 8)));
            }
        }
    };
    static const Pairs PAIRS(MAP);
    if (w.size() < 2 || w[0] < 'a' || w[0] > 'z' || w[1] < 'a' || w[1] > 'z') return nullptr;
    unsigned k = static_cast<unsigned>(w[0] - 'a') * 26 + static_cast<unsigned>(w[1] - 'a');
    if (!(PAIRS.bits[k / 8] & (1u << (k % 8)))) return nullptr;
    return MAP.find(w);
}

bool keep_caps(std::string_view w) {
    static const rv::WordSet SET{
        "ACLU", "AEST", "AKIA", "APAC", "ASPI", "ARPU", "ASAP", "ASIC", "CEST", "CHRO", "CISO", "CMOS", "DELETE",
        "EMEA", "IMHO", "OBGYN", "BESS", "BYOB", "EEPROM", "UCLA", "USAA", "USAF",
        "GET", "HEAD", "HEIC", "IATA", "IEEE", "IOPS", "MAPE", "NIC", "NIST", "NULL", "NYSE", "OLAP", "OLTP",
        "OPTIONS", "PATCH", "POST", "PUT", "README", "REIT", "REST", "ROIC", "STEM", "TIFF", "TODO", "UUID",
    };
    return SET.contains(w);
}

bool emphasis(std::string_view w) {
    static const rv::WordSet SET{"ALL", "AND", "ANY", "ARE", "BAD", "BE", "BIG", "BUT", "BY", "DID", "DO",
                                 "FEW", "FOR", "GO", "GOT", "HAD", "HAS", "HOT", "IF", "LOW", "MY", "NEW",
                                 "NO", "NOR", "NOT", "NOW", "OFF", "OLD", "ONE", "OUR", "OUT", "OWN", "SO", "THE",
                                 "TOO", "TWO", "UP", "USE", "WAS", "WAY", "WE", "WHY", "YES", "YET", "YOU"};
    return SET.contains(w);
}

bool log_level(std::string_view w) {
    static const rv::WordSet SET{"CRITICAL", "DEBUG", "ERROR", "FATAL", "INFO", "TRACE", "WARN", "WARNING"};
    return SET.contains(w);
}

bool unit_word(std::string_view w) {
    static const rv::WordSet SET{
        "Ah", "EB", "EiB", "GB", "GBps", "GHz", "GW", "GWh", "Gb", "Gbps", "GiB", "Hz", "J",
        "KB", "KiB", "L", "MB", "MBps", "MHz", "MJ", "MPa", "MW", "MWh", "Mb", "Mbps", "MiB", "N", "Nm", "PB",
        "Pa", "PiB", "TB", "THz", "TW", "TWh", "Tb", "TiB", "V", "W", "Wh", "dB", "dBA", "dBm", "eV", "GeV",
        "hPa", "kB", "kHz", "kJ", "kN", "kPa", "kV", "kW", "kWh", "kbps", "Kbps", "keV", "mA", "mAh", "mL",
        "mV", "mW", "MeV", "TeV", "mmHg", "kcal", "kg", "km", "mg", "mm", "cm", "nm", "ms", "us", "ns",
    };
    return SET.contains(w);
}

bool version_cue(std::string_view w) {
    static const rv::WordSet SET{
        "alpine", "android", "angular", "ansible", "apache", "bazel", "bluetooth", "build", "builds",
        "candidate", "centos", "chrome", "chromium", "clang", "cuda", "debian", "django", "docker", "dotnet",
        "electron", "elixir", "emacs", "erlang", "fedora", "firefox", "firmware", "flask", "flutter",
        "gcc", "gnome", "golang", "gradle", "hdmi", "helm", "hotfix", "http", "ios", "java", "jdk", "jquery", "js",
        "kafka", "kde", "kernel", "kotlin", "kubernetes", "laravel", "linux", "llvm", "lua", "macos", "mariadb",
        "maven", "mongodb", "mysql", "nginx", "node", "nodejs", "npm", "numpy", "openssl", "pandas", "patch",
        "pcie", "perl", "php", "pip", "podman", "postgres", "postgresql", "python", "pytorch", "rails",
        "react", "redis", "rel", "release", "releases", "rhel", "ruby", "rust", "rustc", "safari", "scala",
        "spark", "spring", "sqlite", "swift", "tag", "tagged", "tags", "tensorflow", "terraform", "tls",
        "typescript", "ubuntu", "update", "upgrade", "usb", "ver", "version", "versions", "vue", "webpack",
        "windows", "xcode",
    };
    return SET.contains(w);
}

bool not_model_cue(std::string_view w) {
    static const rv::WordSet SET{
        "about", "above", "account", "after", "act", "add", "aged", "ages", "apr", "april", "apt", "around", "art",
        "article", "aug", "batch", "booking", "bus", "case", "claim", "confirmation", "dec", "exit", "feb", "file",
        "form", "invoice", "item", "jan", "jul", "jun", "lot", "mar", "nov", "oct", "order", "participant",
        "patient", "receipt", "record", "ref", "reservation", "sample", "sep", "sept", "serial", "specimen",
        "subject", "ticket", "tracking", "train",
        "august", "before", "below", "between", "bldg", "box", "by", "ca", "chapter", "chap", "circa", "class",
        "cohort", "day", "days", "december", "district", "each", "eq", "equation", "every", "exhibit", "february",
        "fig", "figure", "flight", "floor", "for", "from", "gate", "grade", "highway", "hwy", "in", "interstate",
        "issue", "january", "july", "june", "level", "line", "lines", "march", "may", "mile", "november", "no",
        "nos", "num", "number", "october", "of", "on", "only", "op", "opus", "over", "p", "page", "pages", "para",
        "paragraph", "part", "phase", "pier", "platform", "pp", "psalm", "q", "question", "room", "route", "row",
        "rule", "rules", "sec", "section", "sections", "september", "since", "stage", "step", "steps", "suite",
        "table", "terminal", "than", "the", "through", "till", "to", "track", "type", "under", "unit", "until",
        "up", "vol", "volume", "ward", "week", "weeks", "with", "year", "years", "zone",
    };
    return SET.contains(w);
}

bool count_noun(std::string_view w) {
    static const rv::WordSet SET{
        "accounts", "articles", "attendees", "books", "calls", "cases", "characters", "chars", "clicks",
        "clips", "customers", "devices", "documents", "docs", "downloads", "employees", "entries", "events",
        "examples", "files", "followers", "frames", "hits", "hours", "images", "installs", "items", "lines",
        "listeners", "members", "messages", "miles", "operations", "orders", "pages", "pairs", "parameter",
        "parameters", "params", "people", "photos", "players", "points", "posts", "queries", "readers",
        "records", "requests", "residents", "reviews", "rows", "samples", "sentences", "sessions", "shares",
        "signups", "steps", "streams", "students", "subscribers", "tickets", "tokens", "transactions", "trips",
        "units", "uploads", "users", "utterances", "videos", "viewers", "views", "visitors", "visits", "votes",
        "weights", "words", "workers",
    };
    return SET.contains(w);
}

bool model_noun(std::string_view w) {
    static const rv::WordSet SET{
        "adapter", "adapters", "backbone", "base", "baseline", "baselines", "benchmark", "build", "builds",
        "checkpoint", "checkpoints", "class", "classifier", "decoder", "dense", "distill", "draft", "encoder", "entry",
        "fallback", "family", "generator", "instruct", "judge", "llm", "llms", "model", "models", "moe", "option",
        "planner", "proxy", "ranker", "reference", "release", "reranker", "result", "retriever", "router", "run",
        "runs", "scorer", "size", "student", "students", "summarizer", "tagger", "teacher", "tier", "transformer",
        "variant", "variants", "version", "versions", "vocoder",
    };
    return SET.contains(w);
}

bool sentence_opener(std::string_view w) {
    static const rv::WordSet SET{
        "A", "After", "All", "Also", "Although", "An", "And", "Are", "As", "At", "Because", "Before", "Both",
        "But", "By", "Can", "Could", "Did", "Do", "Does", "Don't", "During", "Each", "Even", "Every", "Finally",
        "First", "For", "From", "Had", "Has", "Have", "He", "Her", "Here", "His", "How", "However", "I", "I'll",
        "I'm", "I've", "If", "In", "Inside", "Is", "It", "It's", "Its", "Just", "Last", "Let", "Let's", "Many",
        "Meanwhile", "Most", "My", "Next", "No", "Not", "Now", "Of", "On", "Once", "One", "Only", "Our", "Overall",
        "Please", "She", "Since", "So", "Some", "Still", "That", "That's", "The", "Their", "Then", "There",
        "There's", "These", "They", "They're", "This", "Those", "Though", "To", "Today", "Tomorrow", "Two",
        "Until", "We", "We'll", "We're", "Were", "What", "When", "Where", "Which", "While", "Who", "Why", "Will",
        "With", "Yesterday", "Yet", "You", "Your",
    };
    return SET.contains(w);
}

bool shorthand(std::string_view w) {
    static const rv::WordSet SET{"DoD", "HoH", "MoM", "QoQ", "WoW", "YoY", "YTD", "MTD", "QTD"};
    return SET.contains(w);
}

}  // namespace lp
