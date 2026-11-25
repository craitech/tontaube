// Words of four or five letters that the spoken side of the training pairs writes with one capital inside a
// sentence (names, places, brands), not in english(). Generated; one table.
#include "rules/letters_products/lp.h"

namespace lp {

bool name(std::string_view lower) {
    static const rv::WordSet WORDS{
        "abara", "abebe", "adam", "adams", "adler", "aero", "agnes", "ahmed", "akana", "alder", "anima", "anna",
        "anne", "anouk", "anya", "aoife", "arbor", "arca", "arden", "aris", "arlo", "army", "ashby", "ashe", "asia",
        "asian", "aster", "astro", "async", "auria", "avro", "ayla", "bach", "baker", "barsk", "basel", "berg",
        "bern", "bible", "birch", "bleu", "bliss", "brain", "bran", "brask", "brass", "bravo", "brisa", "brno",
        "brook", "brune", "bson", "buff", "cadet", "cairo", "canon", "canto", "cask", "cedar", "ceres", "chem",
        "chen", "chess", "chile", "china", "chris", "cleo", "cobit", "cobol", "cogs", "cohen", "coppa", "corsa",
        "coso", "covid", "creek", "crime", "cusco", "czech", "dalai", "dana", "daten", "david", "davis", "deco",
        "demir", "deno", "dido", "douce", "dream", "dubai", "dube", "duke", "dune", "dunn", "dutch", "eagle",
        "ebit", "ebita", "ebro", "egypt", "elara", "elena", "elif", "ellis", "ende", "ernst", "evans", "excel",
        "expo", "fable", "falun", "feld", "ferpa", "ferro", "fhir", "fifo", "finch", "finn", "finra", "fisma",
        "fjord", "flac", "foods", "ford", "franz", "frau", "frist", "funny", "gaap", "gale", "gare", "geek",
        "gemma", "gent", "ghent", "gita", "glide", "glory", "gong", "graz", "greco", "greek", "greta", "grove",
        "gulf", "gulls", "guru", "haig", "hale", "halo", "hana", "hanoi", "hanov", "hart", "haven", "hawk", "helen",
        "helio", "helm", "henry", "hepa", "heron", "herr", "hindi", "hipaa", "holy", "homer", "honey", "huis",
        "hydra", "iago", "iliad", "ilsby", "ilse", "ines", "ioana", "ipad", "irene", "iris", "irish", "itar",
        "ivan", "jade", "jahr", "jakub", "james", "jana", "japan", "java", "jesus", "john", "jonah", "jonas",
        "jones", "jsonl", "jude", "julia", "july", "june", "juno", "juris", "kaffe", "karl", "karte", "kaur",
        "kaya", "keel", "kenji", "kerr", "kiss", "kite", "kofi", "kong", "kovac", "kunde", "kwame", "kyiv", "kyoto",
        "ladle", "lagos", "lama", "lark", "latex", "latin", "lear", "leeds", "lemma", "lena", "lense", "lenz",
        "lille", "lima", "lims", "lina", "lind", "linie", "llama", "lora", "lord", "lords", "lorne", "lotte",
        "louis", "lover", "lowe", "lukas", "luke", "lumo", "lund", "lyle", "lyon", "lyra", "mads", "maeve", "malmo",
        "maple", "mara", "march", "marco", "maria", "marta", "mary", "maven", "mavic", "mbeki", "mems", "mers",
        "mesa", "metz", "mews", "mhic", "midi", "milan", "mills", "ming", "mint", "mips", "mira", "mongo", "montt",
        "mooc", "moor", "morel", "mundi", "nacha", "nadia", "nano", "nato", "nemo", "nerc", "nero", "nest", "netz",
        "neuro", "nexa", "nexar", "nexor", "nexus", "ngozi", "nifty", "nils", "noaa", "nopat", "notas", "notch",
        "nowak", "numa", "nuxt", "nwosu", "oake", "ochoa", "odile", "oduya", "ohio", "olaf", "oled", "olmo",
        "olsen", "onnx", "opec", "opex", "orbis", "orca", "oriel", "orin", "orion", "orlo", "orsi", "ortiz",
        "osaka", "osei", "osha", "oskar", "oslo", "ostby", "ostra", "pablo", "palm", "paris", "patel", "paul",
        "paula", "paulo", "pearl", "pell", "petal", "peter", "petra", "phred", "pico", "piet", "pike", "pilar",
        "piotr", "pius", "plato", "porto", "prism", "priya", "psalm", "pylon", "qilon", "quest", "quran", "qwen",
        "rafe", "raman", "rand", "rauf", "ravel", "ravi", "redis", "reims", "reno", "reuss", "revue", "reyes",
        "rhine", "riga", "roas", "robo", "rocky", "roman", "rome", "romeo", "ronin", "rook", "roth", "rouge",
        "rowan", "rowe", "ruben", "ruby", "ruiz", "runes", "rust", "ruth", "ryzen", "saal", "saas", "sabre", "saga",
        "saint", "sami", "santa", "sara", "sars", "sato", "scada", "scala", "scout", "scsi", "scuba", "seng",
        "seoul", "shah", "shema", "siege", "signe", "silva", "simd", "skala", "slate", "smith", "soap", "sonic",
        "sons", "sopot", "spain", "spark", "spelt", "stoxx", "summa", "surah", "swiss", "swot", "syrah", "talon",
        "tarn", "tartu", "tate", "teal", "teil", "tern", "terra", "tesla", "tessa", "thala", "theo", "theta",
        "tiff", "tilde", "titan", "tizen", "tokyo", "tomas", "toml", "topix", "tran", "traum", "troy", "tudor",
        "turbo", "turin", "ultra", "unix", "vale", "vance", "vanta", "varga", "varis", "varna", "varro", "vega",
        "vela", "velto", "vesta", "vexa", "vexon", "vibes", "vireo", "vista", "vogel", "vogt", "volta", "voxel",
        "webb", "weibo", "welsh", "wert", "whiz", "woods", "wren", "xbox", "xeon", "yaml", "yolo", "york", "yusuf",
        "zahl", "zeit", "zeta", "ziel", "zola",
    };
    return WORDS.contains(lower);
}

}  // namespace lp
