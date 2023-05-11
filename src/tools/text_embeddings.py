import pickle
import sys
import nltk
#nltk.download()
import torch
from tqdm import tqdm
from files import join
from collections import Counter
from sentence_transformers import SentenceTransformer


stop_words = (
    "protection",
    "ebay",
    "copy",
    "copyright",
    "this",
    ",",
    "postcard",
    "-",
    ".",
    "ak",
    ")",
    "(",
    "vintage",
    "the",
    "of",
    "view",
    "old",
    "&",
    'cpr',
    "city",
    "new",
    "ca",
    "hotel",
    "ansichtskarte",
    "and",
    "#",
    "cpa",
    "card",
    "state",
    "scene",
    "in",
    "building",
    "post",
    "'s",
    "unused",
    "with",
    ";",
    "photo",
    "house",
    "de",
    "town",
    ":",
    "street",
    "am",
    "from",
    "linen",
    "bad",
    "/",
    "gruss",
    "la",
    "vtg",
    "at",
    "stamps",
    "unp",
    "unposted",
    "dr",
    "pc",
    "jim",
    "chrome",
    "national",
    "mit",
    "stadtkreis",
    "~",
    "der",
    "dc",
    "st",
    "real",
    "und",
    "no",
    "schloss",
    "main",
    "rppc",
    "office",
    "us",
    "to",
    "antique",
    "le",
    "litho",
    "posted",
    "standard",
    "royal",
    "a",
    "um",
    "place",
    "foto",
    "usa",
    "rp",
    "stadt",
    "pub",
    "mission",
    "des",
    "il",
    "postkarte",
    "gel",
    "used",
    "by",
    "ngl",
    "run",
    "central",
    "panorama",
    "tram",
    "im",
    "du",
    "lot",
    "night",
    "car",
    "dom",
    "aerial",
    "m.",
    "wob",
    "court",
    "cars",
    "cancel",
    "die",
    "et",
    "kaiser",
    "white",
    "ancient",
    "co",
    "a.",
    "series",
    "field",
    "postcards",
    "s.",
    "home",
    "united",
    "inn",
    "!",
    "art",
    "souvenir",
    "pm",
    "an",
    "*",
    "von",
    "harz",
    "d.",
    "grand",
    "?",
    "back",
    "rock",
    "burg",
    "historical",
    "partie",
    "stamp",
    "blick",
    "//",
    "golden",
    "salt",
    "i",
    "around",
    "db",
    "train",
    "road",
    "wb",
    "avenue",
    "di",
    "@",
    "near",
    "s",
    "mo",
    "historic",
    "u.",
    "uk",
    "alte",
    "rhine",
    "buildings",
    "'",
    "world",
    "diego",
    "maria",
    "auf",
    "_",
    "early",
    "interior",
    "ma",
    "trier",
    "views",
    "cable",
    "game",
    "stone",
    "goslar",
    "approx",
    "saale",
    "vt",
    "picture",
    "gdr",
    "county",
    "exterior",
    "nach",
    "postal",
    "horse",
    "b",
    "feldpost",
    "public",
    "pre-1980",
    "german",
    "original",
    "prag",
    "mary",
    "den",
    "d",
    "schloß",
    "gardens",
    "entrance",
    "b.",
    "war",
    "gruß",
    "bath",
    "tor",
    "beautiful",
    "vom",
    "markt",
    "great",
    "black",
    "general",
    "george",
    "’",
    "das",
    "ppc",
    "bei",
    "undivided",
    "ship",
    "1940s",
    "map",
    "architecture",
    "1959",
    "government",
    "john",
    "metz",
    "lithographie",
    "spa",
    "note",
    "gare",
    "american",
    "water",
    "multi",
    "me",
    "room",
    "graz",
    "ansicht",
    "c.",
    "houses",
    "e.",
    "w",
    "lkr",
    "rare",
    "turm",
    "large",
    "bodensee",
    "e",
    "pool",
    "postmark",
    "l",
    "saint",
    "alger",
    "halle",
    "wa",
    "gallery",
    "ut",
    "ii",
    "artist",
    "dem",
    "peter",
    "see",
    "hearst",
    "i.",
    "mark",
    "scenic",
    "center",
    "shops",
    "pk",
    "valentine",
    "del",
    "grant",
    "little",
    "nr",
    "ob",
    "sign",
    "haus",
    "1950s",
    "not",
    "side",
    "wall",
    "man",
    "color",
    "bldg",
    "innere",
    "streetview",
    "monastery",
    "long",
    "alt",
    "...",
    "x",
    "1910s",
    "udb",
    "antonio",
    "queen",
    "|",
    "cassel",
    "parade",
    "tree",
    "flag",
    "nj",
    "justice",
    "upper",
    "worms",
    "1930s",
    "mitte",
    "steamer",
    "high",
    "pacific",
    "zum",
    "a.d.",
    "ungel",
    "friedrich",
    "soldiers",
    "foto-ak",
    "hospital",
    "lithograph",
    "people",
    "divided",
    "ville",
    "king",
    "eye",
    "c",
    "red",
    "famous",
    "plaza",
    "christ",
    "vary",
    "municipal",
    "winter",
    "colonial",
    "village",
    "colour",
    "dome",
    "radio",
    "clock",
    "1920s",
    "sailors",
    "bank",
    "cafe",
    "c1910",
    "highway",
    "boat",
    "british",
    "les",
    "printing",
    "promenade",
    "az",
    "small",
    "+",
    "berg",
    "bureau",
    "cross",
    "h.",
    "luebeck",
    "–",
    "chillon",
    "rue",
    "hi",
    "natural",
    "collection",
    "engraving",
    "corner",
    "very",
    "marine",
    "shop",
    "school",
    ">",
    "cpm",
    "looking",
    "over",
    "greater",
    "panoramic",
    "lts",
    "ariel",
    "ltd",
    "..",
    "ref",
    "food",
    "marine",
    "victor",
    "business",
    "pre",
    "linden",
    "+++",
    "m",
    "parish",
    "donau",
    "canal",
    "straße",
    "space",
    "w.",
    "history",
    "marketplace",
    "f.",
    "ddr",
    "air",
    "republic",
    "tn",
    "for",
    "k",
    "straßenbahn",
    "light",
    "terminal",
    "business",
    "zur",
    "ky",
    "tuck",
    "walter",
    "altes",
    "panoramic",
    "g",
    "vw",
    "birds",
    "advertising",
    "auto",
    "depot",
    "arts",
    "co.",
    "tx",
    "ambassador",
    "gold",
    "u.s.",
    "v.",
    "unrun",
    "oh",
    "mont",
    "kirkstall",
    "inner",
    "lahn",
    "detmold",
    "mansion",
    "vue",
    "postkaart",
    "mi",
    "terrace",
    "ga",
    "wroclaw",
    "before",
    "resort",
    "flowers",
    "french",
    "s/w",
    "president",
    "first",
    "marienplatz",
    "speyer",
    "barre",
    "arms",
    "streets",
    "karlsruhe",
    "suspension",
    "continental",
    "havel",
    "governor",
    "dorset",
    "a.m.",
    "chinese",
    "powell",
    "music",
    "nations",
    "coat",
    "photo-old",
    "eglise",
    "emperor",
    "congress",
    "capital",
    "lawn",
    "irish",
    "wayfarers",
    "pa",
    "flower",
    "only",
    "baltic",
    "giant",
    "land",
    "strasse",
    "total",
    "boats",
    "valentines",
    "fine",
    "hof",
    "bus",
    "states",
    "or",
    "d.c.",
    "oil",
    "local",
    "country",
    "rom",
    "scott",
    "rh",
    "conservatory",
    "n.",
    "older",
    "race",
    "amusement",
    "swimming",
    "big",
    "row",
    "n",
    "ab",
    "all",
    "showing",
    "<",
    "ks",
    "t",
    "divided-back",
    "multiview",
    "printed",
    "j.",
    "mint",
    "overlooking",
    "theme",
    "good",
    "times",
    "cottage",
    "hotels",
    "fishing",
    "steam",
    "pencil",
    "motel",
    "auditorium",
    "ships",
    "holiday",
    "snow",
    "free",
    "pan",
    "fleet",
    "tourist",
    "colored",
    "iii",
    "years"
)

max_words_in_text = 8
max_groups_in_scan = 3
digits = "0123456789"
words_delimiters = "-./~,=|+_'"

consonants = "bcdfghjklmnpqrstvwxz"
vowlets = "aeiouy"


def has_symbols_from(word, sss):

    for s in word:
        if s in sss:
            return True
    return False

def cons_vaildate(word):

    return w_vaildate(word, consonants, vowlets, 3)

def vows_vaildate(word):
    return w_vaildate(word, vowlets, consonants, 3)

def w_vaildate(word, sss, anti_sss, len_thr):
    # we can't have 3 or more such symbols in a row and no vowlets in a valid word
    c = 0
    for s in word:
        if s in sss:
            c += 1
        else:
            c = 0
        if c == len_thr:
            if not has_symbols_from(word, anti_sss):
                return False
        elif c > len_thr + 2:
            return False

    return True


def frequency(images_2_texts):

    images_text = images_2_texts.values()

    all_words = []
    for sentence in images_text:
        tokenized_text = nltk.word_tokenize(sentence)
        all_words.extend(map(lambda x: x.lower(), tokenized_text))

    occurence_count = Counter(all_words)
    for w in occurence_count.most_common():

        print(f"{w[0]}-{w[1]}")


def extract_geo_txt(images_2_texts):

    images_2_filtered_texts = {}

    for img_fn in images_2_texts.keys():
        for delimiter in words_delimiters:
            images_2_texts[img_fn] = images_2_texts[img_fn].replace(delimiter, " ")

        tokenized_words = nltk.word_tokenize(images_2_texts[img_fn])

        tokenized_words = map(lambda x: x.lower(), tokenized_words)
        filtered_sentence = []
        for w in tokenized_words:

            if len(w) < 3 or not cons_vaildate(w) or not vows_vaildate(w):
                continue

            try:
                int(w)
                continue
            except:
                pass

            for d in digits:
                try:
                    if w.index(d) != -1:
                        w = ","
                        break
                except:
                    pass

            if w not in stop_words and w not in filtered_sentence:
                filtered_sentence.append(w)

        if filtered_sentence:
            images_2_filtered_texts[img_fn] = " ".join(filtered_sentence[:max_words_in_text])
            # print(images_2_filtered_texts[img_fn], "<-", images_2_texts[img_fn])
    return images_2_filtered_texts


if __name__ == '__main__':

    with open(join(sys.argv[1], "texts.pkl"), 'rb') as f:
        images_2_unfiltered_texts = pickle.load(f)

    images_2_filtered_texts = extract_geo_txt(images_2_unfiltered_texts)

    with open(join(sys.argv[2], sys.argv[4]), 'rb') as f:
        tmp_test_ocr = pickle.load(f)
    test_ocr = {}
    for k in tmp_test_ocr:
        test_ocr[sys.argv[2] + "/" + k] = tmp_test_ocr[k]

    with open(join(sys.argv[3], sys.argv[4]), 'rb') as f:
        tmp_train_ocr = pickle.load(f)

    train_ocr = {}
    for k in tmp_train_ocr:
        train_ocr[sys.argv[3] + "/" + k] = tmp_train_ocr[k]

    images_2_unfiltered_ocr = train_ocr | test_ocr
    images_2_unfiltered_ocr_text = {}

    for image_fn in images_2_unfiltered_ocr:
        image_scanned_text = ""
        word_counter = 0
        for text in images_2_unfiltered_ocr[image_fn]:
            image_scanned_text += text[1] + " "
            if word_counter >= max_groups_in_scan:
                break
        images_2_unfiltered_ocr_text[image_fn] = image_scanned_text

    # frequency(images_2_unfiltered_ocr_text)
    images_2_filtered_ocr_texts = extract_geo_txt(images_2_unfiltered_ocr_text)

    all_img_names_with_text = images_2_filtered_ocr_texts.keys() | images_2_filtered_texts.keys()
    images_2_geo_sentences = {}

    for img_fn in all_img_names_with_text:
        sentence = ""
        if img_fn in images_2_filtered_texts:
            sentence = images_2_filtered_texts[img_fn]

        if img_fn in images_2_filtered_ocr_texts:
            if sentence:
                sentence += ", "

            sentence += images_2_filtered_ocr_texts[img_fn]

        images_2_geo_sentences[img_fn] = sentence
    #
    model = SentenceTransformer('all-MiniLM-L6-v2')
    images_2_texts_embeddings = {}

    for img_fn in tqdm(images_2_geo_sentences.keys()):
        img_name_start_position = img_fn.rfind('/')
        shortened_img_name = img_fn[img_name_start_position + 1:-4]

        images_2_texts_embeddings[shortened_img_name] = model.encode(images_2_geo_sentences[img_fn])

    print(f"Extracted {len(images_2_texts_embeddings)} sentences with geo texts")

    with open(join(sys.argv[1], "images_text_embeddings.pkl"), 'wb') as f:
        pickle.dump(
            images_2_texts_embeddings,
            f
        )
