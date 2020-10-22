# TODO: see if not loading all of these by default is possible
language_resources_path = "/home/mahir256/language-resources"
corpora_path = "/home/mahir256/corpora"

speech_langs = {"bn": "asr_bengali",
                "en": "asr_wsj",
                "jv": "asr_javanese",
                "ne": "asr_nepali",
                "si": "asr_sinhala",
                "su": "asr_sundanese"}

lexicon_langs = {"bn":"data/lexicon.tsv",
                 "en":"data/lexicon.tsv",
                 "jv":"data/lexicon.tsv",
                 "ne":"data/lexicon.tsv",
                 "si":"data/lexicon.tsv",
                 "su":"data/lexicon.tsv"}

feature_classes = ["place", "manner", "voicing", "aspiration", "prenasal",
                   "height", "length", "position", "roundness", "nasal"]
feature_classes_consonant = {
    "place": True,
    "manner": True,
    "voicing": True,
    "aspiration": True,
    "prenasal": True,
    "height": False,
    "length": False,
    "position": False,
    "roundness": False,
    "nasal": False
}
feature_class_values = {
    "place": ["b", "l", "d", "a", "c", "p", "v", "g"], # "bilabial", "labiodental", "dental", "alveolar", "postalveolar", "palatal", "velar", "glottal"
    "manner": ["s", "a", "f", "n", "r", "l"], # "stop", "affricate", "fricative", "nasal", "approximant", "lateral"
    "voicing": ["0", "1"], # false, true
    "aspiration": ["0", "1"], # false, true
    "prenasal": ["0", "1"], # false, true
    "height": ["0", "1", "2", "3", "4", "5", "6"], # "close", "nearclose", "closemid", "mid", "openmid", "nearopen", "open"
    "length": ["0", "1"], # "short", "long"
    "position": ["f", "c", "b"], # "front", "central", "back"
    "roundness": ["0", "1"], # false, true
    "nasal": ["0", "1"] # false, true
}

char_to_int = {}
int_to_char = {}

for feature, value in feature_class_values.items():
    feature_class_values[feature] = list(enumerate(value))
    out_of_class = (len(feature_class_values[feature]),"V" if feature_classes_consonant[feature] else "C")
    feature_class_values[feature].append(out_of_class)
    feature_class_values[feature].append((len(feature_class_values[feature])," "))
    int_to_char[feature] = {x[0]: x[1] for x in feature_class_values[feature]}
    char_to_int[feature] = {x[1]: x[0] for x in feature_class_values[feature]}

common_phone_set = {
  "k": {
    "_type": "C",
    "place": "v",
    "manner": "s",
    "voicing": "0",
    "aspiration": "0",
    "prenasal": "0"
  },
  "kh": {
    "_type": "C",
    "place": "v",
    "manner": "s",
    "voicing": "0",
    "aspiration": "1",
    "prenasal": "0"
    },
  "g": {
    "_type": "C",
    "place": "v",
    "manner": "s",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "0"
  },
  "ng": {
    "_type": "C",
    "place": "v",
    "manner": "s",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "1"
  },
  "gh": {
    "_type": "C",
    "place": "v",
    "manner": "s",
    "voicing": "1",
    "aspiration": "1",
    "prenasal": "0"
  },
  "N": {
    "_type": "C",
    "place": "v",
    "manner": "n",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "0"
  },
  "c": {
    "_type": "C",
    "manner": "a",
    "place": "c",
    "voicing": "0",
    "aspiration": "0",
    "prenasal": "0"
  },
  "ch": {
    "_type": "C",
    "manner": "a",
    "place": "c",
    "voicing": "0",
    "aspiration": "1",
    "prenasal": "0"
  },
  "j": {
    "_type": "C",
    "manner": "a",
    "place": "c",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "0"
  },
  "jh": {
    "_type": "C",
    "manner": "a",
    "place": "c",
    "voicing": "1",
    "aspiration": "1",
    "prenasal": "0"
  },
  "ts": {
    "_type": "C",
    "manner": "a",
    "place": "a",
    "voicing": "0",
    "aspiration": "0",
    "prenasal": "0"
  },
  "tsh": {
    "_type": "C",
    "manner": "a",
    "place": "a",
    "voicing": "0",
    "aspiration": "1",
    "prenasal": "0"
  },
  "dz": {
    "_type": "C",
    "manner": "a",
    "place": "a",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "0"
  },
  "dzh": {
    "_type": "C",
    "manner": "a",
    "place": "a",
    "voicing": "1",
    "aspiration": "1",
    "prenasal": "0"
  },
  "ny": {
    "_type": "C",
    "place": "p",
    "manner": "n",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "0"
  },
  "T": {
    "_type": "C",
    "place": "a",
    "manner": "s",
    "voicing": "0",
    "aspiration": "0",
    "prenasal": "0"
  },
  "Th": {
    "_type": "C",
    "place": "a",
    "manner": "s",
    "voicing": "0",
    "aspiration": "1",
    "prenasal": "0"
  },
  "D": {
    "_type": "C",
    "place": "a",
    "manner": "s",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "0"
  },
  "nD": {
    "_type": "C",
    "place": "a",
    "manner": "s",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "1"
  },
  "Dh": {
    "_type": "C",
    "place": "a",
    "manner": "s",
    "voicing": "1",
    "aspiration": "1",
    "prenasal": "0"
  },
  "nn": {
    "_type": "C",
    "place": "a",
    "manner": "n",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "0"
  },
  "t": {
    "_type": "C",
    "place": "d",
    "manner": "s",
    "voicing": "0",
    "aspiration": "0",
    "prenasal": "0"
  },
  "th": {
    "_type": "C",
    "place": "d",
    "manner": "s",
    "voicing": "0",
    "aspiration": "1",
    "prenasal": "0"
  },
  "d": {
    "_type": "C",
    "place": "d",
    "manner": "s",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "0"
  },
  "nd": {
    "_type": "C",
    "place": "d",
    "manner": "s",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "1"
  },
  "dh": {
    "_type": "C",
    "place": "d",
    "manner": "s",
    "voicing": "1",
    "aspiration": "1",
    "prenasal": "0"
  },
  "n": {
    "_type": "C",
    "place": "d",
    "manner": "n",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "0"
  },
  "p": {
    "_type": "C",
    "place": "b",
    "manner": "s",
    "voicing": "0",
    "aspiration": "0",
    "prenasal": "0"
  },
  "ph": {
    "_type": "C",
    "place": "b",
    "manner": "s",
    "voicing": "0",
    "aspiration": "1",
    "prenasal": "0"
  },
  "b": {
    "_type": "C",
    "place": "b",
    "manner": "s",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "0"
  },
  "mb": {
    "_type": "C",
    "place": "b",
    "manner": "s",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "1"
  },
  "bh": {
    "_type": "C",
    "place": "b",
    "manner": "s",
    "voicing": "1",
    "aspiration": "1",
    "prenasal": "0"
  },
  "m": {
    "_type": "C",
    "place": "b",
    "manner": "n",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "0"
  },
  "r": {
    "_type": "C",
    "place": "a",
    "manner": "r",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "0"
  },
  "l": {
    "_type": "C",
    "place": "a",
    "manner": "l",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "0"
  },
  "sh": {
    "_type": "C",
    "place": "c",
    "manner": "f",
    "voicing": "0",
    "aspiration": "0",
    "prenasal": "0"
  },
  "zh": {
    "_type": "C",
    "place": "c",
    "manner": "f",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "0"
  },
  "s": {
    "_type": "C",
    "place": "a",
    "manner": "f",
    "voicing": "0",
    "aspiration": "0",
    "prenasal": "0"
  },
  "z": {
    "_type": "C",
    "place": "a",
    "manner": "f",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "0"
  },
  "tx": {
    "_type": "C",
    "place": "d",
    "manner": "f",
    "voicing": "0",
    "aspiration": "0",
    "prenasal": "0"
  },
  "dx": {
    "_type": "C",
    "place": "d",
    "manner": "f",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "0"
  },
  "h": {
    "_type": "C",
    "place": "g",
    "manner": "f",
    "voicing": "0",
    "aspiration": "1",
    "prenasal": "0"
  },
  "gl": {
    "_type": "C",
    "place": "g",
    "manner": "s",
    "voicing": "0",
    "aspiration": "0",
    "prenasal": "0"
  },
  "f": {
    "_type": "C",
    "place": "l",
    "manner": "f",
    "voicing": "0",
    "aspiration": "0",
    "prenasal": "0"
  },
  "v": {
    "_type": "C",
    "place": "l",
    "manner": "f",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "0"
  },
  "x": {
    "_type": "C",
    "place": "v",
    "manner": "f",
    "voicing": "0",
    "aspiration": "0",
    "prenasal": "0"
  },
  "i": {
    "_type": "V",
    "height": "0",
    "length": "0",
    "position": "f",
    "roundness": "0",
    "nasal": "0"
  },
  "ii": {
    "_type": "V",
    "height": "0",
    "length": "1",
    "position": "f",
    "roundness": "0",
    "nasal": "0"
  },
  "in": {
    "_type": "V",
    "height": "0",
    "length": "0",
    "position": "f",
    "roundness": "0",
    "nasal": "1"
  },
  "ix": {
    "_type": "V",
    "height": "1",
    "length": "0",
    "position": "f",
    "roundness": "0",
    "nasal": "0"
  },
  "u": {
    "_type": "V",
    "height": "0",
    "length": "0",
    "position": "b",
    "roundness": "1",
    "nasal": "0"
  },
  "uu": {
    "_type": "V",
    "height": "0",
    "length": "1",
    "position": "b",
    "roundness": "1",
    "nasal": "0"
  },
  "un": {
    "_type": "V",
    "height": "0",
    "length": "0",
    "position": "b",
    "roundness": "1",
    "nasal": "1"
  },
  "ux": {
    "_type": "V",
    "height": "1",
    "length": "0",
    "position": "b",
    "roundness": "1",
    "nasal": "0"
  },
  "e": {
    "_type": "V",
    "height": "2",
    "length": "0",
    "position": "f",
    "roundness": "0",
    "nasal": "0"
  },
  "ee": {
    "_type": "V",
    "height": "2",
    "length": "1",
    "position": "f",
    "roundness": "0",
    "nasal": "0"
  },
  "en": {
    "_type": "V",
    "height": "2",
    "length": "0",
    "position": "f",
    "roundness": "0",
    "nasal": "1"
  },
  "eu": {
    "_type": "V",
    "height": "0",
    "length": "0",
    "position": "c",
    "roundness": "0",
    "nasal": "0"
  },
  "ex": {
    "_type": "V",
    "height": "3",
    "length": "0",
    "position": "c",
    "roundness": "0",
    "nasal": "0"
  },
  "exx": {
    "_type": "V",
    "height": "3",
    "length": "1",
    "position": "c",
    "roundness": "0",
    "nasal": "0"
  },
  "e3": {
    "_type": "V",
    "height": "4",
    "length": "0",
    "position": "f",
    "roundness": "0",
    "nasal": "0"
  },
  "o": {
    "_type": "V",
    "height": "2",
    "length": "0",
    "position": "b",
    "roundness": "1",
    "nasal": "0"
  },
  "E": {
    "_type": "V",
    "height": "5",
    "length": "0",
    "position": "f",
    "roundness": "0",
    "nasal": "0"
  },
  "EE": {
    "_type": "V",
    "height": "5",
    "length": "1",
    "position": "f",
    "roundness": "0",
    "nasal": "0"
  },
  "O": {
    "_type": "V",
    "height": "4",
    "length": "0",
    "position": "b",
    "roundness": "1",
    "nasal": "0"
  },
  "OX": {
    "_type": "V",
    "height": "6",
    "length": "0",
    "position": "b",
    "roundness": "0",
    "nasal": "0"
  },
  "ax": {
    "_type": "V",
    "height": "4",
    "length": "0",
    "position": "b",
    "roundness": "0",
    "nasal": "0"
  },
  "axn": {
    "_type": "V",
    "height": "4",
    "length": "0",
    "position": "b",
    "roundness": "0",
    "nasal": "1"
  },
  "a": {
    "_type": "V",
    "height": "6",
    "length": "0",
    "position": "c",
    "roundness": "0",
    "nasal": "0"
  },
  "aa": {
    "_type": "V",
    "height": "6",
    "length": "1",
    "position": "c",
    "roundness": "0",
    "nasal": "0"
  },
  "an": {
    "_type": "V",
    "height": "6",
    "length": "0",
    "position": "c",
    "roundness": "0",
    "nasal": "1"
  },
  "i^": {
    "_type": "C",
    "place": "p",
    "manner": "r",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "0"
  },
  "u^": {
    "_type": "C",
    "place": "b",
    "manner": "r",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "0"
  },
  "e^": {
    "_type": "C",
    "place": "p",
    "manner": "r",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "0"
  },
  "o^": {
    "_type": "C",
    "place": "b",
    "manner": "r",
    "voicing": "1",
    "aspiration": "0",
    "prenasal": "0"
  }
}

feature_class_values["phones"] = list(enumerate(common_phone_set.keys()))
feature_class_values["phones"].append((len(feature_class_values["phones"])," "))
int_to_char["phones"] = {x[0]: x[1] for x in feature_class_values["phones"]}
char_to_int["phones"] = {x[1]: x[0] for x in feature_class_values["phones"]}

festvox_to_phones = {}

festvox_to_phones["bn"] = {
    "k": ["k"],
    "kh": ["kh"],
    "g": ["g"],
    "gh": ["gh"],
    "c": ["c"],
    "ch": ["ch"],
    "j": ["j"],
    "jh": ["jh"],
    "T": ["T"],
    "Th": ["Th"],
    "D": ["D"],
    "Dh": ["Dh"],
    "t": ["t"],
    "th": ["th"],
    "d": ["d"],
    "dh": ["dh"],
    "p": ["p"],
    "b": ["b"],
    "bh": ["bh"],
    "N": ["N"],
    "n": ["n"],
    "m": ["m"],
    "r": ["r"],
    "l": ["l"],
    "sh": ["sh"],
    "s": ["s"],
    "h": ["h"],
    "f": ["f"],
    "i": ["i"],
    "u": ["u"],
    "e": ["e"],
    "o": ["o"],
    "E": ["E"],
    "O": ["O"],
    "a": ["a"],
    "i^": ["i^"],
    "u^": ["u^"],
    "e^": ["e^"],
    "o^": ["o^"],
    " ": [" "]
}

festvox_to_phones["jv"] = {
    "k": ["k"],
    "g": ["g"],
    "ng": ["N"],
    "c": ["c"],
    "j": ["j"],
    "ny": ["ny"],
    "th": ["T"],
    "dh": ["D"],
    "t": ["t"],
    "d": ["d"],
    "n": ["n"],
    "p": ["p"],
    "b": ["b"],
    "m": ["m"],
    "y": ["i^"],
    "r": ["r"],
    "l": ["l"],
    "w": ["u^"],
    "sh": ["sh"],
    "s": ["s"],
    "z": ["z"],
    "hx": ["x"],
    "h": ["h"],
    "f": ["f"],
    "gl": ["gl"],
    "i": ["i"],
    "e": ["e"],
    "ex": ["ex"],
    "a": ["a"],
    "u": ["u"],
    "o": ["o"],
    "ox": ["O"],
    "ai": ["a", "i^"],
    "au": ["a", "u^"],
    "oi": ["o", "i^"],
    " ": [" "]
}

festvox_to_phones["ne"] = {
    "i": ["i"],
    "u": ["u"],
    "e": ["e"],
    "o": ["o"],
    "ax": ["ax"],
    "aa": ["a"],
    "in": ["in"],
    "en": ["en"],
    "un": ["un"],
    "axn": ["axn"],
    "aan": ["an"],
    "ew": ["e", "u^"],
    "oj": ["o", "i^"],
    "ow": ["o", "u^"],
    "axj": ["ax", "i^"],
    "axw": ["ax", "u^"],
    "aaj": ["aa", "i^"],
    "aaw": ["aa", "u^"],
    "ewn": ["e", "un"],
    "ojn": ["o", "in"],
    "own": ["o", "un"],
    "axjn": ["ax", "in"],
    "axwn": ["ax", "un"],
    "aajn": ["a", "in"],
    "aawn": ["a", "un"],
    "k": ["k"],
    "kh": ["kh"],
    "g": ["g"],
    "gh": ["gh"],
    "ng": ["N"],
    "ts": ["ts"],
    "tsh": ["tsh"],
    "jh": ["dz"],
    "jhh": ["dzh"],
    "tt": ["T"],
    "tth": ["Th"],
    "dd": ["D"],
    "ddh": ["Dh"],
    "nn": ["nn"],
    "t": ["t"],
    "th": ["th"],
    "d": ["d"],
    "dh": ["dh"],
    "n": ["n"],
    "p": ["p"],
    "ph": ["ph"],
    "b": ["b"],
    "bh": ["bh"],
    "m": ["m"],
    "y": ["i^"],
    "r": ["r"],
    "l": ["l"],
    "w": ["u^"],
    "s": ["s"],
    "h": ["h"],
    " ": [" "]
}

festvox_to_phones["si"] = {
    "k": ["k"],
    "g": ["g"],
    "ŋ": ["N"],
    "c": ["c"],
    "ɟ": ["j"],
    "ɲ": ["ny"],
    "ʈ": ["T"],
    "ɖ": ["D"],
    "t": ["t"],
    "d": ["d"],
    "n": ["n"],
    "p": ["p"],
    "b": ["b"],
    "m": ["m"],
    "ᵐb": ["mb"],
    "ⁿd": ["nd"],
    "ⁿɖ": ["nD"],
    "ᵑg": ["ng"],
    "y": ["i^"],
    "r": ["r"],
    "l": ["l"],
    "w": ["u^"],
    "s": ["s"],
    "ʃ": ["sh"],
    "h": ["h"],
    "f": ["f"],
    "a": ["a"],
    "aː": ["aa"],
    "æ": ["E"],
    "æː": ["EE"],
    "i": ["i"],
    "iː": ["ii"],
    "u": ["u"],
    "uː": ["uu"],
    "e": ["e"],
    "eː": ["ee"],
    "o": ["o"],
    "oː": ["oː"],
    "ə": ["ex"],
    "əː": ["exx"],
    " ": [" "]
}

festvox_to_phones["su"] = {
    "k": ["k"],
    "g": ["g"],
    "ng": ["N"],
    "c": ["c"],
    "j": ["j"],
    "ny": ["ny"],
    "t": ["t"],
    "d": ["d"],
    "n": ["n"],
    "p": ["p"],
    "b": ["b"],
    "m": ["m"],
    "y": ["i^"],
    "r": ["r"],
    "l": ["l"],
    "w": ["u^"],
    "sy": ["sh"],
    "s": ["s"],
    "h": ["h"],
    "hx": ["x"],
    "f": ["f"],
    "gl": ["gl"],
    "z": ["z"],
    "a": ["a"],
    "ex": ["ex"],
    "e": ["e"],
    "eu": ["eu"],
    "i": ["i"],
    "u": ["u"],
    "o": ["o"],
    "ai": ["a", "i^"],
    "au": ["a", "u^"],
    "oi": ["o", "i^"],
    " ": [" "]
}

festvox_to_phones["en"] = {
  "AA": ["OX"],
  "AA0": ["OX"],
  "AA1": ["OX"],
  "AA2": ["OX"],
  "AE": ["E"],
  "AE0": ["E"],
  "AE1": ["E"],
  "AE2": ["E"],
  "AH": ["ax"],
  "AH0": ["ax"],
  "AH1": ["ax"],
  "AH2": ["ax"],
  "AO": ["O"],
  "AO0": ["O"],
  "AO1": ["O"],
  "AO2": ["O"],
  "AW": ["a", "ux"],
  "AW0": ["a", "ux"],
  "AW1": ["a", "ux"],
  "AW2": ["a", "ux"],
  "AY": ["a", "i^"],
  "AY0": ["a", "i^"],
  "AY1": ["a", "i^"],
  "AY2": ["a", "i^"],
  "B": ["b"],
  "CH": ["c"],
  "D": ["D"],
  "DH": ["dx"],
  "EH": ["e3"],
  "EH0": ["e3"],
  "EH1": ["e3"],
  "EH2": ["e3"],
  "ER": ["r"],
  "ER0": ["r"],
  "ER1": ["r"],
  "ER2": ["r"],
  "EY": ["e", "i^"],
  "EY0": ["e", "i^"],
  "EY1": ["e", "i^"],
  "EY2": ["e", "i^"],
  "F": ["f"],
  "G": ["g"],
  "HH": ["h"],
  "IH": ["ix"],
  "IH0": ["ix"],
  "IH1": ["ix"],
  "IH2": ["ix"],
  "IY": ["i"],
  "IY0": ["i"],
  "IY1": ["i"],
  "IY2": ["i"],
  "JH": ["j"],
  "K": ["k"],
  "L": ["l"],
  "M": ["m"],
  "N": ["n"],
  "NG": ["N"],
  "OW": ["o", "u^"],
  "OW0": ["o", "u^"],
  "OW1": ["o", "u^"],
  "OW2": ["o", "u^"],
  "OY": ["o", "i^"],
  "OY0": ["o", "i^"],
  "OY1": ["o", "i^"],
  "OY2": ["o", "i^"],
  "P": ["p"],
  "R": ["r"],
  "S": ["s"],
  "SH": ["sh"],
  "T": ["T"],
  "TH": ["tx"],
  "UH": ["ux"],
  "UH0": ["ux"],
  "UH1": ["ux"],
  "UH2": ["ux"],
  "UW": ["u"],
  "UW0": ["u"],
  "UW1": ["u"],
  "UW2": ["u"],
  "V": ["v"],
  "W": ["u^"],
  "Y": ["i^"],
  "Z": ["z"],
  "ZH": ["zh"],
  " ": [" "]
}

# AA	vowel
# AE	vowel
# AH	vowel
# AO	vowel
# AW	vowel
# AY	vowel
# B	stop
# CH	affricate
# D	stop
# DH	fricative
# EH	vowel
# ER	vowel
# EY	vowel
# F	fricative
# G	stop
# HH	aspirate
# IH	vowel
# IY	vowel
# JH	affricate
# K	stop
# L	liquid
# M	nasal
# N	nasal
# NG	nasal
# OW	vowel
# OY	vowel
# P	stop
# R	liquid
# S	fricative
# SH	fricative
# T	stop
# TH	fricative
# UH	vowel
# UW	vowel
# V	fricative
# W	semivowel
# Y	semivowel
# Z	fricative
# ZH	fricative
