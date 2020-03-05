language_resources_path = '/home/mahir256/language-resources'
corpora_path = '/home/mahir256/corpora-googleasr'

speech_langs = {'bn': 'asr_bengali',
                'jv': 'asr_javanese',
                'ne': 'asr_nepali',
                'si': 'asr_sinhala',
                'su': 'asr_sundanese'}

lexicon_langs = {'bn':'data/lexicon.tsv',
                 'jv':'data/lexicon.tsv',
                 'ne':'data/lexicon.tsv',
                 'si':'data/lexicon.tsv',
                 'su':'data/lexicon.tsv'}

phonology_langs = {'bn':'festvox/phonology.json',
                   'jv':'festvox/phonology.json',
                   'ne':'festvox/phonology.json',
                   'si':'festvox/ipa_phonology.json',
                   'su':'festvox/phonology.json'}

feature_classes = ["manner", "place", "voice", "height", "length", "frontness", "round"]
feature_classes_consonant = {
    "manner": True,
    "place": True,
    "voice": True,
    "height": False,
    "length": False,
    "frontness": False,
    "round": False
}
feature_class_values = {
    "manner": ["s", "a", "n", "x", "f"],
    "place": ["v", "p", "a", "d", "l", "g", "j"],
    "voice": ["f", "t"],
    "height": ["0", "1", "2", "3", "4"],
    "length": ["s"],
    "frontness": ["f", "c", "b"],
    "round": ["f", "t"]
}

feature_harmonize = {}

feature_harmonize["bn"] = {
    "ctype": {
        "_equivalent": "manner",
        "STOP": "s",
        "AFFRICATE": "a",
        "NASAL": "n",
        "APPROXIMANT": "x",
        "FRICATIVE": "f"
    },
    "poa": {
        "_equivalent": "place",
        "VELAR": "v",
        "POSTALVEOLAR": "p",
        "ALVEOLAR": "a",
        "DENTAL": "d",
        "LABIAL": "l",
        "GLOTTAL": "g",
        "PALATAL": "j"
    },
    "voiced": {
        "_equivalent": "voice",
        "false": "f",
        "true": "t"
    },
    "height": {
        "_equivalent": "height",
        "CLOSE": "0",
        "CLOSEMID": "1",
        "OPENMID": "2",
        "NEAROPEN": "3",
        "OPEN": "4"
    },
    "length": {
        "_equivalent": "length",
        "SHORT": "s"
    },
    "position": {
        "_equivalent": "frontness",
        "FRONT": "f",
        "CENTRAL": "c",
        "BACK": "b"
    },
    "rounded": {
        "_equivalent": "round",
        "false": "f",
        "true": "t"
    }
}
