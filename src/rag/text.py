"""Dutch text normalisation and tokenisation."""

import re
import unicodedata

import snowballstemmer

# Common Dutch function words (based on the standard Snowball/NLTK Dutch list).
STOPWORDS = frozenset(
    """aan al alles als altijd andere ben bij daar dan dat de der deze die dit doch doen door dus
    een eens en er ge geen geweest haar had heb hebben heeft hem het hier hij hoe hun iemand iets
    ik in is ja je kan kon kunnen maar me meer men met mij mijn moet na naar niet niets nog nu of
    om omdat onder ons ook op over reeds te tegen toch toen tot u uit uw van veel voor want waren
    was wat werd wezen wie wil worden wordt zal ze zelf zich zij zijn zo zonder zou""".split()
)

# "dutch_porter" (the classic Snowball algorithm), not the newer default "dutch": the new one
# lengthens vowels inconsistently ("toeslagen" -> "toeslaag" but "toeslag" -> "toeslag"),
# so singular and plural would never match. See tests/test_text.py.
_stemmer = snowballstemmer.stemmer("dutch_porter")
_WORD = re.compile(r"[a-z0-9]+")


def fold(text: str) -> str:
    """Lowercase and strip accents, so 'één' and 'een' or 'geïnd' and 'geind' match."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def tokenize(text: str) -> list[str]:
    """Fold, split into words, drop stopwords, stem.

    Stemming maps e.g. 'toeslagen' and 'toeslag' to the same token, which matters for BM25
    because it only matches exact tokens.
    """
    words = [w for w in _WORD.findall(fold(text)) if w not in STOPWORDS]
    return _stemmer.stemWords(words)


def normalize_for_match(text: str) -> str:
    """Folded text with collapsed whitespace; used to look up evidence quotes in chunks."""
    return " ".join(fold(text).split())
