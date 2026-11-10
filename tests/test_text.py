from rag.text import fold, normalize_for_match, tokenize


def test_fold_strips_accents_and_lowercases():
    assert fold("Één geïnd") == "een geind"


def test_tokenize_drops_stopwords():
    assert "de" not in tokenize("de huur van het huis")
    assert "het" not in tokenize("de huur van het huis")


def test_tokenize_stems_plural_and_singular_to_same_token():
    pairs = [
        ("toeslagen", "toeslag"),
        ("Zorgverzekeringen", "zorgverzekering"),
        ("bedragen", "bedrag"),
        ("aanvragen", "aanvraag"),
        ("huurwoningen", "huurwoning"),
    ]
    for plural, singular in pairs:
        assert tokenize(plural) == tokenize(singular), (plural, singular)


def test_normalize_for_match_collapses_whitespace():
    assert normalize_for_match("U  bent\n18 jaar") == "u bent 18 jaar"
