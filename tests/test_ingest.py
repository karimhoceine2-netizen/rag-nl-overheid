from rag.ingest import doc_id_from_url, parse_html, read_sources

# Synthetic page that mirrors the structure of rijksoverheid.nl pages.
HTML = """
<html><head>
<meta property="og:title" content="Kan ik toeslag krijgen? | Rijksoverheid.nl">
<meta name="DCTERMS.rights" content="CC0 1.0 Universal">
<meta name="DCTERMS.modified" content="2026-10-09T14:11:47.235Z">
</head><body>
<header><h1>Kan ik toeslag krijgen?</h1></header>
<main>
  <div class="intro"><p>Soms kunt u toeslag krijgen.</p></div>
  <div class="rich-text">
    <h2>Voorwaarden</h2>
    <ul><li><p>U bent 18 jaar of ouder.</p></li><li>Meer op <a href="#">toeslagen.nl</a> .</li></ul>
  </div>
  <div class="block video-block"><p>Uitgeschreven tekst van de video</p></div>
  <div class="__cq-feedback-bar"><p>Heeft deze informatie u geholpen?</p></div>
</main></body></html>
"""
URL = "https://www.rijksoverheid.nl/themas/x/vraag-en-antwoord/kan-ik-toeslag-krijgen"


def test_parse_html_extracts_content_and_metadata():
    doc = parse_html(HTML, URL, "2026-11-10")
    assert doc.title == "Kan ik toeslag krijgen?"
    assert doc.license == "CC0 1.0 Universal"
    assert doc.modified == "2026-10-09"
    assert doc.doc_id == "kan-ik-toeslag-krijgen"
    assert doc.text.splitlines() == [
        "Soms kunt u toeslag krijgen.",
        "## Voorwaarden",
        "- U bent 18 jaar of ouder.",
        "- Meer op toeslagen.nl.",
    ]


def test_parse_html_skips_video_and_feedback_blocks():
    doc = parse_html(HTML, URL, "2026-11-10")
    assert "video" not in doc.text
    assert "geholpen" not in doc.text


def test_parse_html_keeps_tables_in_order():
    table = """
    <div class="rich-text"><p>De bedragen zijn:</p>
    <table><caption>Uurprijzen 2026</caption>
      <tr><th>Soort</th><th>Prijs</th></tr>
      <tr><th>dagopvang</th><td><p>€ 11,23</p></td></tr>
    </table><p>Einde.</p></div>"""
    video = '<div class="block video-block">'
    html = HTML.replace(video, table + video)
    lines = parse_html(html, URL, "2026-11-10").text.splitlines()
    assert lines[-5:] == [
        "De bedragen zijn:",
        "Uurprijzen 2026",
        "Soort | Prijs",
        "dagopvang | € 11,23",
        "Einde.",
    ]


def test_parse_html_reports_missing_license():
    html = HTML.replace('<meta name="DCTERMS.rights" content="CC0 1.0 Universal">', "")
    assert parse_html(html, URL, "2026-11-10").license == "unknown"


def test_doc_id_and_sources(tmp_path):
    assert doc_id_from_url("https://x.nl/a/b/c/") == "c"
    sources = tmp_path / "sources.txt"
    sources.write_text("# comment\n\nhttps://x.nl/a\n  https://x.nl/b  \n")
    assert read_sources(sources) == ["https://x.nl/a", "https://x.nl/b"]
