from energy_agent.rag.chunking import chunk_document, clean_line, heading_rank, is_footer


def test_clean_line_strips_markup_and_line_breaks():
    assert clean_line("|**Nr.**|Anmeldung<br>einer<br>Zuordnung|") == (
        "|Nr.|Anmeldung einer Zuordnung|"
    )
    assert clean_line("<!-- Start of picture text -->Use-Case-Name  Neuanlage") == (
        "Use-Case-Name Neuanlage"
    )
    assert clean_line("### <u>Hinweise:</u> ") == "### Hinweise:"


def test_page_footers_are_detected():
    assert is_footer(clean_line("Seite **17** von **111** "))
    assert is_footer("- 9 -")
    assert not is_footer("Seite 17 der Anlage")


def test_heading_ranks_nest_from_letters_to_unnumbered():
    titles = ("A.", "III. Formelle", "2 Zuordnung", "2.1 Use-Case", "Hinweise:")
    ranks = [heading_rank(t) for t in titles]
    assert ranks == sorted(ranks)
    assert heading_rank("2.1.1 UC: Lieferbeginn") > heading_rank("2.1 Use-Case: Lieferbeginn")
    assert heading_rank("1. Allgemeines") == heading_rank("2 Zuordnungsprozesse")


def test_section_path_follows_heading_hierarchy():
    pages = [
        "# **2 Zuordnungsprozesse**\n\n# **2.1 Use-Case: Lieferbeginn**\n\nText A.\n",
        "# **2.1.1 UC: Lieferbeginn**\n\nText B.\n\n# **2.2 Use-Case: Neuanlage**\n\nText C.\n",
    ]
    chunks = chunk_document("gpke-teil2", pages)
    lieferbeginn = "2 Zuordnungsprozesse > 2.1 Use-Case: Lieferbeginn"
    assert [(c.section, c.text, c.page_start) for c in chunks] == [
        (lieferbeginn, "Text A.", 1),
        (f"{lieferbeginn} > 2.1.1 UC: Lieferbeginn", "Text B.", 2),
        ("2 Zuordnungsprozesse > 2.2 Use-Case: Neuanlage", "Text C.", 2),
    ]
    assert chunks[1].id == "gpke-teil2:0001"


def test_footer_does_not_end_up_in_chunk():
    chunks = chunk_document("doc", ["# **1 Titel**\n\nInhalt.\n\nSeite **3** von **9** \n"])
    assert chunks[0].text == "Inhalt."


def test_split_table_keeps_header_row_in_every_chunk():
    row = "|{n}|Aktion {n}|{frist}|Hinweis|"
    table = "\n".join(
        ["|**Nr.**|**Aktion**|**Frist**|**Hinweis/Bemerkung**|", "|---|---|---|---|"]
        + [row.format(n=n, frist="unverzüglich " * 8) for n in range(1, 9)]
    )
    chunks = chunk_document("doc", [f"# **2.1.2 SD: Lieferbeginn**\n\n{table}\n"], max_chars=300)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.startswith("Nr. | Aktion | Frist | Hinweis/Bemerkung\n")
    assert "8 | Aktion 8" in chunks[-1].text


def test_long_header_row_is_kept_once_as_content():
    long_row = "|Nr. 1|Aktion Ankündigung|" + "Frist nach Vorliegen des Grundes " * 10 + "|"
    page = f"# **2.4 Zuordnung**\n\n{long_row}\n|---|---|---|\n|2|Bestätigung|unverzüglich|\n"
    chunks = chunk_document("doc", [page])
    assert len(chunks) == 1
    assert chunks[0].text.count("Ankündigung") == 1
    assert chunks[0].text.endswith("2 | Bestätigung | unverzüglich")


def test_long_paragraph_is_split_near_max_chars():
    sentence = "Der Netzbetreiber prüft die Anmeldung des Lieferanten. "
    chunks = chunk_document("doc", ["# **1 Titel**\n\n" + sentence * 40], max_chars=400)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 for c in chunks)
    assert " ".join(c.text for c in chunks).count("Netzbetreiber") >= 40


def test_overlap_never_produces_a_chunk_of_only_repeated_text():
    paragraphs = "\n\n".join(f"Absatz {i}." * 20 for i in range(6))
    pages = [f"# **1 A**\n\n{paragraphs}\n\n# **2 B**\n\nEnde.\n"]
    chunks = chunk_document("doc", pages, max_chars=500)
    texts = [c.text for c in chunks]
    assert len(texts) == len(set(texts))
    assert chunks[-1].text == "Ende."
