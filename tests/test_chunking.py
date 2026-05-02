from rag.data.chunking import chunk_text


def test_short_text_one_chunk():
    assert chunk_text("Short.") == ["Short."]


def test_empty_text():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_long_text_splits():
    text = " ".join(["This is a sentence."] * 200)
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    # generous upper bound: chunk_size + overlap_carry
    assert all(len(c) <= 220 + 50 for c in chunks)
