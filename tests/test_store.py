import numpy as np

from memory import store
from memory.embed import HashTfidfEmbedder


def test_init_and_add_and_search():
    store.init_db()
    emb = HashTfidfEmbedder()

    samples = [
        ("writing-style", "Terse prose preference",
         "User writes short, direct sentences",
         "Avoid hedging. Default to the active voice. Keep paragraphs to 3-4 lines."),
        ("code-pattern", "Python f-strings",
         "Always use f-strings over .format()",
         "Prefer f-strings; fall back to %-formatting only inside logging."),
        ("decision", "SQLite for storage",
         "Chose SQLite for the memory store",
         "Chose SQLite + numpy because dataset is small and zero ops overhead."),
    ]
    for kind, title, summary, content in samples:
        v = emb.encode([f"{title}\n{summary}\n{content}"])[0]
        store.add_memory(kind, title, summary, content, ["smoke"],
                         "sess-1", "/repo", "main", v, emb.name)

    qv = emb.encode(["how should I write prose"])[0]
    hits = store.top_k(qv, k=2)
    assert len(hits) == 2
    # writing-style item should rank above the unrelated decision
    titles = [h.title for h in hits]
    assert "Terse prose preference" in titles


def test_kind_filter():
    store.init_db()
    emb = HashTfidfEmbedder()
    for kind, t in [("writing-style", "Voice rules"),
                    ("code-pattern", "Pattern A"),
                    ("code-pattern", "Pattern B")]:
        v = emb.encode([t])[0]
        store.add_memory(kind, t, t, t, [], "s", "/r", "main", v, emb.name)

    qv = emb.encode(["pattern"])[0]
    hits = store.top_k(qv, k=5, kind="code-pattern")
    assert all(h.kind == "code-pattern" for h in hits)
    assert len(hits) == 2


def test_session_offset_idempotency():
    store.init_db()
    assert store.get_offset("new-session") == 0
    store.mark_ingested("new-session", 42)
    assert store.get_offset("new-session") == 42
    store.mark_ingested("new-session", 50)
    assert store.get_offset("new-session") == 50


def test_stats():
    store.init_db()
    s = store.stats()
    assert "total" in s and "db_path" in s
    assert s["total"] == 0
