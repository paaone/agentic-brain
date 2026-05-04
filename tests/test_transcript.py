import json
from pathlib import Path

from memory import transcript


def _write_jsonl(path: Path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def test_parse_skips_queue_ops_and_extracts_text(tmp_path):
    p = tmp_path / "t.jsonl"
    _write_jsonl(p, [
        {"type": "queue-operation", "operation": "enqueue"},
        {"type": "user", "message": "hello world", "timestamp": "t1",
         "cwd": "/r", "gitBranch": "main"},
        {"type": "assistant", "message": {
            "content": [
                {"type": "text", "text": "hi back"},
                {"type": "tool_use", "name": "Read"},
            ]}, "timestamp": "t2"},
    ])
    turns = transcript.parse(p)
    assert [t.role for t in turns] == ["user", "assistant"]
    assert "hello world" in turns[0].text
    assert "hi back" in turns[1].text
    assert turns[0].git_branch == "main"


def test_long_tool_result_elided(tmp_path):
    p = tmp_path / "t.jsonl"
    big = "x" * 5000
    _write_jsonl(p, [
        {"type": "user", "message": "go", "timestamp": "t"},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_result", "content": big},
        ]}, "timestamp": "t"},
    ])
    turns = transcript.parse(p)
    assert any("elided" in t.text for t in turns)


def test_derive_path_encodes_cwd():
    p = transcript.derive_path("abc-123", "/home/user/agentic-brain")
    assert p.name == "abc-123.jsonl"
    assert "-home-user-agentic-brain" in str(p)


def test_summarize_truncates(tmp_path):
    turns = [transcript.Turn("user", "x" * 1000, "t") for _ in range(200)]
    out = transcript.summarize(turns, max_chars=5000)
    assert len(out) <= 5200  # allow elision marker overhead
    assert "elided" in out


def test_first_user_text_skips_assistant_turns(tmp_path):
    turns = [
        transcript.Turn("assistant", "hi there", "t"),
        transcript.Turn("user", "first user msg", "t"),
        transcript.Turn("user", "second user msg", "t"),
    ]
    assert transcript.first_user_text(turns) == "first user msg"
