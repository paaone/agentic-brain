from memory import redact


def test_redact_openai_key():
    s = "API_KEY = 'sk-abc123def4567890hijklmn'"
    out = redact.scrub(s)
    assert "sk-abc123def4567890hijklmn" not in out
    assert "REDACTED" in out


def test_redact_github_token():
    s = "token: ghp_abcdefghijklmnopqrstuvwxyz0123"
    out = redact.scrub(s)
    assert "ghp_" not in out


def test_redact_aws_key_id():
    s = "aws AKIAIOSFODNN7EXAMPLE here"
    out = redact.scrub(s)
    assert "AKIA" not in out


def test_redact_jwt():
    s = "Bearer eyJhbGciOiJIUzI1NiIs.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdef123456"
    out = redact.scrub(s)
    assert "eyJhbGciOiJIUzI1NiIs" not in out


def test_redact_env_value():
    s = "DATABASE_URL=postgres://user:supersecretpw@host/db"
    out = redact.scrub(s)
    assert "supersecretpw" not in out
    assert "REDACTED" in out


def test_redact_private_key_block():
    s = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEpAIBAAKCAQEA...\n"
        "-----END RSA PRIVATE KEY-----"
    )
    out = redact.scrub(s)
    assert "MIIEpAIBAAKCAQEA" not in out


def test_looks_sensitive_paths():
    assert redact.looks_sensitive("opening /home/user/.env now")
    assert redact.looks_sensitive("path is /etc/credentials/foo")
    assert redact.looks_sensitive("loaded ./id_rsa")
    assert not redact.looks_sensitive("just regular text here")


def test_does_not_mangle_normal_text():
    s = "The function calculates the user's age based on birthdate."
    assert redact.scrub(s) == s
