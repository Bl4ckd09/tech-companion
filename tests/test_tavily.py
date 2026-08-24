from app.tavily.client import is_explicit_public_lookup, redact_search_query


def test_public_lookup_requires_explicit_search_language() -> None:
    assert is_explicit_public_lookup("Find official Android font-size instructions")
    assert not is_explicit_public_lookup("Something is wrong with my phone")


def test_search_query_redacts_private_values() -> None:
    query = (
        "Search the web for jane@example.com, phone +44 20 7946 0958, "
        "and code 482911"
    )

    redacted = redact_search_query(query)

    assert "jane@example.com" not in redacted
    assert "7946" not in redacted
    assert "482911" not in redacted
    assert "[email]" in redacted
    assert "[phone]" in redacted
