"""Unit tests for hashing and text normalization utilities.

Tests cover text normalization edge cases, cache key generation,
and deterministic hashing behavior.
"""

import hashlib

import pytest

from src.utils.hashing import generate_cache_key, normalize_text


class TestNormalizeText:
    """Tests for normalize_text function."""

    def test_normalize_strips_leading_whitespace(self):
        """normalize_text removes leading whitespace."""
        result = normalize_text("   Hello world")
        assert result == "hello world"

    def test_normalize_strips_trailing_whitespace(self):
        """normalize_text removes trailing whitespace."""
        result = normalize_text("Hello world   ")
        assert result == "hello world"

    def test_normalize_strips_both_leading_and_trailing(self):
        """normalize_text removes both leading and trailing whitespace."""
        result = normalize_text("   Hello world   ")
        assert result == "hello world"

    def test_normalize_collapses_multiple_spaces(self):
        """normalize_text collapses multiple spaces to single space."""
        result = normalize_text("Hello    world")
        assert result == "hello world"

    def test_normalize_collapses_tabs_and_newlines(self):
        """normalize_text collapses tabs and newlines to single space."""
        result = normalize_text("Hello\t\n\r  world")
        assert result == "hello world"

    def test_normalize_converts_to_lowercase(self):
        """normalize_text converts all characters to lowercase."""
        result = normalize_text("HELLO World")
        assert result == "hello world"

    def test_normalize_preserves_single_spaces(self):
        """normalize_text preserves single spaces between words."""
        result = normalize_text("Hello world from Python")
        assert result == "hello world from python"

    def test_normalize_empty_string(self):
        """normalize_text handles empty string."""
        result = normalize_text("")
        assert result == ""

    def test_normalize_whitespace_only(self):
        """normalize_text reduces whitespace-only input to empty string."""
        result = normalize_text("   \t\n   ")
        assert result == ""

    def test_normalize_single_word(self):
        """normalize_text handles single word correctly."""
        result = normalize_text("  HELLO  ")
        assert result == "hello"

    def test_normalize_unicode_characters(self):
        """normalize_text preserves unicode characters."""
        result = normalize_text("  Café naïve résumé  ")
        assert result == "café naïve résumé"

    def test_normalize_special_characters(self):
        """normalize_text preserves special characters."""
        result = normalize_text("Hello, world! How's it going?")
        assert result == "hello, world! how's it going?"

    def test_normalize_numbers_and_symbols(self):
        """normalize_text preserves numbers and symbols."""
        result = normalize_text("Cost: $100.50")
        assert result == "cost: $100.50"

    def test_normalize_idempotent(self):
        """Normalizing twice produces same result as normalizing once."""
        text = "  HELLO   World  "
        normalized_once = normalize_text(text)
        normalized_twice = normalize_text(normalized_once)

        assert normalized_once == normalized_twice

    def test_normalize_case_insensitive_matching(self):
        """Different cases of same text normalize to same result."""
        text1 = "What is the capital of France?"
        text2 = "what is the capital of france?"
        text3 = "WHAT IS THE CAPITAL OF FRANCE?"

        assert normalize_text(text1) == normalize_text(text2)
        assert normalize_text(text2) == normalize_text(text3)


class TestGenerateCacheKey:
    """Tests for generate_cache_key function."""

    def test_generate_cache_key_format(self):
        """Cache key has format: prefix:hash."""
        key = generate_cache_key("test text")

        assert key.startswith("classify:")
        assert len(key.split(":")) == 2

    def test_generate_cache_key_hash_length(self):
        """Cache key hash is 64 characters (SHA256 hex digest)."""
        key = generate_cache_key("test text")
        hash_part = key.split(":")[1]

        assert len(hash_part) == 64

    def test_generate_cache_key_deterministic(self):
        """Same text produces same cache key."""
        text = "What is the capital of France?"

        key1 = generate_cache_key(text)
        key2 = generate_cache_key(text)

        assert key1 == key2

    def test_generate_cache_key_case_insensitive(self):
        """Different cases produce same cache key."""
        key1 = generate_cache_key("Hello World")
        key2 = generate_cache_key("hello world")
        key3 = generate_cache_key("HELLO WORLD")

        assert key1 == key2
        assert key2 == key3

    def test_generate_cache_key_whitespace_insensitive(self):
        """Different whitespace produces same cache key."""
        key1 = generate_cache_key("Hello world")
        key2 = generate_cache_key("Hello    world")
        key3 = generate_cache_key("  Hello\tworld  ")

        assert key1 == key2
        assert key2 == key3

    def test_generate_cache_key_different_text_different_keys(self):
        """Different text produces different cache keys."""
        key1 = generate_cache_key("What is AI?")
        key2 = generate_cache_key("What is ML?")

        assert key1 != key2

    def test_generate_cache_key_custom_prefix(self):
        """Cache key supports custom prefix."""
        key = generate_cache_key("test text", prefix="custom")

        assert key.startswith("custom:")

    def test_generate_cache_key_empty_prefix(self):
        """Cache key works with empty prefix."""
        key = generate_cache_key("test text", prefix="")

        assert key.startswith(":")

    def test_generate_cache_key_matches_expected_hash(self):
        """Cache key hash matches manual SHA256 calculation."""
        text = "test"
        normalized = normalize_text(text)
        expected_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

        key = generate_cache_key(text)
        actual_hash = key.split(":")[1]

        assert actual_hash == expected_hash

    def test_generate_cache_key_unicode_handling(self):
        """Cache key handles unicode text correctly."""
        text = "Café résumé"

        key = generate_cache_key(text)

        assert key.startswith("classify:")
        assert len(key.split(":")[1]) == 64

    def test_generate_cache_key_special_characters(self):
        """Cache key handles special characters correctly."""
        text = "Cost: $100.50 (discounted!)"

        key = generate_cache_key(text)

        assert key.startswith("classify:")
        assert len(key.split(":")[1]) == 64

    def test_generate_cache_key_long_text(self):
        """Cache key works with very long text."""
        text = "word " * 1000  # 5000 characters

        key = generate_cache_key(text)

        assert key.startswith("classify:")
        assert len(key.split(":")[1]) == 64

    def test_generate_cache_key_collision_resistance(self):
        """Similar texts produce different cache keys."""
        key1 = generate_cache_key("The quick brown fox")
        key2 = generate_cache_key("The quick brown box")

        assert key1 != key2

    def test_generate_cache_key_consistency_with_normalization(self):
        """Pre-normalized and non-normalized text produce same key."""
        text = "  HELLO   World  "
        normalized = normalize_text(text)

        key_from_raw = generate_cache_key(text)
        key_from_normalized = generate_cache_key(normalized)

        assert key_from_raw == key_from_normalized
