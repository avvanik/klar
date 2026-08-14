from __future__ import annotations

import pytest

from klar.errors import EmptyAudioError, MissingFileError, UnsupportedFormatError
from klar.hashing import hash_file, short_hash
from klar.ingest import discover_audio, ingest

from .fakes import write_fake_audio


def test_hash_is_content_based_and_stable(tmp_path):
    a = write_fake_audio(tmp_path / "a.mp3")
    b = write_fake_audio(tmp_path / "b.mp3")  # same bytes, different name
    assert hash_file(a) == hash_file(b)
    assert len(short_hash(hash_file(a))) == 12


def test_hash_changes_with_content(tmp_path):
    a = write_fake_audio(tmp_path / "a.mp3", size=1024)
    b = write_fake_audio(tmp_path / "b.mp3", size=2048)
    assert hash_file(a) != hash_file(b)


def test_ingest_missing_file(tmp_path):
    with pytest.raises(MissingFileError):
        ingest(tmp_path / "nope.mp3")


def test_ingest_empty_file(tmp_path):
    p = tmp_path / "empty.mp3"
    p.write_bytes(b"")
    with pytest.raises(EmptyAudioError):
        ingest(p)


def test_ingest_wrong_format(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_bytes(b"x" * 5000)
    with pytest.raises(UnsupportedFormatError):
        ingest(p)


def test_ingest_ok_returns_hash(tmp_path):
    p = write_fake_audio(tmp_path / "call.mp3")
    ing = ingest(p)
    assert ing.content_hash and ing.size_bytes > 256


def test_discover_audio_folder(tmp_path):
    write_fake_audio(tmp_path / "a.mp3")
    write_fake_audio(tmp_path / "b.wav")
    (tmp_path / "readme.txt").write_bytes(b"x" * 300)
    found = discover_audio(tmp_path)
    assert [p.name for p in found] == ["a.mp3", "b.wav"]
