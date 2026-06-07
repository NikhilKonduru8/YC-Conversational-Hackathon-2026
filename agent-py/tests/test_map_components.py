"""Unit tests for mapping component JSON -> Moss knowledge chunks."""

import json

from map_components import (
    chunks_to_documents,
    component_to_documents,
    iter_documents,
)

COMPONENT = {
    "part_number": "LM358",
    "name": "LM358 Dual Op-Amp",
    "category": "op-amp",
    "manufacturer": "TI",
    "package": "PDIP-8",
    "description": "Dual low-power op-amp.",
    "electrical_specs": {
        "absolute_maximum_supply_voltage_v": 32,
        "supply_current_typ_ma": 0.7,
    },
    "pinout": [
        {"pin": 1, "name": "OUT1", "description": "Output A"},
        {"pin": 8, "name": "VCC", "description": "Positive supply"},
    ],
}


def test_produces_three_aspect_chunks():
    docs = component_to_documents(COMPONENT)
    ids = {d["id"] for d in docs}
    assert ids == {"LM358-overview", "LM358-specs", "LM358-pinout"}


def test_chunks_are_self_contained_and_have_string_metadata():
    docs = component_to_documents(COMPONENT)
    for d in docs:
        assert d["text"].strip()
        # part number appears in every chunk so retrieval can attribute it.
        assert "LM358" in d["text"]
        assert d["metadata"]["kind"] == "component"
        assert d["metadata"]["part_number"] == "LM358"


def test_specs_chunk_contains_flattened_values_with_units():
    specs = next(
        d for d in component_to_documents(COMPONENT) if d["id"] == "LM358-specs"
    )
    assert "absolute maximum supply voltage (v): 32" in specs["text"]
    assert "supply current typ (ma): 0.7" in specs["text"]


def test_pinout_chunk_lists_pins():
    pinout = next(
        d for d in component_to_documents(COMPONENT) if d["id"] == "LM358-pinout"
    )
    assert "pin 1 = OUT1" in pinout["text"]
    assert "pin 8 = VCC" in pinout["text"]


def test_component_without_specs_or_pinout_still_has_overview():
    docs = component_to_documents({"part_number": "X1", "name": "Mystery part"})
    assert len(docs) == 1
    assert docs[0]["id"] == "X1-overview"


def test_iter_documents_streams_list_and_single_files(tmp_path):
    # One file with a list of components, one with a single component object.
    (tmp_path / "a.json").write_text(
        json.dumps([COMPONENT, {"part_number": "NE555", "name": "NE555 Timer"}])
    )
    (tmp_path / "b.json").write_text(
        json.dumps({"part_number": "X1", "name": "Mystery part"})
    )
    docs = list(iter_documents(tmp_path))
    ids = {d["id"] for d in docs}
    assert "LM358-overview" in ids
    assert "NE555-overview" in ids
    assert "X1-overview" in ids
    # Source filename is tracked in metadata for traceability.
    assert {d["metadata"]["source"] for d in docs} == {"a.json", "b.json"}


def test_iter_documents_skips_corrupt_files(tmp_path):
    (tmp_path / "good.json").write_text(json.dumps([COMPONENT]))
    (tmp_path / "broken.json").write_text("{ not valid json ")
    docs = list(iter_documents(tmp_path))
    # The corrupt file is skipped; the good one still indexes.
    assert any(d["id"] == "LM358-overview" for d in docs)


def test_iter_documents_empty_dir(tmp_path):
    assert list(iter_documents(tmp_path)) == []


# --- curated_components/ chunk format ---------------------------------------
CURATED = {
    "component_id": "ATmega2560-16C",
    "description": "16MHz, 256kB Flash",
    "ki_keywords": "AVR microcontroller",
    "chunks": [
        {"title": "Section 1", "text": "[Component context: ATmega2560-16C] intro..."},
        {
            "title": "Section 2",
            "text": "[Component context: ATmega2560-16C] 8kB SRAM...",
        },
    ],
}


def test_curated_one_document_per_chunk():
    docs = chunks_to_documents(CURATED, source="ATmega2560-16C.json")
    assert [d["id"] for d in docs] == ["ATmega2560-16C#0", "ATmega2560-16C#1"]
    assert docs[0]["text"].startswith("[Component context: ATmega2560-16C]")
    m = docs[0]["metadata"]
    assert m["component_id"] == "ATmega2560-16C"
    assert m["chunk_index"] == 0
    assert m["chunk_title"] == "Section 1"
    assert m["description"] == "16MHz, 256kB Flash"
    assert m["is_generic"] is False


def test_curated_single_chunk_is_generic():
    doc = {
        "component_id": "LED",
        "chunks": [
            {"title": "Component Overview", "text": "[Component context: LED] ..."}
        ],
    }
    docs = chunks_to_documents(doc)
    assert len(docs) == 1
    assert docs[0]["metadata"]["is_generic"] is True


def test_curated_missing_description_and_keywords():
    # Some files have only component_id + chunks (no description/ki_keywords).
    doc = {"component_id": "74AHC1G00", "chunks": [{"title": "Section 1", "text": "x"}]}
    docs = chunks_to_documents(doc)
    assert docs[0]["metadata"]["description"] == ""
    assert docs[0]["metadata"]["ki_keywords"] == ""


def test_curated_skips_empty_chunk_text():
    doc = {
        "component_id": "X",
        "chunks": [{"title": "a", "text": ""}, {"title": "b", "text": "ok"}],
    }
    docs = chunks_to_documents(doc)
    assert len(docs) == 1
    assert docs[0]["text"] == "ok"


def test_iter_documents_handles_curated_files(tmp_path):
    (tmp_path / "ATmega2560-16C.json").write_text(json.dumps(CURATED))
    docs = list(iter_documents(tmp_path))
    assert {d["id"] for d in docs} == {"ATmega2560-16C#0", "ATmega2560-16C#1"}
