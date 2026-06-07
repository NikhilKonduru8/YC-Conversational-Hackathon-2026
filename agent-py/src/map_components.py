"""Map a folder of component JSON files into Moss knowledge chunks.

Drop your component database — one or many ``*.json`` files, up to ~100 MB
total — into ``agent-py/data/`` (configurable via ``JARVIS_DATA_DIR``). Each file
may be a single component object or a list of them. Every component is flattened
into a few focused, self-contained RAG chunks so semantic retrieval stays sharp:

  * ``<part>-overview``  — what the part is (name, category, package, description)
  * ``<part>-specs``     — flattened electrical specifications / abs-max ratings
  * ``<part>-pinout``    — the pin map

This module is the mapping *library*; ``create_index.py`` streams it into Moss in
batches so memory stays flat regardless of how big ``data/`` gets.

Run directly to sanity-check a data folder without touching Moss:

    uv run src/map_components.py            # stats over data/
    uv run src/map_components.py --sample   # also print a few example chunks
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from config import load_config

_UNITS = {
    "v",
    "ma",
    "a",
    "mv",
    "uv",
    "na",
    "ua",
    "pa",
    "w",
    "mw",
    "hz",
    "khz",
    "mhz",
    "ghz",
    "c",
    "k",
    "f",
    "pf",
    "nf",
    "uf",
    "ohm",
    "kohm",
    "mohm",
    "ns",
    "us",
    "ms",
    "s",
    "db",
    "bits",
    "kb",
    "mb",
    "rpm",
}


def _humanize_key(key: str) -> str:
    """`absolute_max_supply_voltage_v` -> `absolute max supply voltage (v)`."""
    parts = key.replace("-", "_").split("_")
    if parts and parts[-1] in _UNITS:
        unit = parts.pop()
        return f"{' '.join(parts)} ({unit})"
    return " ".join(parts)


def _flatten(value: Any, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            label = _humanize_key(str(k))
            lines.extend(_flatten(v, f"{prefix} {label}".strip()))
    elif isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            lines.append(f"{prefix}: {', '.join(str(i) for i in value)}")
        else:
            for item in value:
                lines.extend(_flatten(item, prefix))
    else:
        lines.append(f"{prefix}: {value}")
    return lines


def _part_id(component: dict) -> str:
    pn = component.get("part_number") or component.get("part") or component.get("id")
    name = component.get("name") or "component"
    return str(pn or name).strip()


def _str_meta(component: dict, source: str = "") -> dict[str, str]:
    meta: dict[str, str] = {"kind": "component"}
    for key in ("part_number", "part", "category", "manufacturer", "package"):
        val = component.get(key)
        if val:
            meta[key] = str(val)
    if source:
        meta["source"] = source
    return meta


def component_to_documents(component: dict, source: str = "") -> list[dict]:
    """Produce the knowledge chunks ({id, text, metadata}) for one component."""
    part = _part_id(component)
    name = component.get("name") or part
    base_meta = _str_meta(component, source)
    docs: list[dict] = []

    overview = [f"{name} ({part})."]
    if component.get("category"):
        overview.append(f"Category: {component['category']}.")
    if component.get("manufacturer"):
        overview.append(f"Manufacturer: {component['manufacturer']}.")
    if component.get("package"):
        overview.append(f"Package: {component['package']}.")
    if component.get("description"):
        overview.append(str(component["description"]))
    docs.append(
        {
            "id": f"{part}-overview",
            "text": " ".join(overview),
            "metadata": {**base_meta, "aspect": "overview"},
        }
    )

    specs = component.get("electrical_specs") or component.get("specs")
    if specs:
        lines = _flatten(specs)
        text = (
            f"Electrical specifications for {name} ({part}): "
            + "; ".join(ln.strip().lstrip(":").strip() for ln in lines if ln)
            + "."
        )
        docs.append(
            {
                "id": f"{part}-specs",
                "text": text,
                "metadata": {**base_meta, "aspect": "electrical_specs"},
            }
        )

    pinout = component.get("pinout") or component.get("pins")
    if pinout:
        pin_lines: list[str] = []
        if isinstance(pinout, list):
            for entry in pinout:
                if isinstance(entry, dict):
                    pin = entry.get("pin") or entry.get("number") or "?"
                    pname = entry.get("name") or entry.get("signal") or ""
                    pdesc = entry.get("description") or entry.get("function") or ""
                    pin_lines.append(
                        f"pin {pin} = {pname}".strip()
                        + (f" ({pdesc})" if pdesc else "")
                    )
                else:
                    pin_lines.append(str(entry))
        elif isinstance(pinout, dict):
            pin_lines = [f"pin {k} = {v}" for k, v in pinout.items()]
        text = (
            f"Pinout / pin configuration for {name} ({part}): "
            + "; ".join(pin_lines)
            + "."
        )
        docs.append(
            {
                "id": f"{part}-pinout",
                "text": text,
                "metadata": {**base_meta, "aspect": "pinout"},
            }
        )

    return docs


def chunks_to_documents(doc: dict, source: str = "") -> list[dict]:
    """Curated `curated_components/` format: one Moss doc per chunk.

    Each file is {component_id, description?, ki_keywords?, chunks:[{title,text}]}.
    The chunk text is already self-contained (it starts with
    "[Component context: <id>]"), so we embed it as-is. `description` and
    `ki_keywords` may be absent — handled with .get().
    """
    component_id = str(doc.get("component_id") or Path(source).stem or "component")
    description = doc.get("description") or ""
    ki_keywords = doc.get("ki_keywords") or ""
    chunks = doc.get("chunks") or []
    is_generic = len(chunks) == 1

    documents: list[dict] = []
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        documents.append(
            {
                # One vector per chunk; share component_id so retrieval can group.
                "id": f"{component_id}#{i}",
                "text": text,
                "metadata": {
                    "component_id": component_id,
                    "chunk_index": i,
                    "chunk_title": chunk.get("title", ""),
                    "description": description,
                    "ki_keywords": ki_keywords,
                    "is_generic": is_generic,
                },
            }
        )
    return documents


def iter_component_files(data_dir: str | Path) -> Iterator[Path]:
    """Yield every real *.json file under the data directory (recursively).

    Skips macOS AppleDouble junk (``._foo.json`` and ``__MACOSX/``) that appears
    when a Mac-made zip is extracted on Linux — those are binary, not JSON.
    """
    root = Path(data_dir)
    if not root.exists():
        return
    for path in sorted(root.rglob("*.json")):
        if path.name.startswith("._") or "__MACOSX" in path.parts:
            continue
        yield path


def _documents_for(data: object, source: str) -> Iterator[dict]:
    """Dispatch a parsed file to the right mapper based on its shape."""
    # Curated chunk format: {component_id, chunks:[...]}
    if isinstance(data, dict) and isinstance(data.get("chunks"), list):
        yield from chunks_to_documents(data, source=source)
    # Legacy: a list of component objects.
    elif isinstance(data, list):
        for component in data:
            if isinstance(component, dict):
                yield from component_to_documents(component, source=source)
    # Legacy: a single component object (electrical_specs/pinout schema).
    elif isinstance(data, dict):
        yield from component_to_documents(data, source=source)


def iter_documents(data_dir: str | Path) -> Iterator[dict]:
    """Stream Moss documents from every component file under `data_dir`.

    Loads one file at a time (peak memory ~= largest single file), so this scales
    to thousands of files / hundreds of MB without holding everything in memory.
    Supports both the curated chunk format and the legacy component schema.
    """
    for path in iter_component_files(data_dir):
        try:
            # Tolerate stray non-UTF-8 bytes (e.g. a ° in a datasheet) instead of
            # crashing the whole build; errors="replace" keeps the file usable.
            text = path.read_bytes().decode("utf-8", errors="replace")
            data = json.loads(text)
        except (OSError, ValueError) as exc:  # ValueError covers JSON + decode
            print(f"  ! skipping {path.name}: {exc}", file=sys.stderr)
            continue
        yield from _documents_for(data, path.name)


def main() -> None:
    cfg = load_config()
    data_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(cfg.moss.data_dir)
    show_sample = "--sample" in sys.argv

    files = list(iter_component_files(data_dir))
    print(f"Data dir: {data_dir}")
    print(f"  JSON files: {len(files)}")

    chunks = 0
    samples: list[dict] = []
    for doc in iter_documents(data_dir):
        chunks += 1
        if show_sample and len(samples) < 5:
            samples.append(doc)
    print(f"  knowledge chunks: {chunks}")

    for doc in samples:
        meta = doc.get("metadata", {})
        label = meta.get("chunk_title") or meta.get("aspect") or ""
        print(f"\n  [{doc['id']}] {label}")
        print(f"    {doc['text'][:160]}")

    if not files:
        print("\n  (empty) — drop component *.json files into the data folder.")
    else:
        print("\nNext: run `uv run src/create_index.py` to build the Moss index.")


if __name__ == "__main__":
    main()
