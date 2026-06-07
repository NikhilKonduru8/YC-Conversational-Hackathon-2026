"""Build the Moss `knowledge` index from the component JSON files in `data/`.

Streams chunks from `data/` (see map_components.iter_documents) and pushes them
to Moss in batches, so a ~100 MB data folder indexes with flat memory use. The
first batch creates the index; the rest are appended with add_docs. An existing
index of the same name is deleted first for a clean rebuild.

    uv run src/map_components.py     # (optional) sanity-check the data folder
    uv run src/create_index.py       # build the Moss index

Needs MOSS_PROJECT_ID / MOSS_PROJECT_KEY in .env.local.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

from dotenv import load_dotenv
from moss import DocumentInfo, MossClient

from config import AGENT_DIR, load_config
from map_components import iter_component_files, iter_documents

load_dotenv(AGENT_DIR / ".env.local")


def _batches(docs: Iterator[dict], size: int) -> Iterator[list[DocumentInfo]]:
    """Group streamed chunk dicts into batches of DocumentInfo."""
    batch: list[DocumentInfo] = []
    for doc in docs:
        metadata = {str(k): str(v) for k, v in (doc.get("metadata") or {}).items()}
        batch.append(
            DocumentInfo(id=str(doc["id"]), text=str(doc["text"]), metadata=metadata)
        )
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


async def build_index() -> None:
    cfg = load_config()
    if not cfg.moss_ready:
        raise OSError(
            "Missing MOSS_PROJECT_ID / MOSS_PROJECT_KEY. Set them in "
            f"{AGENT_DIR / '.env.local'} before running this script."
        )

    data_dir = Path(cfg.moss.data_dir)
    files = list(iter_component_files(data_dir))
    if not files:
        raise SystemExit(
            f"No component *.json files found in {data_dir}. "
            "Add your component database there first."
        )
    print(f"Indexing {len(files)} file(s) from {data_dir} into '{cfg.moss.index}'...")

    client = MossClient(cfg.moss.project_id, cfg.moss.project_key)

    # Clean rebuild: drop an existing index of the same name if present.
    try:
        existing = {ix.name for ix in await client.list_indexes()}
        if cfg.moss.index in existing:
            print(f"  removing existing index '{cfg.moss.index}' for a clean rebuild")
            await client.delete_index(cfg.moss.index)
    except Exception as exc:
        print(f"  (could not check/delete existing index: {exc})")

    total = 0
    created = False
    for batch in _batches(iter_documents(data_dir), cfg.moss.index_batch_size):
        if not created:
            await client.create_index(cfg.moss.index, batch, cfg.moss.model_id)
            created = True
        else:
            await client.add_docs(cfg.moss.index, batch)
        total += len(batch)
        print(f"  indexed {total} chunks...", end="\r", flush=True)

    if not created:
        raise SystemExit("No valid component documents were produced from data/.")

    print(f"\nDone. Moss index '{cfg.moss.index}' built with {total} chunks.")


if __name__ == "__main__":
    asyncio.run(build_index())
