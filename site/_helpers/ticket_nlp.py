"""Prepare a ticket sample with MiniLM embeddings for the docs vignette.

User-facing SQL uses community ``quackformers`` (``embed()`` ≈
``sentence-transformers/all-MiniLM-L6-v2``). This repo tracks DuckDB 2.0 alpha,
where that community binary is not on the CDN yet, so Quarto execution embeds
with the same MiniLM model offline and loads cached Parquet. Printed SQL still
shows the quackformers recipe for builds that can
``INSTALL quackformers FROM community``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / ".cache" / "tickets"
CSV_URL = (
    "https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/"
    "resolve/main/aa_dataset-tickets-multi-lang-5-2-50-version.csv"
)
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# Focused routing demo: two semantically distinct queues, balanced sample.
FOCUS_QUEUES = ("Technical Support", "Billing and Payments")
PER_QUEUE = 400
SEED = 42

QUEUE_PROTOTYPES: list[tuple[str, str]] = [
    (
        "Technical Support",
        "Technical issues with software applications, bugs, crashes, errors, "
        "configuration problems, failed features, and troubleshooting.",
    ),
    (
        "Billing and Payments",
        "Invoices, charges, refunds, payment failures, billing disputes, "
        "subscriptions, receipts, and account balance questions.",
    ),
]


def _csv_path() -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(CSV_URL.encode()).hexdigest()[:12]
    return CACHE / f"tickets-{digest}.csv"


def _sample_paths() -> tuple[Path, Path]:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(
        f"{CSV_URL}|{MODEL_ID}|{FOCUS_QUEUES}|{PER_QUEUE}|{SEED}|v2".encode()
    ).hexdigest()[:12]
    return CACHE / f"sample-{key}.parquet", CACHE / f"prototypes-{key}.parquet"


def download_csv() -> Path:
    path = _csv_path()
    if path.is_file() and path.stat().st_size > 1_000_000:
        return path
    print(f"Downloading ticket CSV to {path} ...", flush=True)
    with urllib.request.urlopen(CSV_URL, timeout=180) as resp:
        path.write_bytes(resp.read())
    return path


def _balanced_focus_rows(csv_path: Path) -> list[dict[str, str]]:
    text = csv_path.read_text(encoding="utf-8", errors="replace")
    by_queue: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in csv.DictReader(io.StringIO(text)):
        if (row.get("language") or "").lower() != "en":
            continue
        queue = (row.get("queue") or "").strip()
        if queue not in FOCUS_QUEUES:
            continue
        if not (row.get("subject") or "").strip() or not (row.get("body") or "").strip():
            continue
        by_queue[queue].append(row)
    rng = random.Random(SEED)
    picked: list[dict[str, str]] = []
    for queue in FOCUS_QUEUES:
        bucket = by_queue.get(queue, [])
        if len(bucket) < PER_QUEUE:
            raise RuntimeError(f"Need >= {PER_QUEUE} English rows for {queue}, got {len(bucket)}")
        rng.shuffle(bucket)
        picked.extend(bucket[:PER_QUEUE])
    rng.shuffle(picked)
    return picked


def prepare_parquets() -> tuple[Path, Path]:
    tickets_path, proto_path = _sample_paths()
    if tickets_path.is_file() and proto_path.is_file():
        return tickets_path, proto_path

    from sentence_transformers import SentenceTransformer
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = _balanced_focus_rows(download_csv())
    model = SentenceTransformer(MODEL_ID)
    ticket_texts = [
        f"{(r.get('subject') or '').strip()}\n{(r.get('body') or '').strip()}" for r in rows
    ]
    ticket_emb = model.encode(ticket_texts, normalize_embeddings=True, show_progress_bar=False)
    proto_texts = [text for _, text in QUEUE_PROTOTYPES]
    proto_emb = model.encode(proto_texts, normalize_embeddings=True, show_progress_bar=False)

    pq.write_table(
        pa.table(
            {
                "ticket_id": pa.array(range(len(rows)), type=pa.int32()),
                "subject": [r["subject"] for r in rows],
                "body": [r["body"] for r in rows],
                "queue": [r["queue"] for r in rows],
                "priority": [r["priority"] for r in rows],
                "language": [r["language"] for r in rows],
                "type": [r.get("type") or "" for r in rows],
                "text": ticket_texts,
                "emb": pa.array(
                    [list(map(float, v)) for v in ticket_emb], type=pa.list_(pa.float32())
                ),
            }
        ),
        tickets_path,
    )
    pq.write_table(
        pa.table(
            {
                "queue": [name for name, _ in QUEUE_PROTOTYPES],
                "prototype": proto_texts,
                "emb": pa.array(
                    [list(map(float, v)) for v in proto_emb], type=pa.list_(pa.float32())
                ),
            }
        ),
        proto_path,
    )
    return tickets_path, proto_path


def load_setup_sql() -> str:
    tickets_path, proto_path = prepare_parquets()
    t = tickets_path.as_posix()
    p = proto_path.as_posix()
    return f"""
LOAD duckboost;
CREATE OR REPLACE TABLE tickets_emb AS
SELECT
  ticket_id,
  subject,
  body,
  queue,
  priority,
  language,
  type,
  text,
  emb::FLOAT[384] AS emb
FROM read_parquet('{t}');
CREATE OR REPLACE TABLE label_prototypes AS
SELECT
  queue,
  prototype,
  emb::FLOAT[384] AS emb
FROM read_parquet('{p}');
"""


def show(display_sql: str, execute_sql: str | None = None) -> None:
    """Print ``display_sql`` and run ``execute_sql`` (default: same) with sample tables."""
    from _helpers.duckrun import run_sql

    display_sql = display_sql.strip()
    execute_sql = (execute_sql or display_sql).strip()
    output = run_sql(f"{load_setup_sql()}\n{execute_sql}").rstrip()
    print(f"```sql\n{display_sql}\n```\n")
    print("```text")
    print(output)
    print("```")
