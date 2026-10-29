"""
Stage 6: append-only, hash-chained forensic log (JSON Lines).

Every entry records the SHA-256 of the entry before it ("prev_hash"), and its
own "hash" is the SHA-256 of its canonical JSON (sorted keys, compact
separators) with the "hash" field left out. Editing, deleting or reordering
any past line breaks the chain from that point on, which verify() reports.

This is tamper-*evident*, not tamper-proof: someone with write access can
rewrite the whole file and recompute every hash. For a real deployment,
periodically anchor the latest hash somewhere the log writer can't touch
(sign it, or push it to a separate write-once store).
"""

import hashlib
import json
import os
import threading
from datetime import datetime, timezone

GENESIS_HASH = "0" * 64


def _canonical(entry):
    return json.dumps(entry, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def compute_hash(entry):
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


class ForensicLog:
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._tail = None  # (seq, hash) of the last entry, loaded lazily
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _read_entries(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _load_tail(self):
        if self._tail is None:
            entries = self._read_entries()
            self._tail = (entries[-1]["seq"], entries[-1]["hash"]) if entries else (0, GENESIS_HASH)
        return self._tail

    def append(self, kind, payload):
        """Append one entry and return it (including its hash)."""
        with self._lock:
            last_seq, last_hash = self._load_tail()
            entry = {
                "seq": last_seq + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "kind": kind,
                "prev_hash": last_hash,
                "payload": payload,
            }
            entry["hash"] = compute_hash(entry)
            line = _canonical(entry)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._tail = (entry["seq"], entry["hash"])
        # Return exactly what was written (default=str may have stringified values).
        return json.loads(line)

    def entries(self, limit=None):
        with self._lock:
            entries = self._read_entries()
        return entries[-limit:] if limit else entries

    def verify(self):
        """Re-walk the chain. Returns {"ok", "count", "broken_at", "reason"}."""
        with self._lock:
            try:
                entries = self._read_entries()
            except json.JSONDecodeError as e:
                return {"ok": False, "count": None, "broken_at": None,
                        "reason": f"unparseable line: {e}"}

        prev_hash = GENESIS_HASH
        for i, entry in enumerate(entries):
            seq = entry.get("seq")
            if seq != i + 1:
                return {"ok": False, "count": len(entries), "broken_at": seq,
                        "reason": f"expected seq {i + 1}, found {seq} (entry missing or reordered)"}
            if entry.get("prev_hash") != prev_hash:
                return {"ok": False, "count": len(entries), "broken_at": seq,
                        "reason": "prev_hash does not match the previous entry"}
            if compute_hash(entry) != entry.get("hash"):
                return {"ok": False, "count": len(entries), "broken_at": seq,
                        "reason": "entry contents do not match its hash"}
            prev_hash = entry["hash"]
        return {"ok": True, "count": len(entries), "broken_at": None, "reason": None}
