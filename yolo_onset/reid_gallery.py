"""
Stage 5: cross-camera matching over OSNet embeddings.

An in-memory gallery of person embeddings seen by each camera. A query from
one camera is compared (cosine similarity) only against entries from *other*
cameras — a hit within the same camera is just the same track, not a
cross-camera link. Embeddings are stored L2-normalized so a dot product is
the cosine similarity.

This is intentionally simple (a Python list, linear scan). Swap it for Redis
or a FAISS index once the gallery grows past a few thousand entries.
"""

import threading
import time

import numpy as np


def _normalize(embedding):
    embedding = np.asarray(embedding, dtype=np.float32).ravel()
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


class EmbeddingGallery:
    def __init__(self, threshold=0.7, max_entries=5000):
        self.threshold = threshold
        self.max_entries = max_entries
        self._entries = []
        self._lock = threading.Lock()

    def match_and_add(self, camera_id, embeddings, meta=None):
        """
        Match each embedding against other cameras' entries, then store them.

        Returns one dict per embedding:
            {"matched", "similarity", "matched_camera", "matched_event"}
        where similarity is the best score against other cameras (None if the
        gallery has no other cameras yet).
        """
        meta = dict(meta or {})
        queries = [_normalize(e) for e in embeddings]
        results = []
        with self._lock:
            others = [e for e in self._entries if e["camera_id"] != camera_id]
            gallery = np.stack([e["embedding"] for e in others]) if others else None
            for query in queries:
                best = {"matched": False, "similarity": None,
                        "matched_camera": None, "matched_event": None}
                if gallery is not None:
                    sims = gallery @ query
                    i = int(np.argmax(sims))
                    best["similarity"] = round(float(sims[i]), 4)
                    if sims[i] >= self.threshold:
                        best.update(matched=True,
                                    matched_camera=others[i]["camera_id"],
                                    matched_event=others[i]["meta"].get("event_ref"))
                results.append(best)

            now = time.time()
            for query in queries:
                self._entries.append({"camera_id": camera_id, "timestamp": now,
                                      "embedding": query, "meta": meta})
            if len(self._entries) > self.max_entries:
                del self._entries[:len(self._entries) - self.max_entries]
        return results

    def cameras(self):
        with self._lock:
            return sorted({e["camera_id"] for e in self._entries})

    def __len__(self):
        with self._lock:
            return len(self._entries)
