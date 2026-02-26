"""
CONSTABLE – FAISS embedding store for face vectors.
Face embeddings (512-d float32 from FaceNet/InceptionResnetV1) are stored in a
flat L2 index.  A parallel JSON sidecar maps FAISS integer IDs → employee IDs.
"""

import os
import json
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[EmbeddingStore] faiss-cpu not installed – using brute-force fallback.")

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "database")
INDEX_PATH = os.path.join(DB_DIR, "face_index.faiss")
META_PATH  = os.path.join(DB_DIR, "face_meta.json")

EMBEDDING_DIM = 512
SIMILARITY_THRESHOLD = 0.85   # cosine similarity threshold (after L2-normalisation)


class EmbeddingStore:
    def __init__(self):
        os.makedirs(DB_DIR, exist_ok=True)
        self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self):
        if FAISS_AVAILABLE and os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH) as f:
                self.meta = json.load(f)   # {str(faiss_id): employee_id}
        else:
            if FAISS_AVAILABLE:
                self.index = faiss.IndexFlatIP(EMBEDDING_DIM)   # inner product on L2-normed vecs = cosine
            else:
                self.index = None
            self.meta = {}

    def _save(self):
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "w") as f:
            json.dump(self.meta, f)

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-10 else vec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, employee_id: str, embeddings: list):
        """Add one or more embeddings for an employee."""
        for emb in embeddings:
            vec = self._normalise(np.array(emb, dtype=np.float32)).reshape(1, -1)
            if FAISS_AVAILABLE and self.index is not None:
                faiss_id = self.index.ntotal
                self.index.add(vec)
                self.meta[str(faiss_id)] = employee_id
            else:
                # Brute-force fallback: store as list in meta
                faiss_id = len(self.meta)
                self.meta[str(faiss_id)] = {"id": employee_id, "vec": vec.tolist()[0]}
        self._save()

    def search(self, embedding: np.ndarray, top_k: int = 1):
        """
        Returns (employee_id, similarity_score) or (None, 0.0) if no match.
        """
        vec = self._normalise(np.array(embedding, dtype=np.float32)).reshape(1, -1)

        if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
            distances, indices = self.index.search(vec, top_k)
            best_idx = int(indices[0][0])
            best_score = float(distances[0][0])
            if best_score >= SIMILARITY_THRESHOLD and best_idx != -1:
                employee_id = self.meta.get(str(best_idx))
                return employee_id, best_score
            return None, best_score

        # Brute-force fallback
        best_score = -1.0
        best_id = None
        for key, val in self.meta.items():
            if isinstance(val, dict):
                stored_vec = np.array(val["vec"], dtype=np.float32)
                score = float(np.dot(vec.flatten(), stored_vec))
                if score > best_score:
                    best_score = score
                    best_id = val["id"]
        if best_score >= SIMILARITY_THRESHOLD:
            return best_id, best_score
        return None, best_score

    def remove_employee(self, employee_id: str):
        """Remove all vectors for an employee (requires index rebuild)."""
        if not FAISS_AVAILABLE or self.index is None:
            self.meta = {k: v for k, v in self.meta.items()
                         if not (isinstance(v, dict) and v.get("id") == employee_id)}
            self._save()
            return

        # Collect surviving entries
        survivors = [(k, v) for k, v in self.meta.items() if v != employee_id]
        new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        new_meta = {}

        # We can't retrieve raw vectors from IndexFlatIP after the fact,
        # so we rebuild from scratch using stored reconstructed vectors.
        # (IndexFlatIP supports reconstruct)
        for old_key, emp_id in self.meta.items():
            if emp_id == employee_id:
                continue
            vec = np.zeros((1, EMBEDDING_DIM), dtype=np.float32)
            self.index.reconstruct(int(old_key), vec.reshape(-1))
            new_id = new_index.ntotal
            new_index.add(vec)
            new_meta[str(new_id)] = emp_id

        self.index = new_index
        self.meta = new_meta
        self._save()

    @property
    def total_vectors(self):
        if FAISS_AVAILABLE and self.index is not None:
            return self.index.ntotal
        return sum(1 for v in self.meta.values() if isinstance(v, dict))
