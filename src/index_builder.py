import faiss
import numpy as np

from src.config import HNSW_PARAMS, IVF_FLAT_PARAMS, IVF_PQ_PARAMS

def build_index(
    index_type: str,
    vectors: np.ndarray,
    params: dict | None = None,
) -> faiss.Index:
    """Build and train a FAISS index using METRIC_L2.

    params overrides defaults from config for the given index_type.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    d = vectors.shape[1]

    if index_type == "HNSW":
        return _build_hnsw(vectors, d, params)
    elif index_type == "IVF_FLAT":
        return _build_ivf_flat(vectors, d, params)
    elif index_type == "IVF_PQ":
        return _build_ivf_pq(vectors, d, params)
    else:
        raise ValueError(f"Unknown index_type: {index_type}")

def _build_hnsw(vectors: np.ndarray, d: int, overrides: dict | None) -> faiss.Index:
    p = {**HNSW_PARAMS, **(overrides or {})}
    index = faiss.IndexHNSWFlat(d, p["M"], faiss.METRIC_L2)
    index.hnsw.efConstruction = p["efConstruction"]
    index.hnsw.efSearch = p["efSearch"]
    index.add(vectors)
    return index

def _build_ivf_flat(vectors: np.ndarray, d: int, overrides: dict | None) -> faiss.Index:
    p = {**IVF_FLAT_PARAMS, **(overrides or {})}
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, p["nlist"], faiss.METRIC_L2)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = p["nprobe"]
    return index

def _build_ivf_pq(vectors: np.ndarray, d: int, overrides: dict | None) -> faiss.Index:
    p = {**IVF_PQ_PARAMS, **(overrides or {})}
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, p["nlist"], p["m"], p["nbits"])
    index.train(vectors)
    index.add(vectors)
    index.nprobe = p["nprobe"]
    return index
