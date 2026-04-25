RANDOM_SEED = 42  # used everywhere: subsampling, splits, baselines

INDEX_TYPES = ["IVF_FLAT", "IVF_PQ", "HNSW"]
DATASETS = ["sift-1M", "gist-1M", "fashion-mnist"]
N_FRACTIONS = [0.02, 0.05, 0.10, 0.15, 0.20]
MEMORY_BUDGETS_MB = [32, 64, 128, 256, 512, 1024]
RECALL_TARGETS = [0.85, 0.90, 0.95, 0.99]
K_VALUES = [5, 10, 50, 100]

# memory violation penalized more: memory is a system resource constraint
MEMORY_VIOLATION_WEIGHT = 2.0
RECALL_VIOLATION_WEIGHT = 1.0

HNSW_PARAMS = {"M": 32, "efConstruction": 200, "efSearch": 128}
IVF_FLAT_PARAMS = {"nlist": 256, "nprobe": 32}
IVF_PQ_PARAMS = {"nlist": 256, "nprobe": 32, "m": 8, "nbits": 8}

DATA_DIR = "data"
ARTIFACTS_DIR = "artifacts"
RESULTS_DIR = "results"

# all Eulidean datasets are available in HDF5 format
DATASET_URLS = {
    "sift-1M": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
    "gist-1M": "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
    "fashion-mnist": "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
}
