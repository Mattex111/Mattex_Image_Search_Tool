import os
import hashlib
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np
import torch
import pyarrow as pa
import lancedb
from PIL import Image
from transformers import AutoModel, AutoProcessor

SUPPORTED_MODELS = {
    "siglip2-so400m": {
        "model_id": "google/siglip2-so400m-patch14-384",
        "dim": 1152,
        "input_size": 384,
        "description": "SigLIP 2 SO400M (384x384, 1152-dim) - Best Accuracy & Detail"
    },
    "siglip2-base": {
        "model_id": "google/siglip2-base-patch16-256",
        "dim": 768,
        "input_size": 256,
        "description": "SigLIP 2 Base (256x256, 768-dim) - Lightweight & Fast"
    }
}

DEFAULT_MODEL_KEY = "siglip2-so400m"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

class SigLIP2Engine:
    """Core multimodal neural engine powered by Google's SigLIP 2 vision-language models."""

    def __init__(self, model_key: str = DEFAULT_MODEL_KEY, device: Optional[str] = None):
        if model_key not in SUPPORTED_MODELS:
            model_key = DEFAULT_MODEL_KEY

        self.model_key = model_key
        self.model_info = SUPPORTED_MODELS[model_key]
        self.model_id = self.model_info["model_id"]
        self.dim = self.model_info["dim"]
        self.input_size = self.model_info["input_size"]

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device == "cuda" and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        print(f"[SigLIP2Engine] Loading model '{self.model_id}' on {self.device} ({self.dtype})...")
        try:
            # Try loading from local cache first for fast offline startup without DNS retries
            self.processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=True)
            self.model = AutoModel.from_pretrained(self.model_id, dtype=self.dtype, local_files_only=True).to(self.device).eval()
            print(f"[SigLIP2Engine] Loaded '{self.model_id}' from local cache (offline ready).")
        except Exception:
            # Fallback to downloading online from HuggingFace Hub if not cached yet
            print(f"[SigLIP2Engine] Local cache miss. Downloading '{self.model_id}' from HuggingFace Hub...")
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModel.from_pretrained(self.model_id, dtype=self.dtype).to(self.device).eval()

        print(f"[SigLIP2Engine] Model loaded successfully. Vector dimension: {self.dim}")

    def encode_images_batch(self, image_paths: List[str]) -> List[List[float]]:
        """Extract L2-normalized image embeddings for a batch of image file paths."""
        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                # Fallback to blank placeholder image on file read error
                images.append(Image.new("RGB", (self.input_size, self.input_size)))

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        if self.dtype in (torch.bfloat16, torch.float16):
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        with torch.inference_mode():
            out = self.model.get_image_features(**inputs)
            feats = out.pooler_output if (hasattr(out, "pooler_output") and out.pooler_output is not None) else out
            feats = feats / feats.norm(dim=-1, keepdim=True)

        feats_fp32 = feats.to(torch.float32).cpu().numpy()
        return feats_fp32.tolist()

    def encode_text_query(self, query: str, use_ensembling: bool = True) -> List[float]:
        """Extract L2-normalized text embedding using multi-prompt ensembling."""
        query_str = query.strip()
        if not query_str:
            return [0.0] * self.dim

        if use_ensembling:
            prompts = [
                query_str,
                f"a photo of {query_str}",
                f"photo of {query_str}"
            ]
        else:
            prompts = [query_str]

        inputs = self.processor(text=prompts, padding="max_length", max_length=64, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.get_text_features(**inputs)
            feats = out.pooler_output if (hasattr(out, "pooler_output") and out.pooler_output is not None) else out
            feats = feats / feats.norm(dim=-1, keepdim=True)
            if use_ensembling and feats.shape[0] > 1:
                feats = feats.mean(dim=0, keepdim=True)
                feats = feats / feats.norm(dim=-1, keepdim=True)

        vec = feats.to(torch.float32).cpu().squeeze(0).numpy()
        return vec.tolist()

    def encode_composed_query(self, image_path: str, text_modification: str, alpha: float = 0.6, beta: float = 0.4) -> List[float]:
        """Composed Multimodal Search: Combines image features z_I and text modification features z_T.
        
        Formula: z_composed = normalize(alpha * z_I + beta * z_T)
        """
        img_vec = np.array(self.encode_images_batch([image_path])[0], dtype=np.float32)
        txt_vec = np.array(self.encode_text_query(text_modification), dtype=np.float32)

        composed = alpha * img_vec + beta * txt_vec
        norm = np.linalg.norm(composed)
        if norm > 0:
            composed = composed / norm

        return composed.tolist()


class LanceDBManager:
    """Manages the transactional LanceDB embedded vector storage and SQL DataFusion filtering engine."""

    def __init__(self, db_path: str, vector_dim: int, table_name: str = "images"):
        self.db_path = db_path
        self.vector_dim = vector_dim
        self.table_name = table_name

        os.makedirs(self.db_path, exist_ok=True)
        self.db = lancedb.connect(self.db_path)

        self.schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self.vector_dim)),
            pa.field("path", pa.string()),
            pa.field("rel_path", pa.string()),
            pa.field("category", pa.string()),
            pa.field("subcategory", pa.string()),
            pa.field("ocr_text", pa.string()),
            pa.field("file_size", pa.int64()),
            pa.field("mtime", pa.float64()),
            pa.field("indexed_at", pa.string()),
        ])

        if self.table_name in self.db.table_names():
            self.table = self.db.open_table(self.table_name)
            # Auto-migrate table schema if new fields (e.g., ocr_text) were added
            existing_names = self.table.schema.names
            for field in self.schema:
                if field.name not in existing_names:
                    try:
                        print(f"[LanceDBManager] Adding missing column '{field.name}' to table schema...")
                        self.table.add_columns({field.name: "''"})
                    except Exception as e:
                        print(f"[LanceDBManager] Could not add column '{field.name}': {e}")
        else:
            self.table = self.db.create_table(self.table_name, schema=self.schema, exist_ok=True)

    def add_records(self, records: List[Dict]):
        if not records:
            return
        self.table.add(records)

    def count(self) -> int:
        return len(self.table)

    def get_existing_paths_map(self) -> Dict[str, float]:
        """Returns a mapping of absolute path -> last modification time of indexed files."""
        if len(self.table) == 0:
            return {}
        df = self.table.search().select(["path", "mtime"]).to_pandas()
        return dict(zip(df["path"], df["mtime"]))

    def search_vector(
        self,
        query_vector: List[float],
        top_k: int = 10,
        category: Optional[str] = None,
        subcategory: Optional[str] = None,
        is_text_search: bool = True
    ) -> List[Dict]:
        """Executes vector similarity search with optional SQL scalar pre-filtering."""
        if len(self.table) == 0:
            return []

        query_builder = self.table.search(query_vector).metric("cosine").limit(top_k)

        conditions = []
        if category and category != "All":
            cat_escaped = category.replace("'", "''")
            conditions.append(f"category = '{cat_escaped}'")
        if subcategory and subcategory != "All":
            sub_escaped = subcategory.replace("'", "''")
            conditions.append(f"subcategory = '{sub_escaped}'")

        if conditions:
            filter_sql = " AND ".join(conditions)
            query_builder = query_builder.where(filter_sql)

        df = query_builder.to_pandas()
        if df.empty:
            return []

        results = []
        for idx, row in df.iterrows():
            dist = row["_distance"] if "_distance" in row else 0.0
            raw_sim = 1.0 - dist

            # Raw true cosine score without artificial warping for 100% transparency & future-proof model upgrades
            sim_pct = max(0.0, min(100.0, raw_sim * 100.0))

            results.append({
                "id": row["id"],
                "path": row["path"],
                "rel_path": row["rel_path"],
                "category": row["category"],
                "subcategory": row["subcategory"],
                "ocr_text": row["ocr_text"] if "ocr_text" in row and row["ocr_text"] else "",
                "score": sim_pct,
                "raw_cosine": raw_sim,
                "distance": dist
            })
        return results

    def delete_record_by_path(self, path: str) -> bool:
        """Deletes an image record from the LanceDB table by its file path."""
        try:
            path_escaped = path.replace("'", "''")
            self.table.delete(f"path = '{path_escaped}'")
            return True
        except Exception as e:
            print(f"[LanceDBManager] Error deleting record for path '{path}': {e}")
            return False

    def find_duplicates(self, similarity_threshold: float = 0.95, max_pairs: int = 100) -> List[Dict]:
        """Find duplicate or near-identical image pairs in the dataset based on vector cosine similarity."""
        if len(self.table) < 2:
            return []

        df = self.table.search().select(["id", "path", "rel_path", "vector"]).to_pandas()
        ids = df["id"].tolist()
        paths = df["path"].tolist()
        rel_paths = df["rel_path"].tolist()
        vectors = np.vstack(df["vector"].values).astype(np.float32)

        # Normalize vectors for fast matrix dot product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_vectors = vectors / norms

        sim_matrix = np.dot(norm_vectors, norm_vectors.T)
        np.fill_diagonal(sim_matrix, 0.0)

        pairs = []
        visited = set()
        num_items = len(ids)

        for i in range(num_items):
            if i in visited:
                continue
            sim_scores = sim_matrix[i]
            match_indices = np.where(sim_scores >= similarity_threshold)[0]

            if len(match_indices) > 0:
                cluster_matches = []
                for idx in match_indices:
                    cluster_matches.append({
                        "id": ids[idx],
                        "path": paths[idx],
                        "rel_path": rel_paths[idx],
                        "similarity": float(sim_scores[idx]) * 100.0
                    })
                    visited.add(idx)

                pairs.append({
                    "primary_path": paths[i],
                    "primary_rel_path": rel_paths[i],
                    "matches": cluster_matches
                })
                visited.add(i)

            if len(pairs) >= max_pairs:
                break

        return pairs

    def create_disk_index(self, num_partitions: int = 256, num_sub_vectors: int = 96):
        """Builds an IVF-PQ approximate nearest neighbor index on disk for scaling to 100k+ datasets."""
        if len(self.table) < 256:
            print("[LanceDBManager] Table has fewer than 256 items. Skipping IVF-PQ index creation.")
            return False
        
        print(f"[LanceDBManager] Building IVF-PQ disk index (partitions={num_partitions}, sub_vectors={num_sub_vectors})...")
        self.table.create_index(metric="cosine", num_partitions=num_partitions, num_sub_vectors=num_sub_vectors)
        print("[LanceDBManager] Disk index created successfully.")
        return True

    def has_disk_index(self) -> bool:
        """Checks if an IVF-PQ vector disk index exists for the current LanceDB table."""
        try:
            if not hasattr(self, 'table') or self.table is None:
                return False
            indices = self.table.list_indices()
            return len(indices) > 0
        except Exception as e:
            print(f"[LanceDBManager] Error checking disk indices: {e}")
            return False

    def get_categories_and_subcategories(self) -> Tuple[List[str], Dict[str, List[str]]]:
        if len(self.table) == 0:
            return ["All"], {"All": ["All"]}

        df = self.table.search().select(["category", "subcategory"]).to_pandas()
        categories = set(df["category"].dropna().unique())
        cats_list = ["All"] + sorted([c for c in categories if c])

        subcat_map = {"All": ["All"]}
        for cat in cats_list:
            if cat == "All":
                continue
            subcats = df[df["category"] == cat]["subcategory"].dropna().unique()
            subcat_map[cat] = ["All"] + sorted([s for s in subcats if s])

        return cats_list, subcat_map


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class IndexManager:
    """Traverses directories, extracts features, and incrementally updates the LanceDB vector store."""

    def __init__(self, engine: SigLIP2Engine, db_manager: LanceDBManager):
        self.engine = engine
        self.db_manager = db_manager

    def index_directory(
        self,
        root_folder: str,
        batch_size: int = 32,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> int:
        root_folder = os.path.normpath(root_folder)
        all_image_paths = []

        for dirpath, _, filenames in os.walk(root_folder):
            for f in filenames:
                ext = os.path.splitext(f.lower())[1]
                if ext in IMAGE_EXTENSIONS:
                    all_image_paths.append(os.path.normpath(os.path.join(dirpath, f)))

        if not all_image_paths:
            return 0

        existing_map = self.db_manager.get_existing_paths_map()
        paths_to_process = []
        for path in all_image_paths:
            try:
                mtime = os.path.getmtime(path)
                if path not in existing_map or existing_map[path] != mtime:
                    paths_to_process.append(path)
            except Exception:
                pass

        total_to_process = len(paths_to_process)
        if total_to_process == 0:
            return 0

        indexed_count = 0
        now_str = datetime.now().isoformat()

        for i in range(0, total_to_process, batch_size):
            batch_paths = paths_to_process[i:i + batch_size]
            vectors = self.engine.encode_images_batch(batch_paths)

            records = []
            for path, vec in zip(batch_paths, vectors):
                rel_path = os.path.relpath(path, root_folder)
                parts = rel_path.split(os.sep)

                if len(parts) > 1:
                    category = parts[0]
                    subcategory = parts[1] if len(parts) > 2 else "General"
                else:
                    category = "Root"
                    subcategory = "General"

                try:
                    file_size = os.path.getsize(path)
                    mtime = os.path.getmtime(path)
                    file_id = compute_sha256(path)[:16]
                except Exception:
                    file_size = 0
                    mtime = time.time()
                    file_id = hashlib.md5(path.encode()).hexdigest()[:16]

                records.append({
                    "id": file_id,
                    "vector": vec,
                    "path": path,
                    "rel_path": rel_path,
                    "category": category,
                    "subcategory": subcategory,
                    "ocr_text": "",
                    "file_size": file_size,
                    "mtime": mtime,
                    "indexed_at": now_str
                })

            self.db_manager.add_records(records)
            indexed_count += len(records)

            if progress_callback:
                last_file = os.path.basename(batch_paths[-1])
                progress_callback(indexed_count, total_to_process, last_file)

        return indexed_count
