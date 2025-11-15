# Minimal, correct, no magic dependencies.
# pip install torch
import os
import io
import math
import zipfile
import urllib.request
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

MODELNET10_URL = "http://modelnet.cs.princeton.edu/ModelNet10.zip"
MODELNET10_CLASSES = [
    "bathtub","bed","chair","desk","dresser",
    "monitor","night_stand","sofa","table","toilet"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(MODELNET10_CLASSES)}

def _download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    tmp = dst.with_suffix(".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        block = 1 << 20
        while True:
            chunk = r.read(block)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dst)

def _extract_zip(zip_path: Path, target_dir: Path):
    # Idempotent extract: if the expected root dir already exists, skip
    if (target_dir / "ModelNet10").exists():
        return
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)

def _read_off(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust OFF parser.
    Returns:
        verts: (V, 3) float64
        faces: (F, 3) int64, triangulated if needed
    """
    with open(path, "r") as f:
        content = f.read().strip().split()
    if not content:
        raise ValueError(f"Empty OFF file: {path}")

    # Handle possible 'OFF' header or single-line variant
    if content[0] != "OFF":
        # Some OFF variants put counts on first line without 'OFF'
        # but ModelNet10 should have 'OFF'. Be strict but helpful.
        raise ValueError(f"Invalid OFF header in {path}")
    # content: ['OFF', nV, nF, nE, v0x, v0y, ...]
    try:
        nV = int(content[1]); nF = int(content[2])  # edges ignored
    except Exception as e:
        # Some files have a separate line for counts; fall back to slower parse
        # Re-read linewise
        with open(path, "r") as f:
            head = f.readline().strip()
            assert head == "OFF", f"Invalid OFF header in {path}"
            # Skip empty/comment lines
            def _next():
                line = f.readline()
                while line and (line.strip() == "" or line.strip().startswith("#")):
                    line = f.readline()
                return line
            counts = _next()
            nV, nF, _ = map(int, counts.split())
            verts = []
            for _ in range(nV):
                vx, vy, vz = map(float, _next().split())
                verts.append([vx, vy, vz])
            faces = []
            for _ in range(nF):
                parts = list(map(int, _next().split()))
                k, idxs = parts[0], parts[1:]
                if k == 3:
                    faces.append(idxs)
                else:
                    # simple fan triangulation
                    for i in range(1, k - 1):
                        faces.append([idxs[0], idxs[i], idxs[i+1]])
            return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)

    # Fast path using the tokenized content
    offset = 4  # OFF nV nF nE uses first 4 tokens
    verts = np.asarray(list(map(float, content[offset:offset + 3*nV])))
    verts = verts.reshape(nV, 3)
    p = offset + 3*nV
    faces = []
    for _ in range(nF):
        k = int(content[p]); p += 1
        idxs = list(map(int, content[p:p+k])); p += k
        if k == 3:
            faces.append(idxs)
        else:
            for i in range(1, k - 1):
                faces.append([idxs[0], idxs[i], idxs[i+1]])
    faces = np.asarray(faces, dtype=np.int64)
    return verts.astype(np.float64), faces

def _normalize_unit_sphere(verts: np.ndarray) -> np.ndarray:
    c = verts.mean(axis=0, keepdims=True)
    verts = verts - c
    r = np.linalg.norm(verts, axis=1).max()
    if r <= 0:
        return verts
    return verts / r

def _triangle_areas(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    a = v[f[:, 0]]
    b = v[f[:, 1]]
    c = v[f[:, 2]]
    # area = 0.5 * ||(b - a) x (c - a)||
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

def _sample_points_on_mesh(verts: np.ndarray, faces: np.ndarray, n: int) -> np.ndarray:
    """
    Sample n points on mesh surface proportional to triangle area.
    Returns (n, 3) float32.
    """
    areas = _triangle_areas(verts, faces)
    if areas.sum() <= 0:
        # degenerate; fallback to uniform vertex jitter
        idx = np.random.choice(len(verts), size=n, replace=len(verts) < n)
        return (verts[idx] + 1e-6 * np.random.randn(n, 3)).astype(np.float32)

    probs = areas / areas.sum()
    tri_idx = np.random.choice(len(faces), size=n, p=probs)
    tri = faces[tri_idx]
    a = verts[tri[:, 0]]
    b = verts[tri[:, 1]]
    c = verts[tri[:, 2]]
    # barycentric sampling with sqrt trick
    u = np.random.rand(n, 1)
    v = np.random.rand(n, 1)
    sqrtu = np.sqrt(u)
    w0 = 1 - sqrtu
    w1 = sqrtu * (1 - v)
    w2 = sqrtu * v
    pts = w0 * a + w1 * b + w2 * c
    return pts.astype(np.float32)

class ModelNet10Dataset(Dataset):
    """
    ModelNet10 loader with optional surface point-cloud sampling and on-disk caching.

    Args:
        root: directory to keep downloads and extractions. Will create `root/ModelNet10`.
        split: "train" or "test".
        as_pointcloud: if True, returns (points[N,3], label). If False, returns ((verts[V,3], faces[F,3]), label).
        n_points: number of points to sample when as_pointcloud is True.
        normalize: "unit_sphere" or "none".
        cache_pointclouds: if True, caches sampled point clouds under root/.cache_modelnet10/.
        seed: RNG seed for deterministic sampling when caching; None means non-deterministic each epoch.

    """
    def __init__(self,
                 root: str,
                 split: str = "train",
                 as_pointcloud: bool = True,
                 n_points: int = 2048,
                 normalize: str = "unit_sphere",
                 cache_pointclouds: bool = True,
                 seed: Optional[int] = 0,
                 download: bool = True):
        assert split in {"train", "test"}
        assert normalize in {"unit_sphere", "none"}
        self.root = Path(root)
        self.split = split
        self.as_pointcloud = as_pointcloud
        self.n_points = int(n_points)
        self.normalize = normalize
        self.cache_pointclouds = cache_pointclouds
        self.seed = seed

        zip_path = self.root / "ModelNet10.zip"
        extracted_root = self.root
        if download:
            _download(MODELNET10_URL, zip_path)
        if not (extracted_root / "ModelNet10").exists():
            if zip_path.exists():
                _extract_zip(zip_path, extracted_root)
            else:
                raise FileNotFoundError(
                    f"ModelNet10 not found at {extracted_root}. Set download=True or put ModelNet10 there."
                )

        self.data_root = extracted_root / "ModelNet10"
        # Build index
        samples = []
        missing_classes = []
        for cls in MODELNET10_CLASSES:
            folder = self.data_root / cls / split
            if not folder.is_dir():
                missing_classes.append(cls)
                continue
            for f in folder.iterdir():
                if f.suffix.lower() == ".off":
                    samples.append((f, CLASS_TO_IDX[cls]))
        if missing_classes:
            # Some zips have lowercase/uppercase mismatches; but official archive matches this list.
            pass
        if not samples:
            raise RuntimeError(f"No OFF files found under {self.data_root} for split={split}")
        self.samples = samples

        # Prepare cache dir
        self.cache_dir = self.root / ".cache_modelnet10" / split
        if self.cache_pointclouds:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Set per-dataset RNG
        self.rng = np.random.default_rng(seed) if seed is not None else None

    def __len__(self) -> int:
        return len(self.samples)

    def _load_mesh(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        verts, faces = _read_off(path)
        if self.normalize == "unit_sphere":
            verts = _normalize_unit_sphere(verts)
        return verts, faces

    def _pc_cache_path(self, mesh_path: Path) -> Path:
        stem = mesh_path.stem  # file name without suffix
        return self.cache_dir / f"{stem}_{self.n_points}_{self.normalize}.pt"

    def __getitem__(self, idx: int):
        mesh_path, label = self.samples[idx]

        if self.as_pointcloud:
            cache_hit = False
            if self.cache_pointclouds:
                cache_file = self._pc_cache_path(mesh_path)
                if cache_file.exists():
                    pts = torch.load(cache_file)
                    cache_hit = True
                else:
                    verts, faces = self._load_mesh(mesh_path)
                    # deterministic sampling when seed is set
                    if self.rng is not None:
                        state = np.random.get_state() if hasattr(np.random, "get_state") else None
                        np.random.seed(self.rng.integers(0, 2**31 - 1))
                        pts_np = _sample_points_on_mesh(verts, faces, self.n_points)
                        if state is not None:
                            np.random.set_state(state)
                    else:
                        pts_np = _sample_points_on_mesh(verts, faces, self.n_points)
                    pts = torch.from_numpy(pts_np)  # (N,3) float32
                    torch.save(pts, cache_file)
            else:
                verts, faces = self._load_mesh(mesh_path)
                pts_np = _sample_points_on_mesh(verts, faces, self.n_points)
                pts = torch.from_numpy(pts_np)

            y = torch.tensor(label, dtype=torch.long)
            return pts, y, {"path": str(mesh_path), "cache": cache_hit if self.as_pointcloud else False}

        else:
            verts, faces = self._load_mesh(mesh_path)
            x = (torch.from_numpy(verts.astype(np.float32)),
                 torch.from_numpy(faces.astype(np.int64)))
            y = torch.tensor(label, dtype=torch.long)
            return x, y, {"path": str(mesh_path)}
