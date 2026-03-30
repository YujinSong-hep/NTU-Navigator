from __future__ import annotations

import heapq
import json
import random
import requests
import re
import difflib
import sys
import tempfile
from collections import deque, Counter
from dataclasses import dataclass
from datetime import time
from math import floor, ceil
from typing import Dict, List, Optional, Tuple, cast

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
import os

# Avoid OpenMP runtime collision (libomp loaded by multiple native deps on macOS).
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import faiss

State = Tuple[int, int, int]
ACTIONS = {0: (0, 1, 0), 1: (0, -1, 0), 2: (0, 0, -1), 3: (0, 0, 1), 4: (1, 0, 0), 5: (-1, 0, 0)}
DEFAULT_PENALTY_TIERS = {"offpeak": 0.1, "shoulder": 4.0, "peak": 100.0}

# 视觉定位系统编号 -> app3 房间号映射
VPR_TO_APP3_ROOM = {1: 6, 2: 1, 3: 4, 4: 5, 5: 8, 6: 3, 7: 2, 8: 7}
APP3_TO_VPR_ROOM = {v: k for k, v in VPR_TO_APP3_ROOM.items()}

# 固定 8 个库名称（VPR 1-8）
VPR_FIXED_NAMES = [
    "AI Lab II",
    "B1 elevator and north stairs",
    "B1 south stairs",
    "B2 elevator and north stairs",
    "B2 south stairs",
    "CBCR",
    "Generative AI Lab",
    "Robotics Lab",
]

VPR_ID_TO_NAME = {i + 1: name for i, name in enumerate(VPR_FIXED_NAMES)}
VPR_NAME_TO_ID = {name: i + 1 for i, name in enumerate(VPR_FIXED_NAMES)}

VPR_INDEX_FILE = "vpr_360_database.index"
VPR_LABELS_FILE = "vpr_360_labels.npy"
VPR_OCR_FILE = "vpr_360_ocr.npy"
VPR_THUMB_FILE = "vpr_360_thumb.npy"
MAX_FEATURES_PER_LOCATION = 800

@dataclass
class Room:
    room_id: int
    floor: int
    y: float
    x: float
    area: int

# =========================================================
# 1. 天气与环境结构
# =========================================================

def _to_slice_bounds(lo: float, hi: float, limit: int) -> Tuple[int, int]:
    """Convert float/int range to safe python slice bounds [start:end)."""
    start = max(0, min(limit, int(floor(float(lo)))))
    end = max(0, min(limit, int(ceil(float(hi)))))
    if end < start:
        start, end = end, start
    return start, end

def fill_rect(structure: np.ndarray, z: float, y0: float, y1: float, x0: float, x1: float, value: int) -> None:
    """Fill one floor rectangle with int/float bounds."""
    d, h, w = structure.shape
    zi = int(round(float(z)))
    zi = max(0, min(d - 1, zi))
    ys, ye = _to_slice_bounds(y0, y1, h)
    xs, xe = _to_slice_bounds(x0, x1, w)
    structure[zi, ys:ye, xs:xe] = value

def fill_rect_z(structure: np.ndarray, z0: float, z1: float, y0: float, y1: float, x0: float, x1: float, value: int) -> None:
    """Fill multi-floor rectangle with int/float bounds."""
    d, h, w = structure.shape
    zs, ze = _to_slice_bounds(z0, z1, d)
    ys, ye = _to_slice_bounds(y0, y1, h)
    xs, xe = _to_slice_bounds(x0, x1, w)
    structure[zs:ze, ys:ye, xs:xe] = value

def get_singapore_weather() -> bool:
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=1.34&longitude=103.68&current=precipitation"
        response = requests.get(url, timeout=3).json()
        precip = response['current']['precipitation']
        return precip > 0.0
    except:
        return False

def load_or_create_vpr_db() -> Tuple[faiss.Index, List[str], List[str], List[bytes]]:
    dim = 384
    if os.path.exists(VPR_INDEX_FILE) and os.path.exists(VPR_LABELS_FILE):
        index = faiss.read_index(VPR_INDEX_FILE)
        labels = list(np.load(VPR_LABELS_FILE, allow_pickle=True))
        if os.path.exists(VPR_OCR_FILE):
            ocr_texts = list(np.load(VPR_OCR_FILE, allow_pickle=True))
        else:
            ocr_texts = [""] * len(labels)
        if os.path.exists(VPR_THUMB_FILE):
            thumbs = list(np.load(VPR_THUMB_FILE, allow_pickle=True))
        else:
            thumbs = [b""] * len(labels)
        if index.ntotal != len(labels):
            # Keep labels aligned to actual vectors count when legacy files drift.
            labels = labels[:index.ntotal]
        if len(ocr_texts) < len(labels):
            ocr_texts.extend([""] * (len(labels) - len(ocr_texts)))
        ocr_texts = ocr_texts[:len(labels)]
        if len(thumbs) < len(labels):
            thumbs.extend([b""] * (len(labels) - len(thumbs)))
        thumbs = thumbs[:len(labels)]
        return index, labels, ocr_texts, thumbs
    return faiss.IndexFlatL2(dim), [], [], []

def save_vpr_db(index: faiss.Index, labels: List[str], ocr_texts: List[str], thumbs: List[bytes]) -> None:
    faiss.write_index(index, VPR_INDEX_FILE)
    np.save(VPR_LABELS_FILE, np.array(labels, dtype=object))
    np.save(VPR_OCR_FILE, np.array(ocr_texts, dtype=object))
    np.save(VPR_THUMB_FILE, np.array(thumbs, dtype=object))

def get_vpr_inventory() -> Tuple[List[str], Counter]:
    if not os.path.exists(VPR_LABELS_FILE):
        return [], Counter()
    labels = list(np.load(VPR_LABELS_FILE, allow_pickle=True))
    counts = Counter([str(x) for x in labels])
    return sorted(counts.keys()), counts

def default_room_name(room_id: int) -> str:
    return f"Room {room_id}"

def resolve_vpr_room_id_from_label(label: str, catalog_labels: Optional[List[str]] = None) -> Optional[int]:
    text = str(label).strip()
    # 只允许固定名称映射，不做数字直读兼容。
    return VPR_NAME_TO_ID.get(text)

def get_fixed_name_counts() -> Dict[str, int]:
    counts = {name: 0 for name in VPR_FIXED_NAMES}
    if not os.path.exists(VPR_LABELS_FILE):
        return counts
    labels = list(np.load(VPR_LABELS_FILE, allow_pickle=True))
    for raw in labels:
        name = str(raw).strip()
        if name in counts:
            counts[name] += 1
    return counts

def get_inventory_rows_app_order() -> List[Tuple[int, int, str, int]]:
    # 输出: [(显示序号1-8, app3房间号, 固定名称, 特征数)]
    counts = get_fixed_name_counts()
    rows = []
    for display_idx, app_room_id in enumerate(range(1, 9), start=1):
        vpr_id = APP3_TO_VPR_ROOM.get(app_room_id)
        if vpr_id is None:
            continue
        name = VPR_ID_TO_NAME[vpr_id]
        rows.append((display_idx, app_room_id, name, counts.get(name, 0)))
    return rows

def get_selector_label_from_app_room(app_room_id: int) -> str:
    for idx, rid, name, _count in get_inventory_rows_app_order():
        if rid == app_room_id:
            return f"{idx}. {name}"
    return default_room_name(app_room_id)

def build_app3_room_name_map() -> Dict[int, str]:
    display_map: Dict[int, str] = {}
    for app_room_id in range(1, 9):
        vpr_id = APP3_TO_VPR_ROOM.get(app_room_id)
        if vpr_id is None:
            continue
        display_map[app_room_id] = VPR_ID_TO_NAME[vpr_id]
    return display_map

@st.cache_resource
def load_vpr_encoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = cast(nn.Module, torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")).to(device)
    model.eval()
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return {
        "device": device,
        "model": model,
        "transform": transform,
    }

def extract_vpr_feature(frame_bgr: np.ndarray, encoder) -> np.ndarray:
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor = encoder["transform"](pil_img).unsqueeze(0).to(encoder["device"])
    with torch.no_grad():
        feat = encoder["model"](tensor).squeeze().cpu().numpy().astype(np.float32)
    feat = feat / (np.linalg.norm(feat) + 1e-10)
    return feat

@st.cache_resource
def load_ocr_reader():
    try:
        import easyocr
    except Exception:
        return None
    device = load_vpr_encoder()["device"]
    return easyocr.Reader(["en", "ch_sim"], gpu=(str(device) != "cpu"))

def extract_frame_text(frame_bgr: np.ndarray) -> str:
    reader = load_ocr_reader()
    if reader is None:
        return ""
    try:
        results = reader.readtext(frame_bgr)
        texts: List[str] = []
        for item in results:
            if isinstance(item, (list, tuple)) and len(item) > 1:
                text = str(item[1])
                if len(text) > 1:
                    texts.append(text)
        return " ".join(texts)
    except Exception:
        return ""

def frame_to_thumb_bytes(frame: np.ndarray, size: int = 120) -> bytes:
    thumb = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if not ok:
        return b""
    return buf.tobytes()

def decode_thumb_bytes(thumb_bytes) -> Optional[np.ndarray]:
    if thumb_bytes is None:
        return None
    if isinstance(thumb_bytes, bytes):
        arr = np.frombuffer(thumb_bytes, dtype=np.uint8)
    elif isinstance(thumb_bytes, np.ndarray):
        arr = thumb_bytes.astype(np.uint8)
    else:
        return None
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def orb_local_match_score(query_frame: np.ndarray, candidate_thumb: Optional[np.ndarray]) -> float:
    if candidate_thumb is None:
        return 0.0
    q = cv2.cvtColor(query_frame, cv2.COLOR_BGR2GRAY)
    c = cv2.cvtColor(candidate_thumb, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=300)  # type: ignore[attr-defined]
    kp1, des1 = orb.detectAndCompute(q, None)
    kp2, des2 = orb.detectAndCompute(c, None)
    if des1 is None or des2 is None:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if not matches:
        return 0.0
    good = [m for m in matches if m.distance < 45]
    denom = max(1, min(len(kp1), len(kp2)))
    return float(len(good)) / float(denom)

def is_smart_text_match(q_text: str, db_text: str) -> bool:
    if not q_text or not db_text:
        return False
    q = str(q_text).lower().replace(" ", "")
    db = str(db_text).lower().replace(" ", "")
    q = q.replace("basement", "b").replace("level", "l").replace("floor", "f")
    db = db.replace("basement", "b").replace("level", "l").replace("floor", "f")
    if len(q) < 2:
        return False

    def extract_alpha_num_map(text: str):
        mapping = {}
        for m in re.finditer(r"([a-z]+)(\d+)", text):
            prefix = m.group(1)
            number = m.group(2).lstrip("0") or "0"
            mapping.setdefault(prefix, set()).add(number)
        return mapping

    q_map = extract_alpha_num_map(q)
    db_map = extract_alpha_num_map(db)
    shared = set(q_map.keys()).intersection(db_map.keys())
    if shared:
        has_overlap_pair = False
        for prefix in shared:
            if q_map[prefix].isdisjoint(db_map[prefix]):
                return False
            has_overlap_pair = True
        if has_overlap_pair:
            return True

    q_nums = set(re.findall(r"\d+", q))
    db_nums = set(re.findall(r"\d+", db))
    q_alpha = re.sub(r"\d+", "", q)
    db_alpha = re.sub(r"\d+", "", db)

    if q_nums and db_nums:
        if not q_nums.intersection(db_nums):
            return False
        if len(q_nums) >= 2 and len(db_nums) >= 2:
            if not (q_nums.issubset(db_nums) or db_nums.issubset(q_nums)):
                return False
        if len(q_alpha) >= 2 and len(db_alpha) >= 2:
            if q_alpha in db_alpha or db_alpha in q_alpha:
                return True
            if difflib.SequenceMatcher(None, q_alpha, db_alpha).ratio() > 0.78:
                return True
        return False

    if len(q_alpha) >= 4 and len(db_alpha) >= 4:
        if q_alpha in db_alpha or db_alpha in q_alpha:
            return True
        if difflib.SequenceMatcher(None, q_alpha, db_alpha).ratio() > 0.8:
            return True
    return False

def expand_views_if_360(frame: np.ndarray, is_360: bool) -> List[np.ndarray]:
    if not is_360:
        return [frame]
    try:
        import py360convert
    except Exception:
        return [frame]
    views: List[np.ndarray] = []
    for yaw in [0, 45, 90, 135, 180, 225, 270, 315]:
        try:
            persp = py360convert.e2p(frame, fov_deg=(90, 90), u_deg=yaw, v_deg=0, out_hw=(800, 800), in_rot_deg=0)
            views.append(persp)
        except Exception:
            continue
    return views if views else [frame]

def backend_search_global_local_orb(feat: np.ndarray, locator, query_frame: np.ndarray, topk: int = 12):
    vector_cache = locator.get("vector_cache")
    if vector_cache is None:
        D, I = locator["index"].search(np.array([feat], dtype=np.float32), k=min(topk, locator["index"].ntotal))
        return D[0], I[0]

    q = feat.astype(np.float32)
    sims = vector_cache @ q
    k2 = min(max(topk * 2, 10), sims.shape[0])
    idx_part = np.argpartition(-sims, k2 - 1)[:k2]
    idx_sorted = idx_part[np.argsort(-sims[idx_part])]

    fused = []
    thumbs = locator.get("thumbs", [])
    for idx in idx_sorted:
        global_sim = float(sims[idx])
        local_sim = 0.0
        if idx < len(thumbs):
            local_sim = orb_local_match_score(query_frame, decode_thumb_bytes(thumbs[idx]))
        score = global_sim + 0.35 * local_sim
        fused.append((int(idx), score))

    fused = sorted(fused, key=lambda x: x[1], reverse=True)[:topk]
    out_i = np.array([x[0] for x in fused], dtype=np.int64)
    out_d = np.array([(1.0 - x[1]) for x in fused], dtype=np.float32)
    return out_d, out_i

def ingest_baseline_video(uploaded_video, location_name: str) -> str:
    location_name = str(location_name or "").strip()
    if not location_name:
        return "Please select a location name."
    if location_name not in VPR_NAME_TO_ID:
        return "Location name must be selected from the fixed 8 options."

    index, labels, ocr_texts, thumbs = load_or_create_vpr_db()
    existing = sum(1 for x in labels if str(x) == location_name)
    if existing >= MAX_FEATURES_PER_LOCATION:
        return f"Location [{location_name}] has reached the cap of {MAX_FEATURES_PER_LOCATION} features."

    encoder = load_vpr_encoder()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.getbuffer())
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return "Failed to read video. Please try another file."

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = max(1, int(fps * 0.5))
        added = 0
        skipped = 0
        idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_interval == 0:
                if existing >= MAX_FEATURES_PER_LOCATION:
                    skipped += 1
                else:
                    feat = extract_vpr_feature(frame, encoder)
                    text = extract_frame_text(frame)
                    thumb = frame_to_thumb_bytes(frame)
                    index.add(np.array([feat], dtype=np.float32))  # type: ignore[call-arg]
                    labels.append(location_name)
                    ocr_texts.append(text)
                    thumbs.append(thumb)
                    added += 1
                    existing += 1
            idx += 1

        cap.release()
        save_vpr_db(index, labels, ocr_texts, thumbs)
        load_vpr_locator.clear()

        if added == 0:
            return f"No new features were added (skipped {skipped} frames)."
        return f"Added {added} features for [{location_name}]. Current total: {existing}/{MAX_FEATURES_PER_LOCATION}."
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def delete_vpr_location(location_name: str) -> str:
    location_name = str(location_name or "").strip()
    if not location_name:
        return "Please select a location to delete."

    if not (os.path.exists(VPR_INDEX_FILE) and os.path.exists(VPR_LABELS_FILE)):
        return "Database does not exist."

    index, labels, ocr_texts, thumbs = load_or_create_vpr_db()
    keep_idx = [i for i, v in enumerate(labels) if str(v) != location_name]
    deleted = len(labels) - len(keep_idx)
    if deleted <= 0:
        return f"Location [{location_name}] was not found."

    if not keep_idx:
        clear_vpr_database()
        return f"Deleted all features for [{location_name}]. Database is now empty."

    new_index = faiss.IndexFlatL2(index.d)
    new_labels: List[str] = []
    new_ocr: List[str] = []
    new_thumbs: List[bytes] = []
    for i in keep_idx:
        vec = np.empty((index.d,), dtype=np.float32)
        index.reconstruct(i, vec)  # type: ignore[call-arg]
        new_index.add(vec.reshape(1, -1))  # type: ignore[call-arg]
        new_labels.append(str(labels[i]))
        new_ocr.append(str(ocr_texts[i]))
        new_thumbs.append(thumbs[i])

    save_vpr_db(new_index, new_labels, new_ocr, new_thumbs)
    load_vpr_locator.clear()
    return f"Deleted {deleted} features for [{location_name}]."

def clear_vpr_database() -> str:
    deleted = 0
    for path in [VPR_INDEX_FILE, VPR_LABELS_FILE, VPR_OCR_FILE, VPR_THUMB_FILE]:
        if os.path.exists(path):
            os.remove(path)
            deleted += 1
    load_vpr_locator.clear()
    if deleted == 0:
        return "Database is already empty."
    return f"Database cleared ({deleted} files removed)."

@st.cache_resource
def load_vpr_locator():
    if not (os.path.exists(VPR_INDEX_FILE) and os.path.exists(VPR_LABELS_FILE)):
        return None

    index, labels, ocr_texts, thumbs = load_or_create_vpr_db()
    if index.ntotal == 0 or not labels:
        return None

    encoder = load_vpr_encoder()
    vec_list = []
    for i in range(index.ntotal):
        v = np.empty((index.d,), dtype=np.float32)
        index.reconstruct(i, v)  # type: ignore[call-arg]
        vec_list.append(v)
    vector_cache = np.array(vec_list, dtype=np.float32)
    vector_cache = vector_cache / (np.linalg.norm(vector_cache, axis=1, keepdims=True) + 1e-8)

    return {
        "index": index,
        "labels": labels,
        "ocr_texts": ocr_texts,
        "thumbs": thumbs,
        "vector_cache": vector_cache,
        "device": encoder["device"],
        "model": encoder["model"],
        "transform": encoder["transform"],
    }

def parse_vpr_room_id(label) -> Optional[int]:
    return resolve_vpr_room_id_from_label(str(label))

def locate_start_room_from_video(uploaded_video) -> Tuple[Optional[int], str]:
    locator = load_vpr_locator()
    if locator is None:
        return None, "Localization database is missing or empty (vpr_360_database.index / vpr_360_labels.npy required)."

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.getbuffer())
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return None, "Failed to read video. Please try another file."

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = max(1, int(fps * 0.33))
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        aspect_ratio = width / height if height > 0 else 1
        is_360 = abs(aspect_ratio - 2.0) < 0.1

        global_text_hits = Counter()
        global_visual_scores: Dict[str, float] = {}
        seen_texts = set()
        location_history = deque(maxlen=15)
        all_blind_results: List[str] = []

        sampled = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if sampled % frame_interval == 0:
                frame_votes: Dict[str, float] = {}
                for sub_frame in expand_views_if_360(frame, is_360):
                    feat = extract_vpr_feature(sub_frame, locator)
                    q_text = extract_frame_text(sub_frame)

                    if len(q_text) > 1:
                        for i, db_text in enumerate(locator.get("ocr_texts", [])):
                            if not db_text:
                                continue
                            if is_smart_text_match(q_text, str(db_text)):
                                global_text_hits[str(locator["labels"][i])] += 1
                                seen_texts.add(q_text)

                    dists, idxs = backend_search_global_local_orb(
                        feat,
                        locator,
                        sub_frame,
                        topk=min(12, locator["index"].ntotal),
                    )
                    if len(idxs) > 0:
                        ws = np.exp(-3.0 * np.array(dists, dtype=np.float32))
                        ws = ws / (float(np.sum(ws)) + 1e-8)
                        for j, idx in enumerate(idxs):
                            if idx == -1 or idx >= len(locator["labels"]):
                                continue
                            label = str(locator["labels"][idx])
                            frame_votes[label] = frame_votes.get(label, 0.0) + float(ws[j])
                            global_visual_scores[label] = global_visual_scores.get(label, 0.0) + float(ws[j])

                if frame_votes:
                    best_now = max(frame_votes.items(), key=lambda x: x[1])[0]
                    location_history.append(best_now)
                    smoothed = Counter(list(location_history)[-5:]).most_common(1)[0][0]
                    all_blind_results.append(smoothed)
            sampled += 1

        cap.release()
        if not global_visual_scores:
            return None, "No valid localization evidence was extracted from the video."

        best_label = None
        if global_text_hits:
            top_hit = max(global_text_hits.values())
            min_hit = max(2, top_hit - 1)
            candidates = {k for k, v in global_text_hits.items() if v >= min_hit}
            if candidates:
                best_label = max(candidates, key=lambda x: global_visual_scores.get(x, 0.0))

        if best_label is None:
            if all_blind_results:
                best_label = Counter(all_blind_results).most_common(1)[0][0]
            else:
                best_label = max(global_visual_scores.items(), key=lambda x: x[1])[0]

        catalog_labels, _ = get_vpr_inventory()
        vpr_room = resolve_vpr_room_id_from_label(best_label, catalog_labels)
        if vpr_room is None:
            return None, f"Unable to resolve label ID: {best_label}"

        app3_room = VPR_TO_APP3_ROOM.get(vpr_room)
        if app3_room is None:
            return None, f"No mapping found for VPR ID {vpr_room}."
        return app3_room, get_selector_label_from_app_room(app3_room)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

@st.cache_data
def build_structure(is_raining_at_all: bool = False):
    """Build the 2-layer map structure used for routing and visualization."""
    D, H, W = 2, 16, 80
    structure = np.ones((D, H, W), dtype=int) # 全是墙壁阴影

    # ==========================================
    # Z=1 (图1，顶层 B1)
    # ==========================================
    structure[1, 12:15, 2:20] = 0   # 左侧基底大厅
    structure[1, 9:12, 6:16] = 0    
    structure[1, 6:9, 11:16] = 0  
    structure[1, 1:15, 20:24] = 0  
    # 【核心】顶部极窄直走廊
    structure[1, 1:3, 24:80] = 0

    # ==========================================
    # Z=0 (图2，底层 B2)
    # ==========================================
    structure[0, 12:15, 2:20] = 0   # 左侧基底大厅
    structure[0, 9:12, 6:16] = 0    
    structure[0, 6:9, 11:16] = 0  
    # 【核心】底部极窄直走廊
    structure[0, 13:15, 20:80] = 0

    # ==========================================
    # 联通设施
    # ==========================================
    structure[0:2, 11:12, 14:15] = 2 # 电梯 (黄块)
    structure[0:2, 11:12, 8:9] = 3   # 左侧楼梯 (三角)

    # 物理联通层：右侧隐形楼梯井 (仅用于AI寻路，画面不显示)
    if is_raining_at_all:
        structure[0:2, 1:15, 78:79] = 1
    else:
        structure[0:2, 1:15, 78:79] = 3

    return structure, (1, 12, 10), (0, 14, 66)

def valid_moves(state: State, structure: np.ndarray) -> List[int]:
    z, y, x = state
    d, h, w = structure.shape
    candidates = []
    for action, (dz, dy, dx) in ACTIONS.items():
        nz, ny, nx = z + dz, y + dy, x + dx
        if not (0 <= nz < d and 0 <= ny < h and 0 <= nx < w): continue
        next_val = int(structure[nz, ny, nx])
        if next_val == 1: continue 
        if action in (4, 5): 
            curr_val = int(structure[z, y, x])
            if curr_val not in (2, 3) or next_val not in (2, 3): continue
        candidates.append(action)
    return candidates

def extract_rooms(structure: np.ndarray) -> List[Room]:
    """Define the eight room anchors used by the planner."""
    rooms = []
    # B1 顶层
    rooms.append(Room(1, 1, 12, 10, 10))  # ① 左侧空地
    rooms.append(Room(2, 1, 1, 38, 10))   # ② 顶走廊
    rooms.append(Room(3, 1, 1, 56, 10))   # ③ 顶走廊
    rooms.append(Room(4, 1, 1, 72, 10))   # ④ 顶走廊靠右

    # B2 底层
    rooms.append(Room(5, 0, 13, 10, 10))  # ⑤ 左侧空地
    rooms.append(Room(6, 0, 14, 24, 10))  # ⑥ 底走廊
    rooms.append(Room(7, 0, 14, 56, 10))  # ⑦ 底走廊
    rooms.append(Room(8, 0, 14, 72, 10))  # ⑧ 底走廊靠右
    return rooms

def snap_room_to_state(room: Room, structure: np.ndarray) -> State:
    """Map possibly-decimal room coordinates to nearest walkable grid cell."""
    z = int(room.floor)
    h, w = structure.shape[1], structure.shape[2]

    y0 = int(round(room.y))
    x0 = int(round(room.x))
    y0 = max(0, min(h - 1, y0))
    x0 = max(0, min(w - 1, x0))

    if int(structure[z, y0, x0]) in (0, 2, 3):
        return (z, y0, x0)

    best = None
    best_d = 10**9
    for y in range(h):
        for x in range(w):
            if int(structure[z, y, x]) not in (0, 2, 3):
                continue
            d = abs(y - room.y) + abs(x - room.x)
            if d < best_d:
                best_d = d
                best = (z, y, x)

    if best is None:
        raise ValueError(f"Room {room.room_id} has no reachable cell on floor {z}.")
    return best

def snap_state_to_grid(state: Tuple[float, float, float], structure: np.ndarray) -> State:
    """Map decimal state (z, y, x) to nearest walkable integer grid state."""
    zf, yf, xf = state
    z = int(round(float(zf)))
    z = max(0, min(structure.shape[0] - 1, z))

    y0 = int(round(float(yf)))
    x0 = int(round(float(xf)))
    h, w = structure.shape[1], structure.shape[2]
    y0 = max(0, min(h - 1, y0))
    x0 = max(0, min(w - 1, x0))

    if int(structure[z, y0, x0]) in (0, 2, 3):
        return (z, y0, x0)

    best = None
    best_d = 10**9
    for y in range(h):
        for x in range(w):
            if int(structure[z, y, x]) not in (0, 2, 3):
                continue
            d = abs(y - float(yf)) + abs(x - float(xf))
            if d < best_d:
                best_d = d
                best = (z, y, x)

    if best is None:
        raise ValueError(f"State {state} has no reachable cell on floor {z}.")
    return best

def elevator_penalty_for_time(dep: time, penalty_tiers: Dict[str, float]) -> Tuple[float, str]:
    m = dep.hour * 60 + dep.minute
    for lo, hi, lbl in [(8*60, 9*60, "Morning peak"), (11*60, 13*60, "Lunch peak"), (17*60, 19*60, "Evening peak")]:
        if lo <= m < hi: return float(penalty_tiers["peak"]), lbl
    return float(penalty_tiers["offpeak"]), "Off-peak"

def vertical_transition_penalty(cur: State, nxt: State, structure: np.ndarray, elev_penalty: float) -> float:
    """Penalty for vertical movement with direction-aware stair costs.

    Peak time exception: keep original behavior (stairs cheap, elevator very expensive).
    Non-peak:
    - Going up by stairs: noticeably higher penalty.
    - Going down by stairs: slightly higher than elevator.
    """
    curr_val = int(structure[cur])
    next_val = int(structure[nxt])
    use_elevator = (curr_val == 2 or next_val == 2)

    is_peak = elev_penalty >= float(DEFAULT_PENALTY_TIERS["peak"])
    if is_peak:
        return elev_penalty if use_elevator else 0.2

    if use_elevator:
        return elev_penalty

    dz = nxt[0] - cur[0]
    if dz > 0:
        # Upstairs by stair: more costly than downstairs.
        return elev_penalty + 0.6
    if dz < 0:
        # Downstairs by stair: slightly higher than elevator.
        return elev_penalty + 0.1
    return 0.2

# =========================================================
# 2. 8D 神经网络大脑 
# =========================================================
def get_tensor(state: State, goal: State, elev_penalty: float, is_raining: bool, d, h, w) -> torch.Tensor:
    return torch.tensor([
        state[0]/(d-1), state[1]/(h-1), state[2]/(w-1),
        goal[0]/(d-1), goal[1]/(h-1), goal[2]/(w-1),
        min(elev_penalty / 100.0, 1.0),
        1.0 if is_raining else 0.0
    ], dtype=torch.float32)

class UVFANetwork(nn.Module):
    def __init__(self, input_dim=8, output_dim=6):
        super(UVFANetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x): return self.net(x)

class ReplayBuffer:
    def __init__(self, cap=80000): self.buffer = deque(maxlen=cap)
    def push(self, sv, a, r, nsv, done): self.buffer.append((sv, a, r, nsv, done))
    def sample(self, b_size):
        batch = random.sample(self.buffer, b_size)
        s, a, r, ns, d = zip(*batch)
        return torch.stack(s), torch.tensor(a), torch.tensor(r, dtype=torch.float32), torch.stack(ns), torch.tensor(d, dtype=torch.float32)
    def __len__(self): return len(self.buffer)

def train_universal_uvfa(structure, walkable_states, episodes=10000, progress_bar=None):
    d, h, w = structure.shape
    device = torch.device("cpu")
    policy_net = UVFANetwork().to(device)
    target_net = UVFANetwork().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    buffer = ReplayBuffer()
    epsilon = 1.0
    epsilon_decay = 0.9995
    
    for ep in range(episodes):
        state = random.choice(walkable_states)
        goal = random.choice(walkable_states)
        elev_penalty = random.choice([0.1, 4.0, 100.0])
        is_raining = random.choice([True, False]) 
        
        ep_trans = []
        for step in range(120):
            if state == goal: break
            sv = get_tensor(state, goal, elev_penalty, is_raining, d, h, w).to(device)
            valid = valid_moves(state, structure)
            if not valid: break
            
            if random.random() < epsilon: action = random.choice(valid)
            else:
                with torch.no_grad():
                    q_vals = policy_net(sv)
                    action = max({a: q_vals[a].item() for a in valid}, key={a: q_vals[a].item() for a in valid}.get)
            
            dz, dy, dx = ACTIONS[action]
            ns = (state[0]+dz, state[1]+dy, state[2]+dx)
            
            reward = -0.1
            done = False
            if ns == goal:
                reward = 100.0
                done = True
            else:
                if action in [4, 5]:
                    reward -= vertical_transition_penalty(state, ns, structure, elev_penalty)
                    
            nsv = get_tensor(ns, goal, elev_penalty, is_raining, d, h, w).to(device)
            ep_trans.append((sv, action, reward, nsv, done, state, ns))
            state = ns
            
        for sv, a, r, nsv, d_flag, _, _ in ep_trans: buffer.push(sv, a, r, nsv, d_flag)

        if len(buffer) > 128:
            b_s, b_a, b_r, b_ns, b_d = buffer.sample(128)
            q_values = policy_net(b_s).gather(1, b_a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = target_net(b_ns).max(1)[0]
                expected_q = b_r + 0.99 * next_q * (1 - b_d)
            loss = F.mse_loss(q_values, expected_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epsilon > 0.05: epsilon *= epsilon_decay
        if ep % 50 == 0 and progress_bar: progress_bar.progress(ep / episodes)
        if ep % 200 == 0: target_net.load_state_dict(policy_net.state_dict())
                
    return policy_net.cpu()

def policy_path_from_uvfa(model, structure, start, end, elev_p, is_rain, max_steps=500):
    start = snap_state_to_grid(start, structure)
    end = snap_state_to_grid(end, structure)
    if start == end: return [start]
    d, h, w = structure.shape
    path, current, visited = [start], start, {start}
    model.eval()
    with torch.no_grad():
        for _ in range(max_steps):
            if current == end: return path
            valid = valid_moves(current, structure)
            if not valid: return None
            sv = get_tensor(current, end, elev_p, is_rain, d, h, w)
            q_vals = model(sv)
            scored = sorted([(q_vals[a].item(), a) for a in valid], key=lambda t: t[0], reverse=True)
            next_state = None
            for _score, action in scored:
                dz, dy, dx = ACTIONS[action]
                cand = (current[0]+dz, current[1]+dy, current[2]+dx)
                if cand not in visited:
                    next_state = cand; break
            if next_state is None:
                dz, dy, dx = ACTIONS[scored[0][1]]
                next_state = (current[0]+dz, current[1]+dy, current[2]+dx)
            path.append(next_state)
            current = next_state
            visited.add(current)
    return None

# =========================================================
# 3. 经典寻路与绘图 (修复并整合)
# =========================================================
def dijkstra_route(structure, start, end, elev_penalty, is_raining):
    start = snap_state_to_grid(start, structure)
    end = snap_state_to_grid(end, structure)
    d, h, w = structure.shape
    dist = np.full((d, h, w), 10**18, dtype=float)
    prev = {}
    pq = [(0.0, start)]
    dist[start] = 0.0
    
    RIGHT_STAIR_X_RANGE = [78]

    while pq:
        cost, cur = heapq.heappop(pq)
        if cur == end: break
        for action in valid_moves(cur, structure):
            dz, dy, dx = ACTIONS[action]
            nxt = (cur[0]+dz, cur[1]+dy, cur[2]+dx)
            terrain = int(structure[nxt])
            
            # 雨天封锁右侧楼梯
            is_right_stair = terrain == 3 and nxt[2] in RIGHT_STAIR_X_RANGE
            if is_right_stair and is_raining: continue

            step_cost = 1.0
            if action in (4, 5):
                step_cost += vertical_transition_penalty(cur, nxt, structure, elev_penalty)
            
            if cost + step_cost < dist[nxt]:
                dist[nxt] = cost + step_cost
                prev[nxt] = cur
                heapq.heappush(pq, (dist[nxt], nxt))
                
    if np.isinf(dist[end]): return None
    path, cur = [end], end
    while cur != start:
        cur = prev[cur]
        path.append(cur)
    return path[::-1]

def route_cost(structure, path, elev_penalty, is_raining):
    if not path or len(path)<2: return 0.0
    total = 0.0

    def cell_val(state):
        z = int(round(state[0]))
        y = int(round(state[1]))
        x = int(round(state[2]))
        z = max(0, min(structure.shape[0]-1, z))
        y = max(0, min(structure.shape[1]-1, y))
        x = max(0, min(structure.shape[2]-1, x))
        return int(structure[z, y, x])

    for cur, nxt in zip(path[:-1], path[1:]):
        c = 1.0
        if cur[0] != nxt[0]:
            cur_i = (int(round(cur[0])), int(round(cur[1])), int(round(cur[2])))
            nxt_i = (int(round(nxt[0])), int(round(nxt[1])), int(round(nxt[2])))
            cur_i = (
                max(0, min(structure.shape[0]-1, cur_i[0])),
                max(0, min(structure.shape[1]-1, cur_i[1])),
                max(0, min(structure.shape[2]-1, cur_i[2])),
            )
            nxt_i = (
                max(0, min(structure.shape[0]-1, nxt_i[0])),
                max(0, min(structure.shape[1]-1, nxt_i[1])),
                max(0, min(structure.shape[2]-1, nxt_i[2])),
            )
            c += vertical_transition_penalty(cur_i, nxt_i, structure, elev_penalty)
        total += c
    return float(total)

def render_labeled_floors(structure, rooms, path=None, start=None, end=None, is_raining=False):
    d, h, w = structure.shape
    titles = ["Floor 1 (B1 - Top Layer)", "Floor 0 (B2 - Bottom Layer)"]
    fig = make_subplots(rows=d, cols=1, subplot_titles=titles, vertical_spacing=0.07)
    
    r_dict = {z: [] for z in range(d)}
    for r in rooms: r_dict[r.floor].append(r)
    
    p_dict = {z: [] for z in range(d)}
    if path:
        for z, y, x in path: 
            # 路线隐身：红线绝不穿透大片阴影区
            if x >= 74:
                if z == 1 and y > 2: continue 
                if z == 0 and y < 13: continue 
            p_dict[z].append((x+0.5, y+0.5))
            
    for z in range(d):
        ridx = d - z 
        
        # 提取物理地图用于显示
        cls = np.ones((h, w), dtype=int) 
        cls[structure[z] != 1] = 0 
        
        # 【视觉强行擦除】：把右侧用于寻路的物理走廊强行盖上阴影
        if z == 1:
            cls[3:16, 74:80] = 1 # B1: 走廊下方全黑
        else:
            cls[0:13, 74:80] = 1 # B2: 走廊上方全黑
            
        # 画出地形
        fig.add_trace(go.Heatmap(z=cls, x=np.arange(w)+0.5, y=np.arange(h)+0.5, 
                                 colorscale=[[0, "#f9f4e8"], [1, "#6e5e4b"]], 
                                 showscale=False, hoverinfo="skip"), row=ridx, col=1)
        
        # ------------------------------------------
        # 放置图标
        # ------------------------------------------
        fig.add_trace(go.Scatter(x=[14], y=[11], mode="markers", 
                                 marker=dict(size=18, color="#f6c244", symbol="square", line=dict(color="black", width=2)), 
                                 name="Elevator", showlegend=(ridx==1)), row=ridx, col=1)
        
        fig.add_trace(go.Scatter(x=[8], y=[11], mode="markers", 
                                 marker=dict(size=16, color="#e98d1c", symbol="triangle-up", line=dict(color="black", width=1)), 
                                 name="Left Stair", showlegend=(ridx==1)), row=ridx, col=1)

        # 最右侧楼梯图标
        rstair_x = [78.5]
        rstair_y = [1.5] if z == 1 else [14.5] 
        r_color = "#999999" if is_raining else "#e98d1c" 
        r_symbol = "x" if is_raining else "triangle-up"
        
        fig.add_trace(go.Scatter(x=rstair_x, y=rstair_y, mode="markers", 
                                 marker=dict(size=18, color=r_color, symbol=r_symbol, line=dict(color="black", width=1)), 
                                 name="Right Stair", showlegend=(ridx==1)), row=ridx, col=1)

        # ------------------------------------------
        # 放置数字和路线
        # ------------------------------------------
        if r_dict[z]:
            fig.add_trace(go.Scatter(x=[r.x+0.5 for r in r_dict[z]], y=[r.y+0.5 for r in r_dict[z]], 
                                     mode="markers+text", text=[str(r.room_id) for r in r_dict[z]], 
                                     textposition="middle center", 
                                     marker=dict(size=22, color="white", line=dict(color="black", width=2)), 
                                     textfont=dict(size=14, color="black"), showlegend=False), row=ridx, col=1)
            
        if p_dict[z]:
            arr = np.array(p_dict[z])
            fig.add_trace(go.Scatter(x=arr[:,0], y=arr[:,1], mode="lines+markers", 
                                     line=dict(color="#b32020", width=4), marker=dict(size=6, color="#b32020"), 
                                     name="Route", showlegend=(ridx==1)), row=ridx, col=1)
            
        if start and start[0] == z: 
            fig.add_trace(go.Scatter(x=[start[2]+0.5], y=[start[1]+0.5], mode="markers", 
                                     marker=dict(size=14, color="#00a66a", symbol="star"), name="Start", showlegend=(ridx==1)), row=ridx, col=1)
        if end and end[0] == z: 
            fig.add_trace(go.Scatter(x=[end[2]+0.5], y=[end[1]+0.5], mode="markers", 
                                     marker=dict(size=14, color="#c33a3a", symbol="x-dot"), name="Goal", showlegend=(ridx==1)), row=ridx, col=1)
        
        fig.update_xaxes(range=[0, w], showgrid=False, zeroline=False, row=ridx, col=1)
        fig.update_yaxes(range=[h, 0], showgrid=False, zeroline=False, row=ridx, col=1)
        
    fig.update_layout(
        height=800,
        paper_bgcolor="#efe9dc",
        plot_bgcolor="#f9f4e8",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(font=dict(color="black", size=13)),
        font=dict(color="black"),
    )
    return fig


# =========================================================
# 4. Streamlit UI (恢复原始侧边栏布局)
# =========================================================
def main():
    st.set_page_config(page_title="Weather-Aware NTU Navigator", layout="wide")
    st.title("⛈️ Weather-Aware AI Route Navigator")
    st.caption("Custom 2-Layer Map matching the hand-drawn sketches.")

    if "uvfa_model" not in st.session_state: 
        st.session_state.uvfa_model = None
    if "located_start_id" not in st.session_state:
        st.session_state.located_start_id = None
    if "locate_msg" not in st.session_state:
        st.session_state.locate_msg = ""

    # --- 恢复侧边栏 ---
    st.sidebar.header("🧠 AI Brain Management")
    model_path = "ntu_8d_uvfa_brain.pth" 
    
    if os.path.exists(model_path):
        st.sidebar.success("✅ Found pre-trained Brain on disk!")
        if st.sidebar.button("📂 Load Brain from Disk", width="stretch"):
            model = UVFANetwork()
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
            st.session_state.uvfa_model = model
            st.sidebar.toast("Brain loaded instantly!")
    else:
        st.sidebar.info("No pre-trained Brain found. Please train one.")

    st.sidebar.markdown("---")
    episodes = st.sidebar.slider("Episodes", 5000, 50000, 20000, 5000)

    st.sidebar.markdown("---")
    st.sidebar.header("🗂️ VPR Database")

    rows = get_inventory_rows_app_order()
    if rows:
        inv_lines = [f"{idx}. {name} ({count} items)" for idx, _app_room_id, name, count in rows]
        st.sidebar.caption("Current Library")
        st.sidebar.text("\n".join(inv_lines))
    else:
        st.sidebar.caption("Library is empty")

    add_video = st.sidebar.file_uploader("Upload baseline video", type=["mp4", "mov", "avi", "mkv"], key="db_add_video")
    add_name = st.sidebar.selectbox("Fixed location name (1-8)", options=VPR_FIXED_NAMES, index=0, key="db_add_name")
    if st.sidebar.button("➕ Add to Library", width="stretch", key="db_add_btn"):
        if add_video is None:
            st.sidebar.warning("Please upload a baseline video first.")
        else:
            with st.spinner("Adding features to library..."):
                st.sidebar.success(ingest_baseline_video(add_video, add_name))
            st.rerun()

    del_choice = st.sidebar.selectbox("Delete by location", options=[""] + VPR_FIXED_NAMES, index=0, key="db_del_choice")
    if st.sidebar.button("✂️ Delete selected location", width="stretch", key="db_del_btn"):
        st.sidebar.info(delete_vpr_location(del_choice))
        st.rerun()

    if st.sidebar.button("🗑️ Clear library", width="stretch", key="db_clear_btn"):
        st.sidebar.info(clear_vpr_database())
        st.rerun()
    
    # 提取结构，根据天气情况
    st.sidebar.header("🌦️ Weather System")
    weather_mode = st.sidebar.radio("Weather Mode", ["Live SG API", "Force Rain 🌧️", "Force Clear ☀️"])
    is_raining = False
    if weather_mode == "Live SG API":
        with st.spinner("Checking Singapore sky..."):
            is_raining = get_singapore_weather()
            st.sidebar.info("Current SG Weather: **Raining 🌧️**" if is_raining else "Current SG Weather: **Clear ☀️**")
    elif weather_mode == "Force Rain 🌧️": is_raining = True
    else: is_raining = False

    struct, _, _ = build_structure(is_raining)
    rooms = extract_rooms(struct)
    # UI 与业务层保留小数坐标；寻路函数内部会自动映射到最近可走网格。
    r_lookup = {r.room_id: (float(r.floor), float(r.y), float(r.x)) for r in rooms}
    room_name_map = build_app3_room_name_map()
    selector_rows = get_inventory_rows_app_order()
    selector_room_ids = [app_room_id for _idx, app_room_id, _name, _count in selector_rows]
    room_selector_label = {app_room_id: f"{idx}. {name}" for idx, app_room_id, name, _count in selector_rows}
    w_states = [(z, y, x) for z in range(struct.shape[0]) for y in range(struct.shape[1]) for x in range(struct.shape[2]) if struct[z,y,x] in (0, 2, 3)]

    if st.sidebar.button("🚀 Train New 8D Brain & Save", type="primary", width="stretch"):
        with st.spinner("Training Neural Network on the Custom Map..."):
            bar = st.progress(0.0)
            model = train_universal_uvfa(struct, w_states, episodes, bar)
            st.session_state.uvfa_model = model
            torch.save(model.state_dict(), model_path)
            bar.empty()
            st.sidebar.success(f"Brain Trained and saved!")

    # --- 恢复主页面左右分栏 ---
    c1, c2 = st.columns([1.2, 2.5])
    with c1:
        st.subheader("📍 Trip")
        opts = selector_room_ids
        st.caption("Start/Destination are shown using fixed mapped names.")
        start_source = st.radio("Start input mode", ["Manual", "Video localization"], horizontal=True)
        start_id = None

        if start_source == "Manual":
            start_id = st.selectbox(
                "Start",
                opts,
                index=0,
                format_func=lambda rid: room_selector_label.get(rid, room_name_map.get(rid, default_room_name(rid))),
            )
            st.session_state.located_start_id = None
            st.session_state.locate_msg = ""
        else:
            uploaded_start_video = st.file_uploader("Upload video for start localization", type=["mp4", "mov", "avi", "mkv"])
            if st.button("📹 Localize Start", width="stretch"):
                if uploaded_start_video is None:
                    st.session_state.locate_msg = "Please upload a video before localization."
                    st.session_state.located_start_id = None
                else:
                    with st.spinner("Running visual localization..."):
                        located_id, msg = locate_start_room_from_video(uploaded_start_video)
                    st.session_state.locate_msg = msg
                    st.session_state.located_start_id = located_id

            if st.session_state.locate_msg:
                if st.session_state.located_start_id is None:
                    st.warning(st.session_state.locate_msg)
                else:
                    st.success(st.session_state.locate_msg)

            if st.session_state.located_start_id is not None:
                name = room_selector_label.get(
                    st.session_state.located_start_id,
                    room_name_map.get(st.session_state.located_start_id, default_room_name(st.session_state.located_start_id)),
                )
                st.caption(f"Current start = {name}")

        end_id = st.selectbox(
            "Destination",
            opts,
            index=len(opts)-1,
            format_func=lambda rid: room_selector_label.get(rid, room_name_map.get(rid, default_room_name(rid))),
        )
        d_time = st.time_input("Departure Time", value=time(12, 30))
        e_pen, lbl = elevator_penalty_for_time(d_time, DEFAULT_PENALTY_TIERS)
        st.info(f"🚦 Traffic: {lbl} (Lift Cost: {e_pen})\n\n🌦️ Right Stair: {'BLOCKED (Rain)' if is_raining else 'OPEN (Clear)'}")
        run_btn = st.button("🗺️ Route Me!", width="stretch")

    with c2:
        if run_btn:
            if start_source == "Video localization":
                if st.session_state.located_start_id is None:
                    st.error("Please complete start localization before generating route.")
                    st.plotly_chart(render_labeled_floors(struct, rooms, is_raining=is_raining), width="stretch")
                    return
                start_id = st.session_state.located_start_id

            if start_id is None:
                st.error("Invalid start point. Please select or localize again.")
                st.plotly_chart(render_labeled_floors(struct, rooms, is_raining=is_raining), width="stretch")
                return

            s_state, e_state = r_lookup[start_id], r_lookup[end_id]
            if st.session_state.uvfa_model is None:
                opt_path = dijkstra_route(struct, s_state, e_state, e_pen, is_raining)
                st.plotly_chart(render_labeled_floors(struct, rooms, path=opt_path, start=s_state, end=e_state, is_raining=is_raining), width="stretch")
            else:
                rl_path = policy_path_from_uvfa(st.session_state.uvfa_model, struct, s_state, e_state, e_pen, is_raining)
                opt_path = dijkstra_route(struct, s_state, e_state, e_pen, is_raining)

                fallback_reason = ""
                c_path = rl_path if rl_path else opt_path
                src = "Deep-RL" if rl_path else "A* Fallback"
                if rl_path is None:
                    fallback_reason = "RL did not return a valid path, so A* was used."
                if rl_path and opt_path:
                    rl_c = route_cost(struct, rl_path, e_pen, is_raining)
                    op_c = route_cost(struct, opt_path, e_pen, is_raining)
                    if rl_c > op_c + 0.5:
                        c_path, src = opt_path, "A* Fallback"
                        fallback_reason = f"RL path cost ({rl_c:.1f}) is higher than A* ({op_c:.1f}), so A* was selected."
                if c_path is None:
                    st.error("No feasible route found.")
                    st.plotly_chart(render_labeled_floors(struct, rooms, start=s_state, end=e_state, is_raining=is_raining), width="stretch")
                    return
                        
                cost = route_cost(struct, c_path, e_pen, is_raining)
                st.success(f"✅ Route generated | Cost: {cost:.1f} | Steps: {len(c_path)}")
                st.plotly_chart(render_labeled_floors(struct, rooms, path=c_path, start=s_state, end=e_state, is_raining=is_raining), width="stretch")
        else:
            st.plotly_chart(render_labeled_floors(struct, rooms, is_raining=is_raining), width="stretch")

if __name__ == "__main__": main()