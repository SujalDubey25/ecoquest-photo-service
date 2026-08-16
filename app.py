# app.py — EcoQuest Photo Validation Service (standalone)
#
# Split out from the combined photo+reflection service so it can be hosted
# independently. Loads the v6 quantized ONNX model (photo/onnx/clip_vision.onnx
# + clip_text.onnx, 90.1MB + 140.9MB = 231MB combined) with a classifier
# retrained on the quantized embedding space (photo_classifier.joblib).
# Confirmed 345.3MB idle RAM, fits free-tier hosting.
#
# HISTORY: float32 was the original safe fallback (676.5MB, too big for free
# tier). fp16 was tried and rejected — no real CPU compute kernel, made RAM
# usage worse, not better. Static int8 quantization (MatMul/Conv only)
# initially collapsed accuracy (0.421 F1, degenerate "always no-match"
# output) — but this was fixed by retraining the classifier head on the
# quantized model's embedding space, not a precision problem. That fix is
# what's deployed here as v6: 220-231MB, 0.684 acc / 0.786 F1 domain-test,
# broader real-world category coverage than earlier versions. See
# PROJECT_STATE_AND_TODO.md §2 for the full experiment log.


import io
import json

import numpy as np
import joblib
import onnxruntime as ort
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from tokenizers import Tokenizer

app = FastAPI(title="EcoQuest Photo Validation Service")

print("Loading photo model (CLIP-base/patch32, float32 ONNX)...")

_vision_sess = ort.InferenceSession("photo/onnx/clip_vision.onnx")
_clip_text_sess = ort.InferenceSession("photo/onnx/clip_text.onnx")

with open("photo/onnx/preprocess_config.json") as f:
    _clip_img_config = json.load(f)

_clip_tokenizer = Tokenizer.from_file("photo/onnx/tokenizer/tokenizer.json")
_clip_tokenizer.enable_truncation(max_length=77)  # CLIP's context length

photo_clf = joblib.load("photo/models/photo_classifier.joblib")

print("Photo model loaded. Ready.")


# ═══════════════════════════════════════════════════════════════════════
# Embedding helpers — verified in verify_manual_pipeline.py (Phase 4a) to
# match the PyTorch pipeline within 0.999+ cosine similarity and identical
# downstream accuracy/F1. Do not change resize/crop/normalize details,
# tokenizer settings, or the pooling formula without re-verifying first.
# ═══════════════════════════════════════════════════════════════════════

def _preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    target_short = _clip_img_config["resize_size"]["shortest_edge"]
    w, h = image.size
    if w <= h:
        new_w = target_short
        new_h = max(1, round(h * target_short / w))
    else:
        new_h = target_short
        new_w = max(1, round(w * target_short / h))
    image = image.resize((new_w, new_h), Image.BICUBIC)

    crop_h = _clip_img_config["crop_size"]["height"]
    crop_w = _clip_img_config["crop_size"]["width"]
    left = (new_w - crop_w) // 2
    top = (new_h - crop_h) // 2
    image = image.crop((left, top, left + crop_w, top + crop_h))

    arr = np.array(image).astype(np.float32) / 255.0  # HWC, [0,1]
    mean = np.array(_clip_img_config["image_mean"], dtype=np.float32)
    std = np.array(_clip_img_config["image_std"], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)  # CHW
    return np.expand_dims(arr, axis=0).astype(np.float32)


def _clip_image_embedding(image: Image.Image) -> np.ndarray:
    pixel_values = _preprocess_image(image)
    out = _vision_sess.run(["image_embedding"], {"pixel_values": pixel_values})
    return out[0][0]


def _clip_text_embedding(text: str) -> np.ndarray:
    enc = _clip_tokenizer.encode(text)
    ids = np.array([enc.ids], dtype=np.int64)
    mask = np.array([enc.attention_mask], dtype=np.int64)
    out = _clip_text_sess.run(["text_embedding"], {"input_ids": ids, "attention_mask": mask})
    return out[0][0]


# ═══════════════════════════════════════════════════════════════════════
# /validate/photo
# ═══════════════════════════════════════════════════════════════════════

class PhotoRequest(BaseModel):
    taskTitle: str
    hint: str = ""
    photoUrl: str


# Decision threshold — v3 (CLIP-base/patch32 + hard negatives): accuracy
# 0.816, precision 0.857, recall 0.818, F1 0.837. NOT portable across
# model/training changes — re-run photo/evaluate.py's threshold sweep
# before trusting a carried-over value if the model ever changes.
PHOTO_MATCH_THRESHOLD = 0.45


def build_photo_features(image: Image.Image, text: str):
    img_emb = _clip_image_embedding(image)
    txt_emb = _clip_text_embedding(text)
    img_emb = img_emb / np.linalg.norm(img_emb)
    txt_emb = txt_emb / np.linalg.norm(txt_emb)
    cosine_sim = float(np.dot(img_emb, txt_emb))
    euclidean_dist = float(np.linalg.norm(img_emb - txt_emb))
    dot_product = float(np.dot(img_emb, txt_emb))
    word_count = len(text.split())
    return np.array([[cosine_sim, euclidean_dist, dot_product, word_count]])


@app.post("/validate/photo")
def validate_photo(req: PhotoRequest):
    try:
        resp = requests.get(
            req.photoUrl,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (EcoQuest-ML-Service; +https://huggingface.co/spaces)"},
        )
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        print(f"[validate_photo] Could not load photo from {req.photoUrl}: {e}")
        return {"score": 0, "feedback": "Could not load the photo — please try submitting again.", "valid": False}

    description = f"{req.taskTitle}. {req.hint}".strip()
    feats = build_photo_features(image, description)
    proba = photo_clf.predict_proba(feats)[0][1]  # probability of "match"
    valid = bool(proba >= PHOTO_MATCH_THRESHOLD)

    # SCORE RESCALING — do not revert without reading this. The Java
    # backend's EcoZoneService.calculatePoints() awards points based on
    # this score in three tiers: >=60 full points, 31-59 half points, <31
    # ZERO points — regardless of our `valid` flag. Rescaled so any `valid`
    # result always maps into 60-100, any invalid result stays in 0-59.
    if valid:
        span = max(1.0 - PHOTO_MATCH_THRESHOLD, 1e-6)
        score = 60 + int(round(40 * (proba - PHOTO_MATCH_THRESHOLD) / span))
        score = max(60, min(100, score))
    else:
        span = max(PHOTO_MATCH_THRESHOLD, 1e-6)
        score = int(round(59 * (proba / span)))
        score = max(0, min(59, score))

    feedback = (
        "Great job — your photo clearly matches this task!" if valid and score >= 70 else
        "Looks like a match, though the photo could be clearer." if valid else
        "This photo doesn't look like it matches the task — please check and try again."
    )
    return {"score": score, "feedback": feedback, "valid": valid}


@app.get("/health")
def health():
    return {"status": "ok", "service": "photo"}
