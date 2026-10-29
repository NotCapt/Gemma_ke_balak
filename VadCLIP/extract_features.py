"""
VadCLIP-compatible feature extractor for your own video clips.

IMPORTANT — READ BEFORE USING:
The VadCLIP authors have not published a full end-to-end extraction script
(see GitHub issue #28 on nwpu-zxr/VadCLIP). What they did ship is
src/crop.py, which shows the spatial preprocessing used for the released
UCF-Crime features: resize every frame to 340x256, take 224x224 crops
(crop 0 = center), then encode with CLIP ViT-B/16 -> 512-dim features.

This script samples every 16th frame (the community-reported temporal
protocol) and, by default, applies the crop.py center crop ("official").
Pass --crop clip to fall back to CLIP's own resize+center-crop transform,
which is what earlier versions of this script did.

Neither mode is guaranteed to bit-match the released features. Validate
against a known UCF-Crime test video (whose official feature file you have)
before trusting scores on your own footage, and re-tune the anomaly
threshold whenever you change the crop mode.

Usage:
    python extract_features.py --video path/to/clip.mp4 --out features.npy
"""

import argparse
import functools
import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image

# The vendored OpenAI CLIP lives in src/clip; make it importable when this file is run directly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import clip

CROP_MODES = ("official", "clip")


def official_center_crop(frame_rgb):
    """Replicates src/crop.py image_crop(type=0): resize to 340x256, center 224x224."""
    img = cv2.resize(frame_rgb, dsize=(340, 256))
    return img[16:240, 58:282, :]


def extract_frames(video_path, sample_every=16, crop="official"):
    """Sample every Nth frame from the video."""
    if crop not in CROP_MODES:
        raise ValueError(f"crop must be one of {CROP_MODES}, got {crop!r}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            # BGR (cv2) -> RGB (PIL/CLIP expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if crop == "official":
                frame_rgb = official_center_crop(frame_rgb)
            frames.append(Image.fromarray(frame_rgb))
        idx += 1
    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames extracted — check video path/codec.")
    return frames


def extract_clip_features(frames, model, preprocess, device, batch_size=32):
    """Encode sampled frames with CLIP visual encoder -> feature vectors."""
    all_features = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_tensors = torch.stack([preprocess(f) for f in batch]).to(device)
            features = model.encode_image(batch_tensors)
            features = features.float().cpu().numpy()
            all_features.append(features)
    return np.concatenate(all_features, axis=0)  # shape: [num_snippets, 512]


@functools.lru_cache(maxsize=2)
def load_clip(clip_model="ViT-B/16", device="cpu"):
    """Load CLIP once per (model, device) — reloading it per request costs seconds."""
    model, preprocess = clip.load(clip_model, device=device)
    model.eval()
    return model, preprocess


def extract_video_features(video_path, device=None, sample_every=16,
                           clip_model="ViT-B/16", crop="official"):
    """Frames -> CLIP features for one video. Returns an [N, 512] float32 array."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = load_clip(clip_model, device)
    frames = extract_frames(video_path, sample_every=sample_every, crop=crop)
    print(f"[INFO] Sampled {len(frames)} frames (every {sample_every}, crop={crop})")
    return extract_clip_features(frames, model, preprocess, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video clip")
    parser.add_argument("--out", required=True, help="Path to save output .npy feature file")
    parser.add_argument("--sample_every", type=int, default=16,
                         help="Sample every Nth frame (community-reported default: 16)")
    parser.add_argument("--clip_model", default="ViT-B/16",
                         help="Must match VadCLIP's backbone — do not change unless you retrained")
    parser.add_argument("--crop", choices=CROP_MODES, default="official",
                         help="official = src/crop.py center crop (340x256 -> 224); "
                              "clip = CLIP's default resize + center crop")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading CLIP model: {args.clip_model}")
    print(f"[INFO] Extracting frames from: {args.video}")

    features = extract_video_features(args.video, device=device, sample_every=args.sample_every,
                                      clip_model=args.clip_model, crop=args.crop)
    print(f"[INFO] Feature array shape: {features.shape}  (expected: [N, 512])")

    np.save(args.out, features)
    print(f"[INFO] Saved features to: {args.out}")
    print("[WARNING] Verify this against a known UCF-Crime test clip's official "
          "feature file before trusting results on your own footage — sampling "
          "protocol is community-inferred, not officially confirmed by the repo authors.")


if __name__ == "__main__":
    main()
