"""
VadCLIP-compatible feature extractor for your own video clips.

IMPORTANT — READ BEFORE USING:
The VadCLIP authors have NOT officially published their exact frame-sampling/
extraction script (see open GitHub issue #28 on nwpu-zxr/VadCLIP, unresolved
as of writing). This script follows the most common community-reported
approach (sample every 16th frame, encode with CLIP ViT-B/16 -> 512-dim
features), consistent with the paper's stated backbone. It is NOT guaranteed
to bit-match the official released features. Validate against a known
UCF-Crime test video (whose official feature+label you have) before trusting
scores on your own footage.

Usage:
    python extract_features.py --video path/to/clip.mp4 --out features.npy
"""

import argparse
import cv2
import numpy as np
import torch
import clip
from PIL import Image


def extract_frames(video_path, sample_every=16):
    """Sample every Nth frame from the video."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video clip")
    parser.add_argument("--out", required=True, help="Path to save output .npy feature file")
    parser.add_argument("--sample_every", type=int, default=16,
                         help="Sample every Nth frame (community-reported default: 16)")
    parser.add_argument("--clip_model", default="ViT-B/16",
                         help="Must match VadCLIP's backbone — do not change unless you retrained")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Loading CLIP model: {args.clip_model}")
    model, preprocess = clip.load(args.clip_model, device=device)
    model.eval()

    print(f"[INFO] Extracting frames from: {args.video} (every {args.sample_every} frames)")
    frames = extract_frames(args.video, sample_every=args.sample_every)
    print(f"[INFO] Sampled {len(frames)} frames")

    print("[INFO] Encoding frames with CLIP visual encoder...")
    features = extract_clip_features(frames, model, preprocess, device)
    print(f"[INFO] Feature array shape: {features.shape}  (expected: [N, 512])")

    np.save(args.out, features)
    print(f"[INFO] Saved features to: {args.out}")
    print("[WARNING] Verify this against a known UCF-Crime test clip's official "
          "feature file before trusting results on your own footage — sampling "
          "protocol is community-inferred, not officially confirmed by the repo authors.")


if __name__ == "__main__":
    main()
