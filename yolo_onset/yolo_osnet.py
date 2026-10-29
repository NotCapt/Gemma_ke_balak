"""
Standalone test for Stage 3 (YOLOv8n person detection) + Stage 4 (OSNet Re-ID).

Run this on ONE image or ONE video frame to sanity-check the stage before
wiring it into the full pipeline.

Setup (run once):
    pip install ultralytics torch torchvision opencv-python
    pip install torchreid          # gives you OSNet
    # if torchreid install fails on your OS, see the fallback note at the
    # bottom of this file.

Re-ID weights:
    torchreid's `pretrained=True` only loads ImageNet classification weights,
    which are NOT trained for person re-identification. Download a Re-ID
    checkpoint from the torchreid model zoo (e.g. osnet_x1_0 trained on
    MSMT17 or Market-1501) and pass it with --reid_weights or the
    OSNET_WEIGHTS environment variable. Without it, cross-camera similarity
    scores are not meaningful.

Usage:
    python yolo_osnet.py --image path/to/test.jpg
    python yolo_osnet.py --video path/to/test.mp4 --frame 30 --reid_weights osnet_x1_0_msmt17.pth
"""

import argparse
import os
import cv2
import numpy as np
import torch


def load_yolo(weights="yolov8n.pt"):
    """Stage 3 model. Imported lazily so this module loads without ultralytics."""
    from ultralytics import YOLO
    return YOLO(weights)  # auto-downloads weights on first run


def get_frame(args):
    """Load a single frame either from an image file or a specific video frame."""
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")
        return frame

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {args.frame} from {args.video}")
    return frame


def detect_people(frame, yolo_model, conf_thresh=0.4, imgsz=640):
    """
    Stage 3: YOLOv8n person detection.
    Input:  a single BGR frame (numpy array, as read by cv2).
    Output: list of boxes [x1, y1, x2, y2, confidence].

    imgsz=1280 finds the small, distant people typical of CCTV far better than
    the default 640 (at ~4x the compute).
    """
    results = yolo_model(frame, imgsz=imgsz, verbose=False)[0]
    boxes = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == 0 and conf >= conf_thresh:  # class 0 = person in COCO
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append([int(x1), int(y1), int(x2), int(y2), conf])
    return boxes


def crop_boxes(frame, boxes):
    """Crop each detected person out of the frame."""
    h, w = frame.shape[:2]
    crops = []
    for x1, y1, x2, y2, conf in boxes:
        # Clamp: a negative index would silently wrap around to the far edge.
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(crop)
    return crops


def load_osnet(weights_path=None, device="cpu"):
    """
    Stage 4: load OSNet from torchreid's model zoo.
    Falls back to printing install instructions if torchreid isn't available.

    Returns (model, weights_label). weights_label is "imagenet" when no Re-ID
    checkpoint was supplied, so callers can flag the embeddings as untrusted.
    """
    try:
        import torchreid
    except ImportError:
        raise ImportError(
            "torchreid not installed. Run: pip install torchreid\n"
            "If that fails, see the fallback note at the bottom of this script."
        )

    weights_path = weights_path or os.environ.get("OSNET_WEIGHTS")
    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=1000,  # unused at inference time, just needed to build the model
        pretrained=True,   # ImageNet init only — Re-ID weights are loaded below
    )
    if weights_path:
        torchreid.utils.load_pretrained_weights(model, weights_path)
        weights_label = os.path.basename(weights_path)
    else:
        print("[WARNING] No OSNet Re-ID weights given (--reid_weights / OSNET_WEIGHTS); "
              "using ImageNet weights — similarity scores will not be reliable.")
        weights_label = "imagenet"
    model.to(device)
    model.eval()
    return model, weights_label


def embed_crop(crop_bgr, model):
    """
    Stage 4: turn one cropped person image into a 512-dim embedding.
    Input:  cropped BGR image (numpy array).
    Output: 1D numpy array, 512 floats, L2-normalized.
    """
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 256))  # OSNet expects (width=128, height=256)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    device = next(model.parameters()).device
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        embedding = model(tensor)

    embedding = embedding.squeeze().cpu().numpy()  # shape: (512,)
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="path to a test image")
    parser.add_argument("--video", type=str, help="path to a test video")
    parser.add_argument("--frame", type=int, default=0, help="frame number if using --video")
    parser.add_argument("--reid_weights", type=str, default=None,
                        help="OSNet Re-ID checkpoint (.pth); defaults to $OSNET_WEIGHTS")
    args = parser.parse_args()

    if not args.image and not args.video:
        raise ValueError("Pass either --image or --video")

    print("Loading YOLOv8n...")
    yolo_model = load_yolo()

    print("Loading OSNet...")
    osnet_model, weights_label = load_osnet(args.reid_weights)
    print(f"OSNet weights: {weights_label}")

    frame = get_frame(args)
    print(f"Frame shape: {frame.shape}")

    boxes = detect_people(frame, yolo_model)
    print(f"\nStage 3 (YOLO) output: {len(boxes)} person(s) detected")
    for i, box in enumerate(boxes):
        print(f"  person {i}: box={box[:4]}, confidence={box[4]:.2f}")

    if not boxes:
        print("No people detected — try a different frame/image.")
        return

    crops = crop_boxes(frame, boxes)

    print(f"\nStage 4 (OSNet) output: embedding per crop")
    embeddings = []
    for i, crop in enumerate(crops):
        emb = embed_crop(crop, osnet_model)
        embeddings.append(emb)
        print(f"  person {i}: embedding shape={emb.shape}, first 5 values={emb[:5]}")
        cv2.imwrite(f"crop_{i}.jpg", crop)  # save so you can visually check the crop

    # if there's more than one person detected in this single frame,
    # this just shows you the mechanic — real matching happens across
    # DIFFERENT camera frames, not within one frame.
    if len(embeddings) >= 2:
        sim = cosine_similarity(embeddings[0], embeddings[1])
        print(f"\nCosine similarity between person 0 and person 1: {sim:.3f}")
        print("(threshold ~0.7 in the pipeline design = 'same person')")


if __name__ == "__main__":
    main()

"""
Fallback if torchreid install fails (common on Windows / some Python versions):

Option A — use a plain torchvision ResNet50 stripped of its classifier head
as a stand-in embedding extractor for testing the PIPELINE MECHANICS
(box -> crop -> embedding -> cosine similarity). It won't have OSNet's
person-Re-ID-specific training, but it lets you test everything except
raw accuracy:

    import torchvision.models as models
    resnet = models.resnet50(pretrained=True)
    resnet.fc = torch.nn.Identity()   # output becomes a 2048-dim embedding
    resnet.eval()

Option B — clone OSNet directly instead of via pip:
    git clone https://github.com/KaiyangZhou/deep-person-reid.git
    cd deep-person-reid && pip install -r requirements.txt && python setup.py develop
"""
