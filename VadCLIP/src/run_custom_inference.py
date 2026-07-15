import torch
import numpy as np
import os
import sys

# Ensure the script can import local files
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import CLIPVAD
import ucf_option
from utils.tools import get_batch_mask, get_prompt_text, process_split

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Load configuration
    args = ucf_option.parser.parse_args([])
    # Override default model path to verify
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "model_ucf.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_ucf.pth")
    print(f"[INFO] Model path: {model_path}")

    label_map = dict({
        'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson', 
        'Assault': 'Assault', 'Burglary': 'Burglary', 'Explosion': 'Explosion', 
        'Fighting': 'Fighting', 'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery', 
        'Shooting': 'Shooting', 'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing', 
        'Vandalism': 'Vandalism'
    })
    prompt_text = get_prompt_text(label_map)

    # Initialize model
    model = CLIPVAD(
        args.classes_num, 
        args.embed_dim, 
        args.visual_length, 
        args.visual_width, 
        args.visual_head, 
        args.visual_layers, 
        args.attn_window, 
        args.prompt_prefix, 
        args.prompt_postfix, 
        device
    )
    print("[INFO] Loading model weights...")
    model_param = torch.load(model_path, map_location=device)
    model.load_state_dict(model_param)
    model.to(device)
    model.eval()

    # Load extracted features
    feat_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "WhatsApp_Video_features.npy")
    if not os.path.exists(feat_path):
        print(f"[ERROR] Features file not found at {feat_path}")
        return
    
    print(f"[INFO] Loading features from: {feat_path}")
    features = np.load(feat_path)
    print(f"[INFO] Loaded features shape: {features.shape}")

    # Process features for the model
    visual, length = process_split(features, args.visual_length)
    visual = torch.tensor(visual).to(device)

    # Prepare inputs
    maxlen = args.visual_length
    len_cur = int(length)
    if len_cur < maxlen:
        visual = visual.unsqueeze(0)
    lengths = torch.zeros(int(len_cur / maxlen) + 1)
    temp_length = len_cur
    for j in range(int(len_cur / maxlen) + 1):
        if j == 0 and temp_length < maxlen:
            lengths[j] = temp_length
        elif j == 0 and temp_length > maxlen:
            lengths[j] = maxlen
            temp_length -= maxlen
        elif temp_length > maxlen:
            lengths[j] = maxlen
            temp_length -= maxlen
        else:
            lengths[j] = temp_length
    
    lengths = lengths.to(int)
    padding_mask = get_batch_mask(lengths, maxlen).to(device)

    # Run forward pass
    print("[INFO] Running inference...")
    with torch.no_grad():
        _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
        logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
        logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
        
        # Calculate anomaly probability profiles
        prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1)).cpu().numpy()
        prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1)).cpu().numpy()
        
        # Class probabilities (excluding normal class 0)
        probs_classes = logits2[0:len_cur].softmax(dim=-1).cpu().numpy()

    # Print out results
    print("\n--- INFERENCE RESULTS ---")
    print(f"Total video snippets (16 frames each): {len_cur}")
    print(f"Mean Anomaly Probability (Score 1 - Binary classifier): {np.mean(prob1):.4f}")
    print(f"Mean Anomaly Probability (Score 2 - Text-matching class-based): {np.mean(prob2):.4f}")
    print(f"Max Anomaly Probability (Score 1): {np.max(prob1):.4f} at snippet {np.argmax(prob1)}")
    print(f"Max Anomaly Probability (Score 2): {np.max(prob2):.4f} at snippet {np.argmax(prob2)}")

    # Find the top class predictions for anomalous frames
    class_names = list(label_map.keys())
    top_class_idx = np.argmax(np.mean(probs_classes[:, 1:], axis=0)) + 1 # ignore Normal at index 0
    top_class_name = class_names[top_class_idx]
    top_class_prob = np.mean(probs_classes[:, top_class_idx])
    print(f"Top predicted anomaly type: {top_class_name} (average probability: {top_class_prob:.4f})")

    # Log snippet predictions
    print("\nSnippet level anomaly scores:")
    print("Snippet | Binary Score | Class-based Score | Top Anomaly Class Prediction")
    print("-" * 75)
    for idx in range(len_cur):
        snippet_class_idx = np.argmax(probs_classes[idx, 1:]) + 1
        snippet_class_name = class_names[snippet_class_idx]
        snippet_class_prob = probs_classes[idx, snippet_class_idx]
        print(f"{idx:7d} | {prob1[idx]:12.4f} | {prob2[idx]:17.4f} | {snippet_class_name} ({snippet_class_prob:.4f})")

if __name__ == "__main__":
    main()
