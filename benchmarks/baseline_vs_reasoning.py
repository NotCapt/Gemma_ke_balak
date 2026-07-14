import os
import time
import json
from datetime import datetime

# =====================================================================
# Alarm Fatigue Benchmark: Baseline CV vs Gemma Reasoning
# =====================================================================
# Goal: Prove that adding Gemma's reasoning layer reduces false positives
# without missing true positives compared to a simple rules-based CV system.

# Simulated Test Set of 100 frames
# "density" (Low, Medium, High)
# "motion" (Calm, Chaotic)
# "ground_truth": True if actual emergency, False if false alarm
TEST_FRAMES = [
    # True Emergencies (Should be flagged)
    {"id": "F001", "density": "High", "motion": "Chaotic", "desc": "Actual stampede", "ground_truth": True},
    {"id": "F002", "density": "Medium", "motion": "Chaotic", "desc": "Fight broke out", "ground_truth": True},
    {"id": "F003", "density": "High", "motion": "Chaotic", "desc": "Fire panic", "ground_truth": True},
    
    # Benign but Tricky (Should NOT be flagged - False Positives for Baseline)
    {"id": "F004", "density": "High", "motion": "Chaotic", "desc": "Flash mob dancing", "ground_truth": False},
    {"id": "F005", "density": "Medium", "motion": "Chaotic", "desc": "Kids playing tag", "ground_truth": False},
    {"id": "F006", "density": "High", "motion": "Calm", "desc": "Crowded concert viewing", "ground_truth": False},
    {"id": "F007", "density": "High", "motion": "Chaotic", "desc": "Mosh pit at concert", "ground_truth": False},
    {"id": "F008", "density": "Medium", "motion": "Chaotic", "desc": "People running from sudden rain", "ground_truth": False},
    
    # Clearly Benign (Should NOT be flagged by either)
    {"id": "F009", "density": "Low", "motion": "Calm", "desc": "Empty corridor", "ground_truth": False},
    {"id": "F010", "density": "Medium", "motion": "Calm", "desc": "People walking normally", "ground_truth": False},
]

# Multiply benign cases to simulate a real crowd monitoring distribution (100 frames)
for i in range(11, 101):
    TEST_FRAMES.append({
        "id": f"F{i:03d}",
        "density": "Medium" if i % 2 == 0 else "Low",
        "motion": "Calm",
        "desc": "Routine monitoring",
        "ground_truth": False
    })


def baseline_system(density, motion):
    """
    Standard Computer Vision rules-based alert system.
    Alerts on any chaotic motion or high density.
    """
    if motion == "Chaotic" or density == "High":
        return True, "Alert Triggered by rules"
    return False, "Safe"

def gemma_reasoning_system(frame_info):
    """
    Simulated Gemma Reasoning Layer.
    In production, this takes the image and asks Gemma. 
    Here we simulate Gemma's ability to recognize context (dancing, playing, rain).
    """
    desc = frame_info["desc"].lower()
    
    # Gemma correctly identifies benign contexts that trigger rules-based alerts
    if "dancing" in desc or "playing" in desc or "rain" in desc or "mosh pit" in desc:
        return False, f"Benign activity recognized: {desc}"
        
    # Gemma still catches actual emergencies
    if "stampede" in desc or "fight" in desc or "fire" in desc:
        return True, f"Emergency confirmed: {desc}"
        
    # Fallback to baseline for remaining
    return baseline_system(frame_info["density"], frame_info["motion"])

def run_benchmark():
    print("="*60)
    print("⚠️ SIMULATION DEMO ONLY ⚠️")
    print("This script simulates the alarm-fatigue reduction of a reasoning layer.")
    print("It uses hard-coded rules and does NOT call the actual Gemma model.")
    print("The results below are theoretical and should not be presented as measured performance.")
    print("="*60)
    print("🚀 Running Alarm Fatigue Benchmark Simulation...")
    print(f"Testing {len(TEST_FRAMES)} simulated frames...\n")
    
    baseline_stats = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "alerts": 0}
    gemma_stats = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "alerts": 0}
    
    for frame in TEST_FRAMES:
        gt = frame["ground_truth"]
        
        # 1. Evaluate Baseline
        base_alert, _ = baseline_system(frame["density"], frame["motion"])
        if base_alert:
            baseline_stats["alerts"] += 1
            if gt: baseline_stats["TP"] += 1
            else:  baseline_stats["FP"] += 1
        else:
            if gt: baseline_stats["FN"] += 1
            else:  baseline_stats["TN"] += 1
            
        # 2. Evaluate Gemma
        gemma_alert, _ = gemma_reasoning_system(frame)
        if gemma_alert:
            gemma_stats["alerts"] += 1
            if gt: gemma_stats["TP"] += 1
            else:  gemma_stats["FP"] += 1
        else:
            if gt: gemma_stats["FN"] += 1
            else:  gemma_stats["TN"] += 1

    # Calculate metrics
    def calc_metrics(stats):
        precision = stats["TP"] / (stats["TP"] + stats["FP"]) if (stats["TP"] + stats["FP"]) > 0 else 0
        recall = stats["TP"] / (stats["TP"] + stats["FN"]) if (stats["TP"] + stats["FN"]) > 0 else 0
        fpr = stats["FP"] / (stats["FP"] + stats["TN"]) if (stats["FP"] + stats["TN"]) > 0 else 0
        return precision, recall, fpr
        
    base_prec, base_rec, base_fpr = calc_metrics(baseline_stats)
    gem_prec, gem_rec, gem_fpr = calc_metrics(gemma_stats)
    
    # Calculate alarm fatigue reduction
    fp_reduction = ((baseline_stats["FP"] - gemma_stats["FP"]) / baseline_stats["FP"] * 100) if baseline_stats["FP"] > 0 else 0
    total_alert_reduction = ((baseline_stats["alerts"] - gemma_stats["alerts"]) / baseline_stats["alerts"] * 100) if baseline_stats["alerts"] > 0 else 0
    
    # Print Report
    print("="*60)
    print("📊 ALARM FATIGUE BENCHMARK RESULTS")
    print("="*60)
    print(f"{'Metric':<25} | {'Baseline CV':<15} | {'Gemma Reasoning':<15}")
    print("-" * 60)
    print(f"{'Total Alerts Fired':<25} | {baseline_stats['alerts']:<15} | {gemma_stats['alerts']:<15}")
    print(f"{'True Positives (Caught)':<25} | {baseline_stats['TP']:<15} | {gemma_stats['TP']:<15}")
    print(f"{'False Positives (Noise)':<25} | {baseline_stats['FP']:<15} | {gemma_stats['FP']:<15}")
    print(f"{'False Negatives (Missed)':<25} | {baseline_stats['FN']:<15} | {gemma_stats['FN']:<15}")
    print("-" * 60)
    print(f"{'Precision':<25} | {base_prec*100:.1f}%{'':<10} | {gem_prec*100:.1f}%")
    print(f"{'Recall':<25} | {base_rec*100:.1f}%{'':<10} | {gem_rec*100:.1f}%")
    print(f"{'False Positive Rate':<25} | {base_fpr*100:.1f}%{'':<10} | {gem_fpr*100:.1f}%")
    print("=" * 60)
    
    print(f"\n🎯 KEY FINDINGS:")
    print(f"1. Gemma reduced False Positives by {fp_reduction:.1f}%")
    print(f"2. Total alert volume (Alarm Fatigue) reduced by {total_alert_reduction:.1f}%")
    print(f"3. True emergencies caught remained at {gem_rec*100:.1f}% (No loss in safety)")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_frames": len(TEST_FRAMES),
        "baseline": baseline_stats,
        "gemma": gemma_stats,
        "metrics": {
            "fp_reduction_percent": fp_reduction,
            "total_alert_reduction_percent": total_alert_reduction
        }
    }
    
    with open("results/benchmark_report.json", "w") as f:
        json.dump(report, f, indent=4)
    print("\n💾 Full report saved to results/benchmark_report.json")

if __name__ == "__main__":
    run_benchmark()
