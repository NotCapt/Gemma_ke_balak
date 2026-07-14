import argparse
import requests
import json
import time

# 50+ hand-written adversarial test cases
TEST_SUITE = [
    # --- Code-switching / Typo Heavy ---
    {"text": "meri bachi nhi mil rhi, blue dress me thi, gate 4 pe the hum", "expected_category": "child_lost"},
    {"text": "yr yhn boht bhid h, log dhakka de rhe h gir gya koi", "expected_category": "crowd_panic"},
    {"text": "aggg lag gyi kude dan me!! jaldi aao", "expected_category": "small_fire"},
    {"text": "uncle behoz ho gye garmi se water chaye", "expected_category": "medical_help"},
    {"text": "mera bag gum gya red color ka", "expected_category": "lost_item"},
    {"text": "tamil translator chahiye please", "expected_category": "need_interpreter"},
    
    # --- Ambiguous / Borderline ---
    # "bhai jaldi aao" - lacks context but implies urgency
    {"text": "bhai jaldi aao kuch samajh nhi aa rha yaha log gir rhe hai", "expected_category": "crowd_panic"},
    # Sarcasm / Exaggeration
    {"text": "kitni bheed hai, saans hi nahi aa rahi mar jayenge hum toh", "expected_category": "crowd_panic"}, 
    {"text": "kya bakwas hai, mera phone nikal liya kisi ne", "expected_category": "lost_item"},
    {"text": "fire! fire! oh wait it's just someone smoking. nvm.", "expected_category": "unknown"}, # Should ideally be ignored/unknown
    {"text": "I can't find my son! Oh wait he's right here.", "expected_category": "unknown"},
    
    # --- Medical variations ---
    {"text": "pregnant lady is having contractions near the main stage", "expected_category": "medical_help"},
    {"text": "chest pain ho raha hai ek uncle ko", "expected_category": "medical_help"},
    {"text": "kisi ko mirgi ka daura pada hai yaha", "expected_category": "medical_help"},
    {"text": "asthma attack! inhaler chahiye", "expected_category": "medical_help"},
    {"text": "pair me moch aa gai chal nhi pa rha", "expected_category": "medical_help"},

    # --- Fire variations ---
    {"text": "sparks coming from the electric panel", "expected_category": "small_fire"},
    {"text": "short circuit near food stall 5", "expected_category": "small_fire"},
    {"text": "kachre me aag", "expected_category": "small_fire"},
    {"text": "smoke coming from the generators", "expected_category": "small_fire"},

    # --- Lost Child variations ---
    {"text": "5 saal ka bacha rora hai uski mummy nhi dikh rhi", "expected_category": "child_lost"},
    {"text": "found a crying toddler near exit 2", "expected_category": "child_lost"},
    {"text": "mera beta gum gaya pink t-shirt pehna tha", "expected_category": "child_lost"},
    {"text": "kid looking for parents", "expected_category": "child_lost"},

    # --- Crowd Panic variations ---
    {"text": "stampede jaisa situation hai barricade tut gya", "expected_category": "crowd_panic"},
    {"text": "people are running towards the exit!", "expected_category": "crowd_panic"},
    {"text": "dhakka mukki ho rhi hai bohot jyada", "expected_category": "crowd_panic"},
    {"text": "too many people pushing at the VIP gate", "expected_category": "crowd_panic"},

    # --- Interpreter variations ---
    {"text": "no hindi no english only telugu", "expected_category": "need_interpreter"},
    {"text": "is there anyone who speaks bengali?", "expected_category": "need_interpreter"},
    {"text": "foreigner trying to ask something in french", "expected_category": "need_interpreter"},
    {"text": "kannada maatadoku yaaru ilva illi?", "expected_category": "need_interpreter"},

    # --- Lost Item variations ---
    {"text": "meri chabi gir gayi", "expected_category": "lost_item"},
    {"text": "dropped my wallet somewhere near the rides", "expected_category": "lost_item"},
    {"text": "kisi ko ID card mila kya mera?", "expected_category": "lost_item"},
    {"text": "left my jacket on the bench", "expected_category": "lost_item"},

    # --- Adversarial / Tricky ---
    {"text": "aag baboola ho raha hai wo aadmi", "expected_category": "unknown"}, # Uses "aag" (fire) but means angry
    {"text": "mera bacha first aaya race me!", "expected_category": "unknown"}, # Uses "bacha" but not lost
    {"text": "bheed me khoya hu mai uski yaadon me", "expected_category": "unknown"}, # Poetic/sarcastic
    {"text": "dil me dard ho raha hai uski baaton se", "expected_category": "unknown"}, # Poetic medical
    {"text": "tumhare dil me aag lag jayegi ye sunke", "expected_category": "unknown"}, # Idiom
    
    # Adding a few more to hit 50
    {"text": "need doctor fast", "expected_category": "medical_help"},
    {"text": "fire brigade bulao", "expected_category": "small_fire"},
    {"text": "police ko bulao dhakka de rhe hai", "expected_category": "crowd_panic"},
    {"text": "kaha gaye mere papa", "expected_category": "child_lost"},
    {"text": "purse chori ho gaya", "expected_category": "lost_item"}, # Theft but handled as lost_item for now
    {"text": "translator required", "expected_category": "need_interpreter"},
    {"text": "koi bangla bolte pare?", "expected_category": "need_interpreter"},
    {"text": "chakkar aagya kisi ko", "expected_category": "medical_help"},
    {"text": "smoke in hall A", "expected_category": "small_fire"},
    {"text": "gate 4 crowding rapidly", "expected_category": "crowd_panic"},
    {"text": "crying baby alone", "expected_category": "child_lost"},
    {"text": "glasses dropped", "expected_category": "lost_item"},
    {"text": "need medical kit", "expected_category": "medical_help"},
    {"text": "burning smell", "expected_category": "small_fire"},
    {"text": "stampede warning", "expected_category": "crowd_panic"}
]

def evaluate(api_url):
    print(f"🚀 Starting Adversarial Evaluation against {api_url}")
    print(f"Testing {len(TEST_SUITE)} hand-written adversarial cases...")
    print("-" * 60)
    
    correct = 0
    total = len(TEST_SUITE)
    errors = 0
    
    results = {
        "child_lost": {"correct": 0, "total": 0},
        "crowd_panic": {"correct": 0, "total": 0},
        "medical_help": {"correct": 0, "total": 0},
        "small_fire": {"correct": 0, "total": 0},
        "need_interpreter": {"correct": 0, "total": 0},
        "lost_item": {"correct": 0, "total": 0},
        "unknown": {"correct": 0, "total": 0}
    }
    
    for i, test_case in enumerate(TEST_SUITE, 1):
        text = test_case["text"]
        expected = test_case["expected_category"]
        
        results[expected]["total"] += 1
        
        # Prepare request
        prompt = f"Classify this emergency into one of these categories: child_lost, crowd_panic, lost_item, medical_help, need_interpreter, small_fire\n\nEmergency: {text}\n\nCategory:"
        payload = {"text": prompt, "max_tokens": 150}
        
        try:
            start = time.time()
            response = requests.post(f"{api_url}/classify", json=payload, timeout=10)
            latency = time.time() - start
            
            if response.status_code == 200:
                res_json = response.json()
                predicted = res_json.get("category", "").lower()
                
                # Check match
                is_match = expected in predicted
                # Also handle unknown correctly
                if expected == "unknown" and not any(cat in predicted for cat in ["child_lost", "crowd_panic", "lost_item", "medical_help", "need_interpreter", "small_fire"]):
                    is_match = True
                
                if is_match:
                    correct += 1
                    results[expected]["correct"] += 1
                    print(f"✅ PASS | {text[:30]:<30} | Expected: {expected:<15} | Pred: {predicted}")
                else:
                    print(f"❌ FAIL | {text[:30]:<30} | Expected: {expected:<15} | Pred: {predicted}")
                    print(f"   Reasoning given: {res_json.get('reasoning')}")
            else:
                errors += 1
                print(f"⚠️ API Error ({response.status_code}) for: {text[:30]}")
                
        except Exception as e:
            errors += 1
            print(f"⚠️ Request Failed for: {text[:30]} - {e}")
            
    # Print Summary
    print("\n" + "=" * 60)
    print(f"📊 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total}")
    print(f"Passed:      {correct}")
    print(f"Failed:      {total - correct - errors}")
    print(f"Errors:      {errors}")
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Accuracy:    {accuracy:.1f}%")
    
    print("\n📈 PER-CATEGORY BREAKDOWN")
    print("-" * 60)
    for cat, stats in results.items():
        if stats["total"] > 0:
            acc = (stats["correct"] / stats["total"]) * 100
            print(f"{cat:<18}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Adversarial Evaluation")
    parser.add_argument("--api", type=str, default="http://localhost:8000", help="GemmaServer URL")
    args = parser.parse_args()
    
    evaluate(args.api)
