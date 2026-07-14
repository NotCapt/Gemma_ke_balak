import json
import random
import os
import argparse
from datasets import Dataset
import google.generativeai as genai

# Seed examples representing the 6 chat categories + 2 vision categories
SEED_DATA = [
    # --- Chat Emergency Categories ---
    {
        "text": "bache ko uski mummy nahi mil rahi hai, wo food court ke paas hai.",
        "category": "child_lost",
        "severity": "CRITICAL",
        "reasoning": "A child is separated from their parents in a crowded area. High risk of abduction or harm.",
        "action": "Dispatch security to food court immediately. Make PA announcement."
    },
    {
        "text": "People are pushing and someone fell down near gate 3! Bhagdad mach gai hai!",
        "category": "crowd_panic",
        "severity": "CRITICAL",
        "reasoning": "Crowd crushing and trampling risk at gate 3. High potential for mass casualties.",
        "action": "Deploy rapid response team to gate 3, open emergency exits, issue calming announcements."
    },
    {
        "text": "There's a small trash can fire near the restrooms. dhua nikal raha hai.",
        "category": "small_fire",
        "severity": "HIGH",
        "reasoning": "Fire can spread quickly in crowded areas causing panic and burns.",
        "action": "Dispatch fire wardens with extinguishers, prepare for localized evacuation."
    },
    {
        "text": "My grandmother fainted in the heat. ek aurat behosh ho gai.",
        "category": "medical_help",
        "severity": "HIGH",
        "reasoning": "Medical emergency (syncope) likely due to heat exhaustion. Requires immediate care.",
        "action": "Send paramedics to location with stretcher and hydration."
    },
    {
        "text": "I lost my blue backpack near the entry.",
        "category": "lost_item",
        "severity": "LOW",
        "reasoning": "Missing personal property. No immediate threat to life or safety.",
        "action": "Direct user to lost and found counter."
    },
    {
        "text": "mujhe kuch samajh nahi aa raha, yahan koi tamil bolne wala hai?",
        "category": "need_interpreter",
        "severity": "MEDIUM",
        "reasoning": "Language barrier causing confusion. Needs translation assistance to navigate or report issues.",
        "action": "Dispatch multilingual volunteer or connect to phone translation service."
    },
    
    # --- Crowd Safety Reasoning (NEW - matches runtime prompts) ---
    {
        "text": "Crowd analysis detected High density and Chaotic motion, leading to a CRITICAL risk level. Explain why this is dangerous and what actions should be taken.",
        "category": "crowd_safety_reasoning",
        "severity": "CRITICAL",
        "reasoning": "High density combined with chaotic motion creates immediate trampling risk. People cannot control their movement in dense crowds, and panic behavior amplifies danger. Mass casualties possible within seconds.",
        "action": "Deploy rapid response team immediately. Open emergency exits. Issue calming announcements. Establish crowd flow control barriers."
    },
    {
        "text": "Crowd analysis detected High density and Calm motion, leading to a MODERATE risk level. Explain why this is dangerous and what actions should be taken.",
        "category": "crowd_safety_reasoning",
        "severity": "MODERATE",
        "reasoning": "High density alone is concerning even without panic. Crowd crush can occur from simple movement like people trying to see something. Requires monitoring to prevent escalation.",
        "action": "Monitor closely. Prepare crowd dispersal plan. Station personnel at choke points. Limit further ingress."
    },
    {
        "text": "Crowd analysis detected Medium density and Chaotic motion, leading to a HIGH risk level. Explain why this is dangerous and what actions should be taken.",
        "category": "crowd_safety_reasoning",
        "severity": "HIGH",
        "reasoning": "Chaotic motion in medium-density areas suggests localized panic or conflict. May spread rapidly if not contained. Could escalate to mass panic event.",
        "action": "Investigate source of chaos immediately. Deploy security to affected zone. Prevent crowd from entering area. Prepare medical response."
    },
    {
        "text": "Crowd analysis detected Low density and Chaotic motion, leading to a MODERATE risk level. Explain why this is dangerous and what actions should be taken.",
        "category": "crowd_safety_reasoning",
        "severity": "MODERATE",
        "reasoning": "Chaotic motion in sparse crowds typically indicates specific incident like fight, medical emergency, or fire. Not mass panic, but requires immediate attention.",
        "action": "Dispatch investigation team. Check for medical emergencies or security incidents. Monitor for crowd gathering around incident."
    },
    {
        "text": "Crowd analysis detected Medium density and Calm motion, leading to a SAFE risk level. Explain the current situation.",
        "category": "crowd_safety_reasoning",
        "severity": "LOW",
        "reasoning": "Moderate crowding with orderly movement indicates normal event operations. No immediate safety concerns detected.",
        "action": "Continue routine monitoring. Maintain current staffing levels."
    },
    {
        "text": "Crowd analysis detected Low density and Calm motion, leading to a SAFE risk level. Explain the current situation.",
        "category": "crowd_safety_reasoning",
        "severity": "LOW",
        "reasoning": "Sparse crowd with calm behavior represents minimal safety risk. Standard conditions for uncrowded areas.",
        "action": "Routine monitoring sufficient. No special measures needed."
    }
]

# Generate more basic permutations if Gemini is not available or to scale up rapidly
def generate_synthetic_variations(seeds, count=1000):
    print(f"Generating {count} synthetic variations locally...")
    variations = []
    locations = ["gate 1", "main stage", "food court", "exit", "parking lot", "restroom area", "vip section", "zone B"]
    
    for i in range(count):
        base = random.choice(seeds)
        loc = random.choice(locations)
        
        new_text = base["text"]
        # Only append locations to Chat emergency categories, not the Crowd reasoning templates
        if base["category"] in ["child_lost", "crowd_panic", "small_fire", "medical_help", "lost_item", "need_interpreter"]:
            if "near" in base["text"] or "paas" in base["text"]:
                new_text = base["text"] + f" around {loc}"
            else:
                new_text = base["text"].replace(".", f" near {loc}.")
        # For crowd_safety_reasoning, keep the template as-is (it's already specific)
        
        variations.append({
            "text": new_text,
            "category": base["category"],
            "severity": base["severity"],
            "reasoning": base["reasoning"],
            "action": base["action"]
        })
        
    return variations

def generate_gemini_dataset(num_samples: int):
    """Generate high-quality diverse dataset using Gemini API (if available)"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️ GEMINI_API_KEY not found. Synthesizing full dataset locally.")
        return generate_synthetic_variations(SEED_DATA, num_samples)
        
    print(f"🚀 Using Gemini to generate diverse, high-quality Hinglish examples...")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    generated_data = []
    
    # We batch generate in chunks of 10. To generate num_samples, we run num_samples // 10 iterations.
    target_api_calls = max(1, num_samples // 10) 
    print(f"Making {target_api_calls} API calls to generate seed variations...")
    
    for i in range(target_api_calls):
        try:
            prompt = """
Generate a JSON list of 10 highly diverse emergency reports in Hinglish (Hindi + English) that might be reported at a large festival or mela.
The categories are: child_lost, crowd_panic, small_fire, medical_help, lost_item, need_interpreter.
Include typos, code-switching, panic, and different locations.
Format as a valid JSON array of objects with keys: "text", "category", "severity", "reasoning", "action".
Do NOT wrap in markdown blocks, return ONLY valid JSON.
Example: [{"text": "meri beti nahi mil rahi gate 2 pe", "category": "child_lost", "severity": "CRITICAL", "reasoning": "Missing child", "action": "Dispatch security"}]
"""
            response = model.generate_content(prompt)
            text_resp = response.text.strip().replace("```json", "").replace("```", "")
            batch = json.loads(text_resp)
            generated_data.extend(batch)
            if i % 10 == 0:
                print(f"✅ Generated {len(generated_data)} samples so far...")
        except Exception as e:
            print(f"⚠️ API Generation error on batch {i+1}: {e}")
            
    print(f"🎉 Generated {len(generated_data)} unique examples via Gemini API.")
    
    # If API failed or yielded fewer samples, pad with synthetic data
    if len(generated_data) < num_samples and len(generated_data) > 0:
        print(f"Padding {len(generated_data)} Gemini seeds to {num_samples} samples using synthetic permutation...")
        scaled_data = generate_synthetic_variations(generated_data, num_samples - len(generated_data))
        generated_data.extend(scaled_data)
    elif len(generated_data) == 0:
        return generate_synthetic_variations(SEED_DATA, num_samples)
        
    return generated_data

def format_prompt(example):
    """Format for Gemma 4 E2B Instruct"""
    categories_str = "child_lost, crowd_panic, lost_item, medical_help, need_interpreter, small_fire, crowd_safety_reasoning"
    
    # For crowd reasoning prompts, the text already contains the full question
    is_crowd_reasoning = example['category'] == "crowd_safety_reasoning"
    
    if is_crowd_reasoning:
        # The prompt already contains "Explain why..." - pass it directly
        prompt = example['text']
    else:
        # For emergency classification, prepend the instruction
        prompt = f"Classify this emergency into one of these categories: {categories_str}\n\nEmergency: {example['text']}\n\nCategory:"
    
    # Build structured JSON target for all categories
    target = json.dumps({
        "category": example['category'],
        "severity": example['severity'],
        "reasoning": example['reasoning'],
        "action": example['action']
    })
    
    return {"text": prompt, "label": target}

def prepare_dataset(num_samples=60000):
    """Prepare the final dataset for HuggingFace Trainer"""
    print(f"Preparing dataset with target size {num_samples}...")
    
    # 1. Generate data
    raw_data = generate_gemini_dataset(num_samples)
    
    # 2. Add exact seeds to ensure perfect learning of core examples
    # Weight seeds more heavily
    raw_data.extend(SEED_DATA * 50) 
    random.shuffle(raw_data)
    
    # 3. Convert to HF Dataset
    dataset = Dataset.from_list(raw_data)
    
    # 4. Apply prompt formatting
    formatted_dataset = dataset.map(format_prompt)
    
    print(f"Dataset ready! Total samples: {len(formatted_dataset)}")
    return formatted_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=60000, help="Target dataset size")
    parser.add_argument("--export", action="store_true", help="Export to JSONL")
    args = parser.parse_args()
    
    ds = prepare_dataset(args.size)
    
    print("\nSample Data:")
    print("-" * 50)
    print("PROMPT:")
    print(ds[0]['text'])
    print("TARGET:")
    print(ds[0]['label'])
    
    if args.export:
        print("\nExporting to data_v2.jsonl...")
        ds.to_json("data_v2.jsonl")
