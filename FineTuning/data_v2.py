import json
from datasets import Dataset

# Training data mapping emergency text to structured reasoning JSON
training_data = [
    {
        "text": "I can't find my 5-year-old son near the food stalls.",
        "expected": json.dumps({
            "category": "child_lost",
            "severity": "CRITICAL",
            "reasoning": "A child is missing in a crowded area. Immediate search is required to prevent abduction or harm.",
            "action": "Dispatch security team to food stalls, announce on PA system, lock down exits."
        })
    },
    {
        "text": "People are pushing and someone fell down near gate 3!",
        "expected": json.dumps({
            "category": "crowd_panic",
            "severity": "CRITICAL",
            "reasoning": "Crowd crushing and trampling risk at gate 3. High potential for mass casualties.",
            "action": "Deploy rapid response team to gate 3, open emergency exits, issue calming announcements."
        })
    },
    {
        "text": "There's a small trash can fire near the restrooms.",
        "expected": json.dumps({
            "category": "small_fire",
            "severity": "HIGH",
            "reasoning": "Fire can spread quickly in crowded areas causing panic and burns.",
            "action": "Dispatch fire wardens with extinguishers, prepare for localized evacuation."
        })
    },
    {
        "text": "My grandmother fainted in the heat.",
        "expected": json.dumps({
            "category": "medical_help",
            "severity": "HIGH",
            "reasoning": "Medical emergency (syncope) likely due to heat exhaustion. Requires immediate care.",
            "action": "Send paramedics to location with stretcher and hydration."
        })
    },
    {
        "text": "I lost my blue backpack.",
        "expected": json.dumps({
            "category": "lost_item",
            "severity": "LOW",
            "reasoning": "Missing personal property. No immediate threat to life or safety.",
            "action": "Direct user to lost and found."
        })
    }
]

def format_prompt(example):
    prompt = f"Classify this emergency into one of these categories: child_lost, crowd_panic, lost_item, medical_help, need_interpreter, small_fire\n\nEmergency: {example['text']}\n\nCategory:"
    return {"text": prompt, "label": example['expected']}

def prepare_dataset():
    dataset = Dataset.from_list(training_data)
    return dataset.map(format_prompt)

if __name__ == "__main__":
    ds = prepare_dataset()
    print("Dataset ready with structured reasoning targets.")
    print(ds[0])
