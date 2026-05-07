import json
from pathlib import Path

# Path to your JSON file
json_file = Path("rag_evaluation.json")

# Load existing JSON
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Add new attributes to each object
for obj in data:
    
    # Optional notes field
    obj.setdefault("Notes", "")

# Save back to JSON
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("JSON updated with RAG and Non-RAG score fields.")
