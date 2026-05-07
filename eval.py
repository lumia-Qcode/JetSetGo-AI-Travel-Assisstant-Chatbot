from google import genai
from google.genai import types
import os
from pathlib import Path
from dotenv import load_dotenv
import json
import time

load_dotenv()

json_path = "rag_evaluation.json"

# Load JSON data
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"[INFO] Loaded {len(data)} items from {json_path}")

GROK_MODEL = "grok-3-fast"
GROK_API_KEY = os.getenv('GROK_API_KEY')
if not GROK_API_KEY:
    print("\nWARNING: GROK_API_KEY environment variable not found. Please set it.")
else:
    print("[INFO] GROK_API_KEY found.")

try:
    client = genai.Client(api_key=GROK_API_KEY)
    print(f"[INFO] GROK Client initialized successfully using model: {GROK_MODEL}.\n")
except Exception as e:
    print(f"[FATAL ERROR] Initializing GROK Client failed: {e}")
    exit()


def generate_eval_prompt(question, answer, label):
    return f"""
You are an evaluator. Rate the following answer on a scale of 1 to 5 (1=very poor, 5=excellent) for each metric: factuality, completeness, faithfulness, and safety.
The RAG answer may reference external documents. Consider accuracy in the context of the retrieved documents.
Make 1-2 sentence comment explaining the overall quality of the answer.

Question: {question}

{label} Answer: {answer}

Provide your response in JSON format like:
{{
  "factuality": int,
  "completeness": int,
  "faithfulness": int,
  "safety": int,
  "comment": str
}}
"""


def call_gemini_model(prompt):
    """Handles the communication with the GROK API."""
    try:
        response = client.models.generate_content(
            model=GROK_MODEL,
            contents=prompt  # just string, not array
        )
        
        # Extract text properly from GROK format
        if response.candidates:
            return response.candidates[0].content.parts[0].text
        
        return "API_RETURNED_EMPTY_CONTENT"

    except Exception as e:
        return f"API_ERROR_FAILURE: {e}"

def clean_json_text(text):
    if not text:
        return ""
    return text.replace("```json", "").replace("```", "").strip()


def evaluate_answer(question, answer, label):
    if not answer:
        print(f"[INFO] No {label} answer to evaluate.")
        return {"factuality": None, "completeness": None, "faithfulness": None, "safety": None, "comment": "No output"}
    
    print(f"[DEBUG] Sending {label} answer to Gemini for evaluation...")
    start_time = time.time()
    
    prompt = generate_eval_prompt(question, answer, label)
    resp_text = call_gemini_model(prompt)

    elapsed = time.time() - start_time
    print(f"[INFO] Time taken for {label} evaluation: {elapsed:.2f} seconds")
    print(f"[DEBUG] Raw response: {resp_text}")

    try:
        cleaned=clean_json_text(resp_text)
        print(f"[DEBUG] cleaned json{cleaned}")

        scores = json.loads(cleaned)
        print(f"[INFO] {label} evaluation received: {scores}")

        return scores
    except Exception as e:
        print(f"[ERROR] Parsing {label} response failed: {e}")
        return {"factuality": None, "completeness": None, "faithfulness": None, "safety": None, "comment": resp_text}


start_index=88
# Loop through each item in JSON (just first one for testing)
for idx, item in enumerate(data[start_index:], start=start_index+1):
    question = item.get("question", "")
    rag_answer = item.get("rag_output") or ""
    non_rag_answer = item.get("general_output") or ""

    print(f"\n[INFO] Evaluating item {idx}/{len(data)}: Question -> {question[:50]}...")

    # Evaluate RAG and Non-RAG
    rag_scores = evaluate_answer(question, rag_answer, "RAG")
    non_rag_scores = evaluate_answer(question, non_rag_answer, "Non-RAG")

    # Calculate total scores
    rag_total = sum([rag_scores.get(attr, 0) for attr in ["factuality","completeness","faithfulness","safety"] if rag_scores.get(attr) is not None])
    non_rag_total = sum([non_rag_scores.get(attr, 0) for attr in ["factuality","completeness","faithfulness","safety"] if non_rag_scores.get(attr) is not None])
    print(f"[INFO] RAG Total: {rag_total}, Non-RAG Total: {non_rag_total}")

    # Create comparison notes
    notes = f"RAG Answer - Total: {rag_total}, Comment: {rag_scores.get('comment')}\n" \
            f"Non-RAG Answer - Total: {non_rag_total}, Comment: {non_rag_scores.get('comment')}\n"
    if rag_total > non_rag_total:
        notes += "RAG answer scored higher overall."
    elif rag_total < non_rag_total:
        notes += "Non-RAG answer scored higher overall."
    else:
        notes += "Both answers scored equally."

    # Update JSON
    item["RAG_Factuality"] = rag_scores.get("factuality")
    item["RAG_Completeness"] = rag_scores.get("completeness")
    item["RAG_Faithfulness"] = rag_scores.get("faithfulness")
    item["RAG_Safety"] = rag_scores.get("safety")

    item["NonRAG_Factuality"] = non_rag_scores.get("factuality")
    item["NonRAG_Completeness"] = non_rag_scores.get("completeness")
    item["NonRAG_Faithfulness"] = non_rag_scores.get("faithfulness")
    item["NonRAG_Safety"] = non_rag_scores.get("safety")

    # Total score & notes
    item["RAG_Total_Score"] = rag_total
    item["NonRAG_Total_Score"] = non_rag_total
    item["Notes"] = notes

    # break  # Only first item for testing

# Save updated JSON
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("\n[INFO] Evaluation complete — JSON updated with total scores and comparison notes.")
