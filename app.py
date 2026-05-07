from flask import Flask, render_template, request, jsonify
from rag import process_user_query  # import existing RAG function
import json
from pathlib import Path

app = Flask(__name__)

# JSON file to store queries and responses
JSON_FILE = Path("rag_evaluation.json")


def append_to_json(question, model_answer, rag_output=None, general_output=None, source=None):

       # Load existing data if file exists
    if JSON_FILE.exists() and JSON_FILE.stat().st_size > 0:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    # Determine next QID
    existing_qids = [entry.get("qid", 0) for entry in data]
    next_qid = max(existing_qids, default=0) + 1

    entry = {
        "qid": next_qid,
        "question": question,
        "model_answer": model_answer,
        "rag_output": rag_output,
        "general_output": general_output,
        "source": source or "",
        "ground_truth": "",
        "RAG_Factuality": "",
        "RAG_Completeness": "",
        "RAG_Faithfulness": "",
        "RAG_Safety": "",
        "RAG_Total_Score": "",
        "NonRAG_Factuality": "",
        "NonRAG_Completeness": "",
        "NonRAG_Faithfulness": "",
        "NonRAG_Safety": "",
        "NonRAG_Total_Score": "",
        "Notes": ""
    }

    data.append(entry)

    # Write back
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({'system_msg': " Please enter a valid query."})
    
    model_answer_text = "Sorry, I couldn't process your request right now."
    

    
    # Call your rag.py function
    try:
        response= process_user_query(user_message)

        # Extract RAG and general outputs safely
        rag_output_text = None
        general_output_text = None

        if isinstance(response, dict):
            rag_output_text = response.get("rag_output")
            general_output_text = response.get("general_output")
            # If model_answer is provided as dict, fallback to rag_output
            model_answer_text = response.get("model_answer") or rag_output_text or general_output_text

        elif isinstance(response, str):
            model_answer_text = response
            rag_output_text = response

        else:
            model_answer_text = str(response)
            rag_output_text = str(response)
        
        if not model_answer_text:
            model_answer_text = "Sorry, I couldn't find a matching tour for your query. Please try adjusting your budget or destination."


        # Append both outputs to JSON
        append_to_json(
           question=user_message,
           model_answer=model_answer_text,
           rag_output=rag_output_text,
           general_output=general_output_text
        )

        # Convert RAG output to HTML for frontend
        import re
        def format_rag_output_to_html(text):
            # Escape HTML special characters first
            text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            
            # Convert bold fields like **Tour Name:** to strong
            text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)

             # Convert URLs to clickable links
            url_pattern = r'(https?://[^\s]+)'
            text = re.sub(url_pattern, r'<a href="\1" target="_blank">\1</a>', text)

            # Convert bullet points • to unordered list
            lines = text.splitlines()
            html_lines = []
            ul_open = False
            for line in lines:
                line = line.strip()
                if line.startswith("•"):
                    if not ul_open:
                        html_lines.append("<ul>")
                        ul_open = True
                    html_lines.append(f"<li>{line[1:].strip()}</li>")
                else:
                    if ul_open:
                        html_lines.append("</ul>")
                        ul_open = False
                    if line:
                        html_lines.append(f"<p>{line}</p>")
            if ul_open:
                html_lines.append("</ul>")
            return "".join(html_lines)

    

    except Exception as e:
        html_response = f"<p> Error processing query: {e}</p>"
        import traceback
        print("ERROR in /chat route:\n", traceback.format_exc())  
        model_answer_text = f"Error processing query: {str(e)}"

    html_response = format_rag_output_to_html(model_answer_text)
    
    return jsonify({'system_msg': html_response,
                    'raw_text': model_answer_text
                    })


if __name__ == '__main__':
    app.run(debug=True)