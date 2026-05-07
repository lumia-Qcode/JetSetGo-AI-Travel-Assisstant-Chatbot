# JetSetGo – AI Travel Assistant 

A Retrieval-Augmented Generation (RAG) powered travel chatbot that helps users find tour packages across popular destinations in northern Pakistan. Built with Flask, Pinecone, Groq (LLaMA 3.3), and a vanilla JS frontend.

---

## Features

- **RAG-based tour search** — Embeds user queries and retrieves the most relevant tours from a Pinecone vector database
- **Smart filtering** — Filters results by destination, budget, duration, and requested activities (hiking, boating, glacier, etc.)
- **Dual-output evaluation** — Every query generates both a RAG answer and a non-RAG (general knowledge) answer for comparison
- **Multilingual support** — Detects Urdu queries, processes them in English, and returns responses in Urdu
- **Voice input & TTS** — Microphone input via Web Speech API; responses are read aloud using SpeechSynthesis
- **Chat history** — Previous conversations are stored in the sidebar for quick re-access
- **Automated evaluation** — `eval.py` scores both RAG and non-RAG answers across Factuality, Completeness, Faithfulness, and Safety using the Grok API

---

## Project Structure

```
├── app.py              # Flask web server and API routes
├── rag.py              # Core RAG pipeline (embedding, retrieval, generation)
├── eval.py             # Automated evaluation of RAG vs non-RAG answers
├── new.py              # Gemini API test/verification script
├── updatejson.py       # Utility to add new fields to the evaluation JSON
├── templates/
│   └── index.html      # Chat UI template
├── static/
│   ├── css/style.css   # Frontend styling
│   └── js/chat.js      # Frontend chat logic
├── data/               # CSV files per destination
│   ├── hunza.csv
│   ├── naran.csv
│   ├── kumrat.csv
│   ├── fairyMedows.csv
│   ├── murree.csv
│   ├── chitral.csv
│   ├── azadKashmir.csv
│   ├── neelum.csv
│   ├── swat.csv
│   └── sakardu.csv
├── rag_evaluation.json # Auto-generated log of all queries and evaluation scores
└── .env                # API keys (not committed to version control)
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- A Pinecone account (free tier works)
- A Groq API key (free tier available at [console.groq.com](https://console.groq.com))
- Optional: A Grok/xAI API key (only needed for `eval.py`)

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. Install dependencies

```bash
pip install flask sentence-transformers pinecone-client openai python-dotenv pandas google-generativeai
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-east-1        # or your Pinecone region
INDEX_NAME=your_index_name
VECTOR_DIMENSION=384                  # dimension for all-MiniLM-L6-v2
METADATA_LIMIT=1000

GROQ_API_KEY=your_groq_api_key

# Optional — only needed for eval.py
GROK_API_KEY=your_grok_api_key

# Optional — only needed for new.py
GEMINI_API_KEY=your_gemini_api_key
```

### 4. Prepare your data

Make sure the `data/` directory contains CSV files for each destination. Each CSV must have at minimum these columns:

| Column      | Description                          |
|-------------|--------------------------------------|
| `Name`      | Tour package name                    |
| `Destination` | Destination name (e.g., Hunza)     |
| `Duration`  | Duration string (e.g., "5 Days 4 Nights") |
| `Price`     | Price string (e.g., "Rs. 45,000")   |
| `Itinerary` | Full itinerary text                  |
| `Link`      | Booking URL                          |

### 5. Run the app

```bash
python app.py
```

The app will start on `http://127.0.0.1:5000`. On first run, `rag.py` will load all CSVs, encode them using SentenceTransformer, and upsert vectors into Pinecone automatically.

---

## How It Works

### RAG Pipeline (`rag.py`)

1. **Query parsing** — Extracts destination(s), budget, duration, and activities from the user's message using regex
2. **Language detection** — Detects if the query is in Urdu; translates to English for processing
3. **Travel intent check** — Determines whether the query is travel-related before hitting the vector DB
4. **Embedding & retrieval** — Encodes the query using `all-MiniLM-L6-v2` and retrieves the top 15 matches from Pinecone
5. **Filtering & ranking** — Applies hard filters (price ≤ budget, duration match, destination match, activity match) and picks the best tour(s) per destination
6. **Response generation** — Uses Groq (LLaMA 3.3 70B) to extract 3–5 highlights from the itinerary and format the final response
7. **Translation** — If the original query was Urdu, the final answer is translated back to Urdu

### Flask Server (`app.py`)

- `GET /` — Serves the chat UI
- `POST /chat` — Accepts a JSON body `{ "message": "..." }`, runs the RAG pipeline, logs the result to `rag_evaluation.json`, and returns an HTML-formatted response

### Evaluation (`eval.py`)

Iterates over all entries in `rag_evaluation.json` and scores each RAG and non-RAG answer on four dimensions (1–5 scale):

| Metric        | Description                                      |
|---------------|--------------------------------------------------|
| Factuality    | Are the facts accurate?                          |
| Completeness  | Does the answer fully address the question?      |
| Faithfulness  | Does the answer stay grounded in the source?     |
| Safety        | Is the content appropriate and non-harmful?      |

Results are written back to the same JSON file.

---

## Destinations Covered

Hunza · Naran/Kaghan · Kumrat · Fairy Meadows · Murree · Chitral · Azad Kashmir · Neelum Valley · Swat · Skardu

---

## Supported Query Types

| Query Example                                    | Behavior                          |
|--------------------------------------------------|-----------------------------------|
| "Show me a 5-day Hunza tour under 50,000 PKR"   | RAG: filtered by destination, duration, budget |
| "I want a tour with glacier hiking in Skardu"   | RAG: filtered by activity         |
| "Find tours in Naran and Swat"                  | RAG: returns one best match per destination |
| "What is the capital of Pakistan?"              | Non-RAG: friendly redirect message |
| اردو میں سوال                                   | Detected, translated, answered, translated back |

---

## Notes

- On startup, `rag.py` re-upserts all tour vectors to Pinecone every time. For production use, add a check to skip upserting if the index is already populated.
- The `eval.py` script uses the Grok API (xAI), which is separate from Groq. Make sure you set the correct key (`GROK_API_KEY`).
- `rag_evaluation.json` grows with every chat message. Back it up periodically.
- Speech recognition and TTS in `chat.js` require a modern browser (Chrome recommended).
