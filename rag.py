from sentence_transformers import SentenceTransformer
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
import os
from openai import OpenAI
import re
from dotenv import load_dotenv
import time


load_dotenv()

# --- Utility Functions (Kept as is) ---

def clean_value(val):
    if pd.isna(val):
        return "Unknown"
    return str(val)

def extract_numeric_price(price_str):
    """Extract numeric value from price string like 'Rs. 275,000'"""
    if pd.isna(price_str) or price_str == "Unknown" or price_str == "N/A":
        return float('inf')
    
    # Remove 'Rs.', commas, and spaces, then convert to float
    cleaned = str(price_str).replace('Rs.', '').replace(',', '').replace(' ', '')
    try:
        return float(cleaned)
    except ValueError:
        return float('inf')

def extract_numeric_duration(duration_str):
    """Extract number of days from duration string"""
    if pd.isna(duration_str) or duration_str == "Unknown" or duration_str == "N/A":
        return 0
    
    # Look for patterns like "5 Days", "7 Days 6 Nights"
    match = re.search(r'(\d+)\s*Days?', str(duration_str))
    if match:
        return int(match.group(1))
    return 0

def preprocess_tour_data(df):
    """Clean and preprocess tour data"""
    # Ensure Price is string and handle missing values
    df['Price'] = df['Price'].fillna('N/A').astype(str)
    df['Duration'] = df['Duration'].fillna('N/A').astype(str)
    df['Name'] = df['Name'].fillna('Unknown Tour')
    df['Itinerary'] = df['Itinerary'].fillna('No itinerary available')
    df['Link'] = df['Link'].fillna('No link available')
    
    # Handle Destination column (some CSVs might not have it)
    if 'Destination' not in df.columns:
        df['Destination'] = 'Unknown'
    else:
        df['Destination'] = df['Destination'].fillna('Unknown')
    
    return df

def load_all_tour_data():
    """Load and combine all tour data from multiple CSV files"""
    csv_files = [
        "data/hunza.csv",
        "data/naran.csv",
        "data/kumrat.csv", 
        "data/fairyMedows.csv",
        "data/murree.csv",
        "data/chitral.csv",
        "data/azadKashmir.csv",
        "data/neelum.csv",
        "data/swat.csv",
        "data/sakardu.csv"
    ]
    
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df = preprocess_tour_data(df)
            all_data.append(df)
            print(f"✓ Loaded {len(df)} tours from {csv_file.split('/')[-1]}")
        except Exception as e:
            print(f"✗ Could not load {csv_file}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n Total tours loaded: {len(combined_df)}")
        print(f" Destinations covered: {combined_df['Destination'].value_counts().to_dict()}")
        return combined_df
    else:
        print(" Error: No tour data could be loaded")
        return pd.DataFrame()

# --- 1. MODIFIED extract_query_parameters ---
def extract_query_parameters(query):
    """Extract destination, duration, and budget from query. Now extracts multiple destinations."""
    query_lower = query.lower()
    
    # Destinations list (Ensure all destination keywords are lower case)
    destinations = ['hunza', 'chitral', 'naran', 'kaghan', 'kumrat', 'neelum', 
                    'fairy meadows', 'murree', 'swat', 'skardu', 'kashmir']
    
    # Extract ALL destinations found in the query
    final_destinations = []
    for dest in destinations:
        if dest in query_lower:
            final_destinations.append(dest)
            
    # Extract duration
    duration_match = re.search(r'(\d+)\s*day', query_lower)
    target_days = int(duration_match.group(1)) if duration_match else None
    
    # Extract budget
    budget_match = re.search(r'(\d+,\d{3}|\d+)\s*(pkr|rs|rs\.|rupess?)', query_lower)
    if budget_match:
        max_price = float(budget_match.group(1).replace(',', ''))
    else:
        # Look for numbers that could be prices
        number_matches = re.findall(r'(\d+,\d{3}|\d{4,})', query)
        if number_matches:
            max_price = float(number_matches[0].replace(',', ''))
        else:
            max_price = None
     # Activities list
    activity_keywords = [
        "boating", "hiking", "trekking", "glacier", "rafting", 
        "sightseeing", "shopping", "fort", "lake", "valley",
        "camping", "jeep ride","jeep", "zipline", "adventure", "resort",
        "waterfall", "forest", "meadows", "guided hike", "nature walk",
    "river"
    ]
    final_activities = [act for act in activity_keywords if act in query_lower]
    
    return final_destinations, target_days, max_price,final_activities

def detect_language(client, text):
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": f"Detect language of this text and return only the ISO code (like 'en' or 'ur'): {text}"}],
        max_tokens=10,
        temperature=0.0
    )
    return response.choices[0].message.content.strip().lower()

def translate_text(client, text, target_lang):
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": f"Translate this to {target_lang}. Only return translation: {text}"}],
        max_tokens=1024,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

# --- 2. MODIFIED filter_and_rank_tours ---
def filter_and_rank_tours(retrieved_docs, query, max_price=None, target_days=None, max_results=3):
    """Filter tours based on destinations, duration, price, and activities."""
    destination_list, extracted_days, extracted_price, extracted_activities = extract_query_parameters(query)

    print("\n--- DEBUG FILTER PARAMETERS ---")
    print("Query:", query)
    print("Target destinations:", destination_list)
    print("Target days:", target_days)
    print("Max price:", max_price)
    print("Activities extracted:", extracted_activities)

    if max_price is None:
        max_price = extracted_price if extracted_price else float('inf')
    if target_days is None:
        target_days = extracted_days if extracted_days else None

    # --- TOP MATCH SELECTION WHEN NO DURATION SPECIFIED ---
    if target_days is None and retrieved_docs:
        top_doc = retrieved_docs[0]  # Pinecone top match
        print("\n--- TARGET DAYS NOT SPECIFIED: Using Pinecone top match ---")
        print(f"Selected Top Match: {top_doc.get('Name')} | Destination: {top_doc.get('Destination')} | Price: {top_doc.get('Price')}")
        return [top_doc]

    best_tours_by_destination = {}
    all_filtered_tours = []

    target_destinations = [d.lower() for d in destination_list]

    i = 0
    for doc in retrieved_docs:
        i += 1
        price = extract_numeric_price(doc.get("Price", "N/A"))
        duration = extract_numeric_duration(doc.get("Duration", "N/A"))
        doc_destination = doc.get("Destination", "").lower()
        doc_itinerary = doc.get("Itinerary", "").lower()

        # Destination check
        destination_match = True
        if target_destinations:
            destination_match = any(d in doc_destination for d in target_destinations)

        # Duration and price check
        price_ok = price <= max_price if max_price != float('inf') else True
        duration_ok = (duration == target_days) if target_days else True

        # Activity check
        activities_ok = True
        if extracted_activities:
            print(f"doc it{doc_itinerary}")
            activities_ok = any(act in doc_itinerary for act in extracted_activities)

        print(f"\n--- CHECKING DOC {i} ---")
        print("Name:", doc.get("Name"))
        print("Destination:", doc_destination, "Match:", destination_match)
        print("Price:", price, "Price OK:", price_ok)
        print("Duration:", duration, "Duration OK:", duration_ok)
        print("Itinerary contains activity:", activities_ok)

        # Combined check
        if price_ok and duration_ok and destination_match and activities_ok:
            score = price + (abs(duration - (target_days if target_days else 0)) * 5000)
            tour_info = {'doc': doc, 'price': price, 'duration': duration, 'score': score}
            print(f"ADDING TO RESULTS:{tour_info}")

            matched_dest = next((d for d in target_destinations if d in doc_destination), None)
            if matched_dest:
                if matched_dest not in best_tours_by_destination or score < best_tours_by_destination[matched_dest]['score']:
                    best_tours_by_destination[matched_dest] = tour_info

            if not target_destinations:
                all_filtered_tours.append(tour_info)

    print(f"selected best tours:{best_tours_by_destination}")

    # --- Final selection with debug logs ---
    if best_tours_by_destination:
        final_tours = [info['doc'] for info in best_tours_by_destination.values()]
        final_tours.sort(key=lambda doc: extract_numeric_price(doc.get("Price", "N/A")))
        print("\n--- FINAL SELECTION: best_tours_by_destination ---")
        for t in final_tours:
            print(f"Selected Tour: {t.get('Name')} | Destination: {t.get('Destination')} | Price: {t.get('Price')}")
        return final_tours[:max_results]

    elif all_filtered_tours:
        all_filtered_tours.sort(key=lambda x: x['score'])
        print("\n--- FINAL SELECTION: all_filtered_tours ---")
        for info in all_filtered_tours[:max_results]:
            t = info['doc']
            print(f"Selected Tour: {t.get('Name')} | Destination: {t.get('Destination')} | Price: {t.get('Price')} | Score: {info['score']}")
        return [info['doc'] for info in all_filtered_tours[:max_results]]

    # Fallback if nothing matches
    fallback_docs = []
    for doc in retrieved_docs:
        doc_destination = doc.get("Destination", "").lower()
        destination_match = True
        if target_destinations:
            destination_match = any(d in doc_destination for d in target_destinations)
        if destination_match:
            fallback_docs.append(doc)

    if fallback_docs:
        fallback_docs.sort(key=lambda doc: extract_numeric_price(doc.get("Price", "N/A")))
        print("\n--- FINAL SELECTION: fallback_docs ---")
        for t in fallback_docs[:1]:
            print(f"Selected Tour: {t.get('Name')} | Destination: {t.get('Destination')} | Price: {t.get('Price')}")
        return fallback_docs[:1]

    print("\n--- FINAL SELECTION: NO MATCH ---")
    return []

# --- 3. MODIFIED generate_with_rag_input ---
def generate_with_rag_input(query,extracted_activities, retrieved_docs, max_tokens=2048):
    """Generates a context-aware response for single or multiple tours."""

    if not retrieved_docs:
        return {"role": "assistant", "content": "No relevant tours found for your query."}

    # If only one tour is found (e.g., single destination query)
    if len(retrieved_docs) == 1:
        best_tour_metadata = retrieved_docs[0] 
        full_itinerary_text = best_tour_metadata.get("Itinerary", "N/A")
        extracted_highlights = get_clean_highlights(full_itinerary_text, extracted_activities)
        
        final_output = f"""
I found the best match for your request! Here are the details for the tour:

**Tour Name:** {best_tour_metadata.get("Name", "N/A")}
**Destination:** {best_tour_metadata.get("Destination", "N/A")}
**Price:** {best_tour_metadata.get("Price", "N/A")}
**Duration:** {best_tour_metadata.get("Duration", "N/A")}
**Link:** {best_tour_metadata.get("Link", "N/A")}
**Highlights:**
{chr(10).join(extracted_highlights)}
"""
        return {
            "role": "assistant",
            "content": final_output.strip()
        }

    # If multiple tours are found (e.g., multiple destination query or general request)
    else:
        output_sections = [f"I found {len(retrieved_docs)} great options based on your request! Here are the best tours I could find:"]
        
        for i, tour in enumerate(retrieved_docs):
            full_itinerary_text = tour.get("Itinerary", "N/A")
            extracted_highlights = get_clean_highlights(full_itinerary_text)
            
            tour_section = f"""
###  Option {i+1}: {tour.get("Name", "N/A")}

* **Destination:** {tour.get("Destination", "N/A")}
* **Price:** {tour.get("Price", "N/A")}
* **Duration:** {tour.get("Duration", "N/A")}
* **Link:** {tour.get("Link", "N/A")}

**Key Highlights:**
{chr(10).join(extracted_highlights)}
---
"""
            output_sections.append(tour_section.strip())
            
        final_output = "\n\n".join(output_sections)
        return {
            "role": "assistant",
            "content": final_output.strip()
        }

# --- Pinecone Setup (Kept as is for context) ---
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
INDEX_NAME = os.getenv('INDEX_NAME')
VECTOR_DIMENSION = int(os.getenv('VECTOR_DIMENSION'))
METADATA_LIMIT = int(os.getenv('METADATA_LIMIT'))

# Load all tour data
print("Loading all tour data...")
df = load_all_tour_data()

if df.empty:
    print("FATAL ERROR: No tour data loaded. Please check your CSV files.")
    exit()

# --- 1. Encoding Model ---
print("Initializing Sentence Transformer model...")
model_embedder = SentenceTransformer('all-MiniLM-L6-v2')

# --- 2. Pinecone Index Setup (Kept as is for context) ---
try:
    print("Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            INDEX_NAME,
            dimension=VECTOR_DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
        )
        print("Index created.")

    index = pc.Index(INDEX_NAME)
    print(f"Pinecone index connected. Status: {index.describe_index_stats()}")

    # Prepare data for upserting
    vectors_to_upsert = []
    for i, row in df.iterrows():
        text_to_embed = f"""
Name: {clean_value(row['Name'])}
Destination: {clean_value(row['Destination'])}
Duration: {clean_value(row['Duration'])}
Price: {clean_value(row['Price'])}
Itinerary: {clean_value(row['Itinerary'])}
"""
        embedding = model_embedder.encode(text_to_embed).tolist()
        
        vectors_to_upsert.append({
            "id": str(i),
            "values": embedding,
            "metadata": {
                "Name": clean_value(row["Name"]),
                "Destination": clean_value(row["Destination"]),
                "Duration": clean_value(row["Duration"]),
                "Price": clean_value(row["Price"]),
                "Link": clean_value(row["Link"]),
                "Itinerary": clean_value(row["Itinerary"]),
                "ShortItinerary": clean_value(row["Itinerary"])[:METADATA_LIMIT],
                "Text": text_to_embed
            }
        })
    
    print("Upserting vectors to Pinecone...")
    index.upsert(vectors=vectors_to_upsert)
    print("Upsert complete.")

except Exception as e:
    print(f"FATAL ERROR setting up Pinecone. Error details: {e}")
    exit()

# --- 3. Language Model Setup (Groq API) ---
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    print("\nWARNING: GROQ_API_KEY environment variable not found. Please set it.")
try:
    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    print(f"\nGroq Client initialized successfully using model: {GROQ_MODEL}.")
except Exception as e:
    print(f"FATAL ERROR initializing Groq Client. Error: {e}")
    exit()

# --- Shared API Call Function (Kept as is) ---
def is_travel_query(query):
    q = query.lower()
    print(f"query:{q}")

    # Strong travel intent keywords (must be combined with a destination OR numbers)
    strong_intent = [
        "give me a", "i want a", "find me a",
        "show me a", "looking for", "plan a",
        "recommend", "suggest", "tour package", "trip package",
        "itinerary", "cheapest", "longest", "shortest", "available tours",  "which tours", "last","which tour", "tour"
    ]

    # Destinations list
    destination_keywords = [
        "hunza", "skardu", "swat", "naran", "kaghan", "gilgit", 
        "neelum", "kashmir", "murree", "chitral", 
        "fairy meadows", "kumrat", 'sightseeing', 'northern areas', 'northern pakistan', 'northern'
    ]

    # Check: does the query contain a destination?
    contains_destination = any(dest in q for dest in destination_keywords)

    # Check: does the query contain numbers (days or budget)?
    contains_number = bool(re.search(r"\d+", q))

    # Check strong intent phrases
    contains_strong_intent = any(phrase in q for phrase in strong_intent)

    # RULE: Trigger RAG only when:
    # (Strong intent AND destination) OR (destination AND numbers)
    if (contains_strong_intent or contains_destination) or \
       (contains_destination and contains_number):
        print("returning true")
        return True

    return False

def call_groq_model(system_prompt, user_prompt, max_tokens):
    max_retries = 3
    backoff_time = 2  # initial 2 seconds
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content
            if content:
                return content
            else:
                return "API_RETURNED_EMPTY_CONTENT"
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str.lower():
                print(f"429 Rate Limit, retrying in {backoff_time} seconds... (Attempt {attempt+1})")
                time.sleep(backoff_time)
                backoff_time *= 2  # exponential backoff
                continue
            else:
                return f"API_ERROR_FAILURE: {e}"
    return "API_ERROR_FAILURE: 429 Rate Limit - retries exhausted"

# --- 4. Generation Functions (Kept as is) ---
def get_clean_highlights(itinerary_text, requested_activities=None):
    """Uses Grok to perform a simple, clean bullet-point extraction."""
    if not itinerary_text or itinerary_text.strip() == "" or itinerary_text == "Unknown":
        return ["• No highlights available"]

    requested_activities=requested_activities or []
    # Improved prompt that works with both formats
    system_prompt_extractor = """You are a travel highlights extractor. Extract 3-5 key highlights from the tour itinerary. 
Focus on unique experiences, main attractions, and user requested_activities if provided.
Return ONLY bullet points starting with •, no other text.
Examples of good highlights:
• Continental breakfast included
• Flight from Islamabad to Gilgit
• Private transport throughout
• Views of Passu Cones
• Optional speed boat adventures at Attabad Lake
• Terrace and basecamp for hiking/trekking"""
    
    user_prompt_extractor = f"Extract 3-5 key highlights from this tour itinerary:\n{itinerary_text}"
    
    content = call_groq_model(system_prompt_extractor, user_prompt_extractor, max_tokens=1000)

    if content.startswith("API_ERROR_FAILURE"):
        # Fallback: extract key phrases manually
        return extract_fallback_highlights(itinerary_text)

    # Clean and format the bullet points
    bullet_points = []
    for line in content.split('\n'):
        line = line.strip()
        if line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
            # Clean the bullet point
            clean_line = line.lstrip('•-* ').strip()
            if clean_line:
                bullet_points.append(f"• {clean_line}")


     # --- Ensure requested activities appear ---
    text_lower = itinerary_text.lower()
    for activity in requested_activities:
        if any(activity.lower() in bp.lower() for bp in bullet_points):
            continue  # Already mentioned
        if activity.lower() in text_lower:
            # Add a bullet with the activity
            bullet_points.append(f"• {activity.capitalize()} included")
    
    # If we got good bullet points, return them (limit to 5)
    if bullet_points and len(bullet_points) >= 2:
        return bullet_points[:5]
    else:
        # Fallback if Grok didn't return proper bullet points
        return extract_fallback_highlights(itinerary_text)

def extract_fallback_highlights(itinerary_text):
    """Fallback method to extract highlights when Grok fails"""
    highlights = []
    
    # Look for key features in the text
    text_lower = itinerary_text.lower()
    
    # Check for common tour features
    features_to_check = [
        "breakfast", "flight", "air ticket", "private transport", 
        "view", "lake", "fort", "valley", "glacier", "hiking", 
        "trekking", "boating", "bazar", "resort", "adventure"
    ]
    
    sentences = re.split(r'[.!?]', itinerary_text)
    for sentence in sentences:
        sentence = sentence.strip()
        if any(feature in sentence.lower() for feature in features_to_check) and len(sentence) > 10:
            # Shorten long sentences
            if len(sentence) > 80:
                words = sentence.split()[:12]  # Take first 12 words
                shortened = ' '.join(words) + '...'
            
                highlights.append(f"• {shortened}")
            else:
                highlights.append(f"• {sentence}")
            
            if len(highlights) >= 5:
                break
    
    # If no features found, create generic highlights
    if not highlights:
        highlights = [
            "• Comprehensive tour package",
            "• Experienced local guides", 
            "• Scenic mountain views",
            "• Cultural experiences",
            "• Comfortable accommodations"
        ]
    
    return highlights[:5]

def generate_without_rag_input(prompt, max_tokens=1024):
    """Generates a response using only the query (no context) via Groq API."""
    # Constrained prompt to prevent hallucination
    system_prompt = "You are a highly constrained travel assistant. Your task is to answer the user query ONLY with general knowledge and a disclaimer about not having specific data. DO NOT invent budgets, specific tour names, or itineraries."
    content = call_groq_model(system_prompt, prompt, max_tokens)
    return {"role": "assistant", "content": content}

def generate_non_travel_input(prompt, language='en', max_tokens=1024):
    lang_instruction="Answer in Urdu." if language=='ur'else "Answer in English."
    """
    Generates a helpful, natural, non-travel response using general reasoning only.
    No hallucinated facts, no invented data — only conversational assistance.
    """

    system_prompt = f"""
You are ViaNova, an AI assistant with a friendly, helpful personality.
{lang_instruction}
You can answer general user questions clearly and conversationally.

Rules:
- You are restricted to travel  only— give a user friendly message that you do not handle non travel queries.
"""

    user_prompt = f"User message: {prompt}\nProvide the best possible helpful reply."

    content = call_groq_model(system_prompt, user_prompt, max_tokens)
    if(content.startswith("API_ERROR_FAILURE") or content == "API_RETURNED_EMPTY_CONTENT"):
        content = "Sorry, I'm having trouble processing your request right now."
    return {"role": "assistant", "content": content}


# --- 5. Interactive Query System ---
def process_user_query(query):
    """
    Process a single user query and return a dict:
    {
        'rag_output': <RAG answer or None>,
        'general_output': <non-RAG answer>,
        'model_answer': <best available answer>
    }

    Multilingual support with robust fallbacks:
    - Detects Urdu queries
    - Translates to English for RAG processing
    - Generates answer
    - Translates back to Urdu if needed
    - Handles Grok failures gracefully
    """

    # --- 0. Detect language ---
    try:
        lang = detect_language(client, query)
    except Exception as e:
        print(f"Language detection failed: {e}")
        lang = "en"

    # --- 1. Translate to English if Urdu ---
    translated_query = query
    if lang == "ur":
        try:
            translated_query = translate_text(client, query, "English")
        except Exception as e:
            print(f"Translation to English failed: {e}")
            translated_query = query  # fallback

    # --- 2. Check if this is a travel query ---
    if not is_travel_query(translated_query):
        try:
            general_response = generate_non_travel_input(translated_query, language=lang)
            final_general_output = general_response.get('content', "معذرت، اس وقت جواب فراہم نہیں کیا جا سکتا۔")
        except Exception as e:
            print(f"Non-RAG generation failed: {e}")
            final_general_output = "معذرت، اس وقت جواب فراہم نہیں کیا جا سکتا۔" if lang=="ur" else "Sorry, could not get a response."

        # Translate to Urdu if needed
        if lang == "ur" and final_general_output and not final_general_output.strip().startswith("معذرت"):
            try:
                final_general_output = translate_text(client, final_general_output, "Urdu")
            except Exception as e:
                print(f"Translation back to Urdu failed: {e}")

        return {
            "rag_output": None,
            "general_output": final_general_output,
            "model_answer": final_general_output
        }

    # --- 3. Extract parameters and retrieve documents ---
    destination, target_days, max_price, activities = extract_query_parameters(translated_query)
    
    try:
        query_embedding = model_embedder.encode([translated_query]).tolist()
        search_results = index.query(vector=query_embedding, top_k=15, include_metadata=True)
        retrieved_docs = [match['metadata'] for match in search_results.get('matches', [])]
    except Exception as e:
        print(f"Pinecone search failed: {e}")
        retrieved_docs = []

    filtered_docs = filter_and_rank_tours(retrieved_docs, translated_query, max_price, target_days, max_results=3)

    # --- 4. Generate outputs ---
    rag_output = None
    general_output = None

    # Generate RAG output
    if filtered_docs:
        try:
            rag_output = generate_with_rag_input(translated_query, activities, filtered_docs).get('content')
        except Exception as e:
            print(f"RAG generation failed: {e}")
            rag_output = None

    # Generate general output
    try:
        general_output = generate_without_rag_input(translated_query).get('content', "معذرت، اس وقت جواب فراہم نہیں کیا جا سکتا۔")
    except Exception as e:
        print(f"Non-RAG generation failed: {e}")
        general_output = "معذرت، اس وقت جواب فراہم نہیں کیا جا سکتا۔" if lang=="ur" else "Sorry, could not get a response."

    # Translate outputs back to Urdu if needed
    if lang == "ur":
        if rag_output:
            try:
                rag_output = translate_text(client, rag_output, "Urdu")
            except Exception as e:
                print(f"Translation of RAG output failed: {e}")
        if general_output:
            try:
                general_output = translate_text(client, general_output, "Urdu")
            except Exception as e:
                print(f"Translation of general output failed: {e}")

    return {
        "rag_output": rag_output,
        "general_output": general_output,
        "model_answer": rag_output or general_output
    }