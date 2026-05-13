import os
import json

from dotenv import load_dotenv
from google import genai

from conversation import (
    build_conversation_query,
    get_clarification_question,
    is_off_topic
)

from retriever import search
from prompts import build_prompt
import time

retriever_initialized = False

def initialize_system():

    global retriever_initialized

    if retriever_initialized:
        return

    from retriever import initialize_retriever

    initialize_retriever()

    retriever_initialized = True

# =========================================================
# GEMINI RESPONSE WITH RETRIES
# =========================================================
def generate_gemini_response(prompt):

    models_to_try = [
        "gemini-3.1-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash"
    ]

    max_retries = 2

    for model_name in models_to_try:

        for attempt in range(max_retries):

            try:

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                return response.text

            except Exception as e:

                error_text = str(e)

                # -------------------------------------------------
                # RETRYABLE ERRORS
                # -------------------------------------------------
                retryable = (
                    "429" in error_text
                    or "503" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                    or "UNAVAILABLE" in error_text
                )

                if retryable:

                    wait_time = (attempt + 1) * 5

                    print(
                        f"[Retry] {model_name} failed. "
                        f"Waiting {wait_time}s..."
                    )

                    time.sleep(wait_time)

                    continue

                else:
                    break

    return None

# =========================================================
# LOAD ENV
# =========================================================
load_dotenv()

GEMINI_API_KEY = os.getenv("Gemini_API_Key")

if not GEMINI_API_KEY:
    raise ValueError("Gemini_API_Key missing in .env")


# =========================================================
# GEMINI CLIENT
# =========================================================
client = genai.Client(api_key=GEMINI_API_KEY)


# =========================================================
# GENERATE RESPONSE
# =========================================================
def generate_response(messages):

    full_text = ""

    for msg in messages:
        if msg["role"] == "user":
            full_text += " " + msg["content"]


    # -----------------------------------------------------
    # OFF TOPIC
    # -----------------------------------------------------
    if is_off_topic(full_text):

        return {
            "reply": "I can only help with SHL assessment recommendations.",
            "recommendations": [],
            "end_of_conversation": False
        }


    # -----------------------------------------------------
    # CLARIFICATION ENGINE
    # -----------------------------------------------------
    clarification = get_clarification_question(messages)

    if clarification:

        return {
            "reply": clarification,
            "recommendations": [],
            "end_of_conversation": False
        }


    # -----------------------------------------------------
    # SYNTHESIZED QUERY
    # -----------------------------------------------------
    synthesized_query = build_conversation_query(messages)


    # -----------------------------------------------------
    # RETRIEVE
    # -----------------------------------------------------
    results = search(synthesized_query, k=25)


    # -----------------------------------------------------
    # PROMPT
    # -----------------------------------------------------
    prompt = build_prompt(
        messages,
        synthesized_query,
        results
    )


    # -----------------------------------------------------
    # GEMINI GENERATION
    # -----------------------------------------------------
    try:

        generated_text = generate_gemini_response(prompt)

        if not generated_text:

            return {
                "reply": (
                    "Temporary AI service overload. "
                    "Please try again shortly."
                ),
                "recommendations": [],
                "end_of_conversation": False
            }

        text = generated_text.strip()

        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

        parsed = json.loads(text)


        # =================================================
        # SAFE RECOMMENDATION MAPPING
        # =================================================
        safe_recommendations = []

        retrieved_lookup = {}

        for r in results:
            retrieved_lookup[r["name"].lower()] = r


        for rec in parsed.get("recommendations", []):

            rec_name = rec.get("name", "").lower()

            if rec_name in retrieved_lookup:

                item = retrieved_lookup[rec_name]

                safe_recommendations.append({
                    "name": item["name"],
                    "url": item["url"],
                    "test_type": item["test_type"]
                })


        # =================================================
        # LIMIT RECOMMENDATIONS
        # =================================================
        safe_recommendations = safe_recommendations[:5]


        # =================================================
        # END OF CONVERSATION LOGIC
        # =================================================
        end_of_conversation = False

        if len(safe_recommendations) > 0:
            end_of_conversation = True


        # =================================================
        # FINAL RESPONSE
        # =================================================
        return {
            "reply": parsed.get(
                "reply",
                "Here are recommended assessments."
            ),

            "recommendations": safe_recommendations,

            "end_of_conversation": end_of_conversation
        }


    except Exception as e:

        fallback_recommendations = []

        for r in results[:3]:

            fallback_recommendations.append({
                "name": r["name"],
                "url": r["url"],
                "test_type": r["test_type"]
            })

        return {
            "reply": (
                "AI reasoning temporarily unavailable. "
                "Here are the top retrieved SHL assessments."
            ),
            "recommendations": fallback_recommendations,
            "end_of_conversation": True
        }
