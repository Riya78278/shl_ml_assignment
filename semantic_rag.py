import os
import json
import asyncio
from dotenv import load_dotenv
from google import genai

from conversation import (
    build_conversation_query,
    get_clarification_question,
    is_off_topic
)

from retriever import search
from prompts import build_prompt

# =========================================================
# LOAD ENV & CLIENT
# =========================================================
load_dotenv()
GEMINI_API_KEY = os.getenv("Gemini_API_Key")

if not GEMINI_API_KEY:
    raise ValueError("Gemini_API_Key missing in .env")

client = genai.Client(api_key=GEMINI_API_KEY)

# =========================================================
# GEMINI RESPONSE WITH RETRIES (Now Async)
# =========================================================
async def generate_gemini_response(prompt):
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
                # Use the asynchronous client wrapper
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text

            except Exception as e:
                error_text = str(e)
                retryable = (
                    "429" in error_text
                    or "503" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                    or "UNAVAILABLE" in error_text
                )

                if retryable:
                    # Massively reduced sleep time. A 5s-10s sleep was killing your response time.
                    wait_time = 1 
                    print(f"[Retry] {model_name} failed. Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    break

    return None

# =========================================================
# GENERATE RESPONSE (Now Async)
# =========================================================
async def generate_response(messages):
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
    # RETRIEVE (Await the async search)
    # -----------------------------------------------------
    results = await search(synthesized_query, k=25)

    # -----------------------------------------------------
    # PROMPT
    # -----------------------------------------------------
    prompt = build_prompt(messages, synthesized_query, results)

    # -----------------------------------------------------
    # GEMINI GENERATION
    # -----------------------------------------------------
    try:
        # Await the async generation
        generated_text = await generate_gemini_response(prompt)

        if not generated_text:
            return {
                "reply": "Temporary AI service overload. Please try again shortly.",
                "recommendations": [],
                "end_of_conversation": False
            }

        text = generated_text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)

        # SAFE RECOMMENDATION MAPPING
        safe_recommendations = []
        retrieved_lookup = {r["name"].lower(): r for r in results}

        for rec in parsed.get("recommendations", []):
            rec_name = rec.get("name", "").lower()
            if rec_name in retrieved_lookup:
                item = retrieved_lookup[rec_name]
                safe_recommendations.append({
                    "name": item["name"],
                    "url": item["url"],
                    "test_type": item["test_type"]
                })

        safe_recommendations = safe_recommendations[:5]
        end_of_conversation = len(safe_recommendations) > 0

        return {
            "reply": parsed.get("reply", "Here are recommended assessments."),
            "recommendations": safe_recommendations,
            "end_of_conversation": end_of_conversation
        }

    except Exception as e:
        fallback_recommendations = [
            {"name": r["name"], "url": r["url"], "test_type": r["test_type"]}
            for r in results[:3]
        ]
        return {
            "reply": "AI reasoning temporarily unavailable. Here are the top retrieved SHL assessments.",
            "recommendations": fallback_recommendations,
            "end_of_conversation": True
        }
