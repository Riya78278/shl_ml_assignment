import json
import numpy as np
import faiss

# =========================================================
# GLOBALS
# =========================================================
embed_model = None
index = None
DATA = None
initialized = False

def initialize_retriever():

    global embed_model
    global index
    global DATA
    global initialized

    if initialized:
        return

    print("Initializing retriever...")

    from sentence_transformers import SentenceTransformer

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------
    with open(
        "data/normalized_assessments.json",
        "r",
        encoding="utf-8"
    ) as f:

        DATA = json.load(f)

    print(f"Loaded {len(DATA)} assessments.")

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------
    embed_model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device="cpu"
    )

    print("Embedding model loaded.")

    # -----------------------------------------------------
    # LOAD FAISS
    # -----------------------------------------------------
    index = faiss.read_index(
        "data/faiss.index"
    )

    print("FAISS loaded.")

    initialized = True

    print("Retriever initialized.")
    
# =========================================================
# CLEAN FUNCTION
# =========================================================
def clean(text):

    if not text:
        return ""

    return (
        text.replace("\n", " ")
        .replace("Login", "")
        .replace("Careers", "")
        .replace("Contact", "")
        .replace("Practice Tests", "")
        .replace("Buy Online", "")
        .strip()
    )

# =========================================================
# PREPARE TEXTS
# =========================================================
texts = []

for item in DATA:

    combined = (
        item.get("name", "")
        + " "
        + clean(item.get("description", ""))[:400]
        + " "
        + item.get("test_type", "")
        + " "
        + " ".join(item.get("keys", []))
    )

    texts.append(combined)

# =========================================================
# HELPERS
# =========================================================
TECH_TERMS = [
    "java",
    "python",
    "react",
    "frontend",
    "backend",
    "sql",
    "cloud",
    "coding",
    "programming",
    "developer"
]

PERSONALITY_TERMS = [
    "personality",
    "teamwork",
    "leadership",
    "behavior",
    "collaboration",
    "opq",
    "communication"
]

COGNITIVE_TERMS = [
    "cognitive",
    "reasoning",
    "aptitude",
    "problem solving",
    "inductive"
]

REPORT_TERMS = [
    "report",
    "profiling guide",
    "participant report",
    "manager report",
    "development report",
    "action planner"
]

ASSESSMENT_TERMS = [
    "assessment",
    "coding",
    "simulation",
    "technical",
    "verify",
    "core java",
    "live coding"
]

# =========================================================
# CLASSIFICATION
# =========================================================
def classify_result(text):

    text = text.lower()

    if any(term in text for term in TECH_TERMS):
        return "technical"

    if any(term in text for term in PERSONALITY_TERMS):
        return "personality"

    if any(term in text for term in COGNITIVE_TERMS):
        return "cognitive"

    return "other"

# =========================================================
# SEARCH
# =========================================================
def search(query, k=30):
    
    if not initialized:
        initialize_retriever()
    
    print("\n=================================================")
    print("SEARCH QUERY:")
    print(query)
    print("=================================================\n")

    query_lower = query.lower()

    # =====================================================
    # EMBED QUERY
    # =====================================================
    query_vec = embed_model.encode([query]).astype("float32")

    distances, indices = index.search(query_vec, k)

    raw_results = []

    # =====================================================
    # BUILD RAW RESULTS
    # =====================================================
    for distance, idx in zip(distances[0], indices[0]):

        item = DATA[idx]

        description = clean(
            item.get("description", "")
        )

        if not description:
            continue

        raw_results.append({
            "name": item.get("name", "Unknown"),
            "description": description,
            "url": item.get("url", ""),
            "test_type": item.get("test_type", "Unknown"),
            "semantic_distance": float(distance),
            "keys": item.get("keys", []),
            "job_levels": item.get("job_levels", [])
        })

    print("=== RAW RETRIEVAL RESULTS ===")

    for r in raw_results:
        print("-", r["name"])

    # =====================================================
    # HYBRID RERANKING
    # =====================================================
    scored_results = []

    for r in raw_results:
        
        score = 0

        text = (
            r["name"]
            + " "
            + r["description"]
            + " "
            + r["test_type"]
            + " "
            + " ".join(r.get("keys", []))
        ).lower()
        
        excluded_terms = [
            "hiring concepts",
            "profiling guide",
            "development report",
            "participant report",
            "manager report",
            "selection report",
            "action planner",
            "assessment report",
            "hipo assessment report",
            "development guide",
            "interview guide"
        ]

        if any(term in text for term in excluded_terms):
            continue

        # -------------------------------------------------
        # TECHNICAL BOOSTS
        # -------------------------------------------------
        if "java" in query_lower and "java" in text:
            score += 10

        if "backend" in query_lower and "backend" in text:
            score += 6

        if "python" in query_lower and "python" in text:
            score += 10

        if "react" in query_lower and "react" in text:
            score += 10

        if "sql" in query_lower and "sql" in text:
            score += 5

        # -------------------------------------------------
        # SENIORITY BOOST
        # -------------------------------------------------
        if "mid-level" in query_lower:

            if any(
                level.lower() == "mid-professional"
                for level in r.get("job_levels", [])
            ):
                score += 5
        
        # -------------------------------------------------
        # PERSONALITY BOOSTS
        # -------------------------------------------------
        if "teamwork" in query_lower:

            if any(
                word in text
                for word in [
                    "teamwork",
                    "collaboration",
                    "opq",
                    "behavior",
                    "personality"
                ]
            ):
                score += 10

        if "communication" in query_lower:

            if any(
                word in text
                for word in [
                    "communication",
                    "personality",
                    "behavior"
                ]
            ):
                score += 8

        if "leadership" in query_lower:

            if any(
                word in text
                for word in [
                    "leadership",
                    "management",
                    "opq"
                ]
            ):
                score += 8

        # -------------------------------------------------
        # COGNITIVE BOOSTS
        # -------------------------------------------------
        if any(
            word in query_lower
            for word in [
                "cognitive",
                "reasoning",
                "aptitude"
            ]
        ):

            if any(
                word in text
                for word in [
                    "cognitive",
                    "reasoning",
                    "aptitude",
                    "inductive"
                ]
            ):
                score += 10

        # -------------------------------------------------
        # ASSESSMENT BOOST
        # -------------------------------------------------
        if any(term in text for term in ASSESSMENT_TERMS):
            score += 4

        # -------------------------------------------------
        # REPORT PENALTY
        # -------------------------------------------------
        if any(term in text for term in REPORT_TERMS):
            score -= 6

        # -------------------------------------------------
        # SEMANTIC BONUS
        # -------------------------------------------------
        semantic_bonus = max(
            0,
            3 - (r["semantic_distance"] * 0.1)
        )

        final_score = score + semantic_bonus

        scored_results.append({
            "score": round(final_score, 2),
            "category": classify_result(text),
            "result": r
        })

    # =====================================================
    # SORT RESULTS
    # =====================================================
    scored_results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # =====================================================
    # CATEGORY BALANCING
    # =====================================================
    technical = []
    personality = []
    cognitive = []
    other = []

    for item in scored_results:

        category = item["category"]

        if category == "technical":
            technical.append(item["result"])

        elif category == "personality":
            personality.append(item["result"])

        elif category == "cognitive":
            cognitive.append(item["result"])

        else:
            other.append(item["result"])

    final_results = []

    # -----------------------------------------------------
    # ENSURE DIVERSITY
    # -----------------------------------------------------
    if technical:
        final_results.append(technical[0])

    if personality:
        final_results.append(personality[0])

    if cognitive:
        final_results.append(cognitive[0])

    # -----------------------------------------------------
    # FILL REMAINING
    # -----------------------------------------------------
    combined = (
        technical[1:]
        + personality[1:]
        + cognitive[1:]
        + other
    )

    seen = set()

    for r in final_results:
        seen.add(r["name"].lower())

    for r in combined:

        key = r["name"].lower()

        if key not in seen:

            seen.add(key)

            final_results.append(r)

    # =====================================================
    # FINAL DEBUG
    # =====================================================
    print("\n=== FINAL RESULTS ===")

    for r in final_results[:5]:
        print("-", r["name"])

    print()

    return final_results[:5]
