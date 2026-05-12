OFF_TOPIC_TERMS = [
    "fire employee",
    "lawsuit",
    "medical advice",
    "bitcoin",
    "stocks",
    "crypto",
    "ignore previous instructions",
    "hack bank",
    "aws certification"
]


TECH_KEYWORDS = [
    "java",
    "python",
    "react",
    "frontend",
    "backend",
    "node",
    "django",
    "sql",
    "cloud",
    "aws",
    "sales",
    "marketing",
    "support",
    "finance"
]


PERSONALITY_KEYWORDS = [
    "personality",
    "teamwork",
    "communication",
    "leadership",
    "behavior",
    "collaboration"
]


COGNITIVE_KEYWORDS = [
    "cognitive",
    "aptitude",
    "reasoning",
    "problem solving"
]


SENIORITY_KEYWORDS = [
    "intern",
    "fresher",
    "entry",
    "mid",
    "senior",
    "lead"
]


def is_off_topic(text):

    text = text.lower()

    for term in OFF_TOPIC_TERMS:
        if term in text:
            return True

    return False


# =========================================================
# EXTRACT CONVERSATION STATE
# =========================================================
def extract_state(messages):

    state = {
        "skills": [],
        "seniority": None,
        "personality": False,
        "cognitive": False,
        "role_text": "",
        "teamwork": False,
        "communication": False,
        "remote": False
    }

    for msg in messages:

        if msg["role"] != "user":
            continue

        text = msg["content"].lower()

        state["role_text"] += " " + text

        for skill in TECH_KEYWORDS:
            if skill in text and skill not in state["skills"]:
                state["skills"].append(skill)

        for word in PERSONALITY_KEYWORDS:
            if word in text:
                state["personality"] = True

        for word in COGNITIVE_KEYWORDS:
            if word in text:
                state["cognitive"] = True

        if "teamwork" in text:
            state["teamwork"] = True

        if "communication" in text:
            state["communication"] = True

        if "remote" in text:
            state["remote"] = True

        if "intern" in text:
            state["seniority"] = "intern"

        elif "fresher" in text:
            state["seniority"] = "fresher"

        elif "entry" in text:
            state["seniority"] = "entry-level"

        elif "mid" in text:
            state["seniority"] = "mid-level"

        elif "senior" in text:
            state["seniority"] = "senior"

    return state


# =========================================================
# CLARIFICATION ENGINE
# =========================================================
def get_clarification_question(messages):

    state = extract_state(messages)

    if len(state["skills"]) == 0:
        return "What role or technical skills are you hiring for?"

    if not state["seniority"]:
        return "What seniority level are you hiring for?"

    if not state["personality"] and not state["cognitive"]:
        return (
            "Do you also want personality or cognitive assessments "
            "such as teamwork, communication, leadership, or reasoning skills?"
        )

    return None


# =========================================================
# BUILD SYNTHESIZED QUERY
# =========================================================
def build_conversation_query(messages):

    state = extract_state(messages)

    query = f"""
    Hiring Context:
    {state['role_text']}

    Skills:
    {' '.join(state['skills'])}

    Seniority:
    {state['seniority']}

    Personality Required:
    {state['personality']}

    Cognitive Required:
    {state['cognitive']}

    Teamwork:
    {state['teamwork']}

    Communication:
    {state['communication']}

    Remote:
    {state['remote']}
    """

    return query.strip()