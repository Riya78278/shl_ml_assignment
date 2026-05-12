import json

# =========================================================
# LOAD RAW CATALOG
# =========================================================
with open(
    "data/catalog.json",
    "r",
    encoding="utf-8"
) as f:

    raw = json.load(f)

normalized = []

# =========================================================
# HELPER
# =========================================================
def derive_test_type(keys):

    keys_text = " ".join(keys).lower()

    # -----------------------------------------------------
    # TECHNICAL
    # -----------------------------------------------------
    if (
        "knowledge & skills" in keys_text
        or "technical" in keys_text
    ):
        return "Technical"

    # -----------------------------------------------------
    # PERSONALITY
    # -----------------------------------------------------
    if (
        "personality & behavior" in keys_text
        or "competencies" in keys_text
    ):
        return "Personality"

    # -----------------------------------------------------
    # COGNITIVE
    # -----------------------------------------------------
    if (
        "ability & aptitude" in keys_text
    ):
        return "Cognitive"

    # -----------------------------------------------------
    # SIMULATION
    # -----------------------------------------------------
    if (
        "assessment exercises" in keys_text
        or "situational judgment" in keys_text
    ):
        return "Simulation"

    return "General"

# =========================================================
# NORMALIZE
# =========================================================
for item in raw:

    keys = item.get("keys", [])

    normalized.append({

        "name": item.get("name", ""),

        "description": (
            item.get("description", "")
        ),

        "url": (
            item.get("link", "")
        ),

        "test_type": derive_test_type(keys),

        "keys": keys,

        "job_levels": item.get(
            "job_levels",
            []
        ),

        "remote": item.get(
            "remote",
            ""
        ),

        "adaptive": item.get(
            "adaptive",
            ""
        )
    })

# =========================================================
# SAVE
# =========================================================
with open(
    "data/normalized_assessments.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        normalized,
        f,
        indent=2,
        ensure_ascii=False
    )

print("Normalization complete.")