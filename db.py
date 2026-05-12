import sqlite3
import json

# =========================================================
# CONNECT DATABASE
# =========================================================
conn = sqlite3.connect("data/shl.db")

cursor = conn.cursor()

# =========================================================
# CREATE TABLE
# =========================================================
cursor.execute("""
CREATE TABLE IF NOT EXISTS assessments (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    name TEXT,
    description TEXT,
    url TEXT,
    test_type TEXT,

    keys TEXT,
    job_levels TEXT,

    remote TEXT,
    adaptive TEXT
)
""")

# =========================================================
# CLEAR OLD DATA
# =========================================================
cursor.execute("DELETE FROM assessments")

# =========================================================
# LOAD NORMALIZED DATA
# =========================================================
with open(
    "data/normalized_assessments.json",
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)

# =========================================================
# INSERT DATA
# =========================================================
for item in data:

    cursor.execute(
        """
        INSERT INTO assessments (

            name,
            description,
            url,
            test_type,

            keys,
            job_levels,

            remote,
            adaptive

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,

        (
            item.get("name", ""),

            item.get("description", ""),

            item.get("url", ""),

            item.get("test_type", "Unknown"),

            json.dumps(
                item.get("keys", [])
            ),

            json.dumps(
                item.get("job_levels", [])
            ),

            item.get("remote", ""),

            item.get("adaptive", "")
        )
    )

# =========================================================
# COMMIT
# =========================================================
conn.commit()

conn.close()

print("SQLite DB created successfully.")