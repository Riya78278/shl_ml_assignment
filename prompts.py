import json


def build_prompt(messages, query, results):

    context = ""

    for i, r in enumerate(results, 1):

        context += f"""
[{i}]
Name: {r['name']}
Description: {r['description'][:250]}
URL: {r['url']}
Test Type: {r['test_type']}
"""

    prompt = f"""
You are an SHL assessment recommendation assistant.

You ONLY recommend SHL assessments.

You NEVER hallucinate assessments.

You support:
- clarification
- refinement
- comparison
- conversational updates

Conversation:
{json.dumps(messages, indent=2)}

Synthesized Query:
{query}

Retrieved SHL Assessments:
{context}

TASKS:
1. Recommend relevant SHL assessments
2. Support conversational refinement
3. Compare assessments if requested
4. Refuse off-topic requests
5. Stay grounded ONLY in context

Rules:-
RETURN STRICT JSON ONLY.
If no suitable personality assessment is retrieved,
DO NOT mention personality assessments generically.
Only discuss retrieved assessments.

Example:

{{
  "reply": "Here are recommended assessments.",
  "recommendations": [
    {{
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/...",
      "test_type": "K"
    }}
  ],
  "end_of_conversation": false
}}
"""

    return prompt