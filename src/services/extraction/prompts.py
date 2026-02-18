"""Extraction prompt templates.

Builds system + user message pairs that instruct the LLM to return
structured JSON matching the ExtractedIntelligence schema exactly.
Includes explicit field definitions, examples, negative examples,
and confidence-score guidance.
"""

from __future__ import annotations

EXTRACTION_SCHEMA_DESCRIPTION = (
    "You MUST return a single JSON object (no markdown, no code fences) "
    "with EXACTLY the following fields:\n"
    "\n"
    "{\n"
    '  "holding": "<string> What the court decided.",\n'
    '  "holding_confidence": <float 0.0-1.0>,\n'
    '  "legal_standard": "<string|null> e.g. strict scrutiny, '
    'rational basis. null if unknown.",\n'
    '  "disposition": "<string> One of: affirmed, reversed, '
    "remanded, vacated, dismissed, affirmed_in_part, "
    'reversed_in_part.",\n'
    '  "disposition_confidence": <float 0.0-1.0>,\n'
    '  "key_authorities": [\n'
    "    {\n"
    '      "citation_string": "<string> Full legal citation, '
    "e.g. '554 U.S. 570 (2008)'\",\n"
    '      "case_name": "<string|null> Case name if '
    'identifiable",\n'
    '      "citation_context": "<string> How this authority was '
    "used: 'followed because...', 'distinguished on the grounds "
    "that...', 'overruled...'\",\n"
    '      "citation_type": "<string> One of: followed, '
    'distinguished, overruled, cited.",\n'
    '      "paragraph_context": "<string> The paragraph where '
    'this citation appears. Copy verbatim."\n'
    "    }\n"
    "  ],\n"
    '  "dissent_present": <boolean>,\n'
    '  "dissent_summary": "<string|null> Summary of the dissent. '
    'null if none.",\n'
    '  "concurrence_present": <boolean>,\n'
    '  "concurrence_summary": "<string|null> Summary of the '
    'concurrence. null if none.",\n'
    '  "legal_topics": ["<string>", ...] At least one legal '
    "topic.\n"
    "}"
)

NEGATIVE_EXAMPLES = """\
IMPORTANT edge cases — follow these rules:
- If the opinion does NOT clearly state a holding, set "holding" to \
"No clear holding identified" and "holding_confidence" to 0.0.
- If you cannot determine the disposition, use "dismissed" with \
"disposition_confidence" of 0.0.
- If there are no cited authorities, return an empty "key_authorities" list.
- If you are unsure about a confidence score, err on the side of LOWER values. \
A 0.4 is better than a hallucinated 0.9.
- Do NOT invent citations. Only include authorities explicitly mentioned in the text.
- Do NOT guess case names — leave as null if not stated in the text.
- "legal_topics" must contain at least one entry. Use general categories like \
"constitutional law", "criminal procedure", "civil rights" if specific topics \
are unclear.\
"""

SYSTEM_PROMPT_TEMPLATE = """\
You are a legal intelligence extraction system. Your task is to read a court \
opinion and extract structured information as JSON.

{schema}

{negative_examples}

You are analyzing opinions from real courts. Accuracy is paramount — false \
precision is worse than honest uncertainty. When in doubt, lower your \
confidence scores.

CRITICAL: Return ONLY valid JSON. No explanation, no markdown fences, no \
additional text outside the JSON object.\
"""

USER_PROMPT_TEMPLATE = """\
Extract structured intelligence from the following court opinion.

Case: {case_name}
Court: {court}

--- BEGIN OPINION TEXT ---
{opinion_text}
--- END OPINION TEXT ---

Return the JSON extraction now.\
"""

CORRECTIVE_PROMPT_TEMPLATE = """\
Your previous extraction attempt was invalid. The specific error was:

{error_message}

Please fix the extraction and return valid JSON matching the schema exactly. \
Remember:
- All confidence scores must be between 0.0 and 1.0.
- "disposition" must be one of: affirmed, reversed, remanded, vacated, \
dismissed, affirmed_in_part, reversed_in_part.
- "citation_type" must be one of: followed, distinguished, overruled, cited.
- "legal_topics" must contain at least one entry.
- Return ONLY valid JSON — no markdown, no explanation.

Here was your previous (invalid) output:
{previous_output}

Return the corrected JSON now.\
"""


def build_extraction_prompt(
    opinion_text: str,
    case_name: str,
    court: str,
) -> tuple[str, str]:
    """Build system + user messages for the extraction LLM call.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        schema=EXTRACTION_SCHEMA_DESCRIPTION,
        negative_examples=NEGATIVE_EXAMPLES,
    )
    user_prompt = USER_PROMPT_TEMPLATE.format(
        case_name=case_name,
        court=court,
        opinion_text=opinion_text,
    )
    return system_prompt, user_prompt


def build_corrective_prompt(
    error_message: str,
    previous_output: str,
) -> str:
    """Build a corrective follow-up prompt after a validation failure."""
    return CORRECTIVE_PROMPT_TEMPLATE.format(
        error_message=error_message,
        previous_output=previous_output,
    )
