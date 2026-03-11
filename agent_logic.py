
# agent_logic.py
# Handles all Gemini API interaction:
#   - Prompt construction (transcript + filtered JSON + definitions)
#   - API call with retry/error handling
#   - Robust multi-strategy JSON response parsing

import json
import time
import re
import google.generativeai as genai
from definitions import PARAMETER_DEFINITIONS


# ---------------------------------------------------------------------------
# Prompt Engineering
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """You are a strict QA Auditor. Your job is to verify whether
AI-extracted call data matches the actual content of the call transcript.
You must be objective, precise, and only base your verdict on what is
explicitly said in the transcript — never assume or infer.

IMPORTANT: You now have THREE verdict options:
- Correct: The extracted value matches the transcript AND the context/intent is right.
- Wrong: The value is missing, hallucinated, or completely incorrect.
- Context Mismatch: The value EXISTS in the transcript but refers to a DIFFERENT
  product, condition, person, or context than what was intended in the extraction.
  Example: buyer asked price for empty bottles, seller quoted price for filled bottles.
  The number '70' is in the transcript, but the context is wrong.
"""


def _build_definitions_block() -> str:
    lines = []
    for param, definition in PARAMETER_DEFINITIONS.items():
        lines.append(f"  - **{param}**: {definition}")
    return "\n".join(lines)


def build_prompt(transcript: str, filtered_json: dict) -> str:
    """
    Construct the full QA audit prompt for Gemini.

    Args:
        transcript: The full conversation transcript text.
        filtered_json: Dict containing only the 12 target parameters.

    Returns:
        A formatted prompt string.
    """
    definitions_block = _build_definitions_block()
    json_str = json.dumps(filtered_json, indent=2, ensure_ascii=False)

    prompt = f"""
## ROLE
You are a strict QA Auditor verifying AI-extracted call data against the original transcript.

## PARAMETER DEFINITIONS
Use these official definitions to evaluate each parameter:
{definitions_block}

## CALL TRANSCRIPT
{transcript}

## EXTRACTED JSON DATA (only the 12 target parameters)
{json_str}

## YOUR TASK
For EACH parameter present in the Extracted JSON above, carefully read the transcript
and determine if the extracted value matches what was actually said.

## OUTPUT FORMAT — CRITICAL RULES
You MUST respond with a VALID JSON array only. No markdown fences, no explanation text.

Each element must have EXACTLY these 5 fields:
1. "parameter"          — the parameter name (lowercase with underscores)
2. "extracted_value"    — the value from the Extracted JSON, as a plain string
3. "transcript_context" — a SHORT excerpt (max 120 chars) from the transcript relevant
                           to this parameter. Use SINGLE QUOTES for any dialogue within
                           this field. NEVER use double quotes inside a field value.
4. "verdict"            — EXACTLY one of these three strings:
                           "Correct"          — value matches and context is right
                           "Wrong"            — value is missing, hallucinated, or wrong
                           "Context Mismatch" — value exists in transcript but wrong context
5. "reason"             — one concise sentence. No double quotes inside.

JSON SAFETY RULES:
- Use ONLY single quotes (') when quoting dialogue inside any field value.
- Do NOT use double quotes (") anywhere inside a field's string value.
- Keep transcript_context under 120 characters.
- Keep reason under 150 characters.

Start your response with [ and end with ]. Return ONLY the JSON array.
"""
    return prompt.strip()


# ---------------------------------------------------------------------------
# Gemini API Call
# ---------------------------------------------------------------------------

def configure_gemini(api_key: str):
    """Configure the Gemini SDK with the provided API key."""
    genai.configure(api_key=api_key)


def call_gemini(
    api_key: str,
    transcript: str,
    filtered_json: dict,
    model_name: str = "gemini-2.5-flash",
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> list[dict]:
    """
    Call the Gemini API and return parsed QA results.

    Args:
        api_key: Gemini API key string.
        transcript: Full call transcript text.
        filtered_json: Dict of the 12 parameters (already filtered).
        model_name: Gemini model to use.
        max_retries: Number of retries on transient failures.
        retry_delay: Seconds to wait between retries.

    Returns:
        List of dicts, each containing:
          {parameter, extracted_value, transcript_context, verdict, reason}

    Raises:
        RuntimeError on unrecoverable API errors.
    """
    configure_gemini(api_key)

    generation_config = genai.GenerationConfig(
        temperature=0.1,               # Low temperature for deterministic QA
        max_output_tokens=8192,        # Enough for 12 params with context
        response_mime_type="application/json",  # Forces Gemini to output valid JSON
    )

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config=generation_config,
    )

    prompt = build_prompt(transcript, filtered_json)
    raw_text = ""

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(prompt)
            raw_text = response.text.strip()
            parsed = _parse_response(raw_text)
            return parsed

        except (json.JSONDecodeError, ValueError) as parse_err:
            last_error = parse_err
            if attempt < max_retries:
                # Parse errors are worth retrying — model might do better next time
                time.sleep(retry_delay)
                continue
            raise RuntimeError(
                f"JSON parse failed after {max_retries} attempts: {parse_err}\n"
                f"Raw response (first 800 chars):\n{raw_text[:800]}"
            )

        except Exception as api_err:
            last_error = api_err
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
            else:
                raise RuntimeError(
                    f"Gemini API call failed after {max_retries} attempts. "
                    f"Last error: {api_err}"
                )

    raise RuntimeError(f"Unexpected exit from retry loop. Last error: {last_error}")


# ---------------------------------------------------------------------------
# Response Parsing — Multi-Strategy Robust Parser
# ---------------------------------------------------------------------------

def _clean_raw_text(text: str) -> str:
    """Strip markdown fences and whitespace from the raw response."""
    text = text.strip()
    # Remove opening fence: ```json, ```JSON, ``` etc.
    text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
    # Remove closing fence
    text = re.sub(r"\n?```\s*$", "", text.strip())
    return text.strip()


def _extract_array_text(text: str) -> str:
    """Find the outermost [...] array in the text."""
    start = text.find("[")
    if start == -1:
        raise ValueError("No JSON array '[' found in response.")
    # Find matching closing bracket
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    # No proper closing bracket — return everything from '[' to end (truncated)
    return text[start:]


def _fix_unescaped_quotes(json_text: str) -> str:
    """
    Attempt to fix unescaped double-quotes inside JSON string values.
    Strategy: scan through the JSON character by character.
    When inside a string, replace any unescaped " that is not the closing
    quote with a single quote character '
    """
    result = []
    in_string = False
    escape_next = False

    for i, ch in enumerate(json_text):
        if escape_next:
            result.append(ch)
            escape_next = False
            continue

        if ch == "\\":
            result.append(ch)
            escape_next = True
            continue

        if ch == '"':
            if in_string:
                # Could be end of string OR an unescaped quote inside.
                # Peek at the next non-space char: if it's , : ] } then it's end
                rest = json_text[i + 1:].lstrip()
                if rest and rest[0] in (',', '}', ']', ':'):
                    in_string = False
                    result.append(ch)
                else:
                    # Unescaped quote inside string — replace with single quote
                    result.append("'")
            else:
                in_string = True
                result.append(ch)
        else:
            result.append(ch)

    return "".join(result)


def _try_close_truncated_json(json_text: str) -> str:
    """
    If JSON is truncated (no closing ]), try to salvage complete objects
    by finding the last complete } and appending ]].
    """
    # Find the last complete object ending
    last_obj_end = json_text.rfind("}")
    if last_obj_end == -1:
        return json_text

    truncated = json_text[: last_obj_end + 1]
    # Count unclosed [ characters and close them
    opens = truncated.count("[") - truncated.count("]")
    closes = "]" * max(0, opens)
    return truncated + closes


def _extract_objects_by_regex(text: str) -> list[dict]:
    """
    Last-resort extraction: use regex to pull out individual JSON objects.
    Returns only objects with at least a 'parameter' and 'verdict' field.
    """
    # Match outermost {...} blocks (handles simple nesting)
    pattern = re.compile(r'\{[^{}]*\}', re.DOTALL)
    objects = []
    for match in pattern.finditer(text):
        raw = match.group(0)
        # Try direct parse
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "parameter" in obj:
                objects.append(obj)
            continue
        except json.JSONDecodeError:
            pass
        # Try with quote fix
        try:
            obj = json.loads(_fix_unescaped_quotes(raw))
            if isinstance(obj, dict) and "parameter" in obj:
                objects.append(obj)
        except json.JSONDecodeError:
            pass
    return objects


def _parse_response(raw_text: str) -> list[dict]:
    """
    Multi-strategy robust parser for Gemini JSON array responses.

    Strategy order:
      1. Direct json.loads (fastest, works when Gemini behaves)
      2. Strip markdown fences + json.loads
      3. Extract [...] array text + json.loads
      4. Fix unescaped double quotes + json.loads
      5. Close truncated JSON + json.loads
      6. Combine quote fix + truncation fix
      7. Last resort: regex extraction of individual objects
    """
    if not raw_text:
        raise ValueError("Empty response from Gemini.")

    attempts = []

    # ── Strategy 1: Direct parse ──────────────────────────────────────────────
    try:
        result = json.loads(raw_text)
        if isinstance(result, list):
            return _normalize_results(result)
    except json.JSONDecodeError as e:
        attempts.append(f"S1 direct: {e}")

    # ── Strategy 2: Strip markdown fences ────────────────────────────────────
    clean = _clean_raw_text(raw_text)
    try:
        result = json.loads(clean)
        if isinstance(result, list):
            return _normalize_results(result)
    except json.JSONDecodeError as e:
        attempts.append(f"S2 strip-fences: {e}")

    # ── Strategy 3: Extract [...] array text ─────────────────────────────────
    try:
        array_text = _extract_array_text(clean)
        result = json.loads(array_text)
        if isinstance(result, list):
            return _normalize_results(result)
    except (json.JSONDecodeError, ValueError) as e:
        attempts.append(f"S3 extract-array: {e}")
        array_text = clean  # fall back for next strategies

    # ── Strategy 4: Fix unescaped double quotes ───────────────────────────────
    try:
        fixed = _fix_unescaped_quotes(array_text)
        result = json.loads(fixed)
        if isinstance(result, list):
            return _normalize_results(result)
    except json.JSONDecodeError as e:
        attempts.append(f"S4 quote-fix: {e}")

    # ── Strategy 5: Close truncated JSON ─────────────────────────────────────
    try:
        closed = _try_close_truncated_json(array_text)
        result = json.loads(closed)
        if isinstance(result, list):
            return _normalize_results(result)
    except json.JSONDecodeError as e:
        attempts.append(f"S5 close-truncated: {e}")

    # ── Strategy 6: Quote fix + truncation fix combined ───────────────────────
    try:
        combined = _try_close_truncated_json(_fix_unescaped_quotes(array_text))
        result = json.loads(combined)
        if isinstance(result, list):
            return _normalize_results(result)
    except json.JSONDecodeError as e:
        attempts.append(f"S6 combined: {e}")

    # ── Strategy 7: Regex extraction of individual objects ───────────────────
    objects = _extract_objects_by_regex(raw_text)
    if objects:
        return _normalize_results(objects)

    # All strategies failed
    raise json.JSONDecodeError(
        f"All 7 parse strategies failed.\nAttempts:\n" + "\n".join(attempts),
        raw_text, 0
    )


def _normalize_results(raw_list: list) -> list[dict]:
    """
    Normalize each item into the expected dict shape.
    Handles 3 verdicts: Correct, Wrong, Context Mismatch.
    """
    required_fields = ["parameter", "extracted_value", "transcript_context", "verdict", "reason"]
    results = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        item_lower = {k.lower().replace(" ", "_"): v for k, v in item.items()}
        normalized = {}
        for field in required_fields:
            normalized[field] = str(item_lower.get(field, "N/A")).strip()

        # Enforce one of the 3 allowed verdict values
        v = normalized["verdict"].lower()
        if "context" in v and "mismatch" in v:
            normalized["verdict"] = "Context Mismatch"
        elif "correct" in v:
            normalized["verdict"] = "Correct"
        else:
            normalized["verdict"] = "Wrong"

        results.append(normalized)
    return results


# ---------------------------------------------------------------------------
# Convenience wrapper for Streamlit
# ---------------------------------------------------------------------------

def process_file_id(
    api_key: str,
    file_id: str,
    mcat_id: str,
    transcript: str,
    filtered_json: dict,
) -> tuple[list[dict], str | None]:
    """
    High-level wrapper called by the Streamlit UI.
    Returns (results_list, error_message).
    """
    if not filtered_json:
        return [], f"file_id {file_id}: No parameters found in Extracted JSON."

    try:
        results = call_gemini(
            api_key=api_key,
            transcript=transcript,
            filtered_json=filtered_json,
        )
        for row in results:
            row["file_id"] = file_id
            row["mcat_id"] = mcat_id
        return results, None
    except RuntimeError as e:
        return [], str(e)
