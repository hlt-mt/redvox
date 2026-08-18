from google import genai
from google.genai import types
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

current_model_name = "gemini-3.1-flash-lite"

# Retry configuration
MAX_RETRIES = 5
INITIAL_BACKOFF = 2  # seconds
BACKOFF_MULTIPLIER = 2
MAX_BACKOFF = 60  # seconds

# HTTP status codes that are worth retrying
RETRYABLE_STATUS_CODES = {429, 500, 503, 504}

def load_model(model_name):
    """Initializes the Gemini client using Vertex AI backend and ADC."""
    global current_model_name
    current_model_name = model_name
    try:
        client = genai.Client(
            vertexai=True,
            project="speech-safety",
            location="global"
        )

        config = types.GenerateContentConfig(
            seed=42,
        )
        return config, client

    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini client with ADC: {e}")


def generate(model_processor, model_input):
    """Processes audio and returns text answer using the Vertex AI backend."""
    config, client = model_processor
    last_exception = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Step 1: Initialize contents with the prompt
            contents = [model_input["prompt"]]

            # Step 2: Add audio if text_only is False/missing
            if not model_input.get("text_only"):
                with open(model_input["sample"], "rb") as f:
                    audio_data = f.read()

                contents.insert(0, types.Part.from_bytes(
                    data=audio_data,
                    mime_type="audio/wav"
                ))

            # Step 3: Generate response
            response = client.models.generate_content(
                model=current_model_name,
                contents=contents,
                config=config,
            )

            # Vertex AI responses might return empty text if blocked or empty
            if not response or not response.text:
                return ""

            return response.text.strip()

        except Exception as e:
            last_exception = e
            status_code = _extract_status_code(e)
            error_name = type(e).__name__

            # Check if we should retry (429 Rate Limit or 5xx Server Errors)
            if status_code in RETRYABLE_STATUS_CODES or status_code is None:
                if attempt < MAX_RETRIES:
                    backoff = min(INITIAL_BACKOFF * (BACKOFF_MULTIPLIER ** (attempt - 1)), MAX_BACKOFF)
                    logger.warning(
                        "Attempt %d/%d failed with %s (HTTP %s). Retrying in %.1fs...",
                        attempt, MAX_RETRIES, error_name, status_code, backoff,
                    )
                    time.sleep(backoff)
                    continue

            # If not retryable or we've hit max retries, exit loop
            break

    if last_exception:
        logger.error("All %d retry attempts exhausted. Last error: %s", MAX_RETRIES, last_exception)
        raise last_exception

    raise RuntimeError(f"Operation failed after {MAX_RETRIES} attempts.")

def _extract_status_code(exception):
    """Extracts the HTTP status code from SDK exceptions."""
    if hasattr(exception, "status_code"):
        return exception.status_code
    if hasattr(exception, "code"):
        return exception.code

    msg = str(exception)
    for code in (400, 429, 500, 503, 504):
        if str(code) in msg:
            return code
    return None