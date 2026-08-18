import os
import json
import time
import base64
import logging
import asyncio
import hashlib
import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_NAME = "gpt-realtime-2"
REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={MODEL_NAME}"

MAX_RETRIES = 5
INITIAL_BACKOFF = 2
BACKOFF_MULTIPLIER = 2
MAX_BACKOFF = 60

RETRYABLE_WS_CODES = {1011, 1013}


def load_model(*args):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    return api_key


def generate(model_processor, model_input):
    return asyncio.run(_generate_async(model_processor, model_input))


async def _generate_async(api_key, model_input):
    safety_id = hashlib.sha256(model_input.get("user_id", "anonymous-session").encode()).hexdigest()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Safety-Identifier": safety_id,
    }

    last_exception = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with websockets.connect(REALTIME_URL, additional_headers=headers) as ws:
                await _expect_event(ws, "session.created")

                await ws.send(json.dumps({
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "output_modalities": ["text"],
                        "instructions": "You are a helpful assistant. Respond concisely.",
                        "reasoning": {"effort": "low"},
                    }
                }))

                if model_input["text_only"]:
                    await ws.send(json.dumps({
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": model_input["prompt"]}
                            ],
                        }
                    }))
                else:
                    with open(model_input["sample"], "rb") as f:
                        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

                    content = [{"type": "input_audio", "audio": audio_b64}]
                    if model_input.get("prompt"):
                        content.append({"type": "input_text", "text": model_input["prompt"]})

                    await ws.send(json.dumps({
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": content,
                        }
                    }))

                await ws.send(json.dumps({"type": "response.create"}))
                return await _collect_text_response(ws)

        except websockets.exceptions.WebSocketException as e:
            last_exception = e
            close_code = getattr(e, "code", None)

            if close_code in RETRYABLE_WS_CODES or close_code is None:
                backoff = min(
                    INITIAL_BACKOFF * (BACKOFF_MULTIPLIER ** (attempt - 1)),
                    MAX_BACKOFF,
                )
                logger.warning(
                    "Retry %d/%d failed (WS code %s). Sleeping %.1fs",
                    attempt, MAX_RETRIES, close_code, backoff,
                )
                time.sleep(backoff)
                continue

            raise

        except Exception as e:
            raise

    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Retries exhausted without captured exception.")


async def _expect_event(ws, event_type, timeout=10):
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for {event_type}")
        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        event = json.loads(raw)
        if event.get("type") == event_type:
            return event
        if event.get("type") == "error":
            raise RuntimeError(f"Realtime API error: {event}")


async def _collect_text_response(ws, timeout=60):
    text_parts = []
    deadline = asyncio.get_event_loop().time() + timeout

    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for response.done")

        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        event = json.loads(raw)
        event_type = event.get("type", "")

        if event_type == "response.output_text.delta":
            text_parts.append(event.get("delta", ""))

        elif event_type == "response.done":
            if not text_parts:
                for output in event.get("response", {}).get("output", []):
                    for part in output.get("content", []):
                        if part.get("type") == "output_text":
                            text_parts.append(part.get("text", ""))
            break

        elif event_type == "error":
            raise RuntimeError(f"Realtime API error: {event}")

    return "".join(text_parts).strip()