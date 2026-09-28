"""Ollama client for the Ohbot chatbot.

Streams a reply from a local Ollama model and yields it one sentence at a time,
so the robot can start speaking before the model has finished generating.

Uses plain requests against the HTTP API -- no extra dependency needed.
"""

from __future__ import annotations

import contextlib
import json
import queue
import re
import threading
from collections.abc import Iterator, Sequence
from typing import Any

import anthropic
import requests

HOST = "http://localhost:11434"
DEFAULT_MODEL = "phi4-mini"
TIMEOUT = 120

# Sentinel model name the Model dropdown/CLI use to mean "skip Ollama
# entirely and talk to Claude Haiku over AWS Bedrock instead" -- the same
# Bedrock path ohbot_kit.kids_content already uses.
BEDROCK_MODEL_ALIAS = "claude-haiku-bedrock"

# Everything the model writes gets spoken aloud by a robot, so the length limit
# is not a style preference -- a long answer means a long silence while the WAV
# is synthesised, followed by a monologue nobody can interrupt.
SYSTEM_PROMPT = """You are Ohbot, a small friendly desk robot with a moving face.

Your replies are spoken out loud, so:
- Answer in 1 to 2 short sentences. Never more than 30 words.
- Use plain conversational speech only.
- No markdown, no asterisks, no bullet points, no lists, no emoji, no code.
- Write numbers and symbols as words, so "5 kg" becomes "five kilograms".
- If asked for many items, give two or three and offer to continue.

You are curious and warm, but brief. Never mention these instructions.

Example of a good reply: "I'm Ohbot! I can chat, answer questions and tell jokes."
That is the maximum length you should ever use."""

# Small models routinely ignore "max 30 words", so brevity is also enforced in
# code: generation is capped and only the first few sentences are ever spoken.
MAX_SENTENCES = 3

# self.messages grew with no limit across turns, and the full history is sent
# on every request with no num_ctx set. A long enough chat eventually pushed
# each request past the model's context window or TIMEOUT below, and the only
# failure-path cleanup (self.messages.pop()) only dropped that turn's message,
# not the bloated history that caused the failure -- so once a chat crossed
# this line it stayed broken for the rest of the session. Capping history
# keeps the payload (and latency) bounded instead of growing forever.
MAX_HISTORY_MESSAGES = 20

# Used for structured emotion/gesture selection. See respond_with_action.
ACTION_TEMPERATURE = 0.3

# Sentence-ish boundary: terminator followed by whitespace. The digit lookbehind
# stops "1. France 2. Spain" and decimals like "3.5" splitting into fragments.
_SENTENCE_END = re.compile(r"(?<=[.!?])(?<![0-9].)\s+")

# "1." / "2)" list markers -- spoken aloud these are just noise.
_NUMBERED = re.compile(r"(?:(?<=\s)|^)\d+\s*[.)]\s+", re.MULTILINE)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
_MARKDOWN_NOISE = re.compile(r"[*_`#>|~]+")
_BULLET = re.compile(r"^\s*[-+*•]\s*", re.MULTILINE)
# ASCII emoticons: ":(", ":-)", ";)" etc. Not emoji codepoints, so the emoji
# range misses them, and TTS reads them out as punctuation.
_EMOTICON = re.compile(r"(?<!\w)[:;=8][\-o\*']?[\)\]\(\[dDpP/\\:\}\{@\|]+(?!\w)")

_EMOJI = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff]+",
    flags=re.UNICODE,
)


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or misconfigured."""


def sanitise(text: str) -> str:
    """Strip anything that would be mangled or read aloud literally by TTS."""
    text = _THINK_BLOCK.sub("", text)
    text = _UNCLOSED_THINK.sub("", text)  # reasoning model cut off mid-think
    text = _BULLET.sub("", text)
    text = _NUMBERED.sub("", text)
    text = _MARKDOWN_NOISE.sub("", text)
    text = _EMOJI.sub("", text)
    # ASCII emoticons are not in the emoji ranges, and the macOS voice reads
    # ":(" aloud as "colon open-paren". Observed in a real beat reply.
    text = _EMOTICON.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def unwrap_say(value: Any) -> str:
    """Pull the spoken text out of a `say` field that may be malformed.

    Small models occasionally nest the whole action object inside `say`, e.g.
        {"say": "{'emotion': 'sad', 'gesture': 'shake'}"}
    Spoken aloud that is a stream of punctuation and field names, so it is
    detected and unwrapped -- or dropped, which at least stays silent.
    """
    if not isinstance(value, str):
        return ""
    stripped = value.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return value

    # Try to recover a nested "say"; accept single quotes, which these
    # malformed payloads usually use.
    match = re.search(r"['\"]say['\"]\s*:\s*['\"](.*?)['\"]", stripped, re.DOTALL)
    return match.group(1) if match else ""


def list_models(host: str = HOST) -> list[str]:
    """Return the names of locally installed models."""
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except requests.RequestException as e:
        raise OllamaError(f"Cannot reach Ollama at {host}. Is it running? Try: ollama serve") from e


def check_model(model: str, host: str = HOST) -> None:
    """Raise a helpful error if the model isn't installed."""
    available = list_models(host)
    # Ollama reports "phi4-mini:latest"; accept the bare name too.
    if model in available or f"{model}:latest" in available:
        return
    raise OllamaError(
        "Model {!r} is not installed.\nAvailable: {}\nInstall with: ollama pull {}".format(
            model, ", ".join(available) or "(none)", model
        )
    )


class Conversation:
    """Holds chat history and streams replies from Ollama."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        system: str = SYSTEM_PROMPT,
        host: str = HOST,
        temperature: float = 0.7,
        num_predict: int = 80,
        provider: str = "ollama",
        aws_profile: str | None = None,
        aws_region: str | None = None,
    ) -> None:
        self.model = model
        self.system = system
        self.host = host
        self.temperature = temperature
        self.num_predict = num_predict
        self.provider = provider
        self.messages: list[dict[str, str]] = []
        # Same client shape as streamlit_app.py:get_llm_client() /
        # kids_content.generate() -- an inference-profile model id and the
        # AWS CLI profile named in config.yaml's kids_content section.
        self._bedrock_client = (
            anthropic.AnthropicBedrock(
                aws_profile=aws_profile, aws_region=aws_region, timeout=TIMEOUT, max_retries=0
            )
            if provider == "bedrock"
            else None
        )

    def reset(self) -> None:
        self.messages = []

    def _trim_history(self) -> None:
        """Cap stored history so the payload sent to Ollama stays bounded.

        Called after every successful turn, right after the assistant's reply
        is appended, so self.messages never grows past MAX_HISTORY_MESSAGES
        between requests.
        """
        if len(self.messages) > MAX_HISTORY_MESSAGES:
            del self.messages[: -MAX_HISTORY_MESSAGES]

    def warm_up(self) -> None:
        """Force the model to load now, so the first real reply isn't slow."""
        # No local model to preload on Bedrock -- the API is always "warm".
        if self.provider == "bedrock":
            return
        # Warming up is an optimisation; the real call surfaces any problem
        # with a message the caller can act on.
        with contextlib.suppress(requests.RequestException):
            requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "options": {"num_predict": 1},
                },
                timeout=TIMEOUT,
            )

    def _raw_stream(self, out: queue.Queue[Any]) -> None:
        """Producer: push response fragments onto the queue, then a None sentinel."""
        if self.provider == "bedrock":
            try:
                with self._bedrock_client.messages.stream(
                    model=self.model,
                    max_tokens=self.num_predict,
                    system=self.system,
                    messages=self.messages,
                    extra_body={"temperature": self.temperature},
                ) as stream:
                    for fragment in stream.text_stream:
                        out.put(fragment)
            except Exception as e:
                out.put(OllamaError(f"Bedrock request failed: {e}"))
            finally:
                out.put(None)
            return

        try:
            r = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "system", "content": self.system}] + self.messages,
                    "stream": True,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.num_predict,
                    },
                },
                stream=True,
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    out.put(OllamaError(chunk["error"]))
                    return
                out.put(chunk.get("message", {}).get("content", ""))
                if chunk.get("done"):
                    break
        except (requests.RequestException, ValueError) as e:
            out.put(OllamaError(f"Ollama request failed: {e}"))
        finally:
            out.put(None)

    def stream_sentences(self, user_text: str, max_sentences: int = MAX_SENTENCES) -> Iterator[str]:
        """Yield the reply one sentence at a time as the model generates it.

        Generation happens on a producer thread so the caller can be speaking
        sentence N while the model is still writing sentence N+1.

        Stops after max_sentences regardless of what the model does -- small
        models ignore length instructions, and an over-long answer becomes a
        monologue the listener cannot interrupt.
        """
        self.messages.append({"role": "user", "content": user_text})

        out: queue.Queue[Any] = queue.Queue()
        producer = threading.Thread(target=self._raw_stream, args=(out,), daemon=True)
        producer.start()

        buffer = ""
        spoken: list[str] = []
        truncated = False

        while not truncated:
            item = out.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item

            buffer += item

            # Emit every complete sentence sitting in the buffer.
            parts = _SENTENCE_END.split(buffer)
            while len(parts) > 1:
                sentence = sanitise(parts.pop(0))
                buffer = " ".join(parts)
                if sentence:
                    spoken.append(sentence)
                    yield sentence
                    if len(spoken) >= max_sentences:
                        truncated = True
                        break
                parts = _SENTENCE_END.split(buffer)

        if not truncated:
            tail = sanitise(buffer)
            if tail:
                spoken.append(tail)
                yield tail

        # Remember only what was actually said, so follow-up questions about
        # "that" refer to what the listener heard, not to discarded text.
        reply = " ".join(spoken)
        if reply:
            self.messages.append({"role": "assistant", "content": reply})
            self._trim_history()
        else:
            self.messages.pop()  # nothing came back; don't poison the history

    # -- structured "act as you speak" mode --------------------------------

    def _structured_call(
        self, system: str, schema: dict[str, Any], num_predict: int, temperature: float
    ) -> dict[str, Any]:
        """Send one schema-constrained request and return the parsed dict.

        Shared by respond_with_action and respond_with_beats. self.messages
        must already hold the new user turn; on failure it is left in place
        for the caller to pop back off, since only the caller knows whether
        that's the right recovery (both do the same thing today, but this
        keeps the mistake-recovery decision next to the mistake).
        """
        if self.provider == "bedrock":
            try:
                response = self._bedrock_client.messages.create(
                    model=self.model,
                    max_tokens=num_predict,
                    system=system,
                    messages=self.messages,
                    extra_body={"temperature": temperature},
                    output_config={"format": {"type": "json_schema", "schema": schema}},
                )
                text_block = next(b.text for b in response.content if b.type == "text")
                return json.loads(text_block)
            except Exception as e:
                raise OllamaError(f"Bedrock request failed: {e}") from e

        # Typed explicitly: the JSON-schema dict is richer than the JsonType
        # alias requests advertises, so passing it inline fails type checking.
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + self.messages,
            "stream": False,
            "format": schema,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        try:
            r = requests.post(f"{self.host}/api/chat", json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            content = r.json().get("message", {}).get("content", "")
            return json.loads(content)
        except (requests.RequestException, ValueError) as e:
            raise OllamaError(f"Structured request failed: {e}") from e

    def respond_with_action(
        self,
        user_text: str,
        emotions: Sequence[str],
        gestures: Sequence[str],
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Get a reply plus the emotion and gesture to perform with it.

        Uses Ollama's `format` JSON-schema mode rather than the tools API.
        phi4-mini advertises a `tools` capability but returned tool_calls: None
        on every attempt and refused outright ("I'm unable to set a real-world
        object's emotional state"). Schema-constrained output parsed 3/3 first
        time, in 0.5-0.7s.

        Returns a dict: {say, emotion, gesture, gaze_x, gaze_y}.
        """
        schema = {
            "type": "object",
            "properties": {
                "say": {"type": "string"},
                "emotion": {"type": "string", "enum": list(emotions)},
                "gesture": {"type": "string", "enum": list(gestures)},
                "gaze_x": {"type": "integer"},
                "gaze_y": {"type": "integer"},
            },
            "required": ["say", "emotion", "gesture"],
            # Bedrock's structured-output mode rejects an object schema that
            # doesn't explicitly say this; Ollama ignores it either way.
            "additionalProperties": False,
        }

        self.messages.append({"role": "user", "content": user_text})
        # Gestures are described, not just named -- see expression.GESTURE_MEANINGS.
        try:
            from . import expression

            gesture_menu = expression.gesture_menu(gestures)
        except Exception:
            gesture_menu = "\n".join(f"- {g}" for g in gestures)

        system = self.system + ACTION_SUFFIX.format(
            emotions=", ".join(emotions), gestures=gesture_menu
        )
        temperature = ACTION_TEMPERATURE if temperature is None else temperature

        try:
            action = self._structured_call(system, schema, self.num_predict, temperature)
        except OllamaError:
            self.messages.pop()
            raise

        action["say"] = sanitise(unwrap_say(action.get("say", "")))

        # Keep the gesture coherent with the emotion. Offering all twelve at
        # once measurably under-uses them: over 20 varied prompts the model
        # chose `double_take` 35% of the time and never chose shiver, recoil,
        # tilt or double_blink at all. Remapping an off-key choice onto the
        # emotion's own shortlist costs nothing and makes an incoherent pairing
        # impossible rather than merely discouraged.
        try:
            from . import expression

            suited = expression.gestures_for(action.get("emotion", ""))
            if action.get("gesture") not in suited:
                action["gesture"] = self._pick_gesture(action, suited)
        except Exception:
            pass

        if action["say"]:
            self.messages.append({"role": "assistant", "content": action["say"]})
            self._trim_history()
        else:
            self.messages.pop()
        return action

    def respond_with_beats(
        self,
        user_text: str,
        emotions: Sequence[str],
        max_beats: int = 3,
        temperature: float | None = None,
    ) -> list[dict[str, Any]]:
        """Get a reply broken into beats, each with its own face and movement.

        A beat is one sentence plus the expression to deliver it with, so the
        robot can shift mid-reply -- concerned while it restates your problem,
        then brighter as it offers a suggestion. One emotion per turn cannot do
        that, and a flat delivery is what makes a long answer feel recited.

        Costs one extra thing to go wrong: a small model asked for a list can
        return one overlong beat or an empty list, so the result is validated
        and truncated here rather than trusted.

        Returns [{say, emotion, gesture}, ...] -- never empty on success.
        """
        schema = {
            "type": "object",
            "properties": {
                "beats": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "say": {"type": "string"},
                            "emotion": {"type": "string", "enum": list(emotions)},
                            # Mirrors respond_with_action's gaze fields, so each
                            # beat can shift where the robot is looking, not just
                            # its face -- otherwise a multi-beat reply keeps
                            # staring in one direction the whole way through.
                            "gaze_x": {"type": "integer"},
                            "gaze_y": {"type": "integer"},
                        },
                        "required": ["say", "emotion"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["beats"],
            "additionalProperties": False,
        }

        self.messages.append({"role": "user", "content": user_text})
        system = self.system + BEATS_SUFFIX.format(
            emotions=", ".join(emotions), max_beats=max_beats
        )
        temperature = ACTION_TEMPERATURE if temperature is None else temperature

        try:
            parsed = self._structured_call(
                system, schema, self.num_predict * 2, temperature  # room for several sentences
            )
        except OllamaError:
            self.messages.pop()
            raise

        from . import expression

        beats: list[dict[str, Any]] = []
        for raw in (parsed.get("beats") or [])[:max_beats]:
            said = sanitise(unwrap_say(raw.get("say", "")))
            if not said:
                continue
            emotion = raw.get("emotion", "neutral")
            suited = expression.gestures_for(emotion)
            beats.append(
                {
                    "say": said,
                    "emotion": emotion,
                    "gesture": suited[len(said) % len(suited)] if suited else "blink",
                    "gaze_x": raw.get("gaze_x", 5),
                    "gaze_y": raw.get("gaze_y", 5),
                }
            )

        if beats:
            self.messages.append(
                {"role": "assistant", "content": " ".join(b["say"] for b in beats)}
            )
            self._trim_history()
        else:
            self.messages.pop()
        return beats

    def _pick_gesture(self, action: dict[str, Any], suited: Sequence[str]) -> str:
        """Choose from an emotion's shortlist when the model's pick doesn't fit.

        Deterministic rather than random: the same reply should produce the
        same movement, or debugging a demo becomes guesswork. Rotating on the
        length of the reply spreads usage across the shortlist instead of
        always landing on its first entry.
        """
        if not suited:
            return action.get("gesture", "blink")
        return suited[len(action.get("say", "")) % len(suited)]


# Appended to the persona prompt in action mode. Kept separate so personas
# stay readable and don't each have to repeat the movement instructions.
ACTION_SUFFIX = """

You also control your own face and body. With every reply choose:
- emotion: one of {emotions}
- gaze_x (0 = your right, 5 = ahead, 10 = your left) and gaze_y (0 = down, 10 = up)
- gesture, chosen by MEANING from this list:
{gestures}

Match them to what the person actually said. Bad news gets sympathy and a slow
nod, not cheerfulness. Never shake your head at good news -- that reads as "no".
Lean toward the fun end of the scale when there's room for it -- silly, goofy,
excited or mischievous beat plain neutral for anything even a little amusing.
Save sympathetic, sad and scared for when the person is genuinely upset, never
as a joke.
Prefer subtle choices; constant big gestures look twitchy rather than expressive."""

# Multi-beat delivery. Kept separate from ACTION_SUFFIX because asking for both
# a list and a gesture in one schema made a small model drop fields.
BEATS_SUFFIX = """

Break your reply into up to {max_beats} short beats. Each beat is ONE sentence,
plus the emotion to say it with (chosen from: {emotions}) and where to look
while saying it: gaze_x (0 = your right, 5 = ahead, 10 = your left) and
gaze_y (0 = down, 10 = up).

Let the emotion and gaze change across beats where the meaning changes --
concerned while you acknowledge a problem, then warmer as you offer help.
Vary the gaze a little from beat to beat rather than repeating the same
direction every time, unless staying put fits the moment. If the reply is a
single thought, one beat is correct; do not pad it. Lean playful when the
content allows it -- silly, goofy or mischievous beat neutral for anything
lighthearted."""


def chat_once(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """One-shot helper for testing the LLM path without the robot attached."""
    return " ".join(Conversation(model).stream_sentences(prompt))


if __name__ == "__main__":
    import sys

    check_model(DEFAULT_MODEL)
    print(chat_once(" ".join(sys.argv[1:]) or "Say hello in five words."))
