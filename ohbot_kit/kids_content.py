"""Kids' performance content: generation, safety filter, fallback library, playback.

Used by streamlit_app.py. A "performance" is always the same shape regardless
of where it came from:

    {"segments": [{"text": str, "expression": str}, ...]}   # 3-5 segments

so perform() never needs to know whether generate() or get_fallback() produced
it. expression must be one of EXPRESSIONS, which is exactly the set of faces
ohbot_kit.expression.POSES supports for this app (see test_kids_content.py for
the check that keeps these in sync).
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from typing import TYPE_CHECKING

import anthropic

from . import expression

if TYPE_CHECKING:
    from .robot import Ohbot

# Cross-region inference profile id for Claude Haiku 4.5. Overridable via
# config.yaml's kids_content.model. This account needs the inference-profile
# form (not the bare on-demand id "anthropic.claude-haiku-4-5-20251001-v1:0"),
# confirmed working in gclaude_train's coach.py with the same AWS profile.
DEFAULT_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
# Kept short and with no retries: a live demo with a child waiting needs a
# failed call to surface immediately so the presenter can hit fallback,
# not silently retry into a much longer wait.
TIMEOUT_SECONDS = 6.0
MAX_TOKENS = 400
# How long to hold, face turned curious, before a "pause_before" segment (a
# joke's punchline, a story's reveal). Long enough to read as a deliberate
# beat, short enough not to feel like a hang for a child waiting on it.
PUNCHLINE_PAUSE_SECONDS = 0.9

EXPRESSIONS = (
    "happy", "surprised", "silly", "sleepy", "neutral",
    "excited", "curious", "goofy", "mischievous", "sympathetic",
)

SEGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            # No minItems/maxItems: Bedrock's structured-outputs schema support
            # only allows minItems of 0 or 1 ("For 'array' type, 'minItems'
            # values other than 0 or 1 are not supported"). The 3-5 count is
            # still enforced by SYSTEM_PROMPT and by the check below.
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "expression": {"type": "string", "enum": list(EXPRESSIONS)},
                    # Where to look while saying it, same convention as the
                    # chat beats schema in ohbot_kit/llm.py: 0-10, defaults to
                    # dead ahead (5, 5) when omitted.
                    "gaze_x": {"type": "integer"},
                    "gaze_y": {"type": "integer"},
                    # At most one segment per performance -- a joke's
                    # punchline, a story's reveal -- for a short suspenseful
                    # hold before it's spoken.
                    "pause_before": {"type": "boolean"},
                },
                "required": ["text", "expression"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["segments"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You write short spoken performances for a friendly desk robot
entertaining a child aged 5 to 9.

Rules, all mandatory:
- Total spoken length must be about 60-80 words (roughly 30 seconds spoken aloud).
- Content must be gentle and age-appropriate for a 5-9 year old, even if the
  requested topic could suggest something scarier -- reinterpret the topic in a
  safe, silly, or comforting way instead.
- No violence, no scary imagery, no peril, nothing frightening, no injury, no death.
- Humour, if any, must be gentle (silly wordplay, funny sounds, harmless
  mix-ups) -- never mockery, insults, or humour at someone's expense.
- Return ONLY the structured output described by the JSON schema you were
  given: a "segments" array of 3 to 5 short segments, each with "text" (a short
  spoken chunk, roughly one sentence) and "expression" (one of: happy,
  surprised, silly, sleepy, neutral, excited, curious, goofy, mischievous,
  sympathetic) matching that segment's mood.
- Optionally, a segment may set "gaze_x" (0 = the robot's right, 5 = ahead, 10
  = the robot's left) and "gaze_y" (0 = down, 10 = up) if looking somewhere
  else fits the moment; omit both to look straight ahead.
- At most one segment -- a joke's punchline, or a story's big reveal -- may
  set "pause_before": true for a short suspenseful pause right before it's
  said. Use this sparingly, never on more than one segment.
"""


class GenerationError(RuntimeError):
    """Any Anthropic-call failure. The UI shows this as "Content generation failed"."""


def generate(
    client: anthropic.AnthropicBedrock,
    character: str,
    topic: str,
    content_type: str,
    model: str = DEFAULT_MODEL,
    persona: dict | None = None,
) -> dict:
    """Ask Claude (via Bedrock) for a segmented, expression-tagged performance.

    Returns {"segments": [{"text": str, "expression": str}, ...]}, 3-5 items.
    Raises GenerationError on any timeout, API, or parsing failure -- callers
    map that straight onto the UI's "error" status.

    persona, if given, is a config.yaml personas.* entry (the same dict
    ohbot_chat.py's --persona resolves to). Its system_prompt is appended,
    subordinated to the rules above -- personas are written in character
    ("You are Ohbot, a ... pirate ...") which is exactly the flavour that
    should colour the wording, but the safety/length rules must still win.
    """
    system = SYSTEM_PROMPT
    if persona:
        system += (
            "\n\nPerform this in the following character, without ever breaking "
            "the rules above:\n" + persona.get("system_prompt", "")
        )
    user_prompt = (
        f"Content type: {content_type}\n"
        f"Character: {character}\n"
        f"Topic: {topic}\n"
        "Write the performance now, following all the rules in the system prompt."
    )
    try:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
            output_config={"format": {"type": "json_schema", "schema": SEGMENT_SCHEMA}},
        )
    except anthropic.APITimeoutError as e:
        raise GenerationError("timed out waiting for a response") from e
    except anthropic.AuthenticationError as e:
        raise GenerationError("authentication failed (check AWS credentials for the genai-agent-user profile)") from e
    except anthropic.RateLimitError as e:
        raise GenerationError("rate limited") from e
    except anthropic.BadRequestError as e:
        raise GenerationError(f"bad request: {e}") from e
    except (anthropic.PermissionDeniedError, anthropic.NotFoundError) as e:
        raise GenerationError(f"API access error: {e}") from e
    except anthropic.APIConnectionError as e:
        raise GenerationError("could not connect to Bedrock") from e
    except anthropic.APIStatusError as e:
        raise GenerationError(f"API error: {e}") from e
    except Exception as e:
        # Bedrock client construction/calls can also raise raw boto3/botocore
        # errors (missing credentials, missing region, signing failures) that
        # aren't anthropic.* exceptions -- these must still become a clean
        # GenerationError rather than crashing the worker thread.
        raise GenerationError(f"Bedrock request failed: {e}") from e

    try:
        text_block = next(b.text for b in response.content if b.type == "text")
        parsed = json.loads(text_block)
    except (StopIteration, json.JSONDecodeError, AttributeError) as e:
        raise GenerationError("model returned unparsable output") from e

    segments = parsed.get("segments") or []
    if not (3 <= len(segments) <= 5):
        raise GenerationError(f"expected 3-5 segments, got {len(segments)}")
    for seg in segments:
        if seg.get("expression") not in EXPRESSIONS:
            raise GenerationError(f"unknown expression {seg.get('expression')!r}")
        if not seg.get("text", "").strip():
            raise GenerationError("empty segment text")

    return {"segments": segments}


# Whole-word blocklist: a last-resort net for anything that slips past the
# system prompt, not a substitute for it. Matched on whole words (not
# substrings) so "classic" doesn't trip on "class".
BLOCKLIST = (
    "kill", "die", "dead", "death", "blood", "gun", "knife", "weapon",
    "monster", "demon", "ghost", "witch", "scary", "scared", "frighten",
    "terrify", "nightmare", "hurt", "pain", "attack", "fight", "war",
    "explode", "explosion", "fire", "burn", "stab", "shoot", "hate",
    "stupid", "dumb", "idiot", "ugly",
)


def filter_content(content: dict) -> list[str]:
    """Return a list of problems found (empty list == passes).

    Only run on LLM-generated content, never on the hand-written fallback
    library, which is reviewed, static copy.
    """
    problems = []
    for i, seg in enumerate(content.get("segments", [])):
        words = re.findall(r"[a-z']+", seg.get("text", "").lower())
        hits = sorted(set(words) & set(BLOCKLIST))
        if hits:
            problems.append(f"segment {i + 1} contains blocked word(s): {', '.join(hits)}")
    return problems


# Pre-written, pre-segmented, pre-approved content for when the LLM path is
# unavailable or fails the filter. Same shape as generate()'s return value, so
# perform() treats both identically.
FALLBACK_LIBRARY: dict[str, list[dict]] = {
    "nursery rhyme": [
        {"segments": [
            {"text": "Hop, hop, hop went the little bunny,", "expression": "happy"},
            {"text": "hopping down the lane so sunny.", "expression": "happy"},
            {"text": "Then he stopped -- oh, what a sight!", "expression": "surprised"},
            {"text": "A carrot patch, just his size, just right!", "expression": "silly"},
        ]},
        {"segments": [
            {"text": "Twinkle, twinkle, robot bright,", "expression": "happy"},
            {"text": "beeping softly through the night.", "expression": "sleepy"},
            {"text": "Up above the world so high,", "expression": "surprised"},
            {"text": "like a diamond in the sky.", "expression": "happy"},
            {"text": "Twinkle, twinkle, off to bed,", "expression": "sleepy"},
        ]},
        {"segments": [
            {"text": "Little duck went for a walk,", "expression": "happy"},
            {"text": "quack quack quack, that's how ducks talk.", "expression": "silly"},
            {"text": "Found a puddle, jumped right in,", "expression": "surprised"},
            {"text": "what a splashy, happy grin!", "expression": "happy"},
        ]},
    ],
    "joke": [
        {"segments": [
            {"text": "Why did the robot go to school?", "expression": "neutral"},
            {"text": "To improve its algo-rhythm!", "expression": "silly", "pause_before": True},
            {"text": "Get it? Rhythm, like dancing?", "expression": "happy"},
        ]},
        {"segments": [
            {"text": "What do you call a sleepy dinosaur?", "expression": "neutral"},
            {"text": "A dino-snore!", "expression": "silly", "pause_before": True},
            {"text": "Rawr... zzz.", "expression": "sleepy"},
        ]},
        {"segments": [
            {"text": "Why don't robots ever panic?", "expression": "neutral"},
            {"text": "They have nerves of steel!", "expression": "silly", "pause_before": True},
            {"text": "Ha! Get it?", "expression": "happy"},
        ]},
        {"segments": [
            {"text": "What did one wall say to the other?", "expression": "neutral"},
            {"text": "I'll meet you at the corner!", "expression": "surprised", "pause_before": True},
            {"text": "Ha ha, walls are so silly.", "expression": "silly"},
        ]},
    ],
    "story": [
        {"segments": [
            {"text": "Once there was a sleepy little cloud,", "expression": "sleepy"},
            {"text": "who floated slowly over town.", "expression": "sleepy"},
            {"text": "Then the sun peeked out and said hello!", "expression": "surprised"},
            {"text": "The cloud woke up and smiled a big smile.", "expression": "happy"},
        ]},
        {"segments": [
            {"text": "A tiny robot loved to sing,", "expression": "happy"},
            {"text": "beep-boop songs about everything.", "expression": "silly"},
            {"text": "One day it sang so very loud,", "expression": "surprised"},
            {"text": "the whole garden danced along, proud!", "expression": "happy"},
        ]},
        {"segments": [
            {"text": "A little star felt very small,", "expression": "sleepy"},
            {"text": "hiding behind a cloud, that's all.", "expression": "sleepy"},
            {"text": "Then it peeked out -- surprise, twinkle time!", "expression": "surprised"},
            {"text": "Turns out small stars can still shine.", "expression": "happy"},
        ]},
    ],
}


def get_fallback(content_type: str) -> dict:
    """One pre-written item for the type, chosen at random so repeats vary."""
    options = FALLBACK_LIBRARY.get(content_type) or FALLBACK_LIBRARY["joke"]
    return random.choice(options)


def perform(bot: "Ohbot", content: dict, stop_event: threading.Event) -> None:
    """Drive Ohbot through a segmented, expression-tagged performance.

    Format-agnostic: content has the same shape whether it came from
    generate() or get_fallback() -- gaze_x/gaze_y/pause_before are all
    optional keys, so fallback-library segments (which never set them) still
    play correctly. Checked for a stop between segments only -- interrupting
    the current segment's audio is the Stop button's own job (calling
    sd.stop() directly), not this loop's.

    Gesture isn't asked of the model (unlike emotion/gaze) -- it's derived
    from the emotion via expression.gestures_for(), the same choice
    ohbot_chat.py's beats mode makes, indexed by segment position so repeated
    emotions don't always play the identical gesture.
    """
    for i, seg in enumerate(content["segments"]):
        if stop_event.is_set():
            return
        emotion = seg["expression"]
        if seg.get("pause_before"):
            bot.express("curious")
            time.sleep(PUNCHLINE_PAUSE_SECONDS)
            if stop_event.is_set():
                return
        suited = expression.gestures_for(emotion)
        bot.gaze(seg.get("gaze_x", 5), seg.get("gaze_y", 5))
        bot.speak(seg["text"], emotion=emotion, gesture=suited[i % len(suited)])
    if not stop_event.is_set():
        bot.express("neutral")


# Which existing config.yaml persona's voice to use for each content type.
VOICE_PERSONA_FOR_TYPE = {
    "nursery rhyme": "storyteller",
    "story": "storyteller",
    "joke": "joker",
}


def voice_for_content_type(cfg, content_type: str, default_voice: str) -> str:
    """Reuse the storyteller/joker persona voices already defined in config.yaml."""
    persona_name = VOICE_PERSONA_FOR_TYPE.get(content_type, "storyteller")
    try:
        return cfg.persona(persona_name).get("voice") or default_voice
    except KeyError:
        return default_voice
