"""Diagnose whether the empathy layer is actually working.

"Does it feel alive" splits into three questions. Two are measurable and this
script measures them; the third needs your eyes, and is printed as a checklist.

    python check_empathy.py             # emotion accuracy + mechanics, no robot needed
    python check_empathy.py --robot     # also runs the on-hardware mechanics checks

1. SEMANTIC   Does the LLM pick a fitting emotion for the sentiment?
              Scored against a labelled set. This is the one that silently rots
              when you change the model, the persona or the pose vocabulary.
2. MECHANICAL Do gestures actually overlap speech? Does anything write to the
              mouth while lip sync owns it? How long until the robot reacts?
3. PERCEPTUAL Does it look right to a person? Checklist at the end.
"""

import argparse
import sys
import time

import _bootstrap  # noqa: F401

from ohbot_kit import expression, llm

# Each case lists the emotions a reasonable person would accept. Deliberately
# generous -- we are testing "not wrong", not "matches my favourite".
CASES = [
    ("My cat died last night.", {"sad", "sympathetic"}),
    ("My grandmother is in hospital.", {"sad", "sympathetic"}),
    ("I'm really scared about my exam.", {"sad", "sympathetic", "curious"}),
    ("I lost my job today.", {"sad", "sympathetic"}),
    ("I just got promoted!", {"happy", "excited"}),
    ("I finally finished my project!", {"happy", "excited"}),
    ("It's my birthday today!", {"happy", "excited"}),
    ("There's a huge spider on my shoulder!", {"surprised", "excited"}),
    ("Guess what just happened!", {"curious", "surprised", "excited"}),
    ("Why is the sky blue?", {"thinking", "curious", "neutral", "happy"}),
    ("What is two plus two?", {"thinking", "neutral", "happy", "curious"}),
    ("I have no idea what you just said.", {"confused", "curious", "thinking"}),
]

# Gesture tone matters as much as the face, and the first version of this check
# was too lenient: it only flagged bouncy gestures on sad input, so it missed
# `shake` (= "no") being chosen for "I finally finished my project!". Both
# directions are checked now.
BAD_ON_SAD = {"perk_up", "double_take"}  # bouncy under bad news
BAD_ON_GOOD = {"shake", "look_away"}  # reads as "no" / disengagement
SAD_CASES = {c[0] for c in CASES[:4]}
GOOD_CASES = {c[0] for c in CASES[4:7]}


def semantic_check(model, host, verbose=True, repeats=1):
    """Score emotion choice. Repeats matter: the model is stochastic, so a
    single pass is a sample, not a measurement -- at temperature 0.7 the same
    12 cases scored 12/12 on one run and 10/12 on the next."""
    convo_system = llm.SYSTEM_PROMPT
    hits, tone_misses, invalid, times = 0, [], [], []
    # Kept so the robot phase performs exactly what was scored here, rather
    # than asking the model again and possibly getting different choices.
    scored = []

    print(f"\n1. SEMANTIC -- does the emotion fit the sentiment?   ({repeats} run(s) per case)\n")
    per_case = {}
    for prompt, acceptable in [c for c in CASES for _ in range(repeats)]:
        # Fresh conversation each time: we're testing the mapping, not memory.
        convo = llm.Conversation(model=model, system=convo_system, host=host)
        t0 = time.time()
        try:
            action = convo.respond_with_action(
                prompt, expression.EMOTIONS, expression.GESTURE_NAMES
            )
        except llm.OllamaError as e:
            print(f"  ERROR {e}")
            return None
        elapsed = time.time() - t0
        times.append(elapsed)

        emotion = action.get("emotion", "")
        gesture = action.get("gesture", "")

        # A value outside the schema enum means the schema isn't binding.
        if emotion not in expression.EMOTIONS or gesture not in expression.GESTURE_NAMES:
            invalid.append((prompt, emotion, gesture))

        ok = emotion in acceptable
        hits += ok
        per_case.setdefault(prompt, []).append((ok, emotion, gesture))
        if prompt in SAD_CASES and gesture in BAD_ON_SAD:
            tone_misses.append((prompt, gesture, "bouncy gesture on bad news"))
        elif prompt in GOOD_CASES and gesture in BAD_ON_GOOD:
            tone_misses.append((prompt, gesture, "negative gesture on good news"))

        if len(per_case[prompt]) == 1:
            scored.append((prompt, action))

        if verbose:
            print(
                "  {} {:5.2f}s  {:<40} -> {:<12} {}".format(
                    "PASS" if ok else "MISS", elapsed, prompt[:38], emotion, gesture
                )
            )

    # Flag cases that are unstable across runs -- those are the ones that will
    # embarrass you on stage, not the ones that are consistently wrong.
    unstable = {p: v for p, v in per_case.items() if len({e for _, e, _ in v}) > 1}

    n = len(CASES) * repeats
    print(f"\n  emotion accuracy : {hits}/{n}  ({100 * hits / n:.0f}%)")
    print(f"  median latency   : {sorted(times)[n // 2]:.2f}s")
    print(f"  schema violations: {len(invalid)}")
    print(f"  gesture tone misses: {len(tone_misses)}")

    # A gesture the model never picks is dead weight in the vocabulary. Before
    # the emotion->gesture affinity map, `double_take` took 35% of all choices
    # while four gestures never fired at all, even on prompts that suited them.
    used = {g for runs in per_case.values() for _, _, g in runs}
    unused = [g for g in expression.GESTURE_NAMES if g not in used]
    print(f"  gestures used      : {len(used)}/{len(expression.GESTURE_NAMES)}")
    if unused:
        print(f"      never chosen: {', '.join(unused)}")
    if repeats > 1:
        print(f"  unstable across runs: {len(unstable)}/{len(CASES)} cases")
        for prompt, runs in unstable.items():
            print(
                "      {!r} -> {}".format(prompt[:38], ", ".join(sorted({e for _, e, _ in runs})))
            )
    if tone_misses:
        for prompt, g, why in tone_misses:
            print(f"      {prompt[:38]!r} -> {g}  ({why})")
    return {
        "accuracy": hits / n,
        "invalid": invalid,
        "tone_misses": tone_misses,
        "scored": scored,
        "unstable": unstable,
    }


def mechanical_check(scored, pause=1.5):
    """Perform every scored case on the robot, so you can watch them all.

    Replays the exact emotion and gesture that were scored above, announcing
    each one so it's obvious when a new case starts. Concurrency is measured
    across every utterance, not just a sample -- a gesture that overlaps on one
    case and not another is exactly the kind of intermittent fault a single
    sample hides.
    """
    from ohbot import ohbot

    from ohbot_kit import audio, tts
    from ohbot_kit import config as config_mod
    from ohbot_kit.robot import Ohbot

    print(f"\n2. MECHANICAL -- performing all {len(scored)} cases on the robot")
    print("   Watch the face. Each case is announced first.\n")

    cfg = config_mod.load(warn=False)
    try:
        audio.install_output(audio.resolve(cfg.get("audio.output_device"), audio.OUTPUT))
        tts.install(tts.KokoroTTS(voice=cfg.get("tts.voice", "af_heart")))
    except Exception as e:
        print(f"  (tts setup: {e})")

    events = []
    original_move = ohbot.move

    def logged(m, pos, spd=5, eye=0):
        events.append((time.time(), m))
        return original_move(m, pos, spd, eye)

    mouth = {expression.TOPLIP, expression.BOTTOMLIP}
    results = []

    ohbot.move = logged
    try:
        with Ohbot(idle=True) as bot:
            for i, (prompt, action) in enumerate(scored, 1):
                emotion = action.get("emotion", "neutral")
                gesture = action.get("gesture", "blink")

                # Return to a clean slate so the next expression is a visible
                # change rather than a drift from the previous one.
                bot.express("neutral")
                bot.set_state("listening")
                time.sleep(pause)

                print(f"  [{i:2}/{len(scored)}] {prompt}")
                print(f"         -> {emotion} / {gesture}")

                # Spoken marker, deliberately flat, so the emotional delivery
                # that follows is unmistakably the robot's response.
                bot.speak(f"Test {i}.")
                time.sleep(0.4)

                # The scenario, so an observer knows what it is reacting to.
                bot.set_state("thinking")
                bot.speak(f"They said: {prompt}")
                time.sleep(0.5)

                events.clear()
                t0 = time.time()
                bot.gaze(action.get("gaze_x", 5), action.get("gaze_y", 5))
                bot.speak(action["say"], emotion=emotion, gesture=gesture)
                t1 = time.time()

                during = [(t, m) for t, m in events if t0 < t < t1]
                lip = [e for e in during if e[1] in mouth]
                gest = [e for e in during if e[1] not in mouth]
                overlapped = bool(lip and gest)
                results.append(
                    {
                        "prompt": prompt,
                        "emotion": emotion,
                        "gesture": gesture,
                        "lip": len(lip),
                        "gest": len(gest),
                        "overlap": overlapped,
                        "seconds": t1 - t0,
                    }
                )
                print(
                    "         {}  {:.1f}s  lip={} gesture={}\n".format(
                        "overlap OK" if overlapped else "NO OVERLAP", t1 - t0, len(lip), len(gest)
                    )
                )

            bot.express("neutral")
            bot.speak("That is all twelve tests.", emotion="happy", gesture="nod")
    finally:
        ohbot.move = original_move

    overlaps = sum(r["overlap"] for r in results)
    total_lip = sum(r["lip"] for r in results)
    total_gest = sum(r["gest"] for r in results)

    print(f"  cases with gesture overlapping speech : {overlaps}/{len(results)}")
    print(f"  total lip-sync moves                  : {total_lip}")
    print(f"  total gesture moves                   : {total_gest}")

    silent = [r for r in results if not r["gest"]]
    if silent:
        print("\n  cases where NOTHING moved during speech:")
        for r in silent:
            print("      {!r} ({})".format(r["prompt"][:40], r["gesture"]))

    ok = overlaps == len(results)
    print(
        "\n  {} gesture and lip sync overlapped on {}".format(
            "PASS --" if ok else "FAIL --",
            "every case" if ok else f"only {overlaps} of {len(results)} cases",
        )
    )
    return {"concurrent": ok, "results": results}


CHECKLIST = """
3. PERCEPTUAL -- only you can judge these. Watch it handle "my cat died":

  [ ] The face changes BEFORE the sentence finishes, not after.
  [ ] The head moves while it talks -- it is not a statue with a moving mouth.
  [ ] Lip movement still tracks the audio (the gesture hasn't disturbed it).
  [ ] Eyes are open at rest, not half-lidded.
  [ ] It blinks occasionally, and nods while YOU are talking.
  [ ] Sad input does not produce a bouncy movement.
  [ ] Motion looks deliberate, not twitchy. If it jitters, lengthen the `hold`
      values in expression.GESTURES rather than lowering the speeds.
  [ ] Nothing buzzes after the program exits (motors detached cleanly).

If any mechanical check fails, fix that first -- perceptual problems usually
follow from a mechanical one rather than from the pose design.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--robot",
        action="store_true",
        help="perform every case on the robot so you can watch them",
    )
    p.add_argument(
        "--pause",
        type=float,
        default=1.5,
        help="seconds held at neutral between cases (default 1.5)",
    )
    p.add_argument(
        "--repeats", type=int, default=1, help="runs per case; >1 exposes run-to-run instability"
    )
    p.add_argument("--model", default=None)
    args = p.parse_args()

    from ohbot_kit import config as config_mod

    cfg = config_mod.load(warn=False)
    model = args.model or cfg.get("llm.model", llm.DEFAULT_MODEL)
    host = cfg.get("llm.host", llm.HOST)

    print(f"Empathy layer check -- model: {model}")
    print(f"  {len(expression.EMOTIONS)} emotions, {len(expression.GESTURE_NAMES)} gestures")

    try:
        llm.check_model(model, host)
    except llm.OllamaError as e:
        print(e, file=sys.stderr)
        return 1

    sem = semantic_check(model, host, repeats=args.repeats)
    mech = None
    if args.robot and sem:
        mech = mechanical_check(sem["scored"], pause=args.pause)

    print(CHECKLIST)

    print("=" * 60)
    failed = []
    # Distinguish "the model is unreachable" from "the model chose badly".
    # Reporting an Ollama outage as low accuracy sends you debugging prompts
    # when the actual problem is that the server isn't running.
    if sem is None:
        failed.append("could not reach the model -- no scores were produced")
    elif sem["accuracy"] < 0.75:
        failed.append(
            "emotion accuracy {:.0f}%, below the 75% threshold".format(100 * sem["accuracy"])
        )
    if sem and sem["invalid"]:
        failed.append("schema not binding -- values outside the enum")
    if mech and not mech["concurrent"]:
        failed.append("gestures not overlapping speech")
    if sem and sem["tone_misses"]:
        failed.append("{} gesture(s) with the wrong tone".format(len(sem["tone_misses"])))

    if failed:
        print("PROBLEMS:")
        for f in failed:
            print(f"  - {f}")
        return 1
    print("Measurable checks pass. Now do the perceptual checklist above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
