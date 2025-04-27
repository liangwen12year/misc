#!/usr/bin/env python3
"""
Audio generation pipeline using the ElevenLabs Python SDK.
Tuning knobs: gender, age, accent.
Accepts previous_text, next_text, previous_request_ids, next_request_ids from CLI args to build conversation context.
"""

import sys
import os
import argparse
from elevenlabs.client import ElevenLabs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TTS audio via ElevenLabs API using split conversation context"
    )
    parser.add_argument(
        "--previous-text", "-PT",
        required=True,
        help="Text of previous conversation segment."
    )
    parser.add_argument(
        "--next-text", "-NT",
        required=True,
        help="Text of next conversation segment."
    )
    parser.add_argument(
        "--previous-request-ids", "-PRI",
        nargs='+',
        required=True,
        help="Space-separated list of previous request IDs."
    )
    parser.add_argument(
        "--next-request-ids", "-NRI",
        nargs='+',
        required=True,
        help="Space-separated list of next request IDs."
    )
    parser.add_argument(
        "--gender",
        choices=["male", "female", "other"],
        default="female",
        help="Voice gender."
    )
    parser.add_argument(
        "--age",
        choices=["child", "young", "adult", "senior"],
        default="adult",
        help="Voice age group."
    )
    parser.add_argument(
        "--accent",
        default="us",
        help="Voice accent (e.g. us, uk, au, in)."
    )
    parser.add_argument(
        "--output", "-o",
        default="output.mp3",
        help="Path for the generated audio file."
    )
    parser.add_argument(
        "--api-key", "-k",
        help=(
            "ElevenLabs API key. If omitted, reads from ELEVEN_LABS_API_KEY env var."
        )
    )
    return parser.parse_args()


def select_voice(client: ElevenLabs, gender: str, age: str, accent: str):
    """
    Pick the first voice matching the given labels or fallback.
    """
    resp = client.voices.get_all()
    all_voices = resp.voices

    def matches(v):
        labels = getattr(v, "labels", {}) or {}
        return (
            labels.get("gender", "").lower() == gender.lower()
            and labels.get("age", "").lower() == age.lower()
            and accent.lower() in labels.get("accent", "").lower()
        )

    filtered = [v for v in all_voices if matches(v)]
    return filtered[0] if filtered else all_voices[0]


def build_prompt(
    previous_text: str,
    next_text: str,
    previous_request_ids: list[str],
    next_request_ids: list[str]
) -> str:
    """
    Construct prompt based on split conversation context and request IDs.
    """
    pri = ", ".join(previous_request_ids)
    nri = ", ".join(next_request_ids)
    return (
        f"Previous Conversation Text:\n{previous_text}\n"
        f"Next Conversation Text:\n{next_text}\n"
        f"Previous Request IDs: {pri}\n"
        f"Next Request IDs: {nri}\n\n"
        "Assistant: "
    )


def generate_audio(
    client: ElevenLabs,
    previous_text: str,
    next_text: str,
    previous_request_ids: list[str],
    next_request_ids: list[str],
    gender: str,
    age: str,
    accent: str,
    output_path: str
) -> str:
    """
    Generate speech from the assembled split context and save to `output_path`.
    """
    prompt = build_prompt(
        previous_text, next_text,
        previous_request_ids, next_request_ids
    )
    voice = select_voice(client, gender, age, accent)

    audio_stream = client.text_to_speech.convert(
        text=prompt,
        voice_id=voice.voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    with open(output_path, "wb") as f:
        for chunk in audio_stream:
            f.write(chunk)

    return output_path


def main():
    args = parse_args()
    key = args.api_key or os.getenv("ELEVEN_LABS_API_KEY")
    if not key:
        sys.exit(
            "⚠️ Error: ElevenLabs API key not provided. Use --api-key or set ELEVEN_LABS_API_KEY."
        )

    client = ElevenLabs(api_key=key)

    try:
        out_file = generate_audio(
            client,
            args.previous_text,
            args.next_text,
            args.previous_request_ids,
            args.next_request_ids,
            args.gender,
            args.age,
            args.accent,
            args.output
        )
        print(f"✅ Generated audio saved to {out_file}")
    except Exception as e:
        print(f"❌ Generation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

