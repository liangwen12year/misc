#!/usr/bin/env python3
"""
Audio generation pipeline using the ElevenLabs Python SDK and Convokit for context lookup.

- For each turn:
    - Generates: {speaker}_{convID}_{turnID}.mp3
      (for all unique speakers, each voicing every turn)
- ✅ No composed audio.
- ✅ Every speaker repeats each turn individually.
- Uses these voices in round‑robin order:
    Ivanna, Mark, Amanda, Grandpa Spuds Oxley, Grandma Muffin, Sassy Aerisita
- Skips generating any file already present under ./samples/final/
"""

import sys
import os
import argparse
import json
from pathlib import Path
from elevenlabs.client import ElevenLabs

VOICE_SLOTS = [
    {"name": "Ivanna - Young & Casual",     "voice_id": "yM93hbw8Qtvdma2wCnJG"},
    {"name": "Mark - Natural Conversations", "voice_id": "UgBBYS2sOqTuMpoF3BR0"},
    {"name": "Amanda",                       "voice_id": "M6N6IdXhi5YNZyZSDe7k"},
    {"name": "Grandpa Spuds Oxley",          "voice_id": "NOpBlnGInO9m6vDvFkFC"},
    {"name": "Grandma Muffin",               "voice_id": "vFLqXa8bgbofGarf6fZh"},
    {"name": "Sassy Aerisita",               "voice_id": "03vEurziQfq3V8WZhQvn"},
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TTS audio per turn (all speakers voice every turn). Skips existing files under samples/final/."
    )
    parser.add_argument(
        "--conversation-id", "-CONV",
        required=True,
        help="Conversation ID to process."
    )
    parser.add_argument(
        "--corpus-name", "-C",
        default="movie-corpus",
        help="Convokit corpus identifier (default: movie-corpus)."
    )
    parser.add_argument(
        "--output", "-o",
        default="./samples/",
        help="Directory to save generated audio files (writes here)."
    )
    parser.add_argument(
        "--api-key", "-k",
        help="ElevenLabs API key. If omitted, reads from ELEVEN_LABS_API_KEY env var."
    )
    return parser.parse_args()


def fetch_utterances_for_conversation(corpus_dir, conv_id):
    utt_file = os.path.join(corpus_dir, "utterances.jsonl")
    utterances = []
    with open(utt_file) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("conversation_id") == conv_id:
                utterances.append({"text": obj["text"], "speaker": obj.get("speaker")})
    if not utterances:
        raise ValueError(f"⚠️ No utterances found for conversation '{conv_id}'.")
    return utterances


def generate_all_speakers_per_turn(client, utterances, conversation_id, output_dir):
    unique_speakers = sorted({utt["speaker"] for utt in utterances if utt.get("speaker")})
    num_speakers = len(unique_speakers)

    print(f"👥 Found {num_speakers} unique speakers: {unique_speakers}")
    if num_speakers == 0:
        raise ValueError("⚠️ No valid speakers found in this conversation.")

    # map speakers to voices round‑robin
    speaker_voice = {
        speaker: VOICE_SLOTS[i % len(VOICE_SLOTS)]
        for i, speaker in enumerate(unique_speakers)
    }
    print(f"🗣 Speaker → Voice mapping: {speaker_voice}")

    out_dir = Path(output_dir)
    final_dir = out_dir / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    for turn_idx, utt in enumerate(utterances, start=1):
        text = utt["text"]
        ssml = f"<speak>{text}</speak>"
        for speaker in unique_speakers:
            voice = speaker_voice[speaker]
            fname = f"{speaker}_{conversation_id}_{turn_idx}.mp3"
            out_path = out_dir / fname
            skip_path = final_dir / fname
            if skip_path.exists():
                print(f"⏭️ Skipping existing in final/: {fname}")
                continue
            print(f"🎙️ Generating {fname} with {voice['name']}")
            stream = client.text_to_speech.convert(
                text=ssml,
                voice_id=voice["voice_id"],
                model_id="eleven_multilingual_v2",
                voice_settings={"ssml": True},
                output_format="mp3_44100_128",
            )
            with open(out_path, "wb") as f:
                for chunk in stream:
                    f.write(chunk)
            print(f"✅ Saved: {fname}")


def main():
    args = parse_args()
    key = args.api_key or os.getenv("ELEVEN_LABS_API_KEY")
    if not key:
        print("⚠️ Error: ElevenLabs API key not provided.")
        return

    corpus_dir = os.path.expanduser(f"~/.convokit/saved-corpora/{args.corpus_name}")
    client = ElevenLabs(api_key=key)
    try:
        utterances = fetch_utterances_for_conversation(corpus_dir, args.conversation_id)
        generate_all_speakers_per_turn(
            client, utterances, args.conversation_id, args.output
        )
        print("✅ Completed generating turn files.")
    except Exception as e:
        print(f"❌ Generation failed: {e}")

if __name__ == "__main__":
    main()

