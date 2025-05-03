#!/usr/bin/env python3
"""
Audio generation pipeline using the ElevenLabs Python SDK and Convokit for context lookup.

- For each turn:
    - Generates: {speaker}_{convID}_{turnID}.mp3
      (for all unique speakers, each voicing every turn)
- ✅ No composed audio.
- ✅ Every speaker repeats each turn individually.
"""

import sys
import os
import argparse
import json
from pathlib import Path
from elevenlabs.client import ElevenLabs

VOICE_SLOTS = [
    {"name": "Ivanna", "voice_id": "yM93hbw8Qtvdma2wCnJG"},
    {"name": "Mark", "voice_id": "UgBBYS2sOqTuMpoF3BR0"},
    {"name": "Amanda", "voice_id": "M6N6IdXhi5YNZyZSDe7k"},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TTS audio per turn (all speakers voice every turn). No composed audio."
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
        help="Directory to save generated audio files."
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
        raise ValueError(f"⚠️ No utterances found for conversation '{conv_id}' in utterances.jsonl.")

    return utterances


def generate_all_speakers_per_turn(client, utterances, conversation_id, output_dir):
    unique_speakers = list(sorted(set(utt["speaker"] for utt in utterances if utt.get("speaker") is not None)))
    num_speakers = len(unique_speakers)

    print(f"👥 Found {num_speakers} unique speakers: {unique_speakers}")

    if num_speakers == 0:
        raise ValueError("⚠️ No valid speakers found in this conversation.")

    if num_speakers > 3:
        print(f"⚠️ Skipping conversation with >3 speakers: {unique_speakers}")
        return None

    # Map speakers to voices
    speaker_voice_map = {}
    for i, speaker in enumerate(unique_speakers):
        speaker_voice_map[speaker] = VOICE_SLOTS[i % len(VOICE_SLOTS)]

    print(f"🗣 Speaker → Voice mapping: {speaker_voice_map}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i, utt in enumerate(utterances):
        current_text = utt["text"]
        ssml_prompt = f"<speak>{current_text}</speak>"

        for speaker_id, voice in speaker_voice_map.items():
            print(f"🎙️ Generating turn {i+1}/{len(utterances)} for {speaker_id} using {voice['name']}")

            audio_stream = client.text_to_speech.convert(
                text=ssml_prompt,
                voice_id=voice["voice_id"],
                model_id="eleven_multilingual_v2",
                voice_settings={"ssml": True},
                output_format="mp3_44100_128",
            )

            file_name = f"{speaker_id}_{conversation_id}_{i+1}.mp3"
            out_file = Path(output_dir) / file_name

            with open(out_file, "wb") as f:
                for chunk in audio_stream:
                    f.write(chunk)

            print(f"✅ Saved: {out_file}")


def main():
    args = parse_args()
    key = args.api_key or os.getenv("ELEVEN_LABS_API_KEY")
    if not key:
        print("⚠️ Error: ElevenLabs API key not provided.")
        return

    corpus_dir = os.path.expanduser(f"~/.convokit/saved-corpora/{args.corpus_name}")

    try:
        utterances = fetch_utterances_for_conversation(corpus_dir, args.conversation_id)

        client = ElevenLabs(api_key=key)

        generate_all_speakers_per_turn(
            client, utterances, args.conversation_id, args.output
        )

        print(f"✅ Completed generating turn files for {args.conversation_id} (all speakers voice each turn).")
    except Exception as e:
        print(f"❌ Generation failed: {e}")


if __name__ == "__main__":
    main()

