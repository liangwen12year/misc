#!/usr/bin/env python3
"""
Audio generation pipeline using the ElevenLabs Python SDK and Convokit for context lookup.
Tuning knobs: gender, age, accent.
Accepts previous and next request IDs to fetch text segments from a Convokit corpus.
Requires: convokit (install via `pip install convokit` or `conda install -c conda-forge convokit`).
Note: Use --list-utterances to view available utterance IDs in the corpus.
Note: If the cache directory for the corpus is missing, it will be automatically downloaded.
Each speaker turn is separated by a 2-second break using SSML.
"""

import sys
import os
import argparse
from elevenlabs.client import ElevenLabs

# Ensure convokit is installed
try:
    from convokit import Corpus, download
except ImportError:
    sys.exit(
        "⚠️ Error: convokit is not installed. "
        "Please install it with `pip install convokit` and try again."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TTS audio via ElevenLabs API using Convokit-based context segments"
    )
    parser.add_argument(
        "--previous-request-ids", "-PRI",
        nargs='+',
        help="List of previous request IDs (utterance IDs in Convokit)."
    )
    parser.add_argument(
        "--next-request-ids", "-NRI",
        nargs='+',
        help="List of next request IDs (utterance IDs in Convokit)."
    )
    parser.add_argument(
        "--corpus-name", "-C",
        default="movie-corpus",
        help="Convokit corpus identifier (default: movie-corpus)."
    )
    parser.add_argument(
        "--list-utterances", action="store_true",
        help="List available utterance IDs and exit."
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


def load_corpus(name: str):
    """
    Ensures the specified Convokit corpus is present: if not cached, downloads it,
    then loads it from the cache.
    """
    cache_root = os.path.expanduser("~/.convokit/saved-corpora")
    cache_dir = os.path.join(cache_root, name)

    os.makedirs(cache_root, exist_ok=True)

    if not os.path.exists(cache_dir):
        print(f"🔄 Corpus '{name}' not found in cache. Downloading now…")
        download(name)

    return Corpus(filename=cache_dir)


def list_utterances(corpus, limit=20):
    """
    Print the first `limit` utterance IDs and their text snippets.
    """
    for i, utt in enumerate(corpus.iter_utterances()):
        print(f"ID: {utt.id}  Text: {utt.text[:60]!r}...")
        if i + 1 >= limit:
            break


def fetch_texts(corpus, ids):
    """
    Given a list of utterance IDs, return a list of their text.
    """
    texts = []
    for uid in ids:
        utt = corpus.get_utterance(uid)
        if utt is None:
            sys.exit(f"⚠️ Utterance ID '{uid}' not found in corpus.")
        texts.append(utt.text)
    return texts


def select_voice(client, gender, age, accent):
    """
    Pick the first voice matching the given labels or fallback.
    """
    resp = client.voices.get_all()
    voices = resp.voices

    def matches(v):
        labels = getattr(v, "labels", {}) or {}
        return (
            labels.get("gender", "").lower() == gender.lower()
            and labels.get("age",    "").lower() == age.lower()
            and accent.lower() in labels.get("accent", "").lower()
        )

    filtered = [v for v in voices if matches(v)]
    return filtered[0] if filtered else voices[0]


def build_prompt(prev_texts, next_texts):
    """
    Combine text segments into an SSML prompt with 2-second breaks between turns.
    """
    ssml_parts = ["<speak>"]
    for segment in prev_texts + next_texts:
        ssml_parts.append(f"{segment}<break time=\"2s\"/>")
    ssml_parts.append("</speak>")
    return "".join(ssml_parts)


def generate_audio(client, prev_ids, next_ids, corpus, gender, age, accent, output_path):
    """
    Fetch context via Convokit, generate speech with SSML breaks, and save to `output_path`.
    """
    prev_segments = fetch_texts(corpus, prev_ids)
    next_segments = fetch_texts(corpus, next_ids)
    prompt = build_prompt(prev_segments, next_segments)
    voice = select_voice(client, gender, age, accent)

    audio_stream = client.text_to_speech.convert(
        text=prompt,
        voice_id=voice.voice_id,
        model_id="eleven_multilingual_v2",
        voice_settings={"ssml": True},
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
        sys.exit("⚠️ Error: ElevenLabs API key not provided.")

    client = ElevenLabs(api_key=key)
    corpus = load_corpus(args.corpus_name)

    if args.list_utterances:
        list_utterances(corpus)
        sys.exit(0)

    if not (args.previous_request_ids and args.next_request_ids):
        sys.exit("⚠️ Error: Provide --previous-request-ids and --next-request-ids, or use --list-utterances.")

    try:
        out = generate_audio(
            client,
            args.previous_request_ids,
            args.next_request_ids,
            corpus,
            args.gender,
            args.age,
            args.accent,
            args.output
        )
        print(f"✅ Generated audio saved to {out}")
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

