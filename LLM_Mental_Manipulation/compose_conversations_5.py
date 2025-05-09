#!/usr/bin/env python3
import argparse
import os
import re
import itertools
from collections import defaultdict
from pydub import AudioSegment

def discover_conversations(input_dir):
    """
    Scan input_dir for files matching:
      {speakerID}_{conversationID}_{turn}.mp3
    Returns:
      convo_map: {
        convoID: {
          'speakers': { speakerID: { turn: filename, … }, … },
          'turns': sorted list of turn numbers common to all speakers
        },
        …
      }
    """
    file_re = re.compile(r'^(u\d+)_([A-Z]+\d+)_([0-9]+)\.mp3$')
    temp = defaultdict(lambda: defaultdict(set))
    filenames = {}

    for fn in os.listdir(input_dir):
        m = file_re.match(fn)
        if not m or not fn.endswith('.mp3'):
            continue
        speaker, convo, turn_s = m.groups()
        turn = int(turn_s)
        temp[convo][speaker].add(turn)
        filenames[(speaker, convo, turn)] = fn

    convo_map = {}
    for convo, sp_map in temp.items():
        speakers = list(sp_map.keys())
        common_turns = set.intersection(*(sp_map[sp] for sp in speakers))
        if not common_turns:
            print(f"⚠️  Conversation {convo} has no common turns; skipping.")
            continue
        turns_sorted = sorted(common_turns)
        speaker_files = {
            sp: { t: filenames[(sp, convo, t)] for t in turns_sorted }
            for sp in speakers
        }
        convo_map[convo] = {
            'speakers': speaker_files,
            'turns': turns_sorted
        }
    return convo_map

def compose_permutation(convo_id, speaker_order, convo_data, input_dir, output_dir):
    """
    Given one permutation of speakers, build the composed audio.
    Cycles through speaker_order for as many turns as there are.
    """
    turns = convo_data['turns']
    speaker_files = convo_data['speakers']
    segments = []

    for idx, turn in enumerate(turns):
        # cycle through the speaker_order:
        speaker = speaker_order[idx % len(speaker_order)]
        fn = speaker_files[speaker][turn]
        path = os.path.join(input_dir, fn)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} not found")
        seg = AudioSegment.from_mp3(path)
        segments.append(seg)

    # reverse the final sequence (remove [::-1] for normal order)
    segments = segments[::-1]

    out = AudioSegment.silent(duration=500)
    for i, seg in enumerate(segments):
        out += seg
        if i < len(segments) - 1:
            out += AudioSegment.silent(duration=1000)

    seq = "_".join(speaker_order)
    out_fn = f"{convo_id}_composed_{seq}.mp3"
    out_path = os.path.join(output_dir, out_fn)
    out.export(out_path, format="mp3")
    print(f"✅ Saved {out_path}")

def main():
    p = argparse.ArgumentParser(
        description="For each conversation, exhaustively alternate speakers & compose audio."
    )
    p.add_argument("--input-dir",  required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    convo_map = discover_conversations(args.input_dir)
    for convo_id, data in convo_map.items():
        speakers = list(data['speakers'].keys())
        print(f"\n🔄 Processing {convo_id}: {len(speakers)} speakers, {len(data['turns'])} turns")

        for perm in itertools.permutations(speakers):
            compose_permutation(convo_id, perm, data, args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()

