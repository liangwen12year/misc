# Audio Pipeline Usage Guide

This guide explains how to use the updated Python TTS script (`gen_sample_11_lab.py`) and the Bash sample generator (`generate_samples.sh`) to synthesize audio from a Convokit corpus via the ElevenLabs API.

---

## Prerequisites

1. **Python 3.12+**
   ```bash
   python3.12 -m venv venv312
   source venv312/bin/activate    # or activate.fish in Fish shell
   pip install --upgrade pip
   ```

2. **Install dependencies**
   ```bash
   pip install convokit elevenlabs jq
   ```

3. **ElevenLabs API Key**
   - Obtain from your ElevenLabs account settings.
   - Export as an environment variable or pass with `-k`.

---

## Python Script: `gen_sample_11_lab.py`

This script will:

1. **Auto-download** the specified Convokit corpus (e.g. `movie-corpus`) on first run.
2. **List utterances** if run with `--list-utterances`.
3. **Fetch** the text of two sets of utterance IDs (previous & next) from the cached corpus.
4. **Build** an SSML prompt with 2‑second breaks between turns.
5. **Generate** speech via ElevenLabs TTS and save as an MP3.

### Command-line arguments

- `-PRI`, `--previous-request-ids`  : Utterance IDs for the previous segment (space-separated).
- `-NRI`, `--next-request-ids`      : Utterance IDs for the next segment (space-separated).
- `-C`,  `--corpus-name`            : Convokit corpus identifier (default: `movie-corpus`).
- `--list-utterances`               : Print a sample of available IDs and exit.
- `--gender`                        : Voice gender (`male`|`female`|`other`, default: `female`).
- `--age`                           : Voice age group (`child`|`young`|`adult`|`senior`, default: `adult`).
- `--accent`                        : Voice accent code (e.g. `us`, `uk`, default: `us`).
- `-o`, `--output`                  : Output MP3 file path (default: `output.mp3`).
- `-k`, `--api-key`                 : ElevenLabs API key (falls back to `$ELEVEN_LABS_API_KEY`).

### Example: synthesize one clip

```bash
python gen_sample_11_lab.py \
  -PRI L1045 L1044 \
  -NRI L985 L984 \
  -C movie-corpus \
  -k sk_... \
  -o sample.mp3
```

On first run for `movie-corpus`, the script auto-downloads and caches the data. Use `--list-utterances` to inspect valid utterance IDs.

---

## Bash Script: `generate_samples.sh`

Automates generation of multiple samples by pairing adjacent utterances.

### Setup

- Place `generate_samples.sh` in the same directory as `gen_sample_11_lab.py`.
- Ensure it’s executable:
  ```bash
  chmod +x generate_samples.sh
  ```
- Install `jq` for JSON parsing:
  ```bash
  sudo apt install jq    # or your distro’s package manager
  ```

### Configuration

Edit the top of `generate_samples.sh` to set:

- `CORPUS_DIR` : Path to your cached corpus (e.g. `~/.convokit/saved-corpora/movie-corpus`).
- `SCRIPT`     : Path to `gen_sample_11_lab.py`.
- `API_KEY`    : Your ElevenLabs API key.
- `OUTPUT_DIR` : Directory to store generated MP3s.
- `NUM_SAMPLES`: Number of utterance pairs to process (default: 10).

### Usage

```bash
./generate_samples.sh
```

This will:

1. Read the first `NUM_SAMPLES + 1` utterance IDs.
2. Pair each ID with the next to form (previous → next) contexts.
3. Invoke `gen_sample_11_lab.py` for each pair, writing MP3 files in `OUTPUT_DIR`.

---

## Tips & Troubleshooting

- **Caching behavior**: The Python script now auto-downloads any missing corpus, so you no longer need a separate `--list-utterances` run just to cache.
- **Python version**: spaCy (a Convokit dependency) may fail to compile on Python 3.13. Use Python 3.12+ for compatibility.
- **Debugging**: Add `set -x` at the top of `generate_samples.sh` to trace execution.

---

Enjoy generating audio from your Convokit conversations! Adjust voice settings, sample count, or corpus as desired.


