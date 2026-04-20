"""Download German sentences from Tatoeba for TTS training."""

import argparse
import bz2
import csv
import io
import os
import random
import re
import tempfile
import urllib.request

TATOEBA_URL = (
    "https://downloads.tatoeba.org/exports/per_language/deu/"
    "deu_sentences_detailed.tsv.bz2"
)

MIN_LENGTH = 20
MAX_LENGTH = 300
_URL_RE = re.compile(r"https?://|www\.")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")


def _is_suitable(text: str) -> bool:
    if len(text) < MIN_LENGTH or len(text) > MAX_LENGTH:
        return False
    if text == text.upper():
        return False
    if not text.rstrip().endswith((".", "!", "?", "…")):
        return False
    if _URL_RE.search(text) or _EMAIL_RE.search(text):
        return False
    return True


def _download_and_extract(url: str) -> list[str]:
    print(f"Downloading {url} ...")
    with tempfile.NamedTemporaryFile(suffix=".tsv.bz2", delete=False) as tmp:
        tmp_path = tmp.name
        urllib.request.urlretrieve(url, tmp_path)

    try:
        with open(tmp_path, "rb") as f:
            raw = bz2.decompress(f.read())
    finally:
        os.unlink(tmp_path)

    sentences = []
    reader = csv.reader(io.StringIO(raw.decode("utf-8")), delimiter="\t")
    for row in reader:
        if len(row) >= 3:
            sentences.append(row[2].strip())

    return sentences


def run(args: argparse.Namespace, config: dict) -> None:
    count = args.count
    seed = args.seed
    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "sentences_tatoeba_de.txt"
    )

    sentences = _download_and_extract(TATOEBA_URL)
    print(f"Downloaded {len(sentences)} sentences")

    filtered = [s for s in sentences if _is_suitable(s)]
    print(f"After filtering: {len(filtered)} suitable sentences")

    rng = random.Random(seed)
    rng.shuffle(filtered)
    selected = filtered[:count]

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("# German sentences from Tatoeba (CC BY 2.0 FR)\n")
        f.write("# https://tatoeba.org — License: https://creativecommons.org/licenses/by/2.0/fr/\n")
        f.write(f"# Selected {len(selected)} of {len(filtered)} filtered sentences\n\n")
        for s in selected:
            f.write(s + "\n")

    print(f"Wrote {len(selected)} sentences to {output}")
