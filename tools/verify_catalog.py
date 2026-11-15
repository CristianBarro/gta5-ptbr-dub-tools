"""Check delivered data against independent known game identifiers and WAV headers."""
import argparse
import collections
import csv
import json
import pathlib
import sqlite3
import wave

from build_catalog import joaat

parser = argparse.ArgumentParser()
parser.add_argument("catalog", type=pathlib.Path)
args = parser.parse_args()
root = args.catalog

def read(name):
    with (root / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

dialogue, links = read("dialogos.csv"), read("audio_map.csv")
by_id = {row["id"]: row for row in dialogue}
assert len(by_id) == len(dialogue), "Repeated dialogue IDs"
assert all(f"{joaat(x['label']):08X}" == x["subtitle_hash"] for x in dialogue)
for link in links:
    assert link["id"] in by_id
    assert joaat(link["candidate_name"]) & 0x1FFFFFFF == int(link["stream_hash_29bit"], 16)
    if link["match_evidence"] == "masked_hash_and_bank_prefix":
        bank = link["awc_path"].split("\\")[-1].removesuffix(".awc")
        assert link["candidate_name"].startswith(bank + "_")
known = by_id["fkn1aud:F1_SRESC2_1"]
assert known["character"] == "FRANKLIN" and known["speaker_slot"] == "1"
assert known["audio_alias"] == "FKN1_DVAA"
assert any(x["id"] == known["id"] and x["stream_hash_29bit"] == "0B8FB153" and x["awc_path"].endswith("ss_ff.rpf\\fkn1.awc") for x in links)
characters = collections.defaultdict(set)
for row in dialogue:
    if row["character"] and row["audio_status"] == "bank_and_hash_match":
        characters[row["audio_alias"]].add(row["character"])
assert not {alias: names for alias, names in characters.items() if len(names) > 1}, "Conflicting speakers for an audio alias"
samples = read("amostras_originais.csv")
assert {row["character"] for row in samples} == {"MICHAEL", "FRANKLIN", "TREVOR"}
for sample in samples:
    with wave.open(str(root / "amostras" / sample["filename"]), "rb") as w:
        assert w.getframerate() == int(sample["sample_rate"])
        assert w.getnframes() == int(sample["samples"]), "ADPCM tail padding must be trimmed"
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        assert any(w.readframes(w.getnframes())), "Silent reference extraction"
queue = read("fila_dublagem.csv")
for row in queue:
    assert row["character"] in {"MICHAEL", "FRANKLIN", "TREVOR"} and row["text_pt"]
    assert abs(float(row["target_duration_seconds"]) - int(row["samples"]) / int(row["sample_rate"])) < 0.00000001
db = sqlite3.connect((root / "gta_dialogue.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
assert db.execute("SELECT COUNT(*) FROM dialogue").fetchone()[0] == len(dialogue)
db.close()
result = dict(status="passed", dialogue_rows=len(dialogue), audio_links_checked=len(links),
              reference_wavs_checked=len(samples), dubbing_queue_variants_checked=len(queue),
              checks=["known Franklin subtitle, speaker slot and stream", "unique IDs", "subtitle hashes", "AWC candidate hashes and banks",
                      "speaker conflicts", "reference WAV formats, exact lengths and nonzero PCM", "dubbing queue durations", "SQLite integrity"])
(root / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
