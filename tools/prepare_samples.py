"""Select one original reference per protagonist, without generating or replacing voices."""
import argparse
import collections
import csv
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("catalog", type=pathlib.Path)
parser.add_argument("selection", type=pathlib.Path)
args = parser.parse_args()

def read(name):
    with (args.catalog / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

lines, links = read("protagonistas.csv"), read("audio_map.csv")
by_id = collections.defaultdict(list)
for link in links:
    if link["match_evidence"] == "masked_hash_and_bank_prefix":
        by_id[link["id"]].append(link)
selection, samples = [], []
for who in ["MICHAEL", "FRANKLIN", "TREVOR"]:
    for row in lines:
        if row["character"] != who or not row["portuguese"] or not 30 < len(row["english"]) < 100:
            continue
        hits = [x for x in by_id[row["id"]] if 1.5 < float(x["duration_seconds"]) < 5]
        if hits:
            hit = hits[0]
            filename = who.lower() + "_original.wav"
            selection.append([filename, hit["awc_path"], hit["stream_hash_29bit"], row["id"]])
            samples.append({**row, **hit, "filename": filename})
            break
if len(samples) != 3:
    raise SystemExit("Nao foi possivel selecionar os tres protagonistas.")
with args.selection.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t", lineterminator="\n")
    writer.writerow(["filename", "awc_path", "stream_hash", "id"])
    writer.writerows(selection)
with (args.catalog / "amostras_originais.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(samples[0]))
    writer.writeheader()
    writer.writerows(samples)
print("Selecionadas tres referencias originais.")
