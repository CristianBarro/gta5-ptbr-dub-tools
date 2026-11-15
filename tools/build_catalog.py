"""Build an evidence-preserving GTA V dialogue catalog using Python's standard library.

Input: local CodeWalker archive scan and an existing decompiled script snapshot.
Never writes to the game directory. Hash matches are candidates, not listening proof.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import csv
import hashlib
import json
import pathlib
import re
import sqlite3

MAIN = {"MICHAEL", "FRANKLIN", "TREVOR"}
SKIP = re.compile(r"^(am_|fm_|fmmc|gb_|mp|pb_|replay_|app|net_|ugc|ob_|freemode$|public_mission)")
STRING = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')
IDENT = re.compile(r"^[A-Za-z][A-Za-z0-9_]{1,30}$")


def joaat(value: str) -> int:
    h = 0
    for c in value.lower().encode("utf-8"):
        h = (h + c) & 0xFFFFFFFF
        h = (h + (h << 10)) & 0xFFFFFFFF
        h ^= h >> 6
    h = (h + (h << 3)) & 0xFFFFFFFF
    h ^= h >> 11
    return (h + (h << 15)) & 0xFFFFFFFF


def split_call(source: str, start: int):
    """start points just after '('. Split arguments respecting nested calls and strings."""
    result, begin, depth, quoted, escaped = [], start, 0, False, False
    for pos in range(start, min(len(source), start + 5000)):
        char = source[pos]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "([{":
            depth += 1
        elif char == ")" and depth == 0:
            return result + [source[begin:pos].strip()]
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            result.append(source[begin:pos].strip())
            begin = pos + 1
    return []


def rows(path):
    with pathlib.Path(path).open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def write_csv(path, values, fields):
    with pathlib.Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)


def clean(text):
    return re.sub(r"~[^~]*~", "", text).strip()


def extract_script(path, metadata_hashes):
    source = path.read_text(encoding="utf-8-sig", errors="replace")
    newlines = [m.start() for m in re.finditer("\n", source)]
    aliases = collections.defaultdict(set)
    for match in re.finditer(r'\b(\w+)\s*=\s*"([A-Z][A-Z0-9_]*)"', source):
        aliases[match[1]].add(match[2])
    registration_names = []
    signature = re.compile(r"^void (func_\d+)\(var uParam0, int iParam1, int iParam2, char\* sParam3,", re.M)
    for match in signature.finditer(source):
        body = source[match.end():match.end() + 1200]
        if "StringCopy" in body and "sParam3" in body and re.search(r"uParam0\[iParam1\s*/\*10\*/\]", body):
            registration_names.append(match[1])
    registrations = collections.defaultdict(set)
    reg_evidence = []
    for name in registration_names:
        for match in re.finditer(r"\b" + name + r"\(", source):
            args = split_call(source, match.end())
            if len(args) != 6 or not args[1].isdigit():
                continue
            voice = args[3].strip('"') if args[3].startswith('"') else None
            if voice is None and len(aliases[args[3]]) == 1:
                voice = next(iter(aliases[args[3]]))
            if not voice or not IDENT.fullmatch(voice):
                continue
            group, slot = args[0].lstrip("&"), int(args[1])
            registrations[group, slot].add(voice)
            reg_evidence.append(dict(script=path.name, group=group, slot=slot, character=voice,
                                     line=bisect.bisect_left(newlines, match.start()) + 1))
    found = {}
    for match in STRING.finditer(source):
        label = match[1]
        if not IDENT.fullmatch(label):
            continue
        h = joaat(label + "SL")
        if h in metadata_hashes:
            found.setdefault(label.upper(), dict(script=path.name, line=bisect.bisect_left(newlines, match.start()) + 1))
    # Calls whose first three args are (conversation struct, text table, conversation label).
    hints = {}
    for match in re.finditer(r'\bfunc_\d+\(\s*&?(\w+)\s*,\s*"([A-Za-z0-9_]+)"\s*,\s*"([A-Za-z0-9_]+)"', source):
        if match[3].upper() in found:
            hints.setdefault(match[3].upper(), set()).add((match[1], match[2].lower() + ".gxt2"))
    return found, hints, registrations, reg_evidence


def build(data, text_data, script_root, output):
    output.mkdir(parents=True, exist_ok=True)
    localized = {}
    raw_text_count = 0
    for item in rows(text_data / "localized_text.csv"):
        raw_text_count += 1
        key = item["language"], item["table"], int(item["hash"], 16)
        old = localized.get(key)
        if old is None or item["path"].startswith("update\\"):
            localized[key] = item
    en_index = collections.defaultdict(dict)
    for (lang, table, h), item in localized.items():
        if lang == "en-US":
            en_index[h][table] = item["text"]
    audio = [x for x in rows(data / "audio_streams.csv") if int(x["sample_rate"]) > 0 and int(x["samples"]) > 0]
    audio_index = collections.defaultdict(list)
    for item in audio:
        audio_index[int(item["stream_hash_29bit"], 16)].append(item)
    roots = collections.defaultdict(list)
    registration_evidence, scanned, skipped = [], [], []
    for n, path in enumerate(sorted(script_root.glob("*.c"))):
        if SKIP.match(path.stem):
            skipped.append(path.name)
            continue
        found, hints, regs, evidence = extract_script(path, en_index)
        registration_evidence.extend(evidence)
        scanned.append(dict(name=path.name, bytes=path.stat().st_size))
        for root, ref in found.items():
            table_hints = hints.get(root, set())
            for table, value in en_index[joaat(root + "SL")].items():
                if not re.fullmatch(r"[0-9A-Zz]+", value) or len(value) % 3:
                    continue
                groups = {g for g, t in table_hints if t == table}
                root_regs = collections.defaultdict(set)
                for (group, slot), voices in regs.items():
                    if not groups or group in groups:
                        root_regs[slot].update(voices)
                roots[table, root].append((ref, dict(root_regs)))
        if len(scanned) % 75 == 0:
            print("SCRIPTS", len(scanned), "CONVERSATIONS", len(roots), flush=True)
    dialogue, links, pauses = [], [], []
    alias_cache = {}
    for (table, root), evidence in sorted(roots.items()):
        metadata = en_index[joaat(root + "SL")][table]
        for pos in range(len(metadata) // 3):
            number = pos + 1
            label = root + "_" + str(number)
            alias = en_index.get(joaat(label + "A"), {}).get(table, "")
            # Engine also supports a single line without a numbered suffix.
            if number == 1 and not alias:
                alias = en_index.get(joaat(root + "A"), {}).get(table, "")
                if alias:
                    label = root
            if not alias:
                continue
            h = joaat(label)
            en = localized.get(("en-US", table, h), {})
            pt = localized.get(("pt-BR", table, h), {})
            line_id = table.removesuffix(".gxt2") + ":" + label
            try:
                slot = -1 if metadata[pos * 3] == "z" else int(metadata[pos * 3], 36)
            except ValueError:
                slot = -1
            voices = set()
            for _, registry in evidence:
                voices.update(registry.get(slot, set()))
            character = next(iter(voices)) if len(voices) == 1 else ""
            row = dict(id=line_id, table=table, conversation=root, line=number, label=label, subtitle_hash=f"{h:08X}",
                       character=character, speaker_slot=slot, speaker_evidence="unique_script_registration" if character else "unresolved_or_conflicting",
                       character_candidates="|".join(sorted(voices)), audio_alias=alias, english=clean(en.get("text", "")),
                       portuguese=clean(pt.get("text", "")), text_source_en=en.get("path", ""), text_source_pt=pt.get("path", ""),
                       script_refs="|".join(sorted({str(ref["script"]) + ":" + str(ref["line"]) for ref, _ in evidence})))
            if alias.startswith("SFX_") or not en:
                row["kind"] = "pause_or_control" if alias.startswith("SFX_") else "no_subtitle"
                pauses.append(row)
                continue
            if alias not in alias_cache:
                hits = []
                for variant in range(1, 100):
                    candidate = alias.lower() + "_" + f"{variant:02d}"
                    for item in audio_index.get(joaat(candidate) & 0x1FFFFFFF, []):
                        bank = item["awc_path"].split("\\")[-1].removesuffix(".awc")
                        bank_match = candidate.startswith(bank + "_")
                        hits.append(dict(item, candidate_name=candidate, variant=variant,
                                         match_evidence="masked_hash_and_bank_prefix" if bank_match else "masked_hash_only_review"))
                # Keep weaker hash matches visible for review, never silently certify them.
                alias_cache[alias] = hits
            hits = alias_cache[alias]
            supported = [x for x in hits if x["match_evidence"] == "masked_hash_and_bank_prefix"]
            row["audio_status"] = "bank_and_hash_match" if supported else ("hash_candidates_review" if hits else "pending_audio_link")
            row["audio_variants"] = len(supported)
            row["kind"] = "dialogue"
            dialogue.append(row)
            for hit in hits:
                links.append(dict(id=line_id, audio_alias=alias, **hit))
    fields = list(dialogue[0]) if dialogue else []
    write_csv(output / "dialogos.csv", dialogue, fields)
    main = [x for x in dialogue if x["character"] in MAIN]
    write_csv(output / "protagonistas.csv", main, fields)
    write_csv(output / "pausas_e_controles.csv", pauses, fields)
    link_fields = ["id", "audio_alias"] + (list(audio[0]) if audio else []) + ["candidate_name", "variant", "match_evidence"]
    write_csv(output / "audio_map.csv", links, link_fields)
    write_csv(output / "speaker_registrations.csv", registration_evidence, ["script", "group", "slot", "character", "line"])
    main_ambient = []
    for item in audio:
        bank = item["awc_path"].split("\\")[-1]
        for who in MAIN:
            if re.match(r"^(?:pain_|breathing_)?" + who.lower() + r"(?:_|\.)", bank) and "s_misc.rpf" in item["awc_path"]:
                main_ambient.append(dict(character=who, status="pending_transcription_or_subtitle_link", **item))
    write_csv(output / "protagonistas_ambiente.csv", main_ambient, ["character", "status"] + list(audio[0]))
    supported_by_id = collections.defaultdict(list)
    for link in links:
        if link["match_evidence"] == "masked_hash_and_bank_prefix":
            supported_by_id[link["id"]].append(link)
    queue = []
    for row in main:
        if not row["portuguese"]:
            continue
        for link in supported_by_id[row["id"]]:
            queue.append(dict(id=row["id"], character=row["character"], text_pt=row["portuguese"], english=row["english"],
                              awc_path=link["awc_path"], stream_hash=link["stream_hash_29bit"], sample_rate=link["sample_rate"],
                              samples=link["samples"], target_duration_seconds=round(int(link["samples"]) / int(link["sample_rate"]), 8),
                              status="awaiting_voice_generation_and_review"))
    queue_fields = ["id", "character", "text_pt", "english", "awc_path", "stream_hash", "sample_rate", "samples", "target_duration_seconds", "status"]
    write_csv(output / "fila_dublagem.csv", queue, queue_fields)
    db_path = output / "gta_dialogue.sqlite"
    # Rebuild generated tables; preserve no user tables in this generated, dedicated database.
    db = sqlite3.connect(db_path)
    for name, values, columns in [("dialogue", dialogue, fields), ("audio_map", links, link_fields),
                                   ("ambient_protagonists", main_ambient, ["character", "status"] + list(audio[0])),
                                   ("localized_text", list(localized.values()), ["language", "path", "table", "hash", "text"]),
                                   ("audio_streams", audio, list(audio[0]))]:
        db.execute('DROP TABLE IF EXISTS "' + name + '"')
        db.execute('CREATE TABLE "' + name + '" (' + ",".join('"' + c + '" TEXT' for c in columns) + ")")
        db.executemany('INSERT INTO "' + name + '" VALUES (' + ",".join("?" for _ in columns) + ")",
                       ([str(r.get(c, "")) for c in columns] for r in values))
    db.execute('CREATE INDEX dialogue_id ON dialogue(id)')
    db.execute('CREATE INDEX dialogue_character ON dialogue(character)')
    db.execute('CREATE INDEX audio_map_id ON audio_map(id)')
    db.execute('CREATE INDEX localized_lookup ON localized_text(language,"table",hash)')
    db.commit()
    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    db.close()
    summary = dict(game_directory=r"D:\SteamLibrary\steamapps\common\Grand Theft Auto V", edition="Legacy",
                   audio_archives_scanned=len({x["awc_path"].split(".rpf")[0] for x in rows(data / "audio_streams.csv")}), awc_containers=len({x["awc_path"] for x in rows(data / "audio_streams.csv")}),
                   raw_stream_entries=sum(1 for _ in rows(data / "audio_streams.csv")), playable_streams=len(audio),
                   localized_rows_raw=raw_text_count, localized_rows_after_precedence=len(localized),
                   dialogue_lines=len(dialogue), protagonist_lines=len(main),
                   protagonists=dict(collections.Counter(x["character"] for x in main)),
                   protagonist_ambient_streams=len(main_ambient),
                   queued_dubbing_variants=len(queue),
                   missing_portuguese_dialogue_lines=sum(not x["portuguese"] for x in dialogue),
                   audio_status=dict(collections.Counter(x["audio_status"] for x in dialogue)),
                   protagonist_audio_status=dict(collections.Counter(x["audio_status"] for x in main)),
                   conversations=len(roots), controls_and_unsubtitled=len(pauses), scripts_scanned=len(scanned),
                   scripts_skipped=len(skipped), sqlite_integrity=integrity,
                   limitations=["Catalog is partial: labels constructed at runtime may be missing.",
                                "Source scripts are an existing local snapshot; compatibility with the installed build is not proved.",
                                "Speaker registrations are static evidence, not in-game or listening validation.",
                                "Audio hashes store only 29 bits; bank-prefix matches reduce ambiguity, hash-only results need review.",
                                "Cutscene channel tracks need segmentation and speaker review.",
                                "No audio generation or game modification was performed."])
    (output / "status.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "script_inventory.json").write_text(json.dumps(dict(scanned=scanned, skipped=skipped), indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=pathlib.Path, required=True)
    parser.add_argument("--text-data", type=pathlib.Path, required=True)
    parser.add_argument("--scripts", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    build(args.data, args.text_data, args.scripts, args.output)
