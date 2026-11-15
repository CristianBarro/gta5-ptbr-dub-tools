"""Search the generated local database. Python 3.10+; no dependencies."""
import argparse
import json
import pathlib
import sqlite3

parser = argparse.ArgumentParser(description="Consultar as falas mapeadas do GTA V")
parser.add_argument("--personagem", choices=["MICHAEL", "FRANKLIN", "TREVOR"], type=str.upper)
parser.add_argument("--texto", default="")
parser.add_argument("--limite", type=int, default=10)
parser.add_argument("--db", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1] / "gta_dialogue.sqlite")
args = parser.parse_args()
db = sqlite3.connect(args.db.as_uri() + "?mode=ro", uri=True)
db.row_factory = sqlite3.Row
clauses, values = [], []
if args.personagem:
    clauses.append("character = ?")
    values.append(args.personagem)
if args.texto:
    clauses.append("(portuguese LIKE ? OR english LIKE ? OR label LIKE ?)")
    values.extend(["%" + args.texto + "%"] * 3)
where = " WHERE " + " AND ".join(clauses) if clauses else ""
for row in db.execute("SELECT id, character, portuguese, english, audio_alias, audio_status, script_refs FROM dialogue" + where + " LIMIT ?", values + [max(1, min(args.limite, 1000))]):
    print(json.dumps(dict(row), ensure_ascii=False))
db.close()
