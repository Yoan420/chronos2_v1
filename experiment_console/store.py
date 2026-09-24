"""Small transactional SQLite metadata store (never writes scientific files)."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, source_key TEXT UNIQUE, data TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS plans (id TEXT PRIMARY KEY, data TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS requests (key TEXT PRIMARY KEY, run_id TEXT NOT NULL)")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA busy_timeout=30000")
            with db:
                yield db
        finally:
            db.close()

    def get(self, run_id):
        with self.connection() as db:
            row = db.execute("SELECT data FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return json.loads(row[0])

    def list(self):
        with self.connection() as db:
            rows = db.execute("SELECT data FROM runs ORDER BY rowid DESC").fetchall()
        return [json.loads(r[0]) for r in rows]

    def insert(self, run):
        with self.connection() as db:
            db.execute("INSERT INTO runs VALUES (?,?,?)", (run['id'], run.get('source_key'), json.dumps(run, ensure_ascii=False)))
        return run

    def update(self, run_id, **changes):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            data = json.loads(row[0])
            data.update(changes)
            data['updated_at'] = now()
            db.execute("UPDATE runs SET data=? WHERE id=?", (json.dumps(data, ensure_ascii=False), run_id))
        return data

    def save_plan(self, plan):
        with self.connection() as db:
            db.execute("INSERT INTO plans VALUES (?,?)", (plan['id'], json.dumps(plan, ensure_ascii=False)))

    def update_if_status(self, run_id, statuses, **changes):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT data FROM runs WHERE id=?', (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            data = json.loads(row[0])
            if data['status'] in statuses:
                data.update(changes, updated_at=now())
                db.execute('UPDATE runs SET data=? WHERE id=?', (json.dumps(data, ensure_ascii=False), run_id))
            return data

    def plan(self, plan_id):
        with self.connection() as db:
            row = db.execute("SELECT data FROM plans WHERE id=?", (plan_id,)).fetchone()
        if row is None:
            raise ValueError('Récapitulatif introuvable. Préparez de nouveau le run.')
        return json.loads(row[0])

    def request_run(self, key):
        with self.connection() as db:
            row = db.execute("SELECT run_id FROM requests WHERE key=?", (key,)).fetchone()
        return self.get(row[0]) if row else None

    def bind_request(self, key, run_id):
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO requests VALUES (?,?)", (key, run_id))
