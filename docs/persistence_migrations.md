# Persistence migrations

The SQLite persistence layer now keeps schema versions in a lightweight
`schema_migrations` table. Each migration is a small Python function registered
in `src/quant_engine/persistence/db.py` and applied automatically when a
connection is opened via `init_db`/`session`.

## How it works

- `migrate()` ensures the `schema_migrations` table exists.
- Each migration has a numeric version and name.
- Pending migrations are executed in order and then recorded.

## Adding a migration

1. Create a new migration function (e.g. `_migration_3`).
2. Append it to the `MIGRATIONS` list with the next version number.
3. Keep migrations idempotent where possible (use `CREATE TABLE IF NOT EXISTS`,
   `CREATE INDEX IF NOT EXISTS`, and guard `ALTER TABLE` with try/except).

This approach avoids manual SQL upgrades and keeps the schema history explicit.
