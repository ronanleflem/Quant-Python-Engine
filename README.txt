Quantitative strategy optimisation engine.

## Database configuration

The engine can persist experiment metadata to a database.  The DSN is provided
via the ``DB_DSN`` environment variable.  When the variable is absent the engine
falls back to a local SQLite file located at ``.db/quant.db``.

To run migrations and start the auxiliary services, ensure the dependencies are
installed and run:

```
alembic upgrade head
```

A sample ``.env.example`` file is provided with sensible defaults.  When using
SQLite no additional services are required.
