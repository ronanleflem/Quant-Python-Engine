# Cleanup legacy (notes)

## Composants legacy encore presents
- Java/FE: endpoints legacy d'optimisation et stats synchrones.
- Python: endpoints historiques `/submit`, `/submit/async`, `/status/{job_id}`, `/result/{job_id}`.

## Safe removals maintenant
- Aucun retire immediat propose (compatibilite historique).

## Deprecation proposee
- Marquer `/submit` et `/submit/async` comme deprecated apres stabilisation canonical.
- Date cible proposee: 2026-04-30.

## TODO
- Confirmer dependances Java/FE sur endpoints legacy.
- Ajouter warnings deprecation dans docs et logs.

## Migration guide
- Voir `docs/migration_legacy_wrappers.md` pour la correspondance old->new des wrappers `quant_engine.stats.*`.
