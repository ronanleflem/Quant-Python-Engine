# Plan d’implémentation des blocs restants

## Constats rapides (par fichier cible)

- **api/app.py** : la file de jobs et l’exécution asynchrone sont présentes via `run_next_job`, mais elles ne sont pas encore câblées à un worker ou à un endpoint dédié (la logique reste interne). Les endpoints exposent les résultats mais ne proposent pas encore un listing/monitoring complet des jobs (pagination, filtres, etc.).
- **integrations/java_client.py** : `get_positions()` avale silencieusement les erreurs réseau, ce qui masque les causes des échecs (absence de logs/retours détaillés).
- **strategies/runner.py** : `_request_ingestion_on_gap()` est un stub (retour immédiat), donc la relance d’ingestion historique côté Java n’est pas encore branchée.
- **filters/utils.py** : le dictionnaire `FILTER_SUMMARIES` contient des descriptions utiles, mais il n’existe pas de surface publique pour les exposer (API/CLI/exports), ni de format “metadata” standard pour les specs.
- **stats/runner.py** : le fichier `stats_details.parquet` est généré vide, ce qui indique qu’un export détaillé (ex. long-form, contributions par événement/condition) reste à implémenter.
- **levels/runner.py** : les sorties sont fonctionnelles mais restent minimalistes (retour global), sans synthèse par symbole/type ni métriques d’exécution (temps, volumes), ce qui limite l’observabilité.

## Plan d’implémentation proposé (court et actionnable)

1. **Orchestration API / jobs (api/app.py)**
   - Créer un worker léger (CLI ou service) qui appelle périodiquement `run_next_job()`.
   - Ajouter un endpoint de monitoring pour lister les jobs (statut, type, pagination).
   - Normaliser la gestion des erreurs (codes/fields communs pour l’état `failed`).

2. **Intégration Java (integrations/java_client.py)**
   - Ajouter un logging standardisé sur `get_positions()` (statut HTTP, message d’erreur).
   - Prévoir un mécanisme d’authentification optionnel (ex. header token via env) afin de sécuriser les appels Java.

3. **Fallback d’ingestion (strategies/runner.py)**
   - Implémenter `_request_ingestion_on_gap()` pour déclencher `request_historical_ingestion()` quand l’OHLC Java est absent.
   - Ajouter des garde-fous (cooldown, seuils de couverture) afin d’éviter les relances excessives.
   - Retourner/propager le statut de l’ingestion pour l’observabilité.

4. **Documentation/metadata des filtres (filters/utils.py)**
   - Exposer `FILTER_SUMMARIES` via une fonction publique ou un endpoint (`/filters/metadata`).
   - Définir un schéma de metadata (ex. colonnes requises, paramètres attendus) réutilisable par la docs et la validation.

5. **Exports détaillés des stats (stats/runner.py)**
   - Produire `stats_details.parquet` à partir du long-form (événements, conditions, targets).
   - Ajouter des clés d’indexation (ex. `row_id`, `split`, `spec_id`) pour relier résumés et détails.

6. **Observabilité des niveaux (levels/runner.py)**
   - Enrichir la réponse (`run_levels_build`/`run_levels_fill`) avec un résumé par symbole et type.
   - Ajouter des métriques d’exécution (temps total, nombre de niveaux analysés).

## Priorisation suggérée

1. **Fallback d’ingestion + logging Java** (impact direct sur la qualité des données).
2. **Worker jobs + monitoring** (stabilise l’API asynchrone).
3. **Stats details + metadata filtres** (améliore l’exploitabilité des analyses).
4. **Observabilité levels** (améliore la supervision des jobs levels).

---

## Ticket research/policy — PY-DCA-POLICY-1

### Objectif

Formaliser une policy de **conclusion stratégique** indépendante du moteur de calcul, afin de garder une séparation claire entre:

- calcul quantitatif (backend Python),
- décision métier (research/policy versionnée).

### Livrables

1. **Spec versionnée**: `specs/examples/decision_policy_dca_example.json`
   - critères « alpha réel vs confort psychologique »,
   - contextes de dominance,
   - conditions d’abandon,
   - validation long terme.
2. **Mapping policy -> métriques backend existantes**.
3. **Liste des métriques manquantes** pour exécution fully-automated de la policy.

### Règle de séparation moteur/policy

- Le moteur publie des métriques et artefacts factuels (aucune conclusion stratégique hardcodée).
- La policy consomme ces métriques et rend un verdict traçable (`decision`, `decision_trace`).
- Toute évolution des règles de décision passe par bump de version de policy et changelog.
