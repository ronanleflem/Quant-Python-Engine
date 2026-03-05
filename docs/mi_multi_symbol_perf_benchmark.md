# PY-MI-4.7 — Benchmark perf MI multi-symboles

## Objectif
Quantifier le coût de **Market Intelligence (MI) ON/OFF** sur des runs backtest multi-symboles, avec des métriques exploitables pour le capacity planning.

## Reproduction
```bash
poetry run pytest -q tests/perf -k mi
```
Le test écrit un artefact JSON dans `artifacts/perf/mi_multi_symbol_benchmark.json`.

## Protocole
- Dataset **synthétique déterministe** (seed fixe) pour minimiser la variance run-to-run.
- 3 répétitions par scénario, utilisation de la médiane (p50).
- Mesures:
  - Latence batch (ms)
  - Mémoire pic `tracemalloc` (MB)
  - Overhead MI absolu et normalisé par symbole
- Matrice benchmarkée:
  - `1m x 1 symbole`
  - `1m x 4 symboles`
  - `5m x 4 symboles`
  - `1h x 8 symboles`

## Résultats (référence locale)
| Cas | Latence OFF p50 (ms) | Latence ON p50 (ms) | Overhead MI (ms) | Overhead/symbole (ms) | Mémoire OFF p50 (MB) | Mémoire ON p50 (MB) | Overhead mémoire (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1m / 1 symbole | 75.99 | 102.91 | 26.92 | 26.92 | 0.231 | 0.361 | 0.130 |
| 1m / 4 symboles | 312.58 | 422.56 | 109.99 | 27.50 | 0.244 | 0.382 | 0.138 |
| 5m / 4 symboles | 308.88 | 418.84 | 109.95 | 27.49 | 0.246 | 0.382 | 0.136 |
| 1h / 8 symboles | 636.49 | 926.33 | 289.83 | 36.23 | 0.259 | 0.398 | 0.139 |

Variabilité run-to-run (coefficient de variation) observée sur ces runs: ~2% à ~11% OFF et ~3.7% à ~6.0% ON.

## Seuils proposés (SLO internes)
Pour capacity planning et alerte précoce de régression:
- **Overhead latence MI/symbole**:
  - cible: `< 30 ms`
  - alerte: `>= 35 ms`
- **Overhead mémoire MI/symbole**:
  - cible: `< 0.04 MB`
  - alerte: `>= 0.06 MB`
- **Stabilité run-to-run (CV)**:
  - cible: `< 10%`
  - alerte: `>= 15%`

## Recommandations
1. Exécuter ce benchmark à chaque release MI (ou changement pipeline features).
2. Garder `bars` et seeds fixes pour comparer les tendances temporelles.
3. Ajouter un suivi historique (CSV/JSON versionné par date) pour détecter la dérive.
4. Si overhead/symbole dépasse 35ms sur `1h x 8`, prioriser:
   - profiling `compute_features` / `label_regimes`,
   - réduction des conversions DataFrame inutiles,
   - mutualisation des calculs cross-symboles si possible.
