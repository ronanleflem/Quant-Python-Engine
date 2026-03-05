# Refactor architecture — Market Intelligence Layer

## 1) Audit de la structure actuelle

### Packages top-level observés

Le package principal `quant_engine` contient actuellement les sous-modules suivants :

- `api`, `backtest`, `cli`, `core`, `datafeeds`, `execution`, `filters`, `integrations`, `io`, `levels`, `live`, `notify`, `optimize`, `performance`, `persistence`, `seasonality`, `signals`, `stats`, `strategies`, `tpsl`, `validate`.

### Points de couplage élevés / "god modules"

- **`api/app.py` (~4000 lignes)** : concentre orchestration, endpoints, jobs async/sync, persistence et dispatch métier.
- **`optimize/variants.py` (~2500 lignes)** : logique très volumineuse orientée variantes.
- **`performance/stress_tests.py` (~1900 lignes)** : contrat + implémentation de stress tests (MC/scénarios) dans un seul fichier.
- **`strategies/runner.py` (~1500 lignes)** : orchestration multi-stratégies + exécution.

Ces fichiers sont des candidats prioritaires pour une extraction par responsabilités.

### Frontières actuellement fragiles

- `api/app.py` importe directement backtest, stats, seasonality, strategies, optimize, stress tests : couche application très couplée aux détails d’implémentation.
- `stats/runner.py` dépend de `api.schemas.StatsSpec`, ce qui inverse la dépendance (le domaine stats devrait dépendre de modèles domaine, pas de l’API HTTP).
- Plusieurs calculs de "régime/liquidité" existent déjà dans `filters/*` alors que ces signaux devraient être centralisés dans une future couche **market_intelligence**.

### Cycles d’import

- Aucune boucle d’import détectée au niveau module lors d’un scan AST interne (bonne base).

### Duplications de responsabilité (fonctionnelles)

- **Stats vs Filters vs Seasonality** : chaque module calcule des transformations de marché (événements, conditions, patterns) avec des conventions différentes.
- **Signal enrichments** dispersés entre `filters`, `stats.events`, `stats.conditions`, `signals`.

Conséquence : difficulté à réutiliser des features de manière cohérente entre backtest, DCA, stress-tests et analytics.

---

## 2) Architecture cible proposée

Objectif : introduire une couche **Market Intelligence Layer** faiblement couplée et consommée via interfaces.

```text
quant_engine/
  core/
    domain/
      models.py            # Candle, Trade, Position, Portfolio, RunResult
      enums.py
    contracts/
      market_intelligence.py
      strategy.py
      feature_store.py
      data_provider.py
    time/
    utils/

  data/
    providers/             # mysql, parquet, api
    loaders/
    alignment/             # multi-asset sync, calendars, joins

  indicators/
    trend/
    vol/
    volume/
    structure/

  market_intelligence/
    pipeline.py
    service.py
    models.py              # FeatureFrame, RegimeLabel, LiquidityFlags...
    correlation/
      rolling.py
      lead_lag.py
    regime/
      detectors.py
      labels.py
    liquidity/
      signals.py
      magnet_failure.py
    structure/
      market_structure.py

  strategy_engine/
    base.py                # StrategyProtocol
    adapters/
    implementations/

  backtest_engine/
    engine.py
    execution.py
    portfolio.py
    event_loop.py

  dca_engine/
    engine.py
    schedulers.py

  stress_test_engine/
    engine.py
    scenarios.py
    monte_carlo.py
    regime_shift.py

  performance_analytics/
    metrics.py
    attribution.py
    segmentation.py        # per-regime, per-flag analytics

  seasonality/
    profiles.py
    compute.py

  api_or_cli/
    api/
    cli/
```

### Pourquoi cette structuration

- `core`: noyau stable (objets métier + contrats abstraits).
- `data`: accès et normalisation des datasets (pas de logique stratégique).
- `indicators`: primitives de calcul réutilisables (EMA/ATR/etc.).
- `market_intelligence`: composition de primitives en features/scores business (corrélation, régime, liquidité, structure).
- `strategy_engine`: décision d’allocation/entry/exit ; **consomme** les features, ne les calcule pas.
- `backtest_engine` + `dca_engine`: moteurs d’exécution indépendants, branchés sur les mêmes contrats.
- `stress_test_engine`: simulation post-run ou in-run, branchée sur outputs standards (returns/equity/trades).
- `performance_analytics`: métriques + segmentation par labels MI.
- `api_or_cli`: uniquement couche d’exposition/orchestration.

---

## 3) Interfaces / contrats

### Modèles communs (core.domain)

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Optional

@dataclass(frozen=True)
class Candle:
    ts: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass(frozen=True)
class Trade:
    ts_open: datetime
    ts_close: Optional[datetime]
    symbol: str
    side: str
    qty: float
    entry: float
    exit: Optional[float]
    pnl: Optional[float]

@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    avg_price: float

@dataclass(frozen=True)
class Portfolio:
    cash: float
    equity: float
    positions: Mapping[str, Position]
```

### FeatureStore / FeaturePipeline

```python
from typing import Protocol
import pandas as pd

class FeatureStore(Protocol):
    def get(self, feature_set: str, symbol: str, timeframe: str) -> pd.DataFrame | None: ...
    def put(self, feature_set: str, symbol: str, timeframe: str, frame: pd.DataFrame) -> None: ...

class FeaturePipeline(Protocol):
    name: str
    version: str
    def compute(self, candles: pd.DataFrame) -> pd.DataFrame: ...
```

### MarketIntelligenceService API

```python
class MarketIntelligenceService(Protocol):
    def compute_features(self, symbol: str, timeframe: str, candles: pd.DataFrame) -> pd.DataFrame: ...
    def get_feature_slice(self, symbol: str, timeframe: str, start_ts, end_ts) -> pd.DataFrame: ...
    def label_regimes(self, symbol: str, timeframe: str, candles: pd.DataFrame) -> pd.Series: ...
    def liquidity_flags(self, symbol: str, timeframe: str, candles: pd.DataFrame) -> pd.DataFrame: ...
```

### Strategy API (consommation sans calcul)

```python
class Strategy(Protocol):
    name: str
    def on_bar(self, candle: Candle, features_row: dict, portfolio: Portfolio) -> list[dict]: ...
```

Le moteur fournit `features_row` ; la stratégie ne connaît ni pipeline ni stockage.

---

## 4) Règles de dépendances explicites

Règles cibles à faire respecter par convention + tests d’architecture :

1. `market_intelligence` -> dépend de `data`, `indicators`, `core`.
2. `strategy_engine` -> dépend de `core` + contrats `market_intelligence` (pas implémentation concrète).
3. `backtest_engine` -> dépend de `strategy_engine`, `core`, contrats MI.
4. `dca_engine` -> dépend de `strategy_engine` (ou `portfolio policies`) + `core` + contrats MI.
5. `performance_analytics` -> dépend de sorties normalisées backtest/DCA + labels MI.
6. `stress_test_engine` -> dépend de sorties normalisées + utilitaires randomization partagés.
7. **Interdits**:
   - `market_intelligence` n’importe jamais `backtest_engine`, `api_or_cli`, `performance_analytics`.
   - `strategy_engine` n’importe jamais `api_or_cli`.
   - `core` n’importe aucun module métier haut niveau.

Exemple de garde-fou (test simple) :

```python
# tests/architecture/test_import_rules.py
FORBIDDEN = {
    "quant_engine.market_intelligence": ["quant_engine.backtest_engine", "quant_engine.api_or_cli"],
}
```

---

## 5) Plan de refactor incrémental (risque minimal)

### PR1 — Poser les contrats sans déplacer le code

- Créer `core/contracts/*` + modèles de base.
- Ajouter des adapters qui enveloppent les modules existants (`stats`, `filters`) sans changer leur API publique.
- Ajouter tests unitaires des contrats (shape des retours, colonnes obligatoires).

### PR2 — Introduire `market_intelligence` en façade

- Créer `market_intelligence/service.py` avec implémentation initiale qui délègue à l’existant (`filters.market_regime`, `filters.liquidity`, etc.).
- Ajouter `FeatureStore` (in-memory + parquet) avec clé `(feature_set, symbol, timeframe, version)`.
- Ajouter CLI: `qe features recompute ...` (non-breaking).

### PR3 — Brancher backtest & DCA sur MI

- Injecter `MarketIntelligenceService` dans les moteurs (dependency injection).
- Les stratégies lisent `features_row` fourni par le moteur.
- Garder fallback legacy via adapter pour éviter régression.

### PR4 — Segmentation performance par régimes/flags

- Ajouter dans `performance_analytics` des vues segmentées :
  - métriques globales
  - métriques conditionnées (`regime=bull`, `magnet_failure=True`)
- Snapshot tests JSON pour stabiliser le format des rapports.

### PR5 — Migrer progressivement `stats` vers MI

- Déplacer d’abord les fonctions purement "feature computation" (events/conditions non causales de target).
- Laisser les agrégateurs statistiques dans `stats`.
- Déprécier API ancienne avec warnings + fenêtre de compatibilité.

### PR6 — Nettoyage et verrouillage d’architecture

- Réduire `api/app.py` en route handlers fins + application services.
- Ajouter tests d’architecture (imports interdits).
- Documenter le graphe de dépendances officiel dans `docs/`.

### Stratégie de tests

- **Unitaires**: détecteurs MI, normalisation features, feature store.
- **Intégration**: run backtest minimal + MI ON/OFF, run DCA minimal + MI labels.
- **Snapshot**: payload performance et stress-tests segmentés.
- **Non-régression**: comparer KPIs principaux avant/après sur 2-3 specs de référence.

---

## 6) Recommandations complémentaires

### Pattern recommandé

- **Ports & Adapters (hexagonal light)** :
  - Ports: `MarketIntelligenceService`, `FeatureStore`, `Strategy`.
  - Adapters: wrappers legacy (`filters`, `stats`, `persistence`).
- DDD “light” sur le noyau : entités/VO dans `core.domain`, services applicatifs dans les engines.

### Configuration

- Centraliser config via fichiers versionnés (YAML/JSON) :
  - pipeline features (`windows`, `thresholds`, `assets`, `timeframes`)
  - params stratégies
- Inclure `config_hash` dans artefacts pour reproductibilité.

### Caching

- L1: cache mémoire par run (dict/LRU).
- L2: cache disque parquet partitionné (`feature_set/symbol/timeframe/date`).
- Invalidation par `(pipeline_version, data_hash)`.

### Multi-actifs et synchronisation temporelle

- Créer `data/alignment` avec :
  - calendrier de référence
  - forward-fill contrôlé (max gap)
  - stratégie explicite des trous de données
- Imposer index temporel UTC unique dans toutes les features.

### CLI utile

- `qe features recompute --feature-set mi_v1 --symbol BTCUSDT --tf 1h --from ... --to ...`
- `qe features inspect --feature-set mi_v1 --symbol BTCUSDT --tf 1h`
- `qe backtest run --with-features mi_v1`

Runbook opératoire associé : `docs/market_intelligence_runbook.md`.

---

## 7) Clarification importante — capacités métier vs modules Python

Tu as raison : dans le projet actuel, tout n’est **pas** un module isolé. Certaines briques sont des
capacités métier réparties sur plusieurs fichiers.

### Cartographie recommandée (état cible simple)

| Capacité métier | Aujourd’hui (principalement) | Cible recommandée |
|---|---|---|
| Backtest | `backtest/*` + morceaux dans `strategies/*`, `performance/*`, `api/app.py` | `backtest_engine/*` |
| DCA | `strategies/dca_*`, `performance/dca_builder.py`, `io/dca_artifacts.py` | `dca_engine/*` + `performance_analytics/*` |
| Stress tests | `performance/stress_tests.py` | `stress_test_engine/*` |
| Performance analytics | `performance/backtest_builder.py`, `performance/dca_builder.py` | `performance_analytics/*` |
| Market stats (proba/conditions/événements) | `stats/*` + partie `filters/*` | `market_intelligence/*` (features/labels) + `stats/*` (agrégation/stat inference) |
| Seasonality | `seasonality/*` | `seasonality/*` (reste module dédié, alimenté par MI au besoin) |
| Optimisation | `optimize/*` | `optimize/*` (orchestrateur de variantes branché sur moteurs + contrats MI) |

### Optimisation : moteur d’exécution ou orchestrateur ?

Réponse courte : **dans cette architecture, `optimize` est un orchestrateur d’expériences (méta-couche), pas le moteur d’exécution principal**.

- Un **moteur d’exécution** consomme une série temporelle + des signaux et simule positions/fills/équité (ex: backtest, DCA).
- L’**optimisation** boucle sur des jeux de paramètres (`trials`), appelle les moteurs, compare des objectifs (Sharpe, drawdown, contraintes), puis promeut les meilleurs candidats.
- Donc `optimize` “exécute des runs”, mais **n’implémente pas lui-même** la mécanique d’exécution des ordres/trades ; il la délègue à `backtest`/`strategy runner`.
- **Oui, cela inclut aussi DCA** : si la stratégie testée est de type DCA, l’optimisation reste la même méta-couche, mais elle pilote alors des runs DCA via le runner stratégie/API canonique.

Dans le code actuel, on voit bien cette délégation : `optimize/variants.py` importe `backtest.runner` et `strategies.runner` puis appelle `run_backtest_from_spec(...)` / `run_backtest_with_payload(...)`. Côté API, le flux canonique supporte aussi `optimize_dca` (converti ensuite vers le runner adapté).

### Décision structurante

- **`stats` ne disparaît pas** : il se spécialise sur l’inférence statistique et les agrégations.
- **`market_intelligence` devient la source unique** des features contextuelles (corrélation, régimes,
  liquidité, structure, magnet failure).
- **`seasonality` reste un module dédié**, mais lit les mêmes données alignées et peut consommer
  des labels MI pour la segmentation.
- **`optimize` reste transverse** (ce n’est pas un moteur d’exécution), il pilote des runs via les
  interfaces stables (`strategy`, `backtest_engine`, `dca_engine`, `market_intelligence`).

En pratique : on garde les modules existants au début, puis on déplace progressivement le code vers
les packages cibles avec des adapters pour éviter la casse.

---

## North-star architecture diagram (texte)

```text
[data providers] -> [data/alignment] -> [indicators] -> [market_intelligence pipelines] -> [feature_store]
                                                                    |                     |
                                                                    v                     v
                                                             [strategy_engine] -----> [backtest_engine / dca_engine]
                                                                    |                     |
                                                                    v                     v
                                                            [orders/trades/positions] -> [performance_analytics]
                                                                                          |
                                                                                          v
                                                                                 [stress_test_engine]
                                                                                          |
                                                                                          v
                                                                                    [api_or_cli]
```

Principe : tout calcul de contexte marché (corrélation/régime/liquidité/structure) vit dans `market_intelligence`; les stratégies et moteurs **consomment** ce contexte via contrats stables.

## 7) Contrat de stabilité des features MI v1 (garanti)

Le contrat entre `core.contracts.MarketIntelligenceService` et `market_intelligence.service.MarketIntelligenceServiceV1` est verrouillé par tests :

- `build_snapshot(symbol, ohlcv)` doit exister et retourner un mapping.
- Clés minimales garanties du snapshot: `symbol`, `timeframe`, `feature_version`, `features`, `regimes`, `liquidity`.
- Les index temporels des DataFrames retournés sont en `DatetimeIndex` UTC, triés croissants.

### Colonnes garanties (snapshot v1)

Les noms et l’ordre des colonnes sont contractuels :

- `features`
  - `feat_return_1`
  - `feat_volatility_5`
  - `feat_corr_close_volume_5`
- `regimes`
  - `label_regime`
- `liquidity`
  - `liq_low_volume`
  - `liq_wide_spread`
  - `liq_illiquid`

### Policy de naming, timezone et NaN

- **Naming**:
  - features numériques préfixées par `feat_`.
  - labels catégoriels préfixés par `label_`.
  - flags de liquidité préfixés par `liq_`.
- **Timezone**:
  - tous les outputs MI sont normalisés en UTC (`DatetimeIndex.tz == UTC`).
- **NaN policy**:
  - `feat_return_1`: NaN initiaux remplacés par `0.0`.
  - `feat_volatility_5`: NaN de fenêtre roulante remplacés par `0.0`.
  - `feat_corr_close_volume_5`: NaN de corrélation roulante remplacés par `0.0`.

Cette politique garantit des features immédiatement consommables par les couches stratégie, filtre et risque sans nettoyage supplémentaire.
