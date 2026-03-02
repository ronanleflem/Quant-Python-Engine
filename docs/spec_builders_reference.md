# Spec Builders Reference (Angular -> Java -> Python)

Objectif: definir des builders clairs cote Java pour transformer des infos UI (pas un JSON deja construit) en specs Python valides.

Ce document complete `docs/parameter_catalog.json`:
- `parameter_catalog.json` = catalogue machine-readable (paths, types, enums).
- ce fichier = design d'implementation des builders (regles, etapes, erreurs, exemples).

## 1) Architecture cible

Flux recommande:
1. Angular envoie un payload metier structure (`RunRequestInput`).
2. Java valide + normalise.
3. Java route vers un builder selon `specType`.
4. Java construit un `Map<String, Object>` spec Python canonique.
5. Java envoie la spec au moteur Python (`/submit` ou `/submit/async`).

`specType` supportes:
- `backtest_signal`
- `strategy_backtest` (DCA/crypto grid)
- `market_stats`
- `seasonality`

Le bloc `stress_tests` est un module partage injecte par les builders qui produisent `performance`.

## 2) Contrat d'entree commun (Angular -> Java)

Definir un DTO principal:

```java
public class RunRequestInput {
  String specType;          // backtest_signal | strategy_backtest | market_stats | seasonality
  String catalogVersion;    // ex: "2026-02-02"
  DataInput data;
  StrategyInput strategy;
  SignalInput signal;
  FiltersInput filters;
  RiskInput risk;
  PerformanceInput performance;
  StatsInput stats;
  SeasonalityInput seasonality;
  OutputInput output;
  PersistenceInput persistence;
}
```

Notes:
- Tous les sous-blocs sont optionnels au niveau transport.
- Le builder impose les requis selon `specType`.
- Eviter les `Map` non types en entree API publique.

## 3) Pipeline standard pour tous les builders

Chaque builder suit la meme pipeline:

1. `validateRequired(input, specType)`
2. `normalizeCommon(input)`:
   - timeframe aliases (`M1` -> `1m`, etc.)
   - bool/string coercion
   - trim strings
3. `validateBusinessRules(input)`:
   - contraintes croisees (ex: `ema_cross` exige `fast` + `slow`)
4. `buildCoreSpec(input)`:
   - compose uniquement les champs utiles
5. `attachOptionalBlocks(input)`:
   - `filters`, `output`, `persistence`, `performance.stress_tests`
6. `finalSanityCheck(spec)`:
   - presence des paths critiques
7. retour spec JSON serializable

## 4) Builder: Backtest Signal

### 4.1 Classe conseillee

`BacktestSpecBuilder implements PythonSpecBuilder`

### 4.2 Requis minimaux

- `data.start`, `data.end`, `data.timeframe`
- une source data valide (`dataset_path` ou `mysql` ou `mysql_env/delta_*`)
- `strategy.strategy_id`
- `signal.type = ema_cross`
- `signal.params.fast` + `signal.params.slow` (ou aliases `ema_fast/ema_slow`)

### 4.3 Mapping principal

- `input.strategy.strategyId` -> `spec.strategy.strategy_id`
- `input.strategy.assetClass` -> `spec.strategy.asset_class`
- `input.data.*` -> `spec.data.*`
- `input.signal.*` -> `spec.signal.*`
- `input.risk.tpSl.*` -> `spec.tpsl.*`
- `input.performance.*` -> `spec.performance.*`

### 4.4 Exemple de sortie

```json
{
  "strategy": {
    "strategy_id": "BT_EURUSD_M1_EMA",
    "asset_class": "FX"
  },
  "data": {
    "dataset_path": "specs/examples/data/eurusd_m1_sample.json",
    "symbols": ["EURUSD"],
    "timeframe": "1m",
    "start": "2024-01-01",
    "end": "2024-01-31"
  },
  "signal": {
    "type": "ema_cross",
    "params": {
      "fast": 5,
      "slow": 20,
      "require_crossing": false
    }
  },
  "tpsl": {
    "atr_window": 14,
    "atr_k": 1.0,
    "r_mult": 2.0
  },
  "performance": {
    "initial_capital": 10000
  }
}
```

## 5) Builder: Strategy Backtest (DCA)

### 5.1 Classe conseillee

`StrategyBacktestSpecBuilder implements PythonSpecBuilder`

### 5.2 Requis minimaux

- `strategy.strategy_id`
- `strategy.type in [dca_equity, dca_etf, crypto_grid]`
- `strategy.params` coherent avec le type
- `data.start`, `data.end`, `data.timeframe`
- `universe[]` recommande (au moins 1 symbole)

### 5.3 Validation specifique par type

`dca_equity`:
- `params.grid[]` non vide
- chaque item: `dd`, `weight`

`dca_etf`:
- `params.grid[]` non vide

`crypto_grid`:
- `params.grid[]` non vide
- items avec `dd` (et optionnel `action`, `intensity`, `weight`)

### 5.4 Mapping principal

- `strategy` -> `spec.strategy`
- `data` -> `spec.data`
- `universe` -> `spec.universe`
- `filters/filter_rules/filter_rules_config` vers `spec.strategy.*` (ou top-level selon votre convention; fixer une seule convention)
- `performance` -> `spec.performance`

### 5.5 Exemple de sortie

```json
{
  "strategy": {
    "strategy_id": "DCA_EQUITY_DEMO",
    "type": "dca_equity",
    "params": {
      "asset_class": "EQUITY",
      "grid": [
        {"dd": -20.0, "weight": 0.2},
        {"dd": -30.0, "weight": 0.2}
      ],
      "tp_sl": {
        "enabled": true,
        "mode": "per_grid_max_dd",
        "rules": [
          {"max_dd_reached": -20.0, "tp_pct": 15.0, "be_pct": 7.0}
        ]
      }
    }
  },
  "data": {
    "start": "2024-01-01",
    "end": "2025-01-01",
    "timeframe": "1D",
    "mysql_env": "QE_MARKETDATA_MYSQL_URL"
  },
  "universe": [
    {"symbol": "AAPL", "asset_class": "STOCK", "exchange": "NASDAQ"}
  ],
  "performance": {
    "initial_capital": 10000,
    "capital_per_unit": 100
  }
}
```

## 6) Builder: Market Stats

### 6.1 Classe conseillee

`MarketStatsSpecBuilder implements PythonSpecBuilder`

### 6.2 Requis minimaux

- `data.start`, `data.end`, `data.timeframe`, source data valide
- `events[]` non vide (recommande)
- `targets[]` non vide (recommande)

### 6.3 Mapping principal

- `stats.events[]` -> `spec.events[]` (`name`, `type`, `params`)
- `stats.conditions[]` -> `spec.conditions[]`
- `stats.targets[]` -> `spec.targets[]`
- `stats.validation` -> `spec.validation`
- `persistence` -> `spec.persistence`
- `artifacts` -> `spec.artifacts`

### 6.4 Exemple de sortie

```json
{
  "data": {
    "dataset_path": "data/eurusd_m1.csv",
    "symbols": ["EURUSD"],
    "timeframe": "M1",
    "start": "2025-01-01",
    "end": "2025-06-30"
  },
  "events": [
    {"name": "k_consecutive", "params": {"k": 2, "direction": "up"}}
  ],
  "conditions": [
    {"name": "session", "params": {"col": "session_id"}}
  ],
  "targets": [
    {"name": "up_next_bar", "params": {}}
  ],
  "validation": {
    "train_months": 2,
    "test_months": 1,
    "folds": 3,
    "embargo_days": 2
  },
  "persistence": {
    "enabled": true,
    "spec_id": "stats_v1",
    "dataset_id": "eurusd_2025_h1"
  }
}
```

### 6.5 Contrat artefacts DCA (EPIC-8)

Quand `artifacts.out_dir` est fourni, les artefacts contractuels DCA JSON/Parquet doivent persister `metadata.universe_rules_version`.

Source de la valeur:
1. `performance.universe_rules_version` si present dans la requete canonique.
2. fallback stable: `asset-universe-rules-v1`.

Ce champ est versionne dans le bloc metadata du contrat et ne remplace aucun champ existant.

## 7) Builder: Seasonality

### 7.1 Classe conseillee

`SeasonalitySpecBuilder implements PythonSpecBuilder`

### 7.2 Requis minimaux

- `data.start`, `data.end`, `data.timeframe`, source data valide
- `seasonality.profile`
- `seasonality.signal`

### 7.3 Mapping principal

- `seasonality.profile` -> `spec.profile`
- `seasonality.signal` -> `spec.signal`
- `seasonality.compute` -> `spec.compute`
- `risk/execution/tp_sl/validation/persistence/artifacts` -> blocs homonymes

### 7.4 Exemple de sortie

```json
{
  "data": {
    "dataset_path": "data/eurusd_m1.csv",
    "symbols": ["EURUSD"],
    "timeframe": "M1",
    "start": "2025-01-01",
    "end": "2025-06-30"
  },
  "profile": {
    "by_hour": true,
    "by_dow": true,
    "measure": "direction",
    "ret_horizon": 1,
    "min_samples_bin": 300
  },
  "signal": {
    "method": "threshold",
    "threshold": 0.54,
    "dims": ["hour", "dow"],
    "combine": "and"
  },
  "compute": {
    "max_trials": 30,
    "search_space": {}
  },
  "execution": {"slippage_bps": 0.5, "commission_bps": 0.2},
  "risk": {"max_positions": 1, "max_allocation": 1.0}
}
```

## 8) Shared Module: Stress Tests Builder

### 8.1 Classe conseillee

`StressTestsBuilder`

Utilisation:
- appele par `BacktestSpecBuilder` et `StrategyBacktestSpecBuilder`
- injecte le bloc `spec.performance.stress_tests`

### 8.2 Mapping principal

- `input.performance.stressTests.enabled` -> `performance.stress_tests.enabled`
- `input.performance.stressTests.monteCarlo.*` -> `performance.stress_tests.monte_carlo.*`
- `input.performance.stressTests.scenarios[]` -> `performance.stress_tests.scenarios[]`

### 8.3 Regles critiques

- `method` dans `bootstrap|iid|shuffle|block|block_bootstrap`
- `source` dans `equity|returns|trades`
- si `method=block|block_bootstrap`: `block_size > 0`
- `n_sims > 0`

### 8.4 Exemple de bloc

```json
{
  "performance": {
    "stress_tests": {
      "enabled": true,
      "monte_carlo": {
        "source": "trades",
        "n_sims": 1000,
        "method": "bootstrap",
        "block_size": 5,
        "seed": 42,
        "output": {
          "mode": "light",
          "max_curves": 30,
          "curve_stride": 10
        }
      },
      "scenarios": [
        {"name": "crash", "type": "crash", "shock_pct": -0.25}
      ]
    }
  }
}
```

## 9) Java package layout conseille

```text
com.yourapp.quant
  /api
    RunController
  /dto
    RunRequestInput + sous DTO
  /validation
    RunRequestValidator
  /builder
    PythonSpecBuilder (interface)
    BacktestSpecBuilder
    StrategyBacktestSpecBuilder
    MarketStatsSpecBuilder
    SeasonalitySpecBuilder
    StressTestsBuilder
    SpecBuilderFactory
  /service
    RunOrchestratorService
  /client
    PythonEngineClient
```

## 10) Interface Java minimale

```java
public interface PythonSpecBuilder {
  boolean supports(String specType);
  Map<String, Object> build(RunRequestInput input);
}
```

Factory:

```java
public class SpecBuilderFactory {
  PythonSpecBuilder resolve(String specType) { ... }
}
```

## 11) Erreurs et retour API

Format recommande pour erreurs de validation:

```json
{
  "code": "VALIDATION_ERROR",
  "errors": [
    {"field": "signal.params.fast", "message": "required"},
    {"field": "performance.stress_tests.monte_carlo.n_sims", "message": "must be > 0"}
  ]
}
```

## 12) Integration avec `parameter_catalog.json`

Utiliser `docs/parameter_catalog.json` pour:
- enums autorises (filters, strategy types, stats events/targets, etc.)
- required paths par `specType`
- generation automatique des validateurs (option avancee)

Approche pragmatique:
1. versionner le catalogue (`catalogVersion`)
2. valider que la version envoyee par Angular est supportee par Java
3. si version non supportee: erreur explicite

## 13) Checklist implementation

1. Creer DTO d'entree types.
2. Creer `RunRequestValidator` (requis + coherence).
3. Implementer `StressTestsBuilder`.
4. Implementer les 4 builders principaux.
5. Ajouter `SpecBuilderFactory`.
6. Ajouter tests unitaires par builder (golden JSON).
7. Integrer dans le service d'orchestration vers Python.


## 10) Contrat de donnees versionne (DCA grid process v1)

Pour les runs `market_stats` utilises par l'initiative `INIT-DCA-GRID-001`, les artefacts de reporting doivent exposer:

- `schema_version`: `dca-grid-process-v1`
- `contract_version`: version semantique du contrat (ex: `1.0.0`)
- `metadata.reproducibility`: `seed`, `dataset_hash`, `config_version`

Artefacts standards produits en JSON + Parquet:

- `metrics`
- `distributions`
- `capital_curves`
- `rolling`
- `score`

### Politique de backward compatibility

- **Patch (`x.y.Z`)**: corrections sans changement de schema (retrocompatible).
- **Minor (`x.Y.z`)**: ajout de champs optionnels uniquement (retrocompatible).
- **Major (`X.y.z`)**: changement breaking (rename/suppression/type change), avec bump de `schema_version`.
- Les consommateurs Angular/export doivent ignorer les champs inconnus et se baser sur `schema_version` pour le routage.
