# High-level Strategies

Ce module introduit trois implémentations de stratégies discrétionnaires dans le
coeur Python du moteur « quant-engine ». Elles partagent un format de signal
commun (`StrategySignal`) et peuvent être utilisées en backtest via
`strategies.runner` ou en live via le `LiveRunner`.

## Stratégies disponibles

### `DcaEquityStrategy`

* **Objectif** : lisser les entrées sur des actions large caps lors des phases de
  drawdown.
* **Logique** : grilles de paliers configurables exprimées en drawdown (%) et en
  poids relatif. Chaque franchissement déclenche un signal `BUY`. Les sorties
  `SELL` sont optionnelles via des règles `tp_sl` (mode `per_grid_max_dd`).
* **Métadonnées** : profondeur atteinte (`dd_pct`), niveau de grille
  (`grid_level`), poids (`palier_used`), règle de take-profit active, etc.
* **Logs** : `log_drawdown_summary=true` pour afficher un résumé de drawdown en
  niveau INFO (par défaut, le résumé passe en DEBUG pour éviter le spam).
* **Référence drawdown** : configurez `drawdown_reference` (ex : `"ATH"`,
  `"3M"`, `"90D"`, ou `{ "mode": "rolling", "window": "90D" }`) pour piloter le
  calcul; les helpers legacy `compute_drawdown` / `compute_reference_high` ne
  sont plus exposés.

### `DcaEtfStrategy`

* **Objectif** : DCA long terme sur ETF globaux avec très peu de sorties.
* **Logique** : mêmes grilles de drawdown configurables, avec possibilité de
  limiter le nombre d’activations sur une période donnée (`activation_limit`).
  Pas de signal de vente par défaut ; un reset est possible en cas de nouvel ATH
  (`reset_on_new_high`).
* **Métadonnées** : drawdown courant, niveau de grille, nombre d’activations sur
  la période et configuration utilisée.

### `CryptoGridStrategy`

* **Objectif** : orchestrer des renforcements/rotations sur un panier crypto en
  combinant contexte macro et drawdown locaux.
* **Logique** : grilles micro (par coin) exprimées en drawdown. Chaque palier
  génère un `BUY` enrichi par l’intensité et l’action définies dans la grille.
  Les prises de profit (`SELL`) se déclenchent via les règles `tp_sl`. Les
  métadonnées incluent également le contexte macro propagé par le runner.

## Format de spécification (`StrategySpec`)

```json
{
  "strategy": {
    "strategy_id": "DCA_EQUITY_DEMO",
    "type": "dca_equity",
    "params": {
      "asset_class": "EQUITY",
      "grid": [
        {"dd": -10.0, "weight": 0.15},
        {"dd": -20.0, "weight": 0.20},
        {"dd": -30.0, "weight": 0.25}
      ],
      "tp_sl": {
        "enabled": true,
        "mode": "per_grid_max_dd",
        "rules": [
          {"max_dd_reached": -10.0, "tp_pct": 5.0},
          {"max_dd_reached": -20.0, "tp_pct": 10.0}
        ]
      }
    }
  },
  "data": {
    "start": "2020-01-01",
    "end": "2023-12-31",
    "timeframe": "1D"
  },
  "universe": [
    {"symbol": "AAPL", "asset_class": "EQUITY"}
  ]
}
```

* `strategy.type` : `dca_equity`, `dca_etf` ou `crypto_grid`.
* `strategy.params` : grilles, règles de TP/SL, limites d’activation, etc. Les
  structures sont libres tant que la stratégie peut les interpréter.
* `data` : fenêtre temporelle et paramètres d’ingestion (MySQL ou backend Java).
* `universe` : liste des instruments à traiter, chacun pouvant surcharger la
  configuration data (ex : timeframe spécifique).
* `output` (optionnel) : `{ "path": "runs/backtest.json", "format": "json" }`.

## Exécution

### Backtest

```bash
poetry run qe strategy backtest --spec specs/strategy_dca_equity_example.json
```

Le runner charge la grille, récupère les données OHLC (MySQL en priorité, Java
en fallback), instancie la stratégie puis sérialise les signaux.

### Live

Le `LiveRunner` lit la section `strategy.impl` d’un `LiveSpec` et construit
automatiquement l’implémentation correspondante.

```json
{
  "strategy": {
    "strategy_id": "LIVE_DCA_EQ",
    "impl": {
      "type": "dca_equity",
      "params": { "grid": [...], "tp_sl": {...} }
    }
  }
}
```

À chaque barre clôturée, le runner :

1. charge les positions actuelles via `java_client.get_positions()` ;
2. construit un contexte (`state`, positions, configuration TP/SL) ;
3. appelle `strategy.evaluate_live_bar(...)` ;
4. transforme les `StrategySignal` en payloads structurés écrits dans
   `quant.trades_live` et/ou publiés au backend Java.

Les signaux restent descriptifs (aucun ordre exécuté) et contiennent toutes les
informations nécessaires pour une supervision côté Java/UI.
