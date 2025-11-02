# Live Trading Runner

La boucle live exécute les stratégies en mode *on-bar-close*. Chaque itération :

1. Lit la dernière barre clôturée depuis MySQL (`marketdata`), sans regarder l'avenir.
2. Passe la fenêtre `(warmup + nouvelles barres)` dans la pipeline `filters → rules → risk gates`.
3. Construit un trade idempotent (1 trade max par barre/stratégie/side) et l'émet vers les destinations configurées.

## Architecture

```
marketdata (MySQL) ──► feed (poll) ──► runner (filters/rules/risk) ──► emitter ──┬─► Java REST
                                                                                └─► quant.trades_live (MySQL)
```

## Variables d'environnement

- `QE_MARKETDATA_MYSQL_URL` : connexion READ (`marketdata.*`).
- `QE_WRITE_MYSQL_URL` : connexion WRITE (`quant.trades_live`).
- `QE_JAVA_LIVE_URL` : endpoint REST Java (ex: `http://localhost:8080`).

## Lancer une stratégie live

```bash
poetry run qe live run --spec specs/live_example.json
```

Le runner charge `warmup_bars` (ex. 300 pour EMA/ADX) pour amorcer les indicateurs, puis boucle toutes `poll_interval_sec` secondes. Les timestamps sont gérés en UTC.

## Idempotence et état

- Cache mémoire par `(strategy_id, symbol, timeframe)` : historique, `last_ts_seen`, graines d'indicateurs.
- Un hash (`uniq_hash`) garantit `1 trade` max par barre/stratégie/side côté base (`quant.trades_live`).
- Redémarrage : le warm-up recharge les barres nécessaires avant de reprendre la boucle.

## Gestion du risque

Les risk gates (`daily_loss_cap`, `equity_dd_lockout`, `atr_risk_gate`, etc.) s'appliquent après les règles. En absence de colonnes requises (`equity`, `pnl`...), les gates deviennent no-op (`True`).

## Latence & extensions

- Mode actuel : `on-bar-close` (latence = clôture barre + poll interval).
- Intra-bar / tick feed et mise à jour incrémentale EMA/ATR sont laissés en TODO.
- Les erreurs REST Java sont journalisées mais non bloquantes.
