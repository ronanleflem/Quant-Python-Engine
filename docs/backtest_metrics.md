# Backtest metrics normalization

Les métriques de backtest sont normalisées pour être comparables entre périodes et timeframes. Les ratios de risque (volatilité, Sharpe, Sortino) sont annualisés à partir des rendements périodiques dérivés de la courbe d'équité et du timeframe.

## Paramètres utiles

Le bloc `performance` d'un run peut accepter :

- `risk_free_rate` : taux sans risque annualisé (exprimé en décimal, ex. `0.02` pour 2%).
- `risk_free_pct` : taux sans risque annualisé en pourcentage (ex. `2.0`).

Si le timeframe est fourni, l'annualisation utilise un nombre de périodes par an cohérent avec l'asset class (crypto 24/7, actions ~252 jours de bourse). Si le timeframe est absent, l'annualisation est déduite de la durée du backtest.
