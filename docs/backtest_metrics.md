# Backtest metrics normalization

Les métriques de backtest sont normalisées pour être comparables entre périodes et timeframes. Les ratios de risque (volatilité, Sharpe, Sortino) sont annualisés à partir des rendements périodiques dérivés de la courbe d'équité et du timeframe.

## Paramètres utiles

Le bloc `performance` d'un run peut accepter :

- `risk_free_rate` : taux sans risque annualisé (exprimé en décimal, ex. `0.02` pour 2%).
- `risk_free_pct` : taux sans risque annualisé en pourcentage (ex. `2.0`).

Si le timeframe est fourni, l'annualisation utilise un nombre de périodes par an cohérent avec l'asset class (crypto 24/7, actions ~252 jours de bourse). Si le timeframe est absent, l'annualisation est déduite de la durée du backtest.

## Métriques DCA supplémentaires (v1)

Pour les runs `spec_type=dca`, le moteur expose aussi :

- `final_performance_normalized` : performance finale normalisée par le capital réellement contribué.
- `capital_efficiency_index` : indice d'efficacité du capital immobilisé, défini (v1) par
  `final_value / max_contributed_capital`.
  - quand `max_contributed_capital <= 0`, la métrique retourne `0.0`.
  - variante future possible: utiliser le capital moyen immobilisé (au lieu du maximum).
- `twr` : time-weighted return basé sur les rendements de cycles.
- `xirr` : rendement annualisé sur cashflows irréguliers (statut séparé `xirr_status` pour non-convergence).
- `max_drawdown_on_contributed_capital` : drawdown max rapporté au capital contribué au fil de l'eau.
- `return_over_stress_ratio` : ratio versionné rendement/stress, défini par
  `final_performance_normalized / max(max_drawdown_on_contributed_capital, epsilon)`.
  - `version`: `return_over_stress_ratio_v1`
  - `numerator`: valeur de `final_performance_normalized`
  - `denominator_raw`: drawdown brut (décimal)
  - `epsilon`: plancher numérique anti-division-par-zéro (par défaut `1e-9`)
  - `denominator`: `max(denominator_raw, epsilon)`
  - `ratio`: `numerator / denominator`
- `time_under_water` : plus longue durée passée sous le dernier plus haut de la courbe de valeur.

### Limites connues

- Les cashflows très irréguliers peuvent entraîner des non-convergences `xirr` (`xirr_status=non_convergent`).
- Certaines séries de cashflows peuvent admettre plusieurs racines IRR; l'algorithme retourne une solution numérique locale si convergence.
- Quand `max_drawdown_on_contributed_capital` est nul (ou très proche de 0),
  `return_over_stress_ratio` utilise `epsilon` comme borne inférieure du dénominateur
  pour garantir un calcul déterministe sans division par zéro.

- `capital_efficiency_index` dépend fortement de l'horizon étudié et de la fréquence des contributions (plus elles sont denses, plus le maximum de capital immobilisé peut augmenter).
