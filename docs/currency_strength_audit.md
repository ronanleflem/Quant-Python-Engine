# Audit d’intégration — Force de monnaie historique (8 majeures FX)

## Objectif
Proposer les meilleurs points d’intégration pour ajouter une **force de monnaie historique** (basket des 8 devises majeures : `USD, EUR, GBP, JPY, CHF, CAD, AUD, NZD`) sans casser le flux actuel backtest/stratégies/filters.

## Lecture de l’architecture actuelle

### 1) Le moteur applique déjà des filtres au bon endroit
- En **backtest signal**, les filtres sont injectés dans `run_backtest_from_spec` après génération du signal brut, via `apply_filter_stack(...)` et `score_filter_rules(...)`.
- C’est l’emplacement le plus naturel pour un filtre de force de devise (gating entrée).

### 2) Le moteur stratégie (DCA / universe) applique aussi des filtres par symbole
- Le runner stratégie parcourt l’univers instrument par instrument, charge l’OHLC, puis applique `filters` et `filter_rules` avant `strategy.backtest(...)`.
- Cela permet d’utiliser exactement le même filtre “currency_strength” sur les runs multi-symboles.

### 3) Le chargement de données permet déjà la multi-source
- Le flux de chargement supporte `dataset_path`, MySQL et Delta/Java fallback.
- C’est un bon point d’ancrage pour une **phase d’enrichissement** qui construit une feature `ccy_strength_*` depuis un panier FX.

### 4) Le module stats peut porter une version “recherche”
- Le pipeline stats construit des conditions/events/targets sur DataFrame long-format.
- Ajouter une condition dédiée (ex: `currency_strength_regime`) permettrait de valider statistiquement l’intérêt de la feature avant de l’utiliser en prod.

## Où l’ajouter (ordre recommandé)

## A. Option 1 (MVP robuste) — Nouveau filtre `currency_strength`
**Pourquoi ici ?**
- Time-to-market rapide.
- Réutilisable en backtest + stratégie sans toucher au cœur de l’exécution des trades.

**Fichiers cibles**
1. `src/quant_engine/filters/currency_strength.py`
   - Calcul d’un score de force par devise sur fenêtre glissante (retours normalisés, ranking cross-section).
   - Exposer une fonction `currency_strength_filter(df, ...) -> pd.Series`.
2. `src/quant_engine/filters/__init__.py`
   - Ajouter import + entrée `filters_registry["currency_strength"]`.
3. `src/quant_engine/filters/utils.py`
   - Ajouter résumé + validation d’inputs (`symbol`, colonnes nécessaires, config basket).
4. `src/quant_engine/api/schemas.py`
   - Ajouter `"currency_strength"` dans `FilterConditionSpec.type`.
5. `docs/filters.md`
   - Documenter paramètres, prérequis, no-op/fail policy.

**Contrat JSON suggéré**
```json
{
  "filters": [
    {
      "type": "currency_strength",
      "params": {
        "base_currency": "EUR",
        "quote_currency": "USD",
        "majors": ["USD","EUR","GBP","JPY","CHF","CAD","AUD","NZD"],
        "lookback": 72,
        "method": "zscore_returns",
        "min_edge": 0.15,
        "invert_for_quote": true
      }
    }
  ]
}
```

## B. Option 2 (propre long terme) — Feature engineering avant filtres
**Pourquoi ?**
- Évite de recalculer la force dans chaque filtre.
- Permet à d’autres briques (signals, risk, stats gate) d’utiliser la même feature canonique.

**Point d’injection recommandé**
- Dans `src/quant_engine/backtest/runner.py`, juste avant l’appel aux filtres (`df_filters`), enrichir le DataFrame avec:
  - `ccy_strength_base`
  - `ccy_strength_quote`
  - `ccy_strength_spread = base - quote`
- Miroir côté `src/quant_engine/strategies/runner.py` pour le flux stratégie.

**Design conseillé**
- Nouveau module `src/quant_engine/core/features/currency_strength.py`.
- Cache par `(timeframe, window, basket, source window)` pour éviter coût O(N * paires).

## C. Option 3 (validation quantitative) — Intégration Stats
**Pourquoi ?**
- Tester objectivement si la force devise améliore `winrate`, `expectancy`, etc.

**Fichiers cibles**
- `src/quant_engine/stats/conditions.py` : `currency_strength_regime(...)`.
- `src/quant_engine/stats/runner.py` : branchement condition standard (déjà prêt par design).
- Specs stats de test dans `specs/tests/`.

## Modèle de calcul recommandé (simple et stable)
1. Construire un basket FX des majeures (idéalement paires croisées liquides).
2. Calculer les retours log par paire sur `lookback`.
3. Transformer paire -> contribution devise (base +, quote -).
4. Agréger par devise (moyenne pondérée volatilité inverse).
5. Normaliser cross-section (z-score ou rank percentile).
6. Pour le symbole tradé `AAA/BBB`, utiliser `strength(AAA) - strength(BBB)`.

## Points d’attention
- **Alignement temporel**: toutes les paires du basket doivent être alignées au même timestamp/timezone.
- **Données manquantes**: fallback no-op explicite ou hard error selon `strict`.
- **Surcoût CPU**: privilégier pré-calcul + cache.
- **Biais look-ahead**: rolling strictement historique (`closed='left'` logique).
- **Convention symboles**: harmoniser `EURUSD` vs `EUR/USD` dès la couche de parsing.

## Plan d’implémentation pragmatique
1. Ajouter le filtre `currency_strength` (Option A).
2. Ajouter 3 tests: valid inputs, missing basket data, gating behavior.
3. Si validé en perf, factoriser vers `core/features` (Option B).
4. Ajouter condition stats pour mesurer le lift (Option C).

## Verdict
Si tu veux l’ajouter **vite et proprement**, commence par un **nouveau filtre** branché sur les pipelines existants (backtest + stratégie). Ensuite, industrialise en feature partagée si l’usage devient central.
