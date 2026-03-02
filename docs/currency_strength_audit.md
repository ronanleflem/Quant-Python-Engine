# Currency strength (8 majeures FX) — audit + état d’implémentation

## Objectif
Documenter **où** et **comment** la force de monnaie historique est branchée dans Quant Engine, et comment l’activer en pratique pour les workflows backtest, stratégie et stats.

Majeures supportées par défaut : `USD, EUR, GBP, JPY, CHF, CAD, AUD, NZD`.

---

## État actuel (implémenté)

Le projet est désormais en mode **Option B + Option C** :

- ✅ **Option B (long terme propre)** : feature engineering partagé via `src/quant_engine/core/features/currency_strength.py`.
- ✅ **Option C (validation quantitative)** : condition stats `currency_strength_regime(...)` dans `src/quant_engine/stats/conditions.py`.
- ⏳ **Option A (filtre dédié `currency_strength`)** : non implémentée volontairement à ce stade, car la feature partagée couvre déjà le besoin principal.

---

## Architecture (où c’est branché)

## 1) Feature partagée
Fichier : `src/quant_engine/core/features/currency_strength.py`

Ce module expose :
- `parse_fx_symbol(symbol)` : supporte `EURUSD`, `EUR/USD`, `EUR_USD`.
- `default_major_pairs(majors)` : génère le basket de paires par défaut.
- `build_strength_table(prices_by_symbol, majors, lookback)` : calcule la force devise par timestamp.
- `enrich_with_currency_strength(df, symbol, prices_by_symbol, majors, lookback)` : ajoute :
  - `ccy_strength_base`
  - `ccy_strength_quote`
  - `ccy_strength_spread`

## 2) Backtest runner
Fichier : `src/quant_engine/backtest/runner.py`

- Si `features.currency_strength.enabled=true`, le runner charge le basket FX, calcule la feature, puis enrichit les rows **avant l’étape filtres**.
- Les colonnes `ccy_strength_*` deviennent disponibles pour le reste du pipeline.

## 3) Strategy runner
Fichier : `src/quant_engine/strategies/runner.py`

- Même activation via `features.currency_strength.enabled=true`.
- Mise en place d’un cache run-level pour éviter de re-fetch le basket à chaque instrument de l’univers.

## 4) Stats conditions
Fichier : `src/quant_engine/stats/conditions.py`

- Nouvelle condition : `currency_strength_regime(...)`.
- Classification du spread en 3 régimes : `long` / `neutral` / `short` selon seuils configurables.

---

## Activation dans une spec (backtest/strategy)

Exemple minimal :

```json
{
  "features": {
    "currency_strength": {
      "enabled": true,
      "lookback": 72,
      "majors": ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]
    }
  }
}
```

Champs supportés :
- `enabled` *(bool)* : active l’enrichissement.
- `lookback` *(int, défaut 72)* : fenêtre de smoothing des retours log.
- `majors` *(list[str])* : basket devises.
- `pairs` *(list[str], optionnel)* : override explicite du panier de paires.

---

## Usage côté stats (Option C)

Exemple de condition :

```json
{
  "conditions": [
    {
      "name": "ccy_regime",
      "type": "currency_strength_regime",
      "params": {
        "spread_col": "ccy_strength_spread",
        "long_threshold": 0.20,
        "short_threshold": -0.20
      }
    }
  ]
}
```

> Important : la condition attend la colonne `ccy_strength_spread` dans le dataset de stats.

---


## Interprétation fonctionnelle (ce que signifient les résultats)

Les 3 colonnes produites ont un sens opérationnel clair :

- `ccy_strength_base` : force normalisée de la devise **base** de la paire (ex: `EUR` sur `EURUSD`).
- `ccy_strength_quote` : force normalisée de la devise **quote** (ex: `USD` sur `EURUSD`).
- `ccy_strength_spread` = `base - quote` : avantage relatif de la base contre la quote.

Lecture rapide du spread :

- **Spread > 0** : biais haussier de la paire (plutôt favorable au **long**).
- **Spread < 0** : biais baissier de la paire (plutôt favorable au **short**).
- **Spread proche de 0** : pas d’avantage clair (zone neutre).

Exemple sur `EURUSD` :
- si `ccy_strength_base=+0.80` et `ccy_strength_quote=-0.20`, alors `spread=+1.00` → contexte pro-long.
- si `ccy_strength_base=-0.40` et `ccy_strength_quote=+0.50`, alors `spread=-0.90` → contexte pro-short.

### Comment réagir en pratique (playbook simple)

Approche recommandée (à adapter au style de stratégie) :

1. Définir une zone neutre par seuils symétriques (ex: `+0.20 / -0.20`).
2. Utiliser le spread comme **gating** des signaux :
   - `spread >= +seuil` → autoriser/prioriser les longs,
   - `spread <= -seuil` → autoriser/prioriser les shorts,
   - sinon → réduire la taille, ou ignorer le trade.
3. En stats, mesurer si les bins `long/neutral/short` améliorent réellement `winrate`, `lift`, `expectancy` avant de durcir les règles.

### Attention à l’interprétation

- Un spread élevé n’est **pas** un signal d’entrée autonome ; c’est un **contexte**.
- En marché très volatil, préférer confirmation par signal primaire (EMA cross, structure, etc.).
- Recalibrer les seuils par timeframe/symbole (M15 vs H1/H4 peuvent avoir des distributions différentes).

## Modèle de calcul (résumé)
1. Retours log par paire du basket.
2. Contribution **base = +retour**, **quote = -retour**.
3. Agrégation cross-pairs par devise.
4. Normalisation cross-section (z-score) pour obtenir des scores comparables.
5. Spread symbole tradé = `force(base) - force(quote)`.

---

## Limites actuelles / prochaines améliorations
- Ajouter un document de référence dédié “currency_strength.md” avec exemples complets de specs.
- Ajouter des exemples prêts à lancer dans `specs/examples`.
- Éventuellement ajouter plus tard un filtre explicite `currency_strength` (Option A) pour du gating direct sans étape intermédiaire.

