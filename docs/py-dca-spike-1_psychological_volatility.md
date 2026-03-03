# PY-DCA-SPIKE-1 — Define quantitative proxy for psychological volatility

## 1) Contexte et objectif

La “volatilité psychologique” vise à approximer la difficulté émotionnelle à tenir une stratégie DCA dans la durée (stress, abandon, overrides manuels). Le backend ne peut pas mesurer l’état mental réel, mais il peut calculer des **proxies comportementaux** robustes à partir de la courbe d’equity.

Objectif du spike : décider si ces proxies sont suffisamment stables et actionnables pour entrer dans le moteur comme métriques de risque/ergonomie (pas comme signal alpha).

---

## 2) Proxies quantitatifs candidats

## Proxy A — Fréquence de nouveaux plus-bas d’equity (NPL)

**Définition**  
Nombre de fois où l’equity atteint un nouveau plus-bas relatif sur une fenêtre roulante (ou sur l’historique cumulé), normalisé par mois/trimestre.

**Intuition psychologique**  
Des nouveaux plus-bas fréquents renforcent la sensation de “stratégie qui empire”, même si l’espérance finale reste positive.

**Avantages**
- Très simple à expliquer aux stakeholders non techniques.
- Facilement reproductible et peu coûteux à calculer.
- Réagit vite aux régimes de marché défavorables.

**Limites / biais**
- Sensible à la granularité temporelle (daily vs hourly).
- Peut sur-réagir au bruit sur des marchés hachés.
- Ne capture pas l’amplitude, seulement la fréquence des épisodes négatifs.

---

## Proxy B — Durée moyenne sous l’eau (Time Under Water, TUW)

**Définition**  
Durée moyenne (et percentiles p50/p90) passée en drawdown avant retour au précédent pic d’equity.

**Intuition psychologique**  
La douleur perçue vient souvent de la **longueur** de la période de récupération plutôt que du drawdown max isolé.

**Avantages**
- Complète naturellement Max Drawdown.
- Mesure la “fatigue de patience”, très pertinente pour stratégies DCA longues.
- Plus stable que des métriques instantanées si on utilise des percentiles.

**Limites / biais**
- Fortement dépendant de l’horizon backtest.
- Les épisodes non clôturés en fin d’échantillon biaisent la moyenne (censure à droite).
- Peut être difficile à comparer entre actifs avec volatilités structurelles différentes sans normalisation.

---

## Proxy C — Variance des drawdowns (instabilité des creux)

**Définition**  
Variance (ou écart-type) des drawdowns observés sur la période, éventuellement en version robuste (winsorisation / MAD).

**Intuition psychologique**  
Des drawdowns d’intensité irrégulière augmentent l’imprévisibilité perçue et le stress décisionnel.

**Avantages**
- Capture l’instabilité du “profil de douleur”, pas seulement son niveau moyen.
- Utile pour distinguer deux stratégies ayant le même MaxDD mais une expérience utilisateur différente.
- Permet des déclinaisons robustes contre outliers.

**Limites / biais**
- Sensible aux événements extrêmes (flash crash) sans version robuste.
- Peut être redondant avec d’autres métriques de dispersion déjà présentes.
- Moins intuitif pour les utilisateurs finaux que TUW.

---

## 3) Recommandation GO/NO-GO

## Décision : **GO (implémentation backend ciblée)**

**Position**  
Implémenter ces proxies comme **métriques descriptives de robustesse comportementale** dans le moteur d’évaluation, avec documentation explicite qu’il ne s’agit pas d’une mesure psychologique clinique.

**Justification**
1. Les 3 proxies sont calculables de manière déterministe à partir de l’equity curve.
2. Ils complètent les métriques risque classiques (Sharpe, MaxDD) sur la dimension “tenabilité utilisateur”.
3. Le coût de calcul est faible, et le risque principal (mésinterprétation) est mitigé par un naming/documentation rigoureux.

**Garde-fous**
- Les exposer comme `behavioral_risk_proxy_*` pour éviter tout abus sémantique.
- Publier des métadonnées de calcul (granularité, horizon, règles de censure).
- Ne pas les utiliser seuls pour une décision de trading automatique.

---

## 4) Mini-plan d’implémentation (si GO)

## 4.1 Fonctions backend (proposées)

- `compute_new_low_frequency(equity_series, window=None, annualization='monthly')`
- `compute_time_under_water_stats(equity_series, include_open_episode=True)`
- `compute_drawdown_variability(equity_series, robust=True)`
- `compute_behavioral_risk_bundle(equity_series, config)` (agrégation payload-ready)

## 4.2 Tests à prévoir

- **Tests unitaires déterministes** sur séries synthétiques simples :
  - equity monotone croissante → NPL=0, TUW=0, variance DD=0.
  - un seul épisode drawdown puis recovery complet → TUW exact attendu.
  - série avec nouveaux plus-bas successifs contrôlés → NPL exact.
- **Tests de robustesse** :
  - impact de granularité (daily vs weekly) documenté.
  - épisode ouvert en fin de série (censure) géré selon config.
  - version robuste de la variance stable face à un outlier extrême.
- **Tests de non-régression** sur un dataset canonique de backtest.

## 4.3 Payload/API (proposition)

Ajouter au bloc `risk_metrics` (ou `diagnostics`) :

```json
{
  "behavioral_risk_proxy": {
    "new_low_frequency": {
      "value": 0.42,
      "unit": "events_per_month"
    },
    "time_under_water": {
      "mean_days": 18.3,
      "p50_days": 9.0,
      "p90_days": 47.0
    },
    "drawdown_variability": {
      "std": 0.061,
      "robust_mad": 0.038
    },
    "metadata": {
      "equity_frequency": "1D",
      "include_open_episode": true,
      "calculation_version": "v1"
    }
  }
}
```

---

## 5) Critères d’acceptation pour implémentation future

1. **Exactitude** : résultats unitaires conformes sur jeux synthétiques (tolérance numérique définie).
2. **Reproductibilité** : mêmes entrées → mêmes sorties, indépendamment de l’environnement d’exécution.
3. **Interprétabilité** : chaque métrique documentée avec définition, unité, limites et exemples.
4. **Traçabilité** : payload inclut métadonnées de granularité/horizon/censure.
5. **Sécurité d’usage** : documentation explicite “proxy comportemental”, non utilisé comme signal d’exécution autonome.

---

## 6) Conclusion exécutable

- **Décision explicite (DoD)** : **implémenter dans le backend** (scope métriques diagnostiques uniquement).
- **Hors scope immédiat** : scoring psychologique composite unique, calibration utilisateur individuelle, usage direct en optimisation de portefeuille sans étude complémentaire.
