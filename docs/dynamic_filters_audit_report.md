# 1) RÉSUMÉ EXÉCUTIF

L’architecture actuelle des filtres est pragmatique et centrée sur des fonctions : chaque filtre est une fonction autonome qui renvoie une `pd.Series` booléenne, et l’orchestration passe par un `filters_registry` global ainsi que deux exécuteurs (`apply_filter_stack` pour les gates purs, `score_filter_rules` pour les règles pondérées hard/soft). L’exécution concrète se fait dans `backtest.runner` et `strategies.runner`, où les masques de filtres sont appliqués avant la consommation des signaux.

La configuration est dynamique à l’exécution (les specs JSON peuvent transmettre des `params` arbitraires), mais la discipline est surtout implicite (valeurs par défaut dans le code + validation d’entrées) plutôt qu’explicite (schémas formels). Cela apporte de la flexibilité, mais crée aussi une surface paramétrique cachée et des contrats de paramètres hétérogènes entre filtres (filtres riches en `**kwargs` vs signatures fortement typées).

Top 5 risques :
1. **Risque de surapprentissage** : paramètres non bornés par filtre + mutations via l’optimisation = explosion des degrés de liberté.
2. **Risque de rigidité mal placée** : sorties principalement booléennes, ce qui pousse à multiplier les gates au lieu d’un scoring calibré.
3. **Risque de fragilité** : logique d’orchestration dupliquée entre runner stratégie/backtest et sémantique mixte strict/allow-if-missing.
4. **Risque de chaos de configuration** : absence de contrat canonique par filtre (schéma/métadonnées/versioning), avec des knobs nommés différemment.
5. **Risque performance/observabilité** : cache existant mais pas de métriques standardisées d’attribution par filtre (couverture/contribution/stabilité) dans les artefacts.

Top 5 quick wins :
1. Ajouter un **schéma canonique de manifeste de filtres** (id/version/bornes de params/whitelist optimisable).
2. Introduire un **Filter Contract** unique avec sorties normalisées (`gate`, `score`, `evidence`).
3. Imposer des **budgets de complexité** dans les runs d’optimisation (max filtres actifs, max paramètres tunables, max overrides).
4. Standardiser le dynamique à 3 modes contrôlés (régime/temps/relatif benchmark).
5. Émettre des métriques obligatoires par filtre dans les payloads de backtest (couverture, pass-rate, contribution score, deltas d’ablation).

---

# 2) INVENTAIRE ACTUEL DES FILTRES

> Légende : Type = `Gate` (booléen), `Score` (continu), `Hybrid` (score interne puis gate par seuil).  
> Les filtres exposés par le moteur sont ceux présents dans `filters_registry`.

| Nom / Chemin | Type | Entrées utilisées | Paramètres hardcodés/par défaut (exemples) | Dépendances | Sortie | Notes (odeurs / incohérences) |
|---|---|---|---|---|---|---|
| `adx` / `src/quant_engine/filters/volatility_trend.py` | Gate | high/low/close | `window=14`, `thresh=20` | aucune | série bool | Dérive de nommage : `thresh` vs `threshold` ailleurs. |
| `atr` / `.../volatility_trend.py` | Gate | high/low/close | `window=14`, `min_mult=0.5`, `max_mult=2.0` | aucune | série bool | Gate ATR redondant avec d’autres modules ATR. |
| `ema_slope` / `.../volatility_trend.py` | Gate | close | `window=50`, `slope_thresh=0.0` | aucune | série bool | Nommage de seuil non harmonisé. |
| `volume_surge` / `.../volume_profile.py` | Hybrid | volume | `window=20`, `mode=z`, `z_thresh=1.5`, `ratio_thresh=1.5` | aucune | série bool | Logique de score interne non exposée. |
| `vwap_side` / `.../volume_profile.py` | Gate | close, volume, levels optionnels | `anchor=day`, `side=above`, `from_levels=True` | repo levels optionnel | série bool | Comportement variable selon dispo des levels. |
| `poc_distance` / `.../volume_profile.py` | Gate | close + levels | `max_distance` requis, `unit=abs`, `level_type=POC` | repo levels | série bool | Params requis non gouvernés par schéma central. |
| `liquidity_sweep` / `.../structure_ict.py` | Gate | high/low/close | `side=high`, `lookback=50` | levels optionnels | série bool | Famille ICT qui chevauche `bos/mss`. |
| `bos` / `.../structure_ict.py` | Gate | high/low/close | `direction=up`, `left=2`, `right=2`, `use_levels=True` | levels optionnels | série bool | `use_levels` change fortement le comportement. |
| `mss` / `.../structure_ict.py` | Gate | high/low/close | `left=2`, `right=2`, `window=50` | levels optionnels | série bool | Flux proche de `bos`; candidat à consolidation. |
| `session_time` / `.../time_seasonality.py` | Gate | DatetimeIndex | `session=london`, `tz=None` | timezone lookup | série bool | Souvent utilisé comme bloqueur hard. |
| `day_of_week` / `.../time_seasonality.py` | Gate | DatetimeIndex | listes allow/deny | aucune | série bool | Candidat à override de régime plutôt que gate statique. |
| `day_of_month` / `.../time_seasonality.py` | Gate | DatetimeIndex | `mode=first`, `n=1` | aucune | série bool | |
| `month_of_year` / `.../time_seasonality.py` | Gate | DatetimeIndex | mois allowed/blocked | aucune | série bool | |
| `intraday_time` / `.../time_seasonality.py` | Gate | DatetimeIndex | `start=09:00`, `end=12:00`, `tz=UTC` | aucune | série bool | |
| `k_consecutive` / `.../stat_prob.py` | Gate | open/close | `k=2`, `direction=up` | aucune | série bool | Primitive de règle très réutilisée. |
| `seasonality_bin` / `.../stat_prob.py` | Gate | DatetimeIndex + stats optionnelles | bins, `min_winrate` optionnel | DB stats optionnelle | série bool | Peut « passer » silencieusement si DB manquante. |
| `hurst_regime` / `.../stat_prob.py` | Gate | close | `window=256`, `min_h=0.45`, `max_h=0.65` | aucune | série bool | Chevauchement Hurst avec `fractal_analysis`. |
| `entropy_window` / `.../stat_prob.py` | Gate | close/open optionnel | `window=64`, bornes d’entropie | aucune | série bool | Proche des contrôles d’entropie de `volatility`. |
| `daily_loss_cap` / `.../risk_mgmt.py` | Gate | index + pnl/equity/ret | cap, mode, notional | aucune | série bool | Gate de risque opérationnel mélangé aux filtres alpha. |
| `daily_trades_cap` / `.../risk_mgmt.py` | Gate | colonne signal + index | `max_trades`, `signal_col` | aucune | série bool | |
| `cooldown_bars` / `.../risk_mgmt.py` | Gate | colonne signal | `bars`, `signal_col` | aucune | série bool | |
| `atr_risk_gate` / `.../risk_mgmt.py` | Gate | high/low/close | `atr_window`, `max_atr_pct` | aucune | série bool | Encore un gate ATR; redondance avec `atr`/`volatility`. |
| `equity_dd_lockout` / `.../risk_mgmt.py` | Gate | equity | `max_dd`, `equity_col` | aucune | série bool | Devrait plutôt vivre dans l’exécution risk engine. |
| `benford_law` / `.../benford.py` | Hybrid | close/open/high/low/volume | fenêtres + seuils métriques | aucune | série bool | Options métriques nombreuses = DoF cachés. |
| `cycles` / `.../cycles.py` | Gate | close | `window=120`, `max_lag=50`, `max_r2=0.5` | aucune | série bool | |
| `donchian_channels` / `.../donchian.py` | Gate | high/low/close | `window=20`, `direction=up` | aucune | série bool | |
| `liquidity_cmf` / `.../liquidity.py` | Gate | high/low/close/volume | `window=20`, `threshold=0.2` | aucune | série bool | |
| `market_manipulation` / `.../market_manipulation.py` | Hybrid | OHLC (+ ATR optionnel) | knobs entropie/kurtosis/ATR | aucune | série bool | Filtre anomalie « parapluie » qui chevauche plusieurs filtres. |
| `htf_poi` / `.../htf_poi.py` | Gate | close + levels | mode, tolerance, distance | repo levels | série bool | `allow_if_missing` permissif peut diminuer la rigueur. |
| `orderflow_delta` / `.../orderflow.py` | Hybrid | delta ou buy/sell/volume | window, ratios, z-score, side | aucune | série bool | Score interne calculé mais non exposé au moteur de scoring. |
| `macro_cot_oi` / `.../macro_cot_oi.py` | Gate | colonnes cot/oi | seuils bias/oi | aucune | série bool | Disponibilité data variable selon classe d’actif. |
| `lower_timeframe_confluence` / `.../lt_confluence.py` | Hybrid | OHLC + volume + proxies delta | nombreux knobs (momentum/adx/vwap/ema/delta/min_score) | aucune | série bool | Surface paramétrique large; risque d’overfit élevé. |
| `psychologic_and_news` / `.../psychologic_news.py` | Gate | DatetimeIndex + news optionnel | fenêtres blackout | colonne news optionnelle | série bool | Comportement permissif par défaut si données absentes. |
| `ict_poi` / `.../ict_poi.py` | Hybrid | close + levels + OHLC optionnel | nombreux sous-modules (OB/FVG/Fib/Model10) | repo levels | série bool | Complexité combinatoire élevée. |
| `statistical_arbitrage` / `.../stat_arbitrage.py` | Gate | close | seuils omega/info | aucune | série bool | |
| `psychologic_ulcer` / `.../psychologic.py` | Gate | close | fenêtre + seuil ulcer | aucune | série bool | |
| `stationarity` / `.../stationarity.py` | Gate | close | `window`, `max_abs_autocorr`, `adf_pvalue`, `kpss_pvalue` | patterns statsmodels optionnels | série bool | Multiples tests -> risque de sprawl paramétrique. |
| `volatility` / `.../volatility.py` | Hybrid | high/low/close | entropie + bornes optionnelles ATR/BB/HV/VIX | aucune | série bool | Chevauche `atr`, `atr_risk_gate`, `market_regime`. |
| `ema_structure` / `.../ema_structure.py` | Gate | close | fenêtres EMA + booléens | aucune | série bool | |
| `rsi_entry` / `.../signal_rules.py` | Gate | close | RSI window/threshold/direction | aucune | série bool | Règle de signal hébergée côté filters. |
| `macd_entry` / `.../signal_rules.py` | Gate | close | paramètres MACD/direction | aucune | série bool | |
| `volume_above_average` / `.../signal_rules.py` | Gate | volume | window + ratio | aucune | série bool | |
| `fractal_analysis` / `.../fractal_analysis.py` | Gate | close | bornes Hurst/skew/kurtosis | aucune | série bool | Chevauchement partiel avec `hurst_regime`. |
| `mean_reversion` / `.../mean_reversion.py` | Gate | high/low/close | knobs BB/Keltner/ATR | aucune | série bool | |
| `contradictory_signals` / `.../contradictory_signals.py` | Hybrid | high/low/close | params RSI/Stoch/Williams/EMA | aucune | série bool | Mélange plusieurs familles d’indicateurs dans un seul gate. |
| `biais_institutional` / `.../biais_institutional.py` | Hybrid | close/volume + macro optionnelle | seuils EMA/VWAP/COT/OI | colonnes macro optionnelles | série bool | Mélange micro + macro dans un même filtre. |
| `atr_rising` / `.../indicator_rules.py` | Gate | high/low/close | `window`, `lookback` | aucune | série bool | |
| `linear_regression_macd_cross` / `.../indicator_rules.py` | Gate | close | MACD + lookback régression | aucune | série bool | |
| `market_regime` / `.../market_regime.py` | Gate | OHLC | seuils adx/atr/bb | aucune | série bool | La logique de régime est elle-même un gate statique. |
| `trend` / `.../trend.py` | Gate | high/low/close | method/lookback/params EMA | aucune | série bool | |
| `stats_gate` / `.../stats_gate.py` | Gate | DB stats + symbol/timeframe + trigger event | threshold/comparator/min_samples/etc. | DB stats + funcs events/conditions | série bool | Dépendance état externe; defaults permissifs possibles. |
| `mtf_anomaly` / `.../mtf_anomaly.py` | Hybrid | OHLC resamplé MTF | window/metric/threshold/params HTF | resampling | série bool | MTF + métriques d’anomalie => espace de tuning large. |
| `stats_gate_score` helper / `.../stats_gate.py` | Score helper (hors registry) | idem stats_gate | sélection/scaling métrique | DB stats | série score | Non exposé au registry; pas d’interface score canonique. |

---

# 3) CONSTATS D’ARCHITECTURE

## 3.1 Contrat des filtres aujourd’hui (de facto)
- Contrat **basé fonctions**, pas classe/interface formelle.
- Forme des signatures hétérogène :
  - Signatures explicites pour certains filtres (ex: `adx_filter(df, window=..., thresh=...)`).
  - Signatures très optionnelles pour d’autres (`htf_poi_filter`, `ict_poi_filter`, `lower_timeframe_confluence_filter`, etc.).
- Attendu runtime côté orchestrateurs :
  1. Callable trouvable dans `filters_registry`.
  2. Accepte `df` + `params`.
  3. Renvoie une `pd.Series` alignée index, interprétée en booléen.

## 3.2 Gestion de la configuration aujourd’hui
- Les specs JSON passent des listes libres :
  - `filters: [{type, params}]`
  - `filter_rules: [{type, params, weight, mode, enabled}]`
  - `filter_rules_config: {min_score|min_score_pct}`
- Pas de schéma central par filtre dans le code (pas de registre canonique bornes/types/defaults hors defaults fonctions + validations ad hoc).
- Validation majoritairement runtime dans `_validate_filter_inputs` (colonnes/index/dépendances externes), pas une gouvernance complète des paramètres.

## 3.3 Composition actuelle
- Deux chemins de composition :
  1. `apply_filter_stack` : AND séquentiel de masques booléens.
  2. `score_filter_rules` : hard mask + score pondéré (`hard`/`soft`, `min_score`/`min_score_pct`).
- Les deux sont exécutés dans **backtest.runner** et **strategies.runner** avec logique largement dupliquée.
- Résultat net : un masque est appliqué au signal d’entrée (avec logique de crossing optionnelle côté backtest).

## 3.4 Observations performance
- Positifs :
  - Cache en mémoire des filtres dans `filters.utils` avec TTL/max-items et clé dataframe + params.
  - Cache OHLC dans strategy runner.
- Gaps :
  - Pas de graphe explicite de pré-calcul/features partagées entre filtres (EMA/ATR/ADX recalculés à plusieurs endroits).
  - Cache opportuniste, pas formellement lié à un versioning deterministic feature store.
  - Pas de temps de calcul par filtre reporté dans les artefacts finaux.

## 3.5 Observations tests
- Il existe des tests smoke/intégration pour de nombreuses specs filtres, et un test unitaire ciblé pour le scoring pondéré.
- Les tests actuels couvrent surtout « ça s’exécute » + comportement hard/soft, mais peu d’éléments sur :
  - golden tests de stabilité de sortie par filtre,
  - batteries d’ablation/stabilité systématiques,
  - tests de gouvernance (bornes paramètres / budgets de complexité max).

---

# 4) PROPOSITION DE DESIGN DYNAMIQUE-MAIS-DISCIPLINÉ

## 4.1 Contrat de filtre (interface rigide)

Définir un contrat canonique déterministe :

```python
class FilterResult(TypedDict):
    gate: pd.Series          # bool alignée index
    score: pd.Series | None  # float alignée index, normalisée en [-1,1] ou [0,100]
    evidence: dict | None    # payload debug borné

class FilterSpecMeta(TypedDict):
    id: str
    name: str
    version: str
    category: str
    parameters_schema: dict
    default_config: dict
```

Champs metadata obligatoires :
- `id` (clé stable unique, ex. `volatility.adx`)
- `name`
- `version` (sémantique, incrément sur changement de comportement)
- `category` (trend/volatility/microstructure/risk/session/etc.)
- `parameters_schema` (typé, borné, avec defaults)
- `default_config`

Règles standard de sortie :
- **A) Gate** : série bool obligatoire.
- **B) Score** : série score optionnelle, normalisée et bornée.
- **C) Evidence** : dict optionnel de petite taille (clés/volume bornés).

Contraintes de déterminisme :
- Pas d’aléatoire caché.
- Le dynamique dérive uniquement de config explicite + data d’entrée.
- Le payload `evidence` ne doit jamais altérer gate/score.

## 4.2 Modèle de configuration (paramètres dynamiques avec garde-fous)

Schéma canonique (YAML/JSON équivalent) :

```yaml
filters:
  - id: volatility.adx
    active: true
    mode: score          # gate|score|hybrid
    weight: 0.8          # borné par schéma, ex [0, 3]
    params:
      window: 14
      threshold: 20.0
    regime_rules:
      - when: {regime: high_vol}
        override: {threshold: 24.0}
      - when: {regime: low_vol}
        override: {threshold: 16.0}
    allowed_overrides: [threshold]   # whitelist stricte
```

Garde-fous globaux (validation hard avant run) :
- Bornes obligatoires pour chaque paramètre tunable (`min`, `max`, `step` ou valeurs discrètes).
- Nombre max de paramètres tunables par filtre : **<= 3** (politique par défaut).
- Nombre max de filtres actifs par stratégie : **<= 8** (politique par défaut).
- Nombre max d’overrides par run : **<= 12**.
- Interdire les paramètres non déclarés (`additionalProperties=false`).
- Interdire l’optimisation des paramètres hors `allowed_overrides`.

Gouvernance optimisation :
- Le search-space référence uniquement des chemins fully-qualified de paramètres whitelistés.
- Les runs qui dépassent le budget de complexité sont rejetés avant exécution du backtest.

## 4.3 Comportements dynamiques autorisés (modes permis)

Autoriser uniquement ces modes dynamiques (opt-in) :

1. **REGIME-BASED OVERRIDES** (mode par défaut le plus sûr)
   - Driver : classifieur de régime déterministe (ex. buckets volatilité/tendance issus de `market_regime`).
   - Sûr si les régimes sont grossiers (2-4 états), stables et pré-déclarés.
   - Dangereux si taxonomie trop granulaire ou fuite d’information (future data).

2. **TIME-BASED OVERRIDES**
   - Driver : phases déterministes du run (`early/mid/late`) ou phases de drawdown bucketisées.
   - Sûr avec des phases grossières et pré-déclarées.
   - Dangereux si multiplication de micro-phases (curve fitting implicite).

3. **BENCHMARK-RELATIVE OVERRIDES**
   - Driver : force/volatilité relatives vs benchmark.
   - Sûr si benchmark + lookback sont fixes et disponibles pour toutes comparaisons.
   - Dangereux si disponibilité benchmark hétérogène selon dataset ou proxy variable entre runs.

Interdictions explicites :
- scripts/formules arbitraires dans la config,
- chaînes d’overrides imbriquées au-delà d’un niveau,
- seuils appris en runtime.

## 4.4 Contrôles anti-overfitting (couche de rigidité)

Introduire des contrôles obligatoires dans optimizer + reporting backtest :

- **Complexity Budget Score** par run :
  - `CBS = active_filters + tunable_params + overrides + score_weights_used`
  - doit rester <= seuil politique (ex. 20).
- **Walk-forward temporel** obligatoire pour tout run avec overrides dynamiques.
- **Minimum de trades** par fold et global (sinon rejet).
- **Pénalité de degrés de liberté** dans l’agrégation d’objectif.
- **Contrôles de stabilité multi-fenêtres** (20y/10y/5y/3y/1y selon disponibilité).
- **Batterie d’ablation** : retirer chaque filtre actif une fois; vérifier delta expectancy non catastrophique + cohérence.
- **Promotion gate** : promouvoir uniquement les configs qui passent performance + stabilité + complexité.

## 4.5 Observabilité & reporting

Métriques obligatoires par filtre et par run/fold :
- Coverage % (barres évaluées)
- Pass rate % (gate vrai)
- Score moyen + écart-type score (si mode score)
- Poids effectif (après overrides)
- Corrélation avec signaux d’entrée
- Proxy de contribution PnL (diff-in-diff via ablation ou impact marginal de masque)
- Temps de calcul (ms) + taux de hit cache
- Compteur de bypass données manquantes (`allow_if_missing`)

Méthode d’attribution :
- Primaire : deltas d’ablation mono-filtre (expectancy, Sharpe, drawdown, trade count).
- Secondaire : contribution conditionnelle par bucket régime/temps.

Structure standard des artefacts de rapport :
1. Métadonnées du run + hash de reproductibilité
2. Contrôles complexité & gouvernance
3. Métriques de performance agrégées
4. Tableau de télémétrie par filtre
5. Résultats d’ablation
6. Stabilité par fenêtre/fold
7. Décision : promote/reject + raisons

---

# 5) TÂCHES DE REFACTOR RECOMMANDÉES (JIRA-ready)

## Tâche 1 — Définir un contrat canonique + métadonnées de filtre
- **Goal** : introduire une interface de filtre rigide sans réécrire tous les filtres d’un coup.
- **Files likely impacted** : `src/quant_engine/filters/__init__.py`, nouveau `src/quant_engine/filters/contracts.py`, `src/quant_engine/filters/utils.py`.
- **Implementation outline** :
  1. Ajouter les types `FilterMeta`, `FilterResult`.
  2. Ajouter un adaptateur qui encapsule les filtres fonctionnels existants.
  3. Stocker un registre metadata parallèle au registre des callables.
- **Definition of Done** : chaque filtre du registry possède metadata + chemin adaptateur.
- **Tests to add** : tests de validation du contrat pour tous les IDs de filtres enregistrés.

## Tâche 2 — Ajouter des schémas stricts de paramètres par filtre
- **Goal** : borner tous les paramètres des filtres et interdire les clés inconnues.
- **Files likely impacted** : nouveau `src/quant_engine/filters/schemas.py`, `src/quant_engine/filters/utils.py`, exemples de specs.
- **Implementation outline** :
  1. Créer un schéma par filtre (type/default/min/max/enum).
  2. Valider `params` avant évaluation.
  3. Échouer en fail-fast avec messages actionnables.
- **Definition of Done** : aucun filtre ne peut tourner avec des paramètres non déclarés.
- **Tests to add** : matrice pass/fail des schémas, tests de rejet des clés inconnues.

## Tâche 3 — Unifier l’orchestration des filtres dans un pipeline unique
- **Goal** : supprimer la duplication entre runners stratégie/backtest.
- **Files likely impacted** : `src/quant_engine/backtest/runner.py`, `src/quant_engine/strategies/runner.py`, nouveau `src/quant_engine/filters/pipeline.py`.
- **Implementation outline** :
  1. Extraire le chemin commun d’exécution des filtres.
  2. Retourner un résultat pipeline standardisé (mask + métriques par filtre).
  3. Remplacer les logiques locales par des appels au pipeline.
- **Definition of Done** : une seule implémentation d’orchestration utilisée dans les deux runners.
- **Tests to add** : tests d’équivalence old-vs-new sur specs fixes.

## Tâche 4 — Introduire une politique score-first
- **Goal** : privilégier les filtres score + seuil global, tout en gardant des gates hard de risque.
- **Files likely impacted** : `src/quant_engine/filters/trade_filter_service.py`, `docs/filters.md`, exemples de specs.
- **Implementation outline** :
  1. Étendre le scorer pour consommer les sorties score normalisées.
  2. Réserver le mode hard aux filtres risque/conformité.
  3. Ajouter des policy checks contre la surutilisation de gates booléens.
- **Definition of Done** : seuil global de score configurable avec plages normalisées.
- **Tests to add** : tests d’agrégation mixte gate+score, tests de bornes de normalisation.

## Tâche 5 — Ajouter un moteur d’overrides dynamiques bornés
- **Goal** : supporter uniquement overrides régime/temps/benchmark.
- **Files likely impacted** : nouveau `src/quant_engine/filters/overrides.py`, parsing config côté runners.
- **Implementation outline** :
  1. Définir schéma d’override + matcher.
  2. Résoudre les paramètres effectifs par barre (déterministe).
  3. Bloquer les types d’overrides non supportés.
- **Definition of Done** : overrides opt-in, bornés et déterministes.
- **Tests to add** : tests d’activation par mode + rejet des overrides invalides.

## Tâche 6 — Validateur de budget de complexité
- **Goal** : empêcher automatiquement le sprawl paramètres/filtres.
- **Files likely impacted** : `src/quant_engine/optimize/variants.py`, `src/quant_engine/backtest/runner.py`, nouveau module de policy.
- **Implementation outline** :
  1. Calculer le score de complexité d’un run.
  2. Appliquer max filtres actifs/tunables/overrides.
  3. Émettre une section gouvernance dans les artefacts.
- **Definition of Done** : les runs non conformes sont rejetés avant exécution.
- **Tests to add** : tests unitaires pass/fail des seuils policy.

## Tâche 7 — Hash de reproductibilité du graphe de config filtres
- **Goal** : même data + même config filtres => même traçabilité de résultat.
- **Files likely impacted** : `src/quant_engine/optimize/variants.py`, `src/quant_engine/backtest/runner.py`, `src/quant_engine/io/artifacts.py`.
- **Implementation outline** :
  1. Canonicaliser la config effective des filtres (après policy d’overrides).
  2. Hasher et persister dans payload/summary.
  3. Inclure version code + versions metadata des filtres.
- **Definition of Done** : chaque artefact de run contient un hash de reproductibilité.
- **Tests to add** : tests de hash déterministe; changement param/version => hash différent.

## Tâche 8 — Télémétrie d’observabilité par filtre
- **Goal** : capturer coverage/pass-rate/score/contribution/coût.
- **Files likely impacted** : `src/quant_engine/filters/pipeline.py` (nouveau), `src/quant_engine/performance/backtest_builder.py`, docs.
- **Implementation outline** :
  1. Collecter des stats runtime par filtre.
  2. Persister dans payload + artefact CSV/JSON optionnel.
  3. Exposer dans logs et summary.
- **Definition of Done** : section de télémétrie renseignée pour tous les filtres actifs.
- **Tests to add** : tests de contrat payload pour les champs de télémétrie.

## Tâche 9 — Ajouter un harnais de tests d’ablation
- **Goal** : quantifier l’utilité marginale et retirer les filtres redondants.
- **Files likely impacted** : `src/quant_engine/optimize/variants.py`, nouveau `src/quant_engine/optimize/ablation.py`.
- **Implementation outline** :
  1. Auto-lancer N+1 backtests (full + retrait d’un filtre à la fois).
  2. Calculer les deltas expectancy/Sharpe/DD.
  3. Ajouter des seuils de promotion.
- **Definition of Done** : rapport d’ablation produit pour les candidats promus.
- **Tests to add** : test d’intégration synthétique sur artefacts d’ablation + calculs de delta.

## Tâche 10 — Stratégie de dépréciation des filtres qui se chevauchent
- **Goal** : réduire la duplication (familles ATR/volatility/regime/Hurst).
- **Files likely impacted** : `docs/filters.md`, `src/quant_engine/filters/__init__.py`, warnings dans la couche orchestration.
- **Implementation outline** :
  1. Déclarer les filtres canoniques par catégorie.
  2. Marquer les recouvrements comme alias dépréciés (phase warning-only).
  3. Retirer après fenêtre de migration.
- **Definition of Done** : map de dépréciation publiée et appliquée via warnings.
- **Tests to add** : tests d’émission de warnings pour IDs dépréciés.

## Tâche 11 — Suite de tests orientée gouvernance
- **Goal** : rendre les règles de discipline non régressibles.
- **Files likely impacted** : `tests/` (nouveaux `test_filters_governance.py`, `test_filters_schema.py`, `test_filters_pipeline_contract.py`).
- **Implementation outline** :
  1. Ajouter des tests max-active-filters/max-tunables/max-overrides.
  2. Ajouter des checks de reproductibilité et déterminisme.
  3. Ajouter des tests d’interdiction de paramètres free-form.
- **Definition of Done** : CI bloque les régressions de gouvernance.
- **Tests to add** : matrice complète de tests gouvernance.

---

# 6) OPTIONNEL : PLAN DE MIGRATION

## Migration pas à pas
1. **Phase 0 (observe-only)** : ajouter registre metadata + télémétrie sans changer le comportement.
2. **Phase 1 (compat mode)** : introduire l’adaptateur de contrat; les filtres legacy restent bool et sont encapsulés.
3. **Phase 2 (schema enforcement soft)** : warning sur params inconnus/hors bornes + rapport de gouvernance.
4. **Phase 3 (schema enforcement hard)** : bloquer les configs invalides; exiger des overrides whitelistés.
5. **Phase 4 (score-first policy)** : promouvoir les patterns d’agrégation score; garder les gates hard de risque.
6. **Phase 5 (deprecation cleanup)** : retirer filtres dépréciés et chemins de config legacy.

## Plan de compatibilité ascendante
- Maintenir le support des formats actuels `filters` et `filter_rules`.
- Fournir un adaptateur :
  - ancien `type` mappé vers nouvelle map d’alias `id`.
  - paramètres legacy auto-mappés vers noms canoniques quand nécessaire.
- Ajouter des warnings de dépréciation avec remplacement cible + version de suppression.

## Chemin de refactor minimal viable d’abord
- MVR = Tâche 1 + Tâche 2 + Tâche 3 + Tâche 8 uniquement.
- Ce lot donne rapidement la rigueur (contrat + schéma), la réduction de duplication (pipeline unique) et la visibilité (télémétrie), avant d’activer les overrides dynamiques.
