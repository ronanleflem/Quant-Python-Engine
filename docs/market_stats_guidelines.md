# Market Stats – Notes & Garde-fous

Ce module calcule des statistiques conditionnelles de marché (probabilités simples, run-lengths, continuation, reversion, etc.) utiles comme **contexte** pour les stratégies.

## Principes de rigueur

- **Définitions figées**  
  - Exemple : "Up candle = close > open"  
  - Exemple : "tf_multiplier=60 ⇒ HTF = 60 × timeframe de base"

- **No lookahead**  
  - Les conditions doivent être calculables à t.  
  - Les cibles (targets) utilisent uniquement t+1..t+n.

- **Échantillon minimum**  
  - n_min (par défaut 300).  
  - Si n < n_min ⇒ résultat marqué `insufficient=true`.

- **WFA (walk-forward analysis)**  
  - Splits train/test.  
  - Les bins (ex. tertiles ATR) doivent être définis sur **train** et appliqués sur test.

- **Multiplicité**  
  - Plusieurs patterns ⇒ risque de faux positifs.  
  - Utiliser un contrôle de FDR (Benjamini–Hochberg).

- **Intervalles de confiance**  
  - Fréquentiste : Wilson 95% CI.  
  - Bayésien : Beta-Binomial (Jeffreys prior) + HDI 95%.

- **Binning fixe**  
  - Pas de re-binning par split test.  
  - Exemple : définir les tertiles de volatilité sur train uniquement.

- **Rolling recalibration**  
  - Rafraîchir les stats sur fenêtres glissantes (ex. 6 mois).  
  - Monitorer le drift temporel des probabilités.

## Pourquoi ces garde-fous ?

- Éviter l’auto-intox (overfit sur un dataset).  
- Savoir si un pattern est **réellement robuste** ou juste du bruit.  
- Pouvoir comparer différentes conditions de marché de manière saine.

---

👉 À lire **avant** d’ajouter de nouveaux events/conditions/targets.
