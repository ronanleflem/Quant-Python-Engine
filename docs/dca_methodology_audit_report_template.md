# Template — Rapport méthodologique DCA Grid (PY-DCA-EPIC-7)

## 1) Métadonnées run

- Run ID:
- Date/heure:
- Reviewer 1:
- Reviewer 2:
- Commit git:
- Version spec:
- Hash dataset:
- Seed(s):

## 2) Checklist anti data-snooping

- [ ] Règles ex-ante gelées avant run.
- [ ] Split temporel calibration/validation/test respecté.
- [ ] Vérification no-lookahead validée.
- [ ] Pas d’optimisation post-hoc sur période test.
- [ ] Paramètres et hypothèses calendaires documentés.

## 3) Matrice risques & mitigations

| Risque | Contrôle appliqué | Résultat | Preuve |
|---|---|---|---|
| Data snooping |  |  |  |
| Look-ahead bias |  |  |  |
| Survivorship bias |  |  |  |
| Leakage calendrier |  |  |  |
| Non-reproductibilité |  |  |  |

## 4) Rejeu reproductible

- Commande exécutée:
- Artefacts produits:
  - [ ] run_manifest.json
  - [ ] metrics.json
  - [ ] trades.parquet/csv
  - [ ] logs.txt
  - [ ] checksums.txt
- Résultat comparaison baseline vs rerun:
  - Statut: `REPRODUCIBLE` | `DRIFT_MINEUR` | `NON_CONFORME`
  - Écarts observés:

## 5) Critères go/no-go

### GO (toutes les conditions vraies)

- [ ] Checklist anti-snooping complète.
- [ ] Rejeu canonique reproductible.
- [ ] Aucune fuite d’information détectée.
- [ ] Critères de performance ex-ante atteints.
- [ ] Signature de revue pairée.

### NO-GO (au moins un point vrai)

- [ ] Changement ex-post des règles ou seuils.
- [ ] Artefacts incomplets/non traçables.
- [ ] Échec de reproductibilité.
- [ ] Fuite d’information non corrigée.

## 6) Décision finale

- Décision: GO / NO-GO
- Rationale synthétique:
- Actions de suivi:
