# Backlog index — DCA Grid (actionnable par agents Codex)

Ce backlog transforme les epics du process DCA en tickets exécutable dans **ce repo Python**.

## Ordre recommandé d’exécution
1. `PY-DCA-EPIC-1-benchmark-calendars-and-passive-runners.md`
2. `PY-DCA-EPIC-2-performance-metrics-engine.md`
3. `PY-DCA-EPIC-3-statistical-robustness-and-dominance.md`
4. `PY-DCA-EPIC-4-rolling-regimes-and-structural-underperformance.md`
5. `PY-DCA-EPIC-5-composite-score-and-edge-classification.md`
6. `PY-DCA-EPIC-6-versioned-data-contract-and-reporting-exports.md`
7. `PY-DCA-EPIC-7-methodology-reproducibility-and-anti-snooping.md`
8. `PY-DCA-EPIC-8-cross-universe-adapter-etf-equity-crypto.md`

## Règles d’exécution pour agents
- Respecter le template `tickets/_templates/TICKET_TEMPLATE.md`.
- Ne pas faire d’UI Angular dans ce repo ; produire un contrat backend stable.
- Pour chaque ticket: coder + tests + commandes de validation + mise à jour docs impactées.
- Préserver déterminisme (`seed`) et reproductibilité (`schema_version`, `universe_rules_version`).

## Dépendances critiques
- EPIC-2 dépend de EPIC-1.
- EPIC-3 dépend de EPIC-1 + EPIC-2.
- EPIC-6 dépend de EPIC-1..5.
- EPIC-8 dépend de EPIC-1 + EPIC-2 + EPIC-6.


## Découpage exécutable (sous-tickets)
- `PY-DCA-SUBTASKS-EXECUTION-PLAN.md`
