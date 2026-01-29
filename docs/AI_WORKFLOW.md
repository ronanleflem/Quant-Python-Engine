# AI_WORKFLOW

## Objectif
Guider un agent IA pour travailler sur ce repo de moteur quant Python de facon fiable, testable, et PR-friendly.

## Workflow standard
1. **Audit du ticket**
   - Utiliser `tickets/_templates/AUDIT_PROMPT.md`.
   - Completer ou corriger le ticket si des elements manquent.
2. **Plan**
   - Produire un plan en petites etapes (2-6), chacune testable.
3. **Implementation**
   - Appliquer le plan et faire des commits logiques si demande.
   - Respecter strictement le scope du ticket.
4. **Validation**
   - Executer les commandes de validation du ticket.
   - Si un test est trop lourd, documenter une alternative (smoke test).
   - Distinguer tests rapides vs tests slow.

## Regles obligatoires
- Tests obligatoires pour chaque changement fonctionnel.
- Aucun refactor hors ticket.
- Aucun changement d'API silencieux.
- Respecter les conventions de nommage et les patterns existants.
- Preferer numpy/pandas vectorise; eviter les boucles Python lentes.

## Gestion des tickets
- Creer un ticket a partir du template: `tickets/_templates/TICKET_TEMPLATE.md`.
- Placer les tickets actifs dans `tickets/active/` (a creer si besoin).
- Deplacer les tickets termines dans `tickets/done/`.

## Branches / PR
- Branches: `feat/EX-###-short-title`, `fix/EX-###-short-title`.
- PR: titre = `[EX-###] short title`.
- Chaque PR doit referencer le ticket.

## Utiliser les templates
- Nouveau ticket: copier `tickets/_templates/TICKET_TEMPLATE.md`.
- Audit: executer la checklist dans `tickets/_templates/AUDIT_PROMPT.md`.
- Exemple de reference: `tickets/examples/EX-001-Add-Indicator-With-Tests.md`.
