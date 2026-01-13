# HTTP client résilient

Le moteur utilise un client HTTP partagé configuré avec pooling, timeouts, retries et backoff pour rendre les appels réseau plus robustes.

## Configuration

Ces variables d'environnement permettent d'ajuster la résilience réseau sans modifier le code :

- `QE_HTTP_CONNECT_TIMEOUT` : timeout de connexion (secondes). Défaut : `3.0`.
- `QE_HTTP_READ_TIMEOUT` : timeout de lecture (secondes). Défaut : `10.0`.
- `QE_HTTP_RETRIES` : nombre total de retries (connexions, lectures, statuts). Défaut : `3`.
- `QE_HTTP_BACKOFF` : facteur de backoff exponentiel. Défaut : `0.4`.
- `QE_HTTP_POOL_CONNECTIONS` : nombre de pools. Défaut : `10`.
- `QE_HTTP_POOL_MAXSIZE` : taille max par pool. Défaut : `10`.

## Comportement

- Les statuts `429, 500, 502, 503, 504` déclenchent un retry.
- Le client est mutualisé pour limiter la création de connexions TCP.
