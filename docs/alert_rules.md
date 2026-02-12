# Alert rules (initial thresholds)

## 5xx rate
- Trigger: 5xx > 1% for 10 min on runs endpoints
- Severity: high

## Latency spike
- Trigger: p95 > 2s or p99 > 5s for 15 min on runs endpoints
- Severity: medium

## Timeout anomaly
- Trigger: 408/504 count > 10 in 10 min
- Severity: high

## Submit success drop
- Trigger: submit 2xx rate < 98% for 15 min
- Severity: medium

## Cancel success drop
- Trigger: cancel 2xx rate < 98% for 15 min
- Severity: low
