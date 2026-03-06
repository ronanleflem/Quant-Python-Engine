from __future__ import annotations

from typing import Any, Callable, Dict, List

from pydantic import ValidationError

from ...io import ids
from ...stats.packs import is_supported_market_stats_pack
from .. import schemas
from ..validation_errors import ApiValidationException, normalize_pydantic_errors, single_validation_error


ParsedRequestValidator = Callable[[Dict[str, Any]], Any]
JobLookup = Callable[[str], Dict[str, Any] | None]
JobInitializer = Callable[..., None]
RequestIdNormalizer = Callable[[Any], str | None]
JobDefaultsResolver = Callable[[], tuple[int | None, int | None]]
StatusMapper = Callable[[str], str]


def enqueue_run_request(
    payload: Dict[str, Any],
    *,
    validate_input: ParsedRequestValidator,
    validate_market_stats_params: Callable[[Dict[str, Any]], None],
    normalize_request_id: RequestIdNormalizer,
    get_job: JobLookup,
    init_job: JobInitializer,
    canonical_job_defaults: JobDefaultsResolver,
    external_job_status: StatusMapper,
    canonical_job_type: str,
    queued_status: str,
) -> schemas.RunEnqueueResponse:
    """Validate and enqueue a canonical run request."""

    try:
        parsed = validate_input(payload)
    except ValidationError as exc:
        raise ApiValidationException(normalize_pydantic_errors(exc)) from exc

    canonical_payload = parsed.model_dump(mode="json")
    validate_market_stats_params(canonical_payload)

    request_id = normalize_request_id(canonical_payload.get("request_id"))
    reused = False
    if request_id is not None:
        existing = get_job(request_id)
        if existing and existing.get("job_type") == canonical_job_type:
            reused = True
            status = external_job_status(str(existing.get("status", queued_status)))
            return schemas.RunEnqueueResponse(run_id=request_id, status=status, reused=True)
        if existing and existing.get("job_type") != canonical_job_type:
            raise ApiValidationException(
                single_validation_error("request_id", "conflict", "request_id already exists")
            )
    else:
        request_id = ids.generate_id()

    max_attempts, timeout_seconds = canonical_job_defaults()
    init_job(
        request_id,
        canonical_job_type,
        payload={"request": canonical_payload},
        status=queued_status,
        max_attempts=max_attempts,
        timeout_seconds=timeout_seconds,
    )
    return schemas.RunEnqueueResponse(run_id=request_id, status=queued_status, reused=reused)


def validate_market_stats_params(canonical_payload: Dict[str, Any]) -> None:
    if str(canonical_payload.get("spec_type") or "").strip().lower() != "market_stats":
        return

    stats_block = canonical_payload.get("stats")
    if stats_block is None:
        stats_block = {}
    if not isinstance(stats_block, dict):
        return

    errors: List[Dict[str, str]] = []
    data_block = canonical_payload.get("data") if isinstance(canonical_payload.get("data"), dict) else {}
    stats_pack = str(data_block.get("stats_pack") or "").strip()
    has_pack = bool(stats_pack)
    if has_pack and not is_supported_market_stats_pack(stats_pack):
        errors.append(
            {
                "field": "market_stats.data.stats_pack",
                "code": "literal_error",
                "message": "Unsupported stats_pack",
            }
        )

    event = stats_block.get("event")
    condition = stats_block.get("condition")
    target = stats_block.get("target")

    has_event = isinstance(event, dict)
    has_condition = isinstance(condition, dict)
    has_target = isinstance(target, dict)
    if not has_pack:
        if not has_event:
            errors.append({"field": "market_stats.stats.event", "code": "missing", "message": "Field required"})
        if not has_condition:
            errors.append({"field": "market_stats.stats.condition", "code": "missing", "message": "Field required"})
        if not has_target:
            errors.append({"field": "market_stats.stats.target", "code": "missing", "message": "Field required"})

    def _require_positive_int(params: Dict[str, Any], field_prefix: str, name: str) -> None:
        value = params.get(name)
        field = f"{field_prefix}.{name}"
        if value in (None, ""):
            errors.append({"field": field, "code": "missing", "message": "Field required"})
            return
        coerced = _coerce_int(value)
        if coerced is None:
            errors.append({"field": field, "code": "int_parsing", "message": "Input should be a valid integer"})
            return
        if coerced < 1:
            errors.append({"field": field, "code": "greater_than_equal", "message": "Input should be >= 1"})

    def _require_direction(params: Dict[str, Any], field_prefix: str, name: str = "direction") -> None:
        value = params.get(name)
        field = f"{field_prefix}.{name}"
        if value in (None, ""):
            errors.append({"field": field, "code": "missing", "message": "Field required"})
            return
        token = str(value).strip().lower()
        if token not in {"up", "down"}:
            errors.append({"field": field, "code": "literal_error", "message": "Input should be 'up' or 'down'"})

    if isinstance(event, dict):
        event_id = str(event.get("id") or "").strip().lower()
        event_params = event.get("params") if isinstance(event.get("params"), dict) else {}
        if event_id == "k_consecutive":
            base = "market_stats.stats.event.params"
            _require_positive_int(event_params, base, "k")
            _require_direction(event_params, base)

    if isinstance(condition, dict):
        condition_id = str(condition.get("id") or "").strip().lower()
        condition_params = condition.get("params") if isinstance(condition.get("params"), dict) else {}
        if condition_id == "htf_trend":
            base = "market_stats.stats.condition.params"
            _require_positive_int(condition_params, base, "tf_multiplier")
            _require_positive_int(condition_params, base, "ema_period")

    if isinstance(target, dict):
        target_id = str(target.get("id") or "").strip().lower()
        target_params = target.get("params") if isinstance(target.get("params"), dict) else {}
        if target_id == "continuation_n":
            base = "market_stats.stats.target.params"
            _require_positive_int(target_params, base, "n")
            _require_direction(target_params, base)
        elif target_id == "time_to_reversal":
            base = "market_stats.stats.target.params"
            _require_positive_int(target_params, base, "max_horizon")

    if errors:
        raise ApiValidationException(errors)


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if "." in text:
                maybe_float = float(text)
                if maybe_float.is_integer():
                    return int(maybe_float)
                return None
            return int(text)
        except Exception:
            return None
    return None
