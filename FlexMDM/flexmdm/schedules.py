"""Insertion and unmasking schedules for FlexMDM."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

import torch


class FlexMDMScheduleNotImplemented(NotImplementedError):
    """Raised when a FlexMDM schedule has not been implemented yet."""


LINEAR_SCHEDULE = "linear"  # alpha(t) = t
QUADRATIC_SCHEDULE = "quadratic"  # alpha(t) = 2t - t^2 = 1 - (1-t)^2
POWER_SCHEDULE = "power"  # alpha(t) = 1 - (1-t)^exponent
LOG_LINEAR_SCHEDULE = "log_linear"  # alpha(t) = (1-lam)*t + lam*log1p(c*t)/log1p(c)
LOGIT_POWER_SCHEDULE = "logit_power"  # alpha(t) = t^a / (t^a + (1-t)^b)

_POWER_FAMILY = (LINEAR_SCHEDULE, QUADRATIC_SCHEDULE, POWER_SCHEDULE)
_SUPPORTED_SCHEDULES = (
    LINEAR_SCHEDULE,
    QUADRATIC_SCHEDULE,
    POWER_SCHEDULE,
    LOG_LINEAR_SCHEDULE,
    LOGIT_POWER_SCHEDULE,
)

_LOG_LINEAR_DEFAULTS: Dict[str, float] = {"lam": 1.0, "c": 75.0}
_LOGIT_POWER_DEFAULTS: Dict[str, float] = {"a": 1.5, "b": 2.0}

# Numerical inverse for closed-form-less schedules (logit_power, log_linear with
# lam in (0, 1)). 40 bisection steps in [0, 1] => ~2^-40 absolute precision in
# fp32, which is far below the eps floors used elsewhere.
_BISECT_NUM_ITERS: int = 40


def normalize_schedule_name(name: Any, *, field: str) -> str:
    """Return a canonical schedule name, defaulting to linear."""
    if name is None:
        return LINEAR_SCHEDULE
    if not isinstance(name, str):
        raise TypeError(f"{field} schedule must be a string, got {type(name).__name__}")

    normalized = name.lower()
    if normalized in _SUPPORTED_SCHEDULES:
        return normalized
    raise FlexMDMScheduleNotImplemented(
        f"{field} schedule {name!r} is not implemented. "
        f"Supported schedules: {_SUPPORTED_SCHEDULES!r}."
    )


def _resolve_exponent(schedule: str, exponent: Optional[float]) -> float:
    """Resolve the effective exponent for the power parameterization.

    Linear and quadratic are special cases (exponent=1 and 2); callers may
    still pass them through uniformly as power schedules. When a schedule
    other than ``power`` is paired with an exponent override, we accept it
    as long as it matches the canonical value; otherwise we raise so the
    caller isn't silently ignored.
    """
    if schedule == POWER_SCHEDULE:
        if exponent is None:
            raise ValueError(
                "power schedule requires an exponent; pass exponent=... "
                "(or insertion_exponent/unmasking_exponent)."
            )
        exp = float(exponent)
        if not exp > 0.0:
            raise ValueError(f"power schedule exponent must be > 0, got {exp}.")
        return exp
    if schedule == LINEAR_SCHEDULE:
        canonical = 1.0
    elif schedule == QUADRATIC_SCHEDULE:
        canonical = 2.0
    else:
        raise AssertionError(f"Unhandled schedule {schedule!r}")
    if exponent is not None and float(exponent) != canonical:
        raise ValueError(
            f"{schedule!r} schedule implies exponent={canonical}, but "
            f"received exponent={exponent}."
        )
    return canonical


def _resolve_log_linear_params(
    params: Optional[Mapping[str, Any]],
) -> tuple[float, float]:
    """Resolve (lam, c) for the log_linear schedule.

    alpha(t) = (1-lam) * t + lam * log(1 + c*t) / log(1 + c)
    """
    if params is None:
        params = {}
    lam = float(params.get("lam", _LOG_LINEAR_DEFAULTS["lam"]))
    c = float(params.get("c", _LOG_LINEAR_DEFAULTS["c"]))
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"log_linear schedule requires lam in [0, 1], got {lam}.")
    if not c > 0.0:
        raise ValueError(f"log_linear schedule requires c > 0, got {c}.")
    return lam, c


def _resolve_logit_power_params(
    params: Optional[Mapping[str, Any]],
) -> tuple[float, float]:
    """Resolve (a, b) for the logit_power schedule.

    alpha(t) = t^a / (t^a + (1-t)^b)
    """
    if params is None:
        params = {}
    a = float(params.get("a", _LOGIT_POWER_DEFAULTS["a"]))
    b = float(params.get("b", _LOGIT_POWER_DEFAULTS["b"]))
    if not (a > 0.0 and b > 0.0):
        raise ValueError(
            f"logit_power schedule requires a > 0 and b > 0, got a={a}, b={b}."
        )
    return a, b


def _bisect_inverse(
    target: torch.Tensor,
    *,
    forward,
    num_iter: int = _BISECT_NUM_ITERS,
) -> torch.Tensor:
    """Bisect alpha-inverse on [0, 1] for a strictly increasing forward.

    ``forward`` maps t in [0, 1] to alpha(t) in [0, 1]. Runs in fp64 so the
    forward function does not saturate near the boundaries — fp32 ``(1-t)^b``
    rounds to 0 well before t reaches 1, which would cap the achievable
    inverse precision at ~1e-4 even with many bisection steps. Casts back
    to the input dtype.
    """
    in_dtype = target.dtype
    target_f = target.clamp(0.0, 1.0).to(torch.float64)
    lo = torch.zeros_like(target_f)
    hi = torch.ones_like(target_f)
    for _ in range(int(num_iter)):
        mid = 0.5 * (lo + hi)
        fmid = forward(mid)
        # alpha is monotone non-decreasing; pull lo up where fmid < target.
        lo = torch.where(fmid < target_f, mid, lo)
        hi = torch.where(fmid >= target_f, mid, hi)
    return (0.5 * (lo + hi)).to(in_dtype)


def _log_linear_alpha(
    t: torch.Tensor, *, lam: float, c: float
) -> torch.Tensor:
    t_safe = t.clamp(min=0.0, max=1.0)
    lin = t_safe
    # log1p(c*t) / log1p(c) is well-defined for c > 0 and t in [0, 1].
    log_term = torch.log1p(c * t_safe) / float(math.log1p(c))
    return (1.0 - lam) * lin + lam * log_term


def _log_linear_alpha_derivative(
    t: torch.Tensor, *, lam: float, c: float
) -> torch.Tensor:
    t_safe = t.clamp(min=0.0, max=1.0)
    log_deriv = c / ((1.0 + c * t_safe) * float(math.log1p(c)))
    return (1.0 - lam) + lam * log_deriv


def _log_linear_alpha_inverse(
    alpha: torch.Tensor, *, lam: float, c: float
) -> torch.Tensor:
    a = alpha.clamp(min=0.0, max=1.0)
    if lam == 0.0:
        return a.clone()
    if lam == 1.0:
        # alpha = log1p(c*t) / log1p(c) => t = expm1(alpha * log1p(c)) / c
        log1p_c = float(math.log1p(c))
        return torch.expm1(a * log1p_c) / float(c)
    return _bisect_inverse(
        a,
        forward=lambda x: _log_linear_alpha(x, lam=lam, c=c),
    )


def _logit_power_alpha(
    t: torch.Tensor, *, a: float, b: float
) -> torch.Tensor:
    t_safe = t.clamp(min=0.0, max=1.0)
    one_minus_t = (1.0 - t_safe).clamp_min(0.0)
    ta = t_safe.pow(a)
    omtb = one_minus_t.pow(b)
    denom = (ta + omtb).clamp_min(torch.finfo(ta.dtype).tiny)
    return ta / denom


def _logit_power_alpha_derivative(
    t: torch.Tensor, *, a: float, b: float
) -> torch.Tensor:
    """alpha'(t) = t^(a-1) * (1-t)^(b-1) * [a*(1-t) + b*t] / (t^a + (1-t)^b)^2."""
    t_safe = t.clamp(min=0.0, max=1.0)
    one_minus_t = (1.0 - t_safe).clamp_min(0.0)
    # Use small floors so the corner cases (t=0 with a-1<0, t=1 with b-1<0)
    # don't produce NaNs from 0**(negative). Defaults a=1.5, b=2 are safe.
    tiny = torch.finfo(t_safe.dtype).tiny
    t_clamped = t_safe.clamp_min(tiny)
    omt_clamped = one_minus_t.clamp_min(tiny)
    ta = t_clamped.pow(a)
    omtb = omt_clamped.pow(b)
    denom_sq = (ta + omtb).clamp_min(tiny).pow(2.0)
    weighted = a * one_minus_t + b * t_safe
    return (
        t_clamped.pow(a - 1.0) * omt_clamped.pow(b - 1.0) * weighted / denom_sq
    )


def _logit_power_alpha_inverse(
    alpha: torch.Tensor, *, a: float, b: float
) -> torch.Tensor:
    a_clamped = alpha.clamp(min=0.0, max=1.0)
    return _bisect_inverse(
        a_clamped,
        forward=lambda x: _logit_power_alpha(x, a=a, b=b),
    )


def sample_linear_time(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Sample stratified linear times in [0, 1)."""
    interval = 1.0 - float(eps)
    interval_size = interval / int(batch_size)
    noise = torch.rand(batch_size, device=device, dtype=dtype)
    offsets = torch.arange(batch_size, device=device, dtype=dtype)
    return (offsets + noise) * interval_size


def sample_globally_stratified_time(
    per_rank_batch_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-6,
    step: Optional[int] = None,
    base_seed: int = 0,
) -> torch.Tensor:
    """Sample times stratified across the global batch.

    Partitions [0, 1 - eps) into ``global_batch = per_rank_batch_size * world_size``
    equal-width strata. Each rank owns ``per_rank_batch_size`` strata and draws
    one uniform sample from each. When ``step`` is provided, a permutation
    seeded identically on every rank (``base_seed + step``) shuffles the
    stratum-to-position assignment so that no rank is permanently tied to any
    region of the time interval.
    """
    global_batch = int(per_rank_batch_size) * int(world_size)
    stratum_width = (1.0 - float(eps)) / float(global_batch)

    if step is not None:
        gen = torch.Generator(device="cpu").manual_seed(int(base_seed) + int(step))
        permutation = torch.randperm(global_batch, generator=gen)
        start = int(rank) * int(per_rank_batch_size)
        stratum_indices = permutation[start : start + int(per_rank_batch_size)]
    else:
        start = int(rank) * int(per_rank_batch_size)
        stratum_indices = torch.arange(
            start, start + int(per_rank_batch_size), dtype=torch.long
        )

    stratum_indices = stratum_indices.to(device=device, dtype=dtype)
    noise = torch.rand(int(per_rank_batch_size), device=device, dtype=dtype)
    return (stratum_indices + noise) * stratum_width


def sample_linear_hitting_times(
    x1: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample linear insertion and unmasking hitting times per token."""
    return sample_schedule_times(
        x1,
        insertion_schedule=LINEAR_SCHEDULE,
        unmasking_schedule=LINEAR_SCHEDULE,
        eps=eps,
    )


def linear_elbo_weight(t: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Linear-schedule ELBO weight."""
    return linear_hazard_rate(t, eps=eps)


def _schedule_alpha(
    t: torch.Tensor,
    schedule: str,
    *,
    exponent: Optional[float] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> torch.Tensor:
    if schedule in _POWER_FAMILY:
        exp = _resolve_exponent(schedule, exponent)
        one_minus_t = (1.0 - t).clamp_min(0.0)
        return 1.0 - one_minus_t.pow(exp)
    if schedule == LOG_LINEAR_SCHEDULE:
        lam, c = _resolve_log_linear_params(params)
        return _log_linear_alpha(t, lam=lam, c=c)
    if schedule == LOGIT_POWER_SCHEDULE:
        a, b = _resolve_logit_power_params(params)
        return _logit_power_alpha(t, a=a, b=b)
    raise AssertionError(f"Unhandled schedule {schedule!r}")


def _schedule_alpha_derivative(
    t: torch.Tensor,
    schedule: str,
    *,
    exponent: Optional[float] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> torch.Tensor:
    if schedule in _POWER_FAMILY:
        exp = _resolve_exponent(schedule, exponent)
        one_minus_t = (1.0 - t).clamp_min(0.0)
        return exp * one_minus_t.pow(exp - 1.0)
    if schedule == LOG_LINEAR_SCHEDULE:
        lam, c = _resolve_log_linear_params(params)
        return _log_linear_alpha_derivative(t, lam=lam, c=c)
    if schedule == LOGIT_POWER_SCHEDULE:
        a, b = _resolve_logit_power_params(params)
        return _logit_power_alpha_derivative(t, a=a, b=b)
    raise AssertionError(f"Unhandled schedule {schedule!r}")


def _schedule_alpha_inverse(
    alpha: torch.Tensor,
    schedule: str,
    *,
    exponent: Optional[float] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> torch.Tensor:
    if schedule in _POWER_FAMILY:
        exp = _resolve_exponent(schedule, exponent)
        alpha = alpha.clamp(min=0.0, max=1.0)
        return 1.0 - (1.0 - alpha).clamp_min(0.0).pow(1.0 / exp)
    if schedule == LOG_LINEAR_SCHEDULE:
        lam, c = _resolve_log_linear_params(params)
        return _log_linear_alpha_inverse(alpha, lam=lam, c=c)
    if schedule == LOGIT_POWER_SCHEDULE:
        a, b = _resolve_logit_power_params(params)
        return _logit_power_alpha_inverse(alpha, a=a, b=b)
    raise AssertionError(f"Unhandled schedule {schedule!r}")


def schedule_alpha(
    t: torch.Tensor,
    *,
    schedule: str = LINEAR_SCHEDULE,
    field: str = "alpha",
    exponent: Optional[float] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> torch.Tensor:
    """Public alpha(t) for a configured schedule. Used by inference-time
    schedule reparameterization (see flexmdm.inference._map_inference_time_to_model_time)."""
    schedule = normalize_schedule_name(schedule, field=field)
    return _schedule_alpha(t, schedule, exponent=exponent, params=params)


def schedule_alpha_inverse(
    alpha: torch.Tensor,
    *,
    schedule: str = LINEAR_SCHEDULE,
    field: str = "alpha inverse",
    exponent: Optional[float] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> torch.Tensor:
    """Public inverse t(alpha)."""
    schedule = normalize_schedule_name(schedule, field=field)
    return _schedule_alpha_inverse(alpha, schedule, exponent=exponent, params=params)


def linear_hazard_rate(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Linear schedule event hazard used by inference tau-leaping."""
    alpha = _schedule_alpha(t, LINEAR_SCHEDULE)
    alpha_prime = _schedule_alpha_derivative(t, LINEAR_SCHEDULE)
    return alpha_prime / (1.0 - alpha + float(eps))


def _sample_schedule_time_after(
    start_t: torch.Tensor,
    *,
    schedule: str,
    exponent: Optional[float] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> torch.Tensor:
    start_alpha = _schedule_alpha(start_t, schedule, exponent=exponent, params=params)
    rand = torch.rand_like(start_t, dtype=torch.float32)
    alpha = start_alpha + rand * (1.0 - start_alpha)
    return _schedule_alpha_inverse(
        alpha, schedule, exponent=exponent, params=params
    ).to(start_t.device)


def schedule_hazard_rate(
    t: torch.Tensor,
    *,
    schedule: str = LINEAR_SCHEDULE,
    eps: float = 1e-6,
    field: str = "hazard_rate",
    exponent: Optional[float] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> torch.Tensor:
    """Return the hazard rate for a configured schedule.

    For the ``power`` schedule with alpha(t) = 1 - (1-t)^n, the hazard
    rate is alpha'/(1-alpha) = n / (1-t), so the eps floor is only a
    numerical safety net near t=1. ``log_linear`` and ``logit_power``
    use the same formula and rely on the same eps floor near t=1.
    """
    schedule = normalize_schedule_name(schedule, field=field)
    alpha = _schedule_alpha(t, schedule, exponent=exponent, params=params)
    alpha_prime = _schedule_alpha_derivative(
        t, schedule, exponent=exponent, params=params
    )
    return alpha_prime / (1.0 - alpha + float(eps))


def sample_schedule_times(
    x1: torch.Tensor,
    *,
    insertion_schedule: str = LINEAR_SCHEDULE,
    unmasking_schedule: str = LINEAR_SCHEDULE,
    eps: float = 1e-6,
    insertion_exponent: Optional[float] = None,
    unmasking_exponent: Optional[float] = None,
    insertion_params: Optional[Mapping[str, Any]] = None,
    unmasking_params: Optional[Mapping[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample insertion and unmasking hitting times for configured schedules.

    For the power family ``(linear, quadratic, power)``, ``*_exponent`` selects
    the parameterization. For ``log_linear`` and ``logit_power``, ``*_params``
    selects the parameterization (defaults: log_linear ``{lam:1, c:75}``,
    logit_power ``{a:1.5, b:2}``). The two parameterization channels are
    independent — exponent is ignored by the new schedules and params is
    ignored by the power family.
    """
    insertion_schedule = normalize_schedule_name(
        insertion_schedule,
        field="insertion sampling",
    )
    unmasking_schedule = normalize_schedule_name(
        unmasking_schedule,
        field="unmasking sampling",
    )
    eps_time = x1.new_full(
        x1.shape,
        float(eps),
        dtype=torch.float32,
    )
    insertion_time = _sample_schedule_time_after(
        eps_time,
        schedule=insertion_schedule,
        exponent=insertion_exponent,
        params=insertion_params,
    )
    unmasking_time = _sample_schedule_time_after(
        insertion_time,
        schedule=unmasking_schedule,
        exponent=unmasking_exponent,
        params=unmasking_params,
    )
    return insertion_time, unmasking_time


def schedule_elbo_weights(
    t: torch.Tensor,
    *,
    insertion_schedule: str = LINEAR_SCHEDULE,
    unmasking_schedule: str = LINEAR_SCHEDULE,
    eps: float = 1e-6,
    insertion_exponent: Optional[float] = None,
    unmasking_exponent: Optional[float] = None,
    insertion_params: Optional[Mapping[str, Any]] = None,
    unmasking_params: Optional[Mapping[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return insertion and unmasking ELBO weights for configured schedules."""
    insertion_weight = schedule_hazard_rate(
        t,
        schedule=insertion_schedule,
        eps=eps,
        field="insertion ELBO",
        exponent=insertion_exponent,
        params=insertion_params,
    )
    unmasking_weight = schedule_hazard_rate(
        t,
        schedule=unmasking_schedule,
        eps=eps,
        field="unmasking ELBO",
        exponent=unmasking_exponent,
        params=unmasking_params,
    )
    return insertion_weight, unmasking_weight


def sample_hitting_times(
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible alias for sample_schedule_times."""
    return sample_schedule_times(*args, **kwargs)
