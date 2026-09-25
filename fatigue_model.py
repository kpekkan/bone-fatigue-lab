"""Constant-amplitude Paris-law teaching model. Stresses MPa; lengths SI internally.

The threshold is an illustrative hard cutoff, not a calibrated bone model.
No crack initiation, closure, remodeling, variable Y, or near-failure correction.
"""
from dataclasses import dataclass, replace
from math import expm1, isfinite, log, pi, sqrt
import numpy as np


@dataclass(frozen=True)
class Parameters:
    delta_sigma: float = 11.0          # MPa, full peak-to-trough range
    sigma_max: float = 11.0            # MPa, maximum tensile stress
    a0_mm: float = 0.01                # pre-existing crack, mm
    Kc: float = 2.2                    # MPa sqrt(m)
    Y: float = 1.0                     # constant geometry factor
    m: float = 2.5
    growth_ref: float = 2.5e-6          # m/cycle at K_ref
    K_ref: float = 1.0                 # MPa sqrt(m); fixes pivot when m changes
    threshold_on: bool = False
    delta_K_th: float = 0.1            # MPa sqrt(m), ILLUSTRATIVE, not measured
    stride_m: float = 2.0              # same foot to same foot = one tibial cycle
    cycles_per_day: float = 5000.0     # scheduling conversion only
    section_mm: float = 15.0          # textbook section scale, warning only
    af_override_mm: float | None = None  # optional earlier endpoint, e.g. 12.7


def validate(p):
    positive = ('a0_mm', 'Kc', 'Y', 'm', 'growth_ref',
                'K_ref', 'stride_m', 'cycles_per_day', 'section_mm')
    for name in positive:
        if not isfinite(getattr(p, name)) or getattr(p, name) <= 0:
            raise ValueError(f'{name} must be positive and finite.')
    for name in ('sigma_max', 'delta_sigma', 'delta_K_th'):
        if not isfinite(getattr(p, name)) or getattr(p, name) < 0:
            raise ValueError(f'{name} must be nonnegative and finite.')
    if p.af_override_mm is not None:
        if not isfinite(p.af_override_mm) or p.af_override_mm <= p.a0_mm:
            raise ValueError('The specified final crack must exceed the initial crack.')


def paris_cycles(a0, af, delta_sigma, C, m, Y=1.0):
    """Analytic integral. a0 and af in m; C uses MPa sqrt(m), not Pa sqrt(m).

    Handles m=2 and values arbitrarily close to 2 using expm1 for accuracy.
    C units: m / (cycle * (MPa sqrt(m))**m).
    """
    if not all(isfinite(x) for x in (a0, af, delta_sigma, C, m, Y)):
        raise ValueError('All arguments must be finite.')
    if min(a0, C, m, Y) <= 0 or af < a0 or delta_sigma < 0:
        raise ValueError('Invalid crack lengths, material properties, or stress range.')
    if af == a0:
        return 0.0
    if delta_sigma == 0:
        return float('inf')
    b = 1.0 - m / 2.0
    L = log(af / a0)
    integral = L if b == 0 else a0**b * expm1(b * L) / b
    return integral / (C * (Y * delta_sigma)**m * pi**(m/2.0))


def solve(p=Parameters()):
    validate(p)
    a0 = p.a0_mm * 1e-3
    ac = (p.Kc / (p.Y * p.sigma_max))**2 / pi if p.sigma_max > 0 else float('inf')
    af = ac if p.af_override_mm is None else min(ac, p.af_override_mm * 1e-3)
    K0 = p.Y * p.sigma_max * sqrt(pi*a0)
    dK0 = p.Y * p.delta_sigma * sqrt(pi*a0)
    C = p.growth_ref / p.K_ref**p.m
    endpoint = 'Fast-fracture criterion' if af == ac else 'Specified crack length'
    if K0 >= p.Kc:
        N, status = 0.0, 'Immediate fast fracture'
    elif p.delta_sigma == 0:
        N, status = float('inf'), 'No cyclic driving force'
    elif p.sigma_max == 0:
        N, status = float('inf'), 'No tensile crack opening'
    elif p.threshold_on and dK0 <= p.delta_K_th:
        N, status = float('inf'), 'No growth in threshold model'
    else:
        N = paris_cycles(a0, af, p.delta_sigma, C, p.m, p.Y)
        status = 'Finite propagation life'
    return dict(N=N, distance_km=N*p.stride_m/1000,
                days=N/p.cycles_per_day, ac_mm=ac*1e3, af_mm=af*1e3,
                K0=K0, dK0=dK0, C=C, status=status, endpoint=endpoint,
                outside_geometry=isfinite(af) and af*1e3 > p.section_mm)


def trajectory(p, points=500):
    """Return physical cycles and crack length. Log-spaced cracks resolve acceleration."""
    r = solve(p)
    if r['N'] == 0:
        return np.array([0.0]), np.array([p.a0_mm])
    if not isfinite(r['N']):
        # The flat trace is shown over a reference window, not over an infinite axis.
        baseline = solve(replace(p, threshold_on=False))['N']
        horizon = baseline if isfinite(baseline) else 20000.0
        return np.linspace(0, horizon, points), np.full(points, p.a0_mm)
    a = np.geomspace(p.a0_mm*1e-3, r['af_mm']*1e-3, points)
    n = np.array([paris_cycles(a[0], x, p.delta_sigma, r['C'], p.m, p.Y) for x in a])
    return n, a*1000


def growth_rate(p, delta_K):
    k = np.asarray(delta_K, dtype=float)
    rate = p.growth_ref*(k/p.K_ref)**p.m
    return np.where(p.threshold_on & (k <= p.delta_K_th), 0, rate)


def stress_sweep(p, link_peak=True, points=180):
    """Sweep range. Linked: sigma_max=range; unlinked: sigma_max remains fixed."""
    ranges = np.geomspace(0.01, 100, points)
    raw, limited = [], []
    for stress in ranges:
        q = replace(p, delta_sigma=stress,
                    sigma_max=stress if link_peak else p.sigma_max)
        raw.append(solve(replace(q, threshold_on=False))['N'])
        limited.append(solve(replace(q, threshold_on=True))['N'])
    return ranges, np.array(raw), np.array(limited)


def format_number(x, decimals=0):
    if not isfinite(x):
        return 'No finite endpoint'
    if x != 0 and (abs(x) >= 1e8 or abs(x) < 0.01):
        return f'{x:.2e}'
    return f'{x:,.{decimals}f}'
