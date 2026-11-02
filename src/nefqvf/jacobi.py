"""Shared Jacobi recurrence for all six families, in the positive
off-diagonal convention."""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit, prange

from .broadcasting import outer_values, paired_values

NORMAL = 0
POISSON = 1
GAMMA = 2
BINOMIAL = 3
NEGATIVE_BINOMIAL = 4
GHS = 5


@njit(inline="always")
def _a_scalar(family: int, n: int, mean: float, fixed: float) -> float:
    """Return one positive off-diagonal coefficient inside Numba kernels."""

    if n == 0:
        return 0.0
    if family == NORMAL:
        return fixed * np.sqrt(n)
    if family == POISSON:
        return np.sqrt(mean * n)
    if family == GAMMA:
        return (mean / fixed) * np.sqrt(n * (n + fixed - 1.0))
    if family == BINOMIAL:
        p = mean / fixed
        return np.sqrt(p * (1.0 - p) * n * (fixed - n + 1.0))
    if family == NEGATIVE_BINOMIAL:
        c = mean / (fixed + mean)
        return np.sqrt(c * n * (n + fixed - 1.0)) / (1.0 - c)

    # GHS: cos(eta / 2) = r / sqrt(r**2 + mean**2).
    sec_half_eta = np.sqrt(fixed * fixed + mean * mean) / fixed
    return 0.5 * sec_half_eta * np.sqrt(n * (n + 2.0 * fixed - 1.0))


@njit(inline="always")
def _b_scalar(family: int, n: int, mean: float, fixed: float) -> float:
    """Return one diagonal coefficient inside Numba kernels."""

    if family == NORMAL:
        return mean
    if family == POISSON:
        return mean + n
    if family == GAMMA:
        return (mean / fixed) * (2.0 * n + fixed)
    if family == BINOMIAL:
        p = mean / fixed
        return p * (fixed - n) + (1.0 - p) * n
    if family == NEGATIVE_BINOMIAL:
        c = mean / (fixed + mean)
        return (c * fixed + (1.0 + c) * n) / (1.0 - c)
    return (n + fixed) * mean / fixed


@njit(cache=True, parallel=True)
def _basis_kernel(
    family: int,
    x: np.ndarray,
    mean: np.ndarray,
    fixed: np.ndarray,
    n_max: int,
) -> np.ndarray:
    """Evaluate the forward Jacobi recurrence for flattened instances."""

    result = np.empty((x.size, n_max + 1), dtype=np.float64)
    for index in prange(x.size):
        result[index, 0] = 1.0
        previous = 0.0
        current = 1.0
        for n in range(n_max):
            a_n = _a_scalar(family, n, mean[index], fixed[index])
            a_next = _a_scalar(family, n + 1, mean[index], fixed[index])
            b_n = _b_scalar(family, n, mean[index], fixed[index])
            following = ((x[index] - b_n) * current - a_n * previous) / a_next
            result[index, n + 1] = following
            previous = current
            current = following
    return result


@njit(cache=True, parallel=True)
def _basis_dot_kernel(
    family: int,
    x: np.ndarray,
    mean: np.ndarray,
    fixed: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Evaluate one expansion per flattened instance by Clenshaw recurrence."""

    degree = coefficients.shape[1] - 1
    result = np.empty(x.size, dtype=np.float64)

    for index in prange(x.size):
        if degree == 0:
            result[index] = coefficients[index, 0]
            continue

        u_k_plus_1 = coefficients[index, degree]
        u_k_plus_2 = 0.0
        for k in range(degree - 1, -1, -1):
            a_next = _a_scalar(family, k + 1, mean[index], fixed[index])
            alpha = (
                x[index] - _b_scalar(family, k, mean[index], fixed[index])
            ) / a_next
            u_k = coefficients[index, k] + alpha * u_k_plus_1
            if k + 2 <= degree:
                ratio = _a_scalar(family, k + 1, mean[index], fixed[index]) / _a_scalar(
                    family, k + 2, mean[index], fixed[index]
                )
                u_k -= ratio * u_k_plus_2
            u_k_plus_2 = u_k_plus_1
            u_k_plus_1 = u_k
        result[index] = u_k_plus_1
    return result


def _evaluation_values(
    x: Any, mean: Any, fixed: Any, grid: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Prepare paired or grid-shaped values for the compiled kernels."""

    if grid:
        x_view, parameter_views, batch_shape, observation_shape = outer_values(
            x, mean, fixed
        )
        output_shape = batch_shape + observation_shape
        x_array, mean_array, fixed_array = np.broadcast_arrays(x_view, *parameter_views)
        return x_array, mean_array, fixed_array, output_shape, batch_shape

    (x_array, mean_array, fixed_array), output_shape = paired_values(x, mean, fixed)
    return x_array, mean_array, fixed_array, output_shape, output_shape


def basis(
    family: int,
    x: Any,
    n_max: int,
    mean: Any,
    fixed: Any,
    *,
    grid: bool = False,
) -> np.ndarray:
    """Evaluate ``phi_0, ..., phi_n_max`` along a final degree axis.

    The compiled kernel sees one flattened row per broadcast instance. This
    wrapper performs all shape work and restores the user-facing axes.
    """
    if not isinstance(n_max, (int, np.integer)) or n_max < 0:
        raise ValueError("n_max must be a nonnegative integer")

    x_array, mean_array, fixed_array, output_shape, _ = _evaluation_values(
        x, mean, fixed, grid
    )
    result = _basis_kernel(
        family,
        np.asarray(x_array, dtype=float).ravel(),
        np.asarray(mean_array, dtype=float).ravel(),
        np.asarray(fixed_array, dtype=float).ravel(),
        int(n_max),
    )
    return result.reshape(output_shape + (n_max + 1,))


def basis_dot(
    family: int,
    x: Any,
    coefficients: Any,
    mean: Any,
    fixed: Any,
    *,
    grid: bool = False,
) -> np.ndarray:
    """Evaluate a polynomial expansion without materializing its basis tensor.

    ``coefficients`` stores polynomial degree on its final axis. In grid mode,
    its leading axes must broadcast to the parameter batch rather than to the
    observation axes.
    """
    coefficient_array = np.asarray(coefficients)
    if coefficient_array.ndim == 0 or coefficient_array.shape[-1] == 0:
        raise ValueError("coefficients must have a nonempty final degree axis")

    if grid:
        x_view, parameter_views, batch_shape, observation_shape = outer_values(
            x, mean, fixed
        )
        coefficient_shape = coefficient_array.shape[:-1]
        coefficient_batch_shape = np.broadcast_shapes(batch_shape, coefficient_shape)
        if coefficient_batch_shape != batch_shape:
            raise ValueError(
                "For grid evaluation, coefficient batch dimensions must "
                "broadcast to the parameter batch shape"
            )
        coefficient_view = np.broadcast_to(
            coefficient_array, batch_shape + (coefficient_array.shape[-1],)
        ).reshape(
            batch_shape + (1,) * len(observation_shape) + (coefficient_array.shape[-1],)
        )
        output_shape = batch_shape + observation_shape
        x_array, mean_array, fixed_array = np.broadcast_arrays(x_view, *parameter_views)
        coefficients_array = np.broadcast_to(
            coefficient_view, output_shape + (coefficient_array.shape[-1],)
        )
    else:
        values = (np.asarray(x), np.asarray(mean), np.asarray(fixed))
        output_shape = np.broadcast_shapes(
            *(value.shape for value in values), coefficient_array.shape[:-1]
        )
        x_array, mean_array, fixed_array = (
            np.broadcast_to(value, output_shape) for value in values
        )
        coefficients_array = np.broadcast_to(
            coefficient_array, output_shape + (coefficient_array.shape[-1],)
        )

    coeff_dtype = coefficients_array.dtype

    result = _basis_dot_kernel(
        family,
        np.asarray(x_array, dtype=coeff_dtype).ravel(),
        np.asarray(mean_array, dtype=coeff_dtype).ravel(),
        np.asarray(fixed_array, dtype=coeff_dtype).ravel(),
        np.asarray(coefficients_array, dtype=coeff_dtype).reshape(
            -1, coefficient_array.shape[-1]
        ),
    )
    return result.reshape(output_shape)


def jacobi_coefficients(
    family: int, n: Any, mean: Any, fixed: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(a_n, b_n)`` in the positive off-diagonal convention.

    ``family`` is an internal integer code so the same Numba kernels can serve
    all systems without Python-level dispatch in the recurrence loop.
    """
    n_array = np.asarray(n)
    if (
        np.any(~np.isfinite(n_array))
        or np.any(n_array < 0)
        or np.any(n_array != np.floor(n_array))
    ):
        raise ValueError("n must contain nonnegative integers")

    (n_array, mean_array, fixed_array), _ = paired_values(n_array, mean, fixed)
    n_float = n_array.astype(float)

    if family == NORMAL:
        a_n = fixed_array * np.sqrt(n_float)
        b_n = mean_array + np.zeros_like(n_float)
    elif family == POISSON:
        a_n = np.sqrt(mean_array * n_float)
        b_n = mean_array + n_float
    elif family == GAMMA:
        theta = mean_array / fixed_array
        a_n = theta * np.sqrt(n_float * (n_float + fixed_array - 1.0))
        b_n = theta * (2.0 * n_float + fixed_array)
    elif family == BINOMIAL:
        if np.any(n_float > fixed_array):
            raise ValueError("Binomial Jacobi coefficients require n <= N")
        p = mean_array / fixed_array
        a_n = np.sqrt(p * (1.0 - p) * n_float * (fixed_array - n_float + 1.0))
        b_n = p * (fixed_array - n_float) + (1.0 - p) * n_float
    elif family == NEGATIVE_BINOMIAL:
        c = mean_array / (fixed_array + mean_array)
        a_n = np.sqrt(c * n_float * (n_float + fixed_array - 1.0)) / (1.0 - c)
        b_n = (c * fixed_array + (1.0 + c) * n_float) / (1.0 - c)
    else:
        sec_half_eta = np.sqrt(fixed_array**2 + mean_array**2) / fixed_array
        a_n = (
            0.5 * sec_half_eta * np.sqrt(n_float * (n_float + 2.0 * fixed_array - 1.0))
        )
        b_n = (n_float + fixed_array) * mean_array / fixed_array
    return np.asarray(a_n), np.asarray(b_n)
