"""JAX/GPU implementation of the boundary-finding portion of ``Profile.Jtor_part1``.

``freegs4e.jtor.Profile.Jtor_part1`` returns Python lists and ``None`` when an
X-point is absent.  Those values are not JIT compatible, so this module keeps
all outputs at fixed shape and represents that case with ``has_boundary=False``.

The implementation follows the same high-level procedure as FreeGS4E:

1. find the magnetic axis and the primary X-point;
2. use the X-point flux as ``psi_bndry`` (unless supplied explicitly);
3. flood-fill the connected region around the axis below the separatrix.

The critical-point calculation uses the same central finite differences and
sub-grid stationary-point correction as ``freegs4e.critical.scan_for_crit``.
The point lists are represented by fixed-size grid-shaped arrays internally so
that the whole calculation can be compiled by JAX.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Protocol

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import beta
import numpy as np

import freegsnke

from .utils import point_in_polygon_grid

class ProfileLike(Protocol):
    """The ``freegs4e.jtor.Profile`` attributes used by this adapter."""

    Ip: float
    mask_inside_limiter: np.ndarray

@dataclass(frozen=True, slots=True)
class Jtor_build_result:

    jtor: jax.Array
    opt: jax.Array
    xpt: jax.Array
    psi_bndry: jax.Array
    diverted_core_mask: jax.Array
    limiter_core_mask: jax.Array
    flag_limiter: jax.Array
    flood_fill_steps: jax.Array

    def __iter__(self):
        yield self.jtor
        yield self.opt
        yield self.xpt
        yield self.psi_bndry
        yield self.diverted_core_mask
        yield self.limiter_core_mask
        yield self.flag_limiter

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class Jtor_part1_result:
    """Fixed-shape, JIT-compatible equivalent of ``Profile.Jtor_part1``.

    ``opt`` and ``xpt`` have shape ``(1, 3)``.  If no X-point is found,
    ``xpt`` is NaN, ``diverted_core_mask`` is all False, and
    ``has_boundary`` is False.  This replaces the CPU API's empty ``xpt`` and
    ``None`` core mask.
    """

    opt: jax.Array
    xpt: jax.Array
    diverted_core_mask: jax.Array
    psi_bndry: jax.Array
    has_xpt: jax.Array
    has_boundary: jax.Array

    def tree_flatten(self) -> tuple[tuple[jax.Array, ...], None]:
        return (
            (
                self.opt,
                self.xpt,
                self.diverted_core_mask,
                self.psi_bndry,
                self.has_xpt,
                self.has_boundary,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls, aux_data: None, children: tuple[jax.Array, ...]
    ) -> "Jtor_part1_result":
        del aux_data
        return cls(*children)

    def __iter__(self):
        yield self.opt
        yield self.xpt
        yield self.diverted_core_mask
        yield self.psi_bndry
        yield self.has_xpt
        yield self.has_boundary


def Jtor_part1(
    R: jax.Array,
    Z: jax.Array,
    psi: jax.Array,
    mask_inside_limiter: jax.Array,
    ip: float | jax.Array,
    *,
    psi_bndry: float | jax.Array | None = None,
    max_iterations: int | None = None,
    los_top_k: int = 20,
) -> Jtor_part1_result:
    """Run the JAX version of ``freegs4e.jtor.Profile.Jtor_part1``.

    Inputs should already be JAX device arrays to avoid a host-to-device copy
    on every equilibrium iteration.  ``max_iterations`` defaults to the number
    of grid points, which is sufficient for four-neighbour propagation across
    a rectangular grid.
    """

    if los_top_k < 1:
        raise ValueError("los_top_k must be at least one.")
    iteration_count = psi.shape[0] + psi.shape[1] - 2 \
        if max_iterations is None else max_iterations
    provided_boundary = psi_bndry is not None
    boundary_value = psi[0, 0] if psi_bndry is None else psi_bndry
    outputs = _jtor_part1_kernel(
        R,
        Z,
        psi,
        mask_inside_limiter,
        ip, 
        boundary_value, 
        provided_boundary,
        int(iteration_count),
        los_top_k,
    )
    # ``flood_fill_steps`` is an internal diagnostic appended by the kernel;
    # retain the established public Jtor_part1 result shape.
    return Jtor_part1_result(*outputs[:6])


@partial(jax.jit, static_argnames=("max_iterations", "los_top_k"))
def _jtor_part1_kernel(
    R: jax.Array,
    Z: jax.Array,
    psi: jax.Array,
    mask_inside_limiter: jax.Array,
    ip: jax.Array,
    supplied_psi_bndry: jax.Array,
    use_supplied_psi_bndry: jax.Array,
    max_iterations: int,
    los_top_k: int = 20,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:

    opt, has_opt, xpt, has_xpt, xpt_cells = _critical_points(
        R, Z, psi, mask_inside_limiter, ip, los_top_k
    )
    one = jnp.ones((), dtype=psi.dtype)
    sign_direction = jnp.where(ip < 0, -one, one)
    oriented_psi = psi * sign_direction
    psi_bndry_oriented = jnp.where(
        use_supplied_psi_bndry,
        supplied_psi_bndry * sign_direction,
        xpt[2] * sign_direction,
    )
    axis_r_index = jnp.argmin(jnp.abs(R[:, 0] - opt[0]))
    axis_z_index = jnp.argmin(jnp.abs(Z[0, :] - opt[1]))
    blocked = _dilate_cells(xpt_cells)
    # ``score > psi_bndry`` in the former maximin propagation is equivalent
    # to reachability from the axis through cells strictly above the
    # separatrix.  Carrying only this boolean reachability map avoids a
    # float64 min/max map on every flood-fill sweep.
    allowed = (
        (oriented_psi > psi_bndry_oriented)
        & ~blocked
    )
    reachable = jnp.zeros_like(allowed).at[axis_r_index, axis_z_index].set(
        allowed[axis_r_index, axis_z_index]
    )

    def cond_fn(state):
        _, changed, step = state
        return changed & (step < max_iterations)

    def body_fn(state):
        reachable, _, step = state
        next_reachable = _propagate_reachable(reachable, allowed)
        changed = jnp.any(next_reachable != reachable)
        return next_reachable, changed, step + 1

    reachable, _, flood_fill_steps = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (reachable, jnp.array(True), jnp.array(0)),
    )
    # Preserve the established JAX behaviour: limiter membership is applied
    # to the final core, while flood-fill may traverse the surrounding grid.
    core_mask = reachable & mask_inside_limiter
    core_mask = core_mask & _geometric_core_side(R, Z, opt, xpt, has_xpt)
    has_core = jnp.any(core_mask)
    has_boundary = has_opt & has_core & (use_supplied_psi_bndry | has_xpt)
    core_mask = core_mask & has_boundary
    psi_bndry = jnp.where(
        has_boundary, psi_bndry_oriented / sign_direction, psi[0, 0]
    )
    return (
        opt[jnp.newaxis, :],
        jnp.where(has_xpt, xpt, jnp.full((3,), jnp.nan, dtype=psi.dtype))[jnp.newaxis, :],
        core_mask,
        psi_bndry,
        has_xpt,
        has_boundary,
        flood_fill_steps,
    )


def _critical_points(
    R: jax.Array,
    Z: jax.Array,
    psi: jax.Array,
    mask_inside_limiter: jax.Array,
    ip: jax.Array,
    los_top_k: int = 20,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """JAX translation of ``scan_for_crit`` + ``find_critical`` selection."""

    d_r = R[1, 0] - R[0, 0]
    d_z = Z[0, 1] - Z[0, 0]
    psi_r = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * d_r)
    psi_z = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * d_z)
    bp2 = psi_r**2 + psi_z**2
    center = bp2[1:-1, 1:-1]
    local_min = (
        (center < bp2[:-2, :-2]) & (center < bp2[:-2, 1:-1])
        & (center < bp2[:-2, 2:]) & (center < bp2[1:-1, :-2])
        & (center < bp2[1:-1, 2:]) & (center < bp2[2:, :-2])
        & (center < bp2[2:, 1:-1]) & (center < bp2[2:, 2:])
    )
    values = psi[2:-2, 2:-2]
    f_r = psi_r[1:-1, 1:-1]
    f_z = psi_z[1:-1, 1:-1]
    f_rr = (psi[3:-1, 2:-2] - 2 * values + psi[1:-3, 2:-2]) / d_r**2
    f_zz = (psi[2:-2, 3:-1] - 2 * values + psi[2:-2, 1:-3]) / d_z**2
    # Keep FreeGS4E's historical 0.5 convention exactly: its determinant
    # uses ``0.25 * f_rz**2`` and its correction uses ``0.5 * f_rz``.
    f_rz = 0.5 * (
        psi[3:-1, 3:-1] + psi[1:-3, 1:-3]
        - psi[3:-1, 1:-3] - psi[1:-3, 3:-1]
    ) / (d_r * d_z)
    determinant = f_rr * f_zz - 0.25 * f_rz**2
    safe_determinant = jnp.where(determinant == 0, 1.0, determinant)
    delta_r = -(f_r * f_zz - 0.5 * f_rz * f_z) / safe_determinant
    delta_z = -(f_z * f_rr - 0.5 * f_rz * f_r) / safe_determinant
    valid = (
        local_min & (determinant != 0)
        & (jnp.abs(delta_r) < 1.5 * d_r) & (jnp.abs(delta_z) < 1.5 * d_z)
    )
    estimated_psi = values + 0.5 * (f_r * delta_r + f_z * delta_z)
    candidate_r = R[2:-2, 2:-2] + delta_r
    candidate_z = Z[2:-2, 2:-2] + delta_z

    o_candidates = valid & (determinant > 0)
    # ``fastcrit`` discards O-points whose closest grid cell is outside the
    # limiter, then chooses the remaining point nearest the domain centre.
    o_candidates = o_candidates & mask_inside_limiter[2:-2, 2:-2]
    r_mid = 0.5 * (R[-1, 0] + R[0, 0])
    z_mid = 0.5 * (Z[0, -1] + Z[0, 0])
    o_cost = (candidate_r - r_mid) ** 2 + (candidate_z - z_mid) ** 2
    o_index = jnp.argmin(jnp.where(o_candidates, o_cost, jnp.inf))
    has_opt = jnp.any(o_candidates)
    width = o_candidates.shape[1]
    o_r, o_z = o_index // width, o_index % width
    opt = jnp.stack([candidate_r[o_r, o_z], candidate_z[o_r, o_z], estimated_psi[o_r, o_z]])

    # ``find_critical`` retains every correctly ordered saddle after its
    # primary X-point has been accepted.  All of them must remain barriers in
    # ``inside_mask_``.  The line-of-sight check is only used to decide which
    # saddle is the primary X-point returned to the caller.
    xpoint_cells = valid & (determinant < 0)
    xpoint_cells = xpoint_cells & ((estimated_psi - opt[2]) * ip < 0)
    x_cost = (estimated_psi - opt[2]) ** 2
    # Numerical ``find_critical`` sorts the valid X-points by x_cost and
    # checks their lines of sight in that order.  JAX cannot create a
    # variable-length candidate list inside JIT, so retain only a fixed number
    # of the best valid saddles.  On MAST-U this is much smaller than the full
    # 61 x 125 candidate grid, while ``los_top_k`` remains user-configurable.
    k = min(los_top_k, xpoint_cells.size)
    flat_cost = x_cost.reshape(-1)
    flat_xpoint_cells = xpoint_cells.reshape(-1)
    top_scores, top_indices = jax.lax.top_k(
        jnp.where(flat_xpoint_cells, -flat_cost, -jnp.inf), k
    )
    top_valid = jnp.isfinite(top_scores)
    top_candidate_r = candidate_r.reshape(-1)[top_indices]
    top_candidate_z = candidate_z.reshape(-1)[top_indices]
    top_candidate_psi = estimated_psi.reshape(-1)[top_indices]
    top_cost = flat_cost[top_indices]
    top_los = top_valid & _monotonic_line_of_sight(
        R,
        Z,
        psi,
        opt,
        top_candidate_r,
        top_candidate_z,
        top_candidate_psi,
    )
    x_index = jnp.argmin(jnp.where(top_los, top_cost, jnp.inf))
    has_xpt = has_opt & jnp.any(top_los)
    xpt = jnp.stack([
        top_candidate_r[x_index],
        top_candidate_z[x_index],
        top_candidate_psi[x_index],
    ])
    # Keep the full X-point set for the flood-fill barrier.  ``xpt`` above is
    # still the primary X-point selected by the original API, but
    # ``critical.inside_mask_`` protects the neighbourhood of every X-point
    # supplied to it.  The CPU implementation first refines a saddle to its
    # continuous coordinate, then calls ``argmin(abs(grid - coordinate))``.
    # The local-minimum stencil centre is not always that nearest grid point,
    # so reproduce the latter explicitly.
    x_cell_r = jnp.argmin(
        jnp.abs(R[:, 0, jnp.newaxis, jnp.newaxis] - candidate_r), axis=0
    )
    x_cell_z = jnp.argmin(
        jnp.abs(Z[0, :, jnp.newaxis, jnp.newaxis] - candidate_z), axis=0
    )
    xpt_cells = jnp.zeros(psi.size, dtype=bool).at[
        (x_cell_r * psi.shape[1] + x_cell_z).reshape(-1)
    ].max(xpoint_cells.reshape(-1)).reshape(psi.shape)
    return opt, has_opt, xpt, has_xpt, xpt_cells


def _monotonic_line_of_sight(
    R: jax.Array, Z: jax.Array, psi: jax.Array, opt: jax.Array,
    candidate_r: jax.Array, candidate_z: jax.Array, candidate_psi: jax.Array,
) -> jax.Array:
    """Vectorised JAX form of FreeGS4E's ``discard_xpoints_f``."""

    d_r, d_z = R[1, 0] - R[0, 0], Z[0, 1] - Z[0, 0]
    samples = max(2, psi.shape[0], psi.shape[1])
    steps = jnp.arange(samples)
    # This is the exact point count used by ``discard_xpoints_f``.  The
    # fixed-size trailing part is masked only to keep the JIT output shape
    # static; it does not participate in the reductions below.
    count = (
        jnp.floor(
            jnp.maximum(
                jnp.abs(candidate_r - opt[0]) / d_r,
                jnp.abs(candidate_z - opt[1]) / d_z,
            )
        ).astype(jnp.int32)
        + 2
    )
    active = steps < count[..., None]
    t = steps / (count[..., None] - 1)
    r_line = opt[0] + (candidate_r[..., None] - opt[0]) * t
    z_line = opt[1] + (candidate_z[..., None] - opt[1]) * t
    i = jnp.clip(jnp.floor((r_line - R[0, 0]) / d_r).astype(jnp.int32), 0, psi.shape[0] - 2)
    j = jnp.clip(jnp.floor((z_line - Z[0, 0]) / d_z).astype(jnp.int32), 0, psi.shape[1] - 2)
    r_fraction = (r_line - R[i, 0]) / d_r
    z_fraction = (z_line - Z[0, j]) / d_z
    line_psi = (
        (1 - r_fraction) * (1 - z_fraction) * psi[i, j]
        + r_fraction * (1 - z_fraction) * psi[i + 1, j]
        + (1 - r_fraction) * z_fraction * psi[i, j + 1]
        + r_fraction * z_fraction * psi[i + 1, j + 1]
    )
    line_psi = jnp.where(candidate_psi[..., None] < opt[2], -line_psi, line_psi)
    maximum = jnp.max(jnp.where(active, line_psi, -jnp.inf), axis=-1)
    endpoint = jnp.take_along_axis(line_psi, (count - 1)[..., None], axis=-1)[..., 0]
    ratio = (maximum - endpoint) / (maximum - line_psi[..., 0])
    min_index = jnp.argmin(jnp.where(active, line_psi, jnp.inf), axis=-1)
    min_r = jnp.take_along_axis(r_line, min_index[..., None], axis=-1)[..., 0]
    min_z = jnp.take_along_axis(z_line, min_index[..., None], axis=-1)[..., 0]
    return (ratio < 0.001) & ((min_r - opt[0]) ** 2 + (min_z - opt[1]) ** 2 < 1e-4)


def _dilate_cells(mask: jax.Array) -> jax.Array:
    padded = jnp.pad(mask, ((1, 1), (1, 1)), constant_values=False)
    return (
        padded[:-2, :-2] | padded[:-2, 1:-1] | padded[:-2, 2:]
        | padded[1:-1, :-2] | padded[1:-1, 1:-1] | padded[1:-1, 2:]
        | padded[2:, :-2] | padded[2:, 1:-1] | padded[2:, 2:]
    )


def _geometric_core_side(
    R: jax.Array, Z: jax.Array, opt: jax.Array, xpt: jax.Array, has_xpt: jax.Array
) -> jax.Array:
    slope = -(opt[0] - xpt[0]) / (opt[1] - xpt[1])
    intercept = xpt[1] - slope * xpt[0]
    mask = ((opt[1] - (slope * opt[0] + intercept)) * (Z - (slope * R + intercept))) > 0
    return jnp.where(has_xpt, mask, jnp.ones_like(mask, dtype=bool))


def _propagate_reachable(
    reachable: jax.Array, allowed: jax.Array
) -> jax.Array:
    """One four-neighbour boolean flood-fill step, suitable for ``lax``."""

    padded_reachable = jnp.pad(reachable, ((1, 1), (1, 1)), constant_values=False)
    neighbour_reachable = (
        padded_reachable[:-2, 1:-1]
        | padded_reachable[2:, 1:-1]
        | padded_reachable[1:-1, :-2]
        | padded_reachable[1:-1, 2:]
    )
    return reachable | (allowed & neighbour_reachable)


@jax.jit
def _interpolate_limiter_points(psi, cell_ids, weights_r, weights_z, d_r_d_z):
    r, z = cell_ids[:, 0], cell_ids[:, 1]
    z_left = psi[r, z] * weights_z[:, 0] + psi[r, z + 1] * weights_z[:, 1]
    z_right = psi[r + 1, z] * weights_z[:, 0] + psi[r + 1, z + 1] * weights_z[:, 1]
    return (z_left * weights_r[:, 0] + z_right * weights_r[:, 1]) / d_r_d_z


@jax.jit
def _core_mask_limiter_kernel(psi, psi_bndry, core_mask, limiter_cells, cell_ids, weights_r, weights_z, d_r_d_z):
    core = core_mask.astype(psi.dtype)
    neighbours = core[:-1, :-1] + core[1:, :-1] + core[:-1, 1:] + core[1:, 1:]
    offending = jnp.zeros_like(core, dtype=bool).at[:-1, :-1].set((neighbours > 0) & (neighbours < 4))
    offending = offending & limiter_cells
    values = _interpolate_limiter_points(psi, cell_ids, weights_r, weights_z, d_r_d_z)
    active = offending[cell_ids[:, 0], cell_ids[:, 1]]
    psi_on_limiter = jnp.max(jnp.where(active, values, -jnp.inf))
    flag = psi_on_limiter > psi_bndry
    boundary = jnp.where(flag, psi_on_limiter, psi_bndry)
    corrected = jnp.where(flag, (psi > boundary) & core_mask.astype(bool), core_mask.astype(bool))
    return boundary, corrected, flag, offending, values, active


@partial(jax.jit, static_argnames=("max_iterations", "los_top_k"))
def _jtor_build_kernel(
    R: jax.Array,
    Z: jax.Array,
    psi: jax.Array,
    mask_inside_limiter: jax.Array,
    limiter_cells: jax.Array,
    cell_ids: jax.Array,
    weights_r: jax.Array,
    weights_z: jax.Array,
    d_r_d_z: jax.Array,
    ip: jax.Array,
    supplied_psi_bndry: jax.Array,
    use_supplied_psi_bndry: jax.Array,
    paxis: jax.Array,
    Ip: jax.Array,
    Raxis: jax.Array,
    alpha_m: jax.Array,
    alpha_n: jax.Array,
    max_iterations: int,
    los_top_k: int = 20,
) -> tuple[jax.Array, ...]:
    """Build the current profile without any data-dependent Python control flow.

    All limiter geometry inputs are fixed for an equilibrium grid.  Keeping
    this composition inside one JIT boundary prevents a topology change from
    dispatching eager, variable-length diagnostic operations between Picard
    iterations.
    """
    (
        opt,
        xpt,
        diverted_core_mask,
        diverted_psi_bndry,
        _,
        _,
        flood_fill_steps,
    ) = _jtor_part1_kernel(
        R,
        Z,
        psi,
        mask_inside_limiter,
        ip,
        supplied_psi_bndry,
        use_supplied_psi_bndry,
        max_iterations,
        los_top_k,
    )

    def no_diverted_core(_):
        return (
            diverted_psi_bndry,
            jnp.zeros_like(diverted_core_mask),
            jnp.array(False),
        )

    def apply_limiter(_):
        boundary, limiter_core_mask, flag, _, _, _ = _core_mask_limiter_kernel(
            psi,
            diverted_psi_bndry,
            diverted_core_mask & mask_inside_limiter,
            limiter_cells,
            cell_ids,
            weights_r,
            weights_z,
            d_r_d_z,
        )

        # Match the reference fallback: if limiter correction removed every
        # core point, retain the diverted core and its separatrix value.
        def retain_limiter_core(_):
            return boundary, limiter_core_mask, flag

        def restore_diverted_core(_):
            return diverted_psi_bndry, diverted_core_mask & mask_inside_limiter, flag

        return jax.lax.cond(
            jnp.any(limiter_core_mask & mask_inside_limiter),
            retain_limiter_core,
            restore_diverted_core,
            operand=None,
        )

    psi_bndry, limiter_core_mask, flag_limiter = jax.lax.cond(
        jnp.any(diverted_core_mask),
        apply_limiter,
        no_diverted_core,
        operand=None,
    )
    jtor, jtorshape = _ConstrainPaxisIp_kernel(
        R,
        Z,
        psi,
        opt[0, 2],
        psi_bndry,
        limiter_core_mask,
        paxis,
        Ip,
        Raxis,
        alpha_m,
        alpha_n,
    )
    return (
        jtor,
        jtorshape,
        opt,
        xpt,
        psi_bndry,
        diverted_core_mask,
        limiter_core_mask,
        flag_limiter,
        flood_fill_steps,
    )


@partial(jax.jit, static_argnames=("max_iterations",))
def _jtor_build_from_core_mask_kernel(
    R: jax.Array,
    Z: jax.Array,
    psi: jax.Array,
    core_mask: jax.Array,
    limiter_cells: jax.Array,
    cell_ids: jax.Array,
    weights_r: jax.Array,
    weights_z: jax.Array,
    d_r_d_z: jax.Array,
    paxis: jax.Array,
    Ip: jax.Array,
    Raxis: jax.Array,
    alpha_m: jax.Array,
    alpha_n: jax.Array,
    psi_axis: jax.Array,
    psi_bndry: jax.Array,
    max_iterations: int,
) -> tuple[jax.Array, ...]:
    """Build a current profile from an externally supplied core mask."""
    def no_core(_):
        return psi_bndry, jnp.zeros_like(core_mask), jnp.array(False)

    def apply_limiter(_):
        boundary, corrected, flag, _, _, _ = _core_mask_limiter_kernel(
            psi,
            psi_bndry,
            core_mask,
            limiter_cells,
            cell_ids,
            weights_r,
            weights_z,
            d_r_d_z,
        )
        return boundary, corrected, flag

    boundary, limiter_core_mask, flag_limiter = jax.lax.cond(
        jnp.any(core_mask), apply_limiter, no_core, operand=None
    )
    jtor, jtorshape = _ConstrainPaxisIp_kernel(
        R,
        Z,
        psi,
        psi_axis,
        boundary,
        limiter_core_mask,
        paxis,
        Ip,
        Raxis,
        alpha_m,
        alpha_n,
    )
    nan_point = jnp.full((1, 3), jnp.nan, dtype=psi.dtype)
    opt = jnp.concatenate(
        (jnp.zeros((1, 2), dtype=psi.dtype), psi_axis.reshape(1, 1)), axis=1
    )
    return (
        jtor,
        jtorshape,
        opt,
        nan_point,
        boundary,
        core_mask,
        limiter_core_mask,
        flag_limiter,
        jnp.array(0, dtype=jnp.int32),
    )

class Limiter_handler:
    """JAX-compatible counterpart of ``freegsnke.limiter_func.Limiter_handler``.

    Geometry methods retain the FreeGSNKE method names.  They run while the
    limiter is constructed and pack its variable number of fine boundary
    points.  The per-equilibrium methods, especially ``core_mask_limiter``,
    operate only on fixed-shape JAX arrays and can therefore execute on a GPU.
    """

    def __init__(self, eq, limiter, device: jax.Device | None = None):
        self.limiter = limiter
        self.eqR = jax.device_put(jnp.asarray(eq.R), device)
        self.eqZ = jax.device_put(jnp.asarray(eq.Z), device)
        self.eqR_1D = self.eqR[:, 0]
        self.eqZ_1D = self.eqZ[0, :]
        self.dR = self.eqR[1, 0] - self.eqR[0, 0]
        self.dZ = self.eqZ[0, 1] - self.eqZ[0, 0]
        self.dRdZ = self.dR * self.dZ
        self.nx, self.ny = eq.R.shape
        self.nxny = self.nx * self.ny
        self.map2d = jnp.zeros_like(self.eqR)
        self.eqRidx, self.eqZidx = jnp.meshgrid(
            jnp.arange(self.nx), jnp.arange(self.ny), indexing="ij"
        )
        self.vertices = jax.device_put(
            jnp.stack([jnp.asarray(limiter.R), jnp.asarray(limiter.Z)], axis=1), device
        )
        self.build_mask_inside_limiter()
        self.limiter_points()
        self.plasma_pts = self.extract_plasma_pts(self.eqR, self.eqZ, self.mask_inside_limiter)
        self.idxs_mask = self.extract_index_mask(self.mask_inside_limiter)

    def extract_index_mask(self, mask):
        """Return the fixed geometry indices selected by ``mask``."""
        indices = np.argwhere(np.asarray(mask))
        return jnp.asarray(indices.T, dtype=jnp.int32)

    def extract_plasma_pts(self, R, Z, mask):
        """Return ``(R, Z)`` points inside the limiter."""
        idxs = self.extract_index_mask(mask)
        return jnp.stack([R[idxs[0], idxs[1]], Z[idxs[0], idxs[1]]], axis=1)

    def reduce_rect_domain(self, map):
        return map[self.Rrange[0]:self.Rrange[1], self.Zrange[0]:self.Zrange[1]]

    def build_reduced_rect_domain(self):
        indices = np.asarray(self.idxs_mask)
        self.Rrange = (int(indices[0].min()), int(indices[0].max()) + 1)
        self.Zrange = (int(indices[1].min()), int(indices[1].max()) + 1)
        self.mask_inside_limiter_red = self.reduce_rect_domain(self.mask_inside_limiter)

    def build_mask_inside_limiter(self):
        """Build the limiter containment mask on the selected JAX device."""
        self.mask_inside_limiter = point_in_polygon_grid(self.eqR, self.eqZ, self.vertices)

    def broaden_mask(self, mask, layer_size=3):
        """JAX form of FreeGSNKE's square-neighbourhood dilation."""
        padded = jnp.pad(mask.astype(bool), ((layer_size, layer_size),) * 2)
        result = jnp.zeros_like(mask, dtype=bool)
        for r_offset in range(2 * layer_size + 1):
            for z_offset in range(2 * layer_size + 1):
                result = result | padded[
                    r_offset:r_offset + mask.shape[0], z_offset:z_offset + mask.shape[1]
                ]
        return result

    def make_layer_mask(self, mask, layer_size=3):
        return self.broaden_mask(mask, layer_size) & ~mask.astype(bool)

    def limiter_points(self, refine=16):
        """Pack limiter/grid intersections and bilinear interpolation weights.

        ``refine`` is retained for FreeGSNKE API compatibility; the reference
        implementation uses exact intersections with grid lines rather than
        this parameter.
        """
        del refine
        r_grid = np.asarray(self.eqR_1D)
        z_grid = np.asarray(self.eqZ_1D)
        vertices = np.asarray(self.vertices)
        points: list[np.ndarray] = []
        for start, end in zip(vertices[:-1], vertices[1:], strict=True):
            delta = end - start
            segment: list[np.ndarray] = []
            if delta[0] == 0:
                values = z_grid[(z_grid > min(start[1], end[1])) & (z_grid <= max(start[1], end[1]))]
                segment.append(np.column_stack([np.full_like(values, start[0]), values]))
            elif delta[1] == 0:
                values = r_grid[(r_grid > min(start[0], end[0])) & (r_grid <= max(start[0], end[0]))]
                segment.append(np.column_stack([values, np.full_like(values, start[1])]))
            else:
                slope = delta[1] / delta[0]
                intercept = start[1] - slope * start[0]
                r_values = r_grid[(r_grid > min(start[0], end[0])) & (r_grid <= max(start[0], end[0]))]
                z_values = z_grid[(z_grid > min(start[1], end[1])) & (z_grid <= max(start[1], end[1]))]
                segment.append(np.column_stack([r_values, slope * r_values + intercept]))
                segment.append(np.column_stack([(z_values - intercept) / slope, z_values]))
            segment.append(end[None, :])
            local = np.concatenate(segment, axis=0)
            points.append(local[np.argsort(np.linalg.norm(local - end, axis=1))])
        fine_points = np.concatenate(points, axis=0)
        r_ids = np.searchsorted(r_grid, fine_points[:, 0], side="left") - 1
        z_ids = np.searchsorted(z_grid, fine_points[:, 1], side="left") - 1
        r_ids = np.clip(r_ids, 0, self.nx - 2)
        z_ids = np.clip(z_ids, 0, self.ny - 2)
        self.grid_per_limiter_fine_point = jnp.asarray(np.column_stack([r_ids, z_ids]), dtype=jnp.int32)
        self.mask_limiter_cells = jnp.zeros((self.nx, self.ny), dtype=bool).at[r_ids, z_ids].set(True)
        self.limiter_mask_out = self.make_layer_mask(~self.mask_inside_limiter, 1)
        self.offending_mask = jnp.zeros((self.nx, self.ny), dtype=bool)
        r_weights = np.column_stack([r_grid[r_ids + 1] - fine_points[:, 0], fine_points[:, 0] - r_grid[r_ids]])
        z_weights = np.column_stack([z_grid[z_ids + 1] - fine_points[:, 1], fine_points[:, 1] - z_grid[z_ids]])
        self.fine_point = jnp.asarray(fine_points)
        self.fine_cell_ids = self.grid_per_limiter_fine_point
        self.fine_point_per_cell = {tuple(cell): [] for cell in np.unique(np.column_stack([r_ids, z_ids]), axis=0)}
        self.fine_point_per_cell_R = {}
        self.fine_point_per_cell_Z = {}
        for index, cell in enumerate(np.column_stack([r_ids, z_ids])):
            key = tuple(cell)
            self.fine_point_per_cell[key].append(index)
            self.fine_point_per_cell_R.setdefault(key, []).append(r_weights[index])
            self.fine_point_per_cell_Z.setdefault(key, []).append(z_weights[index][None, :])
        for key in self.fine_point_per_cell:
            self.fine_point_per_cell_R[key] = np.asarray(
                self.fine_point_per_cell_R[key]
            )
            self.fine_point_per_cell_Z[key] = np.asarray(
                self.fine_point_per_cell_Z[key]
            )
        self.fine_weights_r = jnp.asarray(r_weights)
        self.fine_weights_z = jnp.asarray(z_weights)

    def interp_on_limiter_points_cell(self, id_R, id_Z, psi):
        """Interpolate ``psi`` at limiter points belonging to one grid cell.

        This keeps FreeGSNKE's public return contract: ``vals`` contains only
        the values in cell ``(id_R, id_Z)`` and ``idxs`` contains their indices
        in ``fine_point``.  The packed arrays remain available internally for
        the JIT-compatible limiter correction.
        """
        values = _interpolate_limiter_points(
            psi, self.fine_cell_ids, self.fine_weights_r, self.fine_weights_z, self.dRdZ
        )
        active = (self.fine_cell_ids[:, 0] == id_R) & (self.fine_cell_ids[:, 1] == id_Z)
        return values[active], jnp.where(active)[0]

    def interp_on_limiter_points(self, id_R, id_Z, psi):
        """Interpolate limiter points in the 3-by-3 cells around a cell.

        ``Limiter_handler`` historically returns variable-length values and
        fine-point indices here.  This method is intentionally a host-side
        compatibility helper; the per-equilibrium GPU path uses the packed
        arrays directly in :func:`_core_mask_limiter_kernel`.
        """
        values = _interpolate_limiter_points(
            psi, self.fine_cell_ids, self.fine_weights_r, self.fine_weights_z, self.dRdZ
        )
        active = (
            (jnp.abs(self.fine_cell_ids[:, 0] - id_R) <= 1)
            & (jnp.abs(self.fine_cell_ids[:, 1] - id_Z) <= 1)
        )
        return values[active], jnp.where(active)[0]

    def core_mask_limiter(self, psi, psi_bndry, core_mask, limiter_mask_out):
        """GPU/JIT implementation of FreeGSNKE's limiter correction."""
        del limiter_mask_out  # Present in the original API but unused there.
        result = _core_mask_limiter_kernel(
            jnp.asarray(psi), jnp.asarray(psi_bndry), jnp.asarray(core_mask),
            self.mask_limiter_cells, self.fine_cell_ids, self.fine_weights_r,
            self.fine_weights_z, self.dRdZ,
        )
        psi_bndry_out, core_mask_out, flag, offending, interpolated, active = result
        self.flag_limiter = flag
        self.offending_mask = offending
        # Preserve the diagnostics exposed by FreeGSNKE while keeping the
        # fixed-size packed values available for JIT-oriented debugging.
        self.interpolated_on_limiter = interpolated[active]
        self.interpolated_idxs = jnp.where(active)[0]
        self.interpolated_on_limiter_packed = interpolated
        self.interpolated_on_limiter_active = active
        return psi_bndry_out, core_mask_out, flag

    def Iy_from_jtor(self, jtor):
        return jtor[self.idxs_mask[0], self.idxs_mask[1]] * self.dRdZ

    def normalize_sum(self, Iy, epsilon=1e-6):
        return Iy / (jnp.sum(Iy) + epsilon)

    def hat_Iy_from_jtor(self, jtor):
        return self.normalize_sum(jtor[self.idxs_mask[0], self.idxs_mask[1]])

    def rebuild_map2d(self, reduced_vector, map_dummy, idxs_mask):
        return jnp.zeros_like(map_dummy, dtype=reduced_vector.dtype).at[
            idxs_mask[0], idxs_mask[1]
        ].set(reduced_vector)


class ConstrainPaxisIp:
    def __init__(
        self,
        eq: freegsnke.equilibrium_update.Equilibrium,
        paxis: float,
        Ip: float,
        fvac: float,
        alpha_m: float,
        alpha_n: float,
        Raxis: float = 1.0,
        los_top_k: int = 20,
    ):

        self.paxis = paxis
        self.Ip = Ip
        self._fvac = fvac

        if alpha_m < 0 or alpha_n < 0:
            raise ValueError("alpha_m and alpha_n must be positive.")
        self.alpha_m = alpha_m
        self.alpha_n = alpha_n

        if Raxis < 0:
            raise ValueError("Raxis must be positive.")
        self.Raxis = Raxis
        if los_top_k < 1:
            raise ValueError("los_top_k must be at least one.")
        self.los_top_k = los_top_k
        self.limiter_handler = Limiter_handler(eq, eq.tokamak.limiter)
    
    def Jtor(
        self,
        R: jax.Array,
        Z: jax.Array,
        psi: jax.Array,
        psi_bndry: float = None,
        core_mask: jax.Array | None = None,
        psi_axis: jax.Array | None = None,
    ):
        result = self.Jtor_build(
            R, Z, psi, psi_bndry, self.limiter_handler, self.Ip, core_mask, psi_axis
        )
        self.jtor, self.opt, self.xpt, self.psi_bndry, self.diverted_core_mask, \
            self.limiter_core_mask, self.flag_limiter = result
        self.flood_fill_steps = result.flood_fill_steps
        return self.jtor

    def Jtor_build(
        self,
        R: jax.Array,
        Z: jax.Array,
        psi: jax.Array,
        psi_bndry: float,
        limiter_handler: Limiter_handler,
        ip: float,
        core_mask: jax.Array | None = None,
        psi_axis: jax.Array | None = None,
    ):
        """Universal function that calculates the plasma current distribution,
        common to all of the different types of profile parametrizations used in FreeGSNKE.
        """
        (
            jtor,
            self.jtorshape,
            opt,
            xpt,
            psi_bndry,
            diverted_core_mask,
            limiter_core_mask,
            flag_limiter,
            flood_fill_steps,
        ) = self._jtor_build_kernel_result(
            R, Z, psi, psi_bndry, limiter_handler, ip, core_mask, psi_axis
        )
        return Jtor_build_result(
            jtor=jtor,
            opt=opt,
            xpt=xpt,
            psi_bndry=psi_bndry,
            diverted_core_mask=diverted_core_mask,
            limiter_core_mask=limiter_core_mask,
            flag_limiter=flag_limiter,
            flood_fill_steps=flood_fill_steps,
        )

    def Jtor_pure(
        self,
        R: jax.Array,
        Z: jax.Array,
        psi: jax.Array,
        psi_bndry: float | None = None,
    ) -> jax.Array:
        """Return ``Jtor`` without mutating profile diagnostics.

        Newton--Krylov calls this method from a traced ``lax`` body.  Use
        :meth:`Jtor` outside the solver when ``opt``, ``xpt``, or limiter
        diagnostics need to be retained on the profile object.
        """
        return self._jtor_build_kernel_result(
            R, Z, psi, psi_bndry, self.limiter_handler, self.Ip
        )[0]

    def _jtor_build_kernel_result(
        self,
        R: jax.Array,
        Z: jax.Array,
        psi: jax.Array,
        psi_bndry: float | None,
        limiter_handler: Limiter_handler,
        ip: float,
        core_mask: jax.Array | None = None,
        psi_axis: jax.Array | None = None,
    ):
        """Prepare scalar arguments and call the side-effect-free JIT kernel."""

        if core_mask is not None:
            max_iterations = psi.shape[0] + psi.shape[1] - 2
            return _jtor_build_from_core_mask_kernel(
                R,
                Z,
                psi,
                core_mask.astype(bool),
                limiter_handler.mask_limiter_cells,
                limiter_handler.fine_cell_ids,
                limiter_handler.fine_weights_r,
                limiter_handler.fine_weights_z,
                limiter_handler.dRdZ,
                self.paxis,
                self.Ip,
                self.Raxis,
                self.alpha_m,
                self.alpha_n,
                psi[0, 0] if psi_axis is None else jnp.asarray(psi_axis, dtype=psi.dtype),
                psi[0, 0] if psi_bndry is None else jnp.asarray(psi_bndry, dtype=psi.dtype),
                max_iterations,
            )

        # ``None`` cannot enter a JIT as an array.  Its original meaning is
        # represented by the current corner flux plus a dynamic selector.
        use_supplied_psi_bndry = jnp.asarray(psi_bndry is not None)
        supplied_psi_bndry = (
            psi[0, 0]
            if psi_bndry is None
            else jnp.asarray(psi_bndry, dtype=psi.dtype)
        )
        max_iterations = psi.shape[0] + psi.shape[1] - 2
        return _jtor_build_kernel(
            R,
            Z,
            psi,
            limiter_handler.mask_inside_limiter,
            limiter_handler.mask_limiter_cells,
            limiter_handler.fine_cell_ids,
            limiter_handler.fine_weights_r,
            limiter_handler.fine_weights_z,
            limiter_handler.dRdZ,
            ip,
            supplied_psi_bndry,
            use_supplied_psi_bndry,
            self.paxis,
            self.Ip,
            self.Raxis,
            self.alpha_m,
            self.alpha_n,
            max_iterations,
            self.los_top_k,
        )

    def Jtor_part2(self, R, Z, psi, psi_axis, psi_bndry, mask):
        jtor, self.jtorshape = _ConstrainPaxisIp_kernel(
            R, Z, psi, psi_axis, psi_bndry, mask, 
            self.paxis, self.Ip, self.Raxis, self.alpha_m, self.alpha_n, 
        )
        return jtor

@jax.jit
def _ConstrainPaxisIp_kernel(
    R: jax.Array, 
    Z: jax.Array,
    psi: jax.Array,
    psi_axis: float,
    psi_bndry: float,
    mask: jax.Array,
    paxis: float,
    Ip: float,
    Raxis: float,
    alpha_m: float,
    alpha_n: float,
    ):

    if psi_bndry is None:
        psi_bndry = psi[0, 0]

    # grid sizes
    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]

    # calculate normalised psi
    psi_norm = jnp.clip((psi - psi_axis) / (psi_bndry - psi_axis), 0.0, 1.0)

    # shape function
    jtorshape = (
        1.0 - psi_norm ** alpha_m
    ) ** alpha_n

    # if there is a masking function, use it
    if mask is not None:
        jtorshape *= mask
        mask = mask

    # now apply constraints to define constants
    shapeintegral = (
        beta(1.0 / alpha_m, 1.0 + alpha_n) / alpha_m
    )
    shapeintegral *= psi_bndry - psi_axis

    # integrate current density components
    IR = (
        jnp.sum(jtorshape * R / Raxis) * dR * dZ
    )  # romb(romb(jtorshape * R / Raxis)) * dR * dZ
    I_R = (
        jnp.sum(jtorshape * Raxis / R) * dR * dZ
    )  # romb(romb(jtorshape * Raxis / R)) * dR * dZ

    # find L scaling parameter and scaled beta
    LBeta0 = -paxis * Raxis / shapeintegral
    L = Ip / I_R - LBeta0 * (IR / I_R - 1)
    Beta0 = LBeta0 / L

    # calculate final toroidal current density
    Jtor = (
        L
        * (Beta0 * R / Raxis + (1 - Beta0) * Raxis / R)
        * jtorshape
    )

    return Jtor, jtorshape

class nksolver:
    """Implementation of Newton Krylow algorithm for solving
    a generic root problem of the type
    F(x, other args) = 0
    in the variable x -- F(x) should have the same dimensions as x.
    Problem must be formulated so that x is a 1d np.array.

    In practice, given a guess x_0 and F(x_0) = R_0
    it aims to find the best step dx such that
    F(x_0 + dx) is minimum.
    """

    def __init__(
        self, problem_dimension, l2_reg=1e-6, collinearity_reg=1e-6, verbose=False
    ):
        """Instantiates the class.

        Parameters
        ----------
        problem_dimension : int
            Dimension of independent variable.
            np.shape(x) = problem_dimension
            x is a 1d vector.
        l2_reg : float
            Tychonoff regularization coeff
        collinearity_reg : float
            Tychonoff regularization coeff which further penalizes collinear terms

        """

        self.problem_dimension = problem_dimension
        dummy_hessenberg_residual = jnp.zeros(problem_dimension)
        self.dummy_hessenberg_residual = dummy_hessenberg_residual.at[0].set(1.0)
        self.verbose = verbose
        self.set_regularization(l2_reg, collinearity_reg)
        # self.force_sign_alignment = force_sign_alignment

    def _legacy_Arnoldi_unit(
        self,
        x0,
        dx,
        R0,
        # nR0,
        F_function,
        args,
        build_next=True,
    ):
        """Explores direction dx and proposes new direction for next exploration.

        Parameters
        ----------
        x0 : 1d np.array, np.shape(x0) = self.problem_dimension
            The expansion point x_0
        dx : 1d np.array, np.shape(dx) = self.problem_dimension
            The first direction to be explored. This will be sized appropriately.
        R0 : 1d np.array, np.shape(R0) = self.problem_dimension
            Residual of the root problem F_function at expansion point x_0
        F_function : 1d np.array, np.shape(x0) = self.problem_dimension
            Function representing the root problem at hand
        args : list
            Additional arguments for using function F
            F = F(x, *args)

        Returns
        -------
        new_candidate_step : 1d np.array, with same self.problem_dimension
            The direction to be explored next

        """

        # res_now = np.copy(R0)
        # calculate residual at explored point x0+dx
        try_reduce = 0
        res_calculated = False
        dx1 = jnp.copy(dx)
        while res_calculated is False:
            try:
                candidate_x = x0 + dx1
                R_dx = F_function(candidate_x, *args)
                res_calculated = True
            except:
                dx1 *= 0.75
                self.Q[:, self.n_it] *= 0.75
                try_reduce += 1
                if try_reduce >= 10:
                    raise ValueError(
                        f"Failed to calculate residual after 10 reductions of step size. Last tried step size was {np.linalg.norm(dx1)}. Check if the function is well defined in the explored region."
                    )
        useful_residual = R_dx - R0

        self.n_G = self.n_G.at[self.n_it].set(jnp.linalg.norm(useful_residual))
        self.G = self.G.at[:, self.n_it].set(useful_residual)
        self.Gn = self.Gn.at[:, self.n_it].set(useful_residual / self.n_G[self.n_it])
        self.collinearity = self.collinearity.at[:self.n_it, self.n_it].set(jnp.sum(
                self.Gn[:, self.n_it, jnp.newaxis] * self.Gn[:, : self.n_it], 
                axis=0
                ))
        # print('coll', self.n_it, self.collinearity[:self.n_it, self.n_it])

        if build_next:
            # append to Hessenberg matrix
            self.Hm = self.Hm.at[: self.n_it + 1, self.n_it].set(jnp.sum(
                self.Qn[:, : self.n_it + 1] * useful_residual[:, jnp.newaxis], axis=0
            ))

            # ortogonalise wrt previous directions
            next_candidate = useful_residual - jnp.sum(
                self.Qn[:, : self.n_it + 1]
                * self.Hm[: self.n_it + 1, self.n_it][jnp.newaxis, :],
                axis=1,
            )

            # append to Hessenberg matrix and normalize
            self.Hm = self.Hm.at[self.n_it + 1, self.n_it].set(
                jnp.linalg.norm(next_candidate)
                )
            # normalise the candidate direction for next iteration
            next_candidate /= self.Hm[self.n_it + 1, self.n_it]

            return next_candidate

    def set_regularization(self, l2_reg, collinearity_reg):
        """Sets the regularization coeffs

        Parameters
        ----------
        l2_reg : float
            Tychonoff regularization coeff
        collinearity_reg : float
            Tychonoff regularization coeff which further penalizes collinear terms
        """
        self.l2_reg = l2_reg
        self.collinearity_reg = collinearity_reg

    def Arnoldi_iteration(
        self,
        x0,
        dx,
        R0,
        F_function,
        args,
        step_size,
        scaling_with_n,
        target_relative_unexplained_residual,
        max_n_directions,
        clip,
        explore_all=False,
    ):
        """Run the functional Arnoldi implementation and expose legacy fields.

        ``_arnoldi_iteration_lax`` owns all iterative state.  Assigning
        these fields afterwards keeps the existing simulation call sites
        working while the next step moves the pure function into ``lax``.
        """
        result = _arnoldi_iteration_kernel(
            x0,
            dx,
            R0,
            F_function,
            args,
            l2_reg=self.l2_reg,
            collinearity_reg=self.collinearity_reg,
            step_size=step_size,
            scaling_with_n=scaling_with_n,
            target_relative_unexplained_residual=target_relative_unexplained_residual,
            max_n_directions=max_n_directions,
            clip=clip,
            explore_all=explore_all,
        )
        self.last_result = result
        self.x0 = result.x0
        self.R0 = result.R0
        self.nR0 = result.nR0
        self.max_dim = result.Q.shape[1]
        self.Q = result.Q
        self.Qn = result.Qn
        self.G = result.G
        self.Gn = result.Gn
        self.n_G = result.n_G
        self.collinearity = result.collinearity
        self.Hm = result.Hm
        self.collinear_aware_regulariz = result.collinear_aware_regulariz
        # The public compatibility facade exposes a Python integer, while the
        # JIT result intentionally keeps this value on device.
        self.n_it = int(result.n_it)
        self.n_it_tot = 0
        # Keep fixed-size diagnostics too. ``n_it`` identifies the valid
        # prefix; this avoids reintroducing dynamic-size JAX arrays at the
        # compatibility boundary.
        self.coeffs = result.coeffs
        self.relative_unexplained_residuals = result.relative_unexplained_residuals
        self.dx = result.dx
        self.success = result.success
        return result


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class NKResult:
    """Explicit state produced by one Newton--Krylov Arnoldi expansion.

    All arrays have a fixed ``max_dim``-based shape.  Coefficients after
    ``n_it`` are zero and residual-history entries after it are NaN.  That
    layout is ready to become a ``lax.while_loop`` carry next.
    """

    dx: jax.Array
    coeffs: jax.Array
    n_it: jax.Array
    x0: jax.Array
    R0: jax.Array
    nR0: jax.Array
    Q: jax.Array
    Qn: jax.Array
    G: jax.Array
    Gn: jax.Array
    n_G: jax.Array
    collinearity: jax.Array
    Hm: jax.Array
    collinear_aware_regulariz: jax.Array
    relative_unexplained_residuals: jax.Array
    success: jax.Array

    def tree_flatten(self):
        return (
            (
                self.dx,
                self.coeffs,
                self.n_it,
                self.x0,
                self.R0,
                self.nR0,
                self.Q,
                self.Qn,
                self.G,
                self.Gn,
                self.n_G,
                self.collinearity,
                self.Hm,
                self.collinear_aware_regulariz,
                self.relative_unexplained_residuals,
                self.success,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


def _arnoldi_unit_functional(
    x0,
    dx,
    R0,
    F_function,
    args,
    Q,
    Qn,
    G,
    Gn,
    n_G,
    collinearity,
    Hm,
    n_it,
):
    """Evaluate one direction and return every modified Arnoldi array."""
    try_reduce = 0
    trial_dx = jnp.copy(dx)
    while True:
        try:
            residual_at_trial = F_function(x0 + trial_dx, *args)
            break
        except Exception:
            trial_dx = trial_dx * 0.75
            Q = Q.at[:, n_it].set(Q[:, n_it] * 0.75)
            try_reduce += 1
            if try_reduce >= 10:
                raise ValueError(
                    "Failed to calculate the residual after 10 reductions "
                    "of the Arnoldi step."
                )

    useful_residual = residual_at_trial - R0
    useful_norm = jnp.linalg.norm(useful_residual)
    n_G = n_G.at[n_it].set(useful_norm)
    G = G.at[:, n_it].set(useful_residual)
    Gn = Gn.at[:, n_it].set(useful_residual / useful_norm)
    column_ids = jnp.arange(Q.shape[1])
    previous_columns = column_ids < n_it
    used_columns = column_ids <= n_it
    collinearity_column = jnp.sum(
        Gn[:, n_it, jnp.newaxis] * Gn, axis=0
    )
    collinearity = collinearity.at[:, n_it].set(
        jnp.where(previous_columns, collinearity_column, 0.0)
    )

    # Keep a fixed-size Hessenberg column.  Only coefficients belonging to
    # already explored directions are meaningful; the rest stay zero.
    projection = jnp.sum(Qn * useful_residual[:, jnp.newaxis], axis=0)
    projection = jnp.pad(projection, (0, 1))
    row_ids = jnp.arange(Hm.shape[0])
    Hm = Hm.at[:, n_it].set(
        jnp.where(row_ids <= n_it, projection, 0.0)
    )
    next_direction = useful_residual - jnp.sum(
        Qn * jnp.where(used_columns, Hm[:-1, n_it], 0.0)[jnp.newaxis, :],
        axis=1,
    )
    next_norm = jnp.linalg.norm(next_direction)
    Hm = Hm.at[n_it + 1, n_it].set(next_norm)
    return (
        next_direction / next_norm,
        Q,
        G,
        Gn,
        n_G,
        collinearity,
        Hm,
    )


def _arnoldi_iteration_functional(
    x0,
    dx,
    R0,
    F_function,
    args,
    *,
    l2_reg,
    collinearity_reg,
    step_size,
    scaling_with_n,
    target_relative_unexplained_residual,
    max_n_directions,
    clip,
    explore_all=False,
) -> NKResult:
    """Functional equivalent of the previous ``Arnoldi_iteration`` method.

    This first migration intentionally retains Python control flow.  The
    Arnoldi matrices and least-squares problem are already fixed-size and use
    masks, so replacing this loop with ``lax.while_loop`` is the next isolated
    change.
    """
    problem_dimension = x0.size
    max_dim = int(max_n_directions) + 1
    x0 = jnp.copy(x0)
    R0 = jnp.copy(R0)
    nR0 = jnp.linalg.norm(R0)
    dtype = R0.dtype

    Q = jnp.zeros((problem_dimension, max_dim), dtype=dtype)
    Qn = jnp.zeros_like(Q)
    G = jnp.zeros_like(Q)
    Gn = jnp.zeros_like(Q)
    n_G = jnp.zeros(max_dim, dtype=dtype)
    collinearity = jnp.zeros((max_dim, max_dim), dtype=dtype)
    Hm = jnp.zeros((max_dim + 1, max_dim), dtype=dtype)
    coeffs = jnp.zeros(max_dim, dtype=dtype)
    relative_unexplained_residuals = jnp.full(max_dim, jnp.nan, dtype=dtype)

    adjusted_step_size = step_size * nR0
    n_it = 0
    direction = jnp.asarray(dx, dtype=dtype) / jnp.linalg.norm(dx)
    Qn = Qn.at[:, n_it].set(direction)
    direction = direction * adjusted_step_size
    Q = Q.at[:, n_it].set(direction)
    collinear_aware_regulariz = jnp.zeros((1, 1), dtype=dtype)

    explore = True
    while explore:
        (
            direction,
            Q,
            G,
            Gn,
            n_G,
            collinearity,
            Hm,
        ) = _arnoldi_unit_functional(
            x0, direction, R0, F_function, args,
            Q, Qn, G, Gn, n_G, collinearity, Hm, n_it,
        )

        column_ids = jnp.arange(max_dim)
        used_columns = column_ids <= n_it
        used_pairs = used_columns[:, None] & used_columns[None, :]
        # Outside the explored subspace the value is exactly one before the
        # ``- 1`` below, hence those columns get no collinearity penalty.
        masked_abs_collinearity = jnp.where(
            used_pairs, jnp.abs(collinearity), 0.0
        )
        collinearity_penalty = jnp.max(
            1 / (1 - masked_abs_collinearity) ** 2, axis=0
        ) - 1
        active_regularization = (
            l2_reg + collinearity_penalty * collinearity_reg
        ) * nR0**2
        # Inactive columns have zero Gram/RHS entries.  Give them a unit
        # diagonal instead of a zero pivot, making the full solve equivalent
        # to the old active-submatrix solve with zero inactive coefficients.
        diagonal_regularization = jnp.where(
            used_columns, active_regularization, jnp.ones((), dtype=dtype)
        )
        collinear_aware_regulariz = jnp.diag(diagonal_regularization)
        coeffs = jnp.linalg.solve(
            G.T @ G + collinear_aware_regulariz,
            G.T @ (-R0),
        )
        coeffs = jnp.clip(coeffs, -clip, clip)
        explained_residual = jnp.sum(G * coeffs[None, :], axis=1)
        relative_error = jnp.linalg.norm(R0 + explained_residual) / nR0
        relative_unexplained_residuals = relative_unexplained_residuals.at[n_it].set(
            relative_error
        )

        explore = n_it < max_n_directions
        if not explore_all:
            explore = explore and bool(
                relative_error > target_relative_unexplained_residual
            )

        if explore:
            n_it += 1
            Qn = Qn.at[:, n_it].set(direction)
            direction = direction * (
                adjusted_step_size * (1 + n_it) ** scaling_with_n
            )
            Q = Q.at[:, n_it].set(direction)

    return NKResult(
        dx=jnp.sum(Q * coeffs[None, :], axis=1),
        coeffs=coeffs,
        n_it=n_it,
        x0=x0,
        R0=R0,
        nR0=nR0,
        Q=Q,
        Qn=Qn,
        G=G,
        Gn=Gn,
        n_G=n_G,
        collinearity=collinearity,
        Hm=Hm,
        collinear_aware_regulariz=collinear_aware_regulariz,
        relative_unexplained_residuals=relative_unexplained_residuals,
        success=jnp.array(True),
    )


def _residual_with_backoff(x0, direction, F_function, args):
    """Evaluate ``F`` with a device-side finite-value backoff policy."""
    residual = F_function(x0 + direction, *args)
    valid = jnp.all(jnp.isfinite(residual))

    def cond_fn(state):
        attempts, _, _, is_valid = state
        return (~is_valid) & (attempts < 9)

    def body_fn(state):
        attempts, trial_direction, _, _ = state
        trial_direction = trial_direction * 0.75
        trial_residual = F_function(x0 + trial_direction, *args)
        trial_valid = jnp.all(jnp.isfinite(trial_residual))
        return attempts + 1, trial_direction, trial_residual, trial_valid

    _, direction, residual, valid = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (jnp.array(0, dtype=jnp.int32), direction, residual, valid),
    )
    return direction, residual, valid


@partial(
    jax.jit,
    static_argnames=("F_function", "max_n_directions", "explore_all"),
)
def _arnoldi_iteration_kernel(
    x0,
    dx,
    R0,
    F_function,
    args,
    *,
    l2_reg,
    collinearity_reg,
    step_size,
    scaling_with_n,
    target_relative_unexplained_residual,
    max_n_directions,
    clip,
    explore_all=False,
) -> NKResult:
    """Fixed-shape, JIT-compatible Newton--Krylov Arnoldi expansion.

    ``F_function`` is static Python code, while ``args`` must be a pytree of
    JAX arrays.  No object mutation, Python list, or data-dependent Python
    branch is allowed in ``F_function``.
    """
    max_dim = max_n_directions + 1
    problem_dimension = x0.size
    dtype = R0.dtype
    nR0 = jnp.linalg.norm(R0)
    direction_norm = jnp.linalg.norm(dx)
    # To block the Nan error
    tiny = jnp.finfo(dtype).tiny
    initial_active = (nR0 > tiny) & (direction_norm > tiny)
    direction = jnp.where(
        direction_norm > tiny,
        dx / direction_norm,
        jnp.zeros_like(x0),
    )

    Q = jnp.zeros((problem_dimension, max_dim), dtype=dtype)
    Qn = jnp.zeros_like(Q).at[:, 0].set(direction)
    G = jnp.zeros_like(Q)
    Gn = jnp.zeros_like(Q)
    n_G = jnp.zeros(max_dim, dtype=dtype)
    collinearity = jnp.zeros((max_dim, max_dim), dtype=dtype)
    Hm = jnp.zeros((max_dim + 1, max_dim), dtype=dtype)
    coeffs = jnp.zeros(max_dim, dtype=dtype)
    regulariz = jnp.eye(max_dim, dtype=dtype)
    history = jnp.full(max_dim, jnp.nan, dtype=dtype)
    adjusted_step_size = step_size * nR0
    direction = direction * adjusted_step_size
    Q = Q.at[:, 0].set(direction)

    # n_it is the column evaluated by the next body call.  It is only
    # incremented when another direction is actually explored.
    initial_state = (
        jnp.array(0, dtype=jnp.int32),  # n_it
        initial_active,
        initial_active,  # success: all residual evaluations were finite
        direction,
        Q,
        Qn,
        G,
        Gn,
        n_G,
        collinearity,
        Hm,
        coeffs,
        regulariz,
        history,
        jnp.asarray(jnp.inf, dtype=dtype),
    )

    def cond_fn(state):
        _, active, _, _, _, _, _, _, _, _, _, _, _, _, _ = state
        return active

    def body_fn(state):
        (
            n_it,
            _,
            success,
            direction,
            Q,
            Qn,
            G,
            Gn,
            n_G,
            collinearity,
            Hm,
            _,
            _,
            history,
            _,
        ) = state
        trial_direction, residual_at_trial, residual_valid = _residual_with_backoff(
            x0, direction, F_function, args
        )
        Q = Q.at[:, n_it].set(trial_direction)
        useful_residual = jnp.where(
            jnp.isfinite(residual_at_trial - R0), residual_at_trial - R0, 0.0
        )
        useful_norm = jnp.linalg.norm(useful_residual)
        useful_valid = residual_valid & (useful_norm > tiny) & jnp.isfinite(useful_norm)
        safe_useful_norm = jnp.where(useful_valid, useful_norm, 1.0)
        n_G = n_G.at[n_it].set(useful_norm)
        G = G.at[:, n_it].set(useful_residual)
        Gn = Gn.at[:, n_it].set(useful_residual / safe_useful_norm)

        column_ids = jnp.arange(max_dim, dtype=n_it.dtype)
        used_columns = column_ids <= n_it
        previous_columns = column_ids < n_it
        collinearity_column = jnp.sum(Gn[:, n_it, None] * Gn, axis=0)
        collinearity = collinearity.at[:, n_it].set(
            jnp.where(previous_columns, collinearity_column, 0.0)
        )
        projection = jnp.pad(
            jnp.sum(Qn * useful_residual[:, None], axis=0), (0, 1)
        )
        row_ids = jnp.arange(max_dim + 1, dtype=n_it.dtype)
        Hm = Hm.at[:, n_it].set(jnp.where(row_ids <= n_it, projection, 0.0))
        next_direction = useful_residual - jnp.sum(
            Qn * jnp.where(used_columns, Hm[:-1, n_it], 0.0)[None, :], axis=1
        )
        next_norm = jnp.linalg.norm(next_direction)
        safe_next_norm = jnp.where(next_norm > tiny, next_norm, 1.0)
        next_direction = next_direction / safe_next_norm
        Hm = Hm.at[n_it + 1, n_it].set(next_norm)

        used_pairs = used_columns[:, None] & used_columns[None, :]
        masked_abs_collinearity = jnp.where(
            used_pairs, jnp.abs(collinearity), 0.0
        )
        collinearity_penalty = jnp.max(
            1 / (1 - masked_abs_collinearity) ** 2, axis=0
        ) - 1
        active_regularization = (
            l2_reg + collinearity_penalty * collinearity_reg
        ) * nR0**2
        regulariz = jnp.diag(jnp.where(
            used_columns, active_regularization, jnp.ones((), dtype=dtype)
        ))
        coeffs = jnp.linalg.solve(G.T @ G + regulariz, G.T @ (-R0))
        coeffs = jnp.clip(coeffs, -clip, clip)
        relative_error = jnp.linalg.norm(R0 + jnp.sum(G * coeffs[None, :], axis=1)) / nR0
        history = history.at[n_it].set(relative_error)

        target_reached = relative_error <= target_relative_unexplained_residual
        requested_next_direction = n_it < max_n_directions
        next_is_valid = (next_norm > tiny) & jnp.isfinite(next_norm)
        keep_exploring = useful_valid & requested_next_direction & \
            next_is_valid & (explore_all | ~target_reached)
        next_slot = jnp.minimum(n_it + 1, max_n_directions)

        def append_direction(values):
            q, qn = values
            next_step = adjusted_step_size * (1 + next_slot) ** scaling_with_n
            qn = qn.at[:, next_slot].set(next_direction)
            q = q.at[:, next_slot].set(next_direction * next_step)
            return q, qn

        Q, Qn = jax.lax.cond(
            keep_exploring, append_direction, lambda values: values, (Q, Qn)
        )
        next_direction = jnp.where(
            keep_exploring,
            next_direction * (adjusted_step_size * (1 + next_slot) ** scaling_with_n),
            direction,
        )
        return (
            jnp.where(keep_exploring, n_it + 1, n_it),
            keep_exploring,
            success & residual_valid,
            next_direction,
            Q,
            Qn,
            G,
            Gn,
            n_G,
            collinearity,
            Hm,
            coeffs,
            regulariz,
            history,
            relative_error,
        )

    (
        n_it,
        _,
        success,
        _,
        Q,
        Qn,
        G,
        Gn,
        n_G,
        collinearity,
        Hm,
        coeffs,
        regulariz,
        history,
        _,
    ) = jax.lax.while_loop(cond_fn, body_fn, initial_state)
    return NKResult(
        dx=jnp.sum(Q * coeffs[None, :], axis=1),
        coeffs=coeffs,
        n_it=n_it,
        x0=x0,
        R0=R0,
        nR0=nR0,
        Q=Q,
        Qn=Qn,
        G=G,
        Gn=Gn,
        n_G=n_G,
        collinearity=collinearity,
        Hm=Hm,
        collinear_aware_regulariz=regulariz,
        relative_unexplained_residuals=history,
        success=success,
    )
