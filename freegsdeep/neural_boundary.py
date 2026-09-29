"""FP32 adapter from a trained :class:`XPlimnet` to the JAX boundary solver.

The network estimates magnetic-axis flux, separatrix flux, and whether the
plasma is limiter limited.  Its separatrix estimate is supplied to the
existing Jtor implementation; the latter still performs the geometrical
X-point/limiter checks, which prevents a bad neural prediction from directly
creating an unphysical current mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from freegsdeep.model import XPlimnet
from freegsdeep.utils.jtor import _ConstrainPaxisIp_kernel


@dataclass(frozen=True)
class BoundaryPrediction:
    psi_axis: jax.Array
    psi_bndry: jax.Array
    limiter_probability: jax.Array
    core_mask: jax.Array


def _cast_inexact_leaves(tree: Any, dtype: Any) -> Any:
    """Cast stored FP64 checkpoint leaves without touching integer/static data."""
    return jax.tree_util.tree_map(
        lambda leaf: leaf.astype(dtype)
        if eqx.is_inexact_array(leaf)
        else leaf,
        tree,
    )


class NeuralPlasmaBoundary:
    """Use a trained ``XPlimnet`` to define the plasma region directly."""

    def __init__(self, model: XPlimnet, limiter_mask: jax.Array) -> None:
        self.model = model
        self.limiter_mask = jnp.asarray(limiter_mask, dtype=jnp.float32)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        nx: int,
        ny: int,
        limiter_mask: jax.Array,
        *,
        key: jax.Array | None = None,
        dtype: Any = jnp.float32,
    ) -> "NeuralPlasmaBoundary":
        """Load either an FP32 or legacy FP64 Equinox checkpoint as FP32."""
        model = XPlimnet(nx=nx, ny=ny, key=jax.random.PRNGKey(0) if key is None else key)
        with Path(checkpoint).open("rb") as stream:
            model = eqx.tree_deserialise_leaves(stream, model)
        return cls(_cast_inexact_leaves(model, dtype), limiter_mask)

    def predict(self, psi: jax.Array) -> BoundaryPrediction:
        """Predict physical fluxes from one ``(nx, ny)`` total-flux field."""
        return predict_boundary(self.model, self.limiter_mask, psi)

    def jtor(self, profile: Any, R: jax.Array, Z: jax.Array, psi: jax.Array) -> jax.Array:
        """Build Jtor from the neural flux threshold without geometry repair."""
        jtor, _, prediction = build_neural_jtor(
            self.model,
            self.limiter_mask,
            R,
            Z,
            psi,
            paxis=profile.paxis,
            Ip=profile.Ip,
            Raxis=profile.Raxis,
            alpha_m=profile.alpha_m,
            alpha_n=profile.alpha_n,
        )
        self.last_prediction = prediction
        return jtor


def predict_boundary(
    model: XPlimnet,
    limiter_mask: jax.Array,
    psi: jax.Array,
) -> BoundaryPrediction:
    """Predict boundary values without mutating a wrapper object."""
    limiter_mask = jnp.asarray(limiter_mask, dtype=jnp.float32)
    psi = jnp.asarray(psi, dtype=jnp.float32)
    if psi.shape != limiter_mask.shape:
        raise ValueError(
            f"psi has shape {psi.shape}; expected {limiter_mask.shape}."
        )
    masked = psi * limiter_mask
    offset = jnp.min(masked)
    scale = jnp.maximum(jnp.max(masked) - offset, jnp.finfo(psi.dtype).eps)
    normalized = (masked - offset) / scale * limiter_mask
    reconstruction, classification = model(normalized[jnp.newaxis, ...])
    physical = reconstruction * scale + offset
    psi_axis = physical[0]
    psi_bndry = physical[1]
    psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis + jnp.finfo(psi.dtype).eps)
    # The predicted separatrix itself defines the plasma region.  A previous
    # 0.95 cutoff inserted an artificial buffer inside that separatrix.
    core_mask = (psi_norm < 1.0) & (limiter_mask > 0.5)
    return BoundaryPrediction(
        psi_axis=psi_axis,
        psi_bndry=psi_bndry,
        limiter_probability=classification.reshape(()),
        core_mask=core_mask,
    )


def build_neural_jtor(
    model: XPlimnet,
    limiter_mask: jax.Array,
    R: jax.Array,
    Z: jax.Array,
    psi: jax.Array,
    *,
    paxis: jax.Array | float,
    Ip: jax.Array | float,
    Raxis: jax.Array | float,
    alpha_m: jax.Array | float,
    alpha_n: jax.Array | float,
) -> tuple[jax.Array, jax.Array, BoundaryPrediction]:
    """Predict a boundary and form Jtor from its threshold-defined region.

    This deliberately bypasses critical-point detection, flood-fill, and the
    limiter-contact correction.  ``core_mask`` is exactly the neural flux
    threshold returned by :func:`predict_boundary`.
    """
    prediction = predict_boundary(model, limiter_mask, psi)
    jtor, jtorshape = _ConstrainPaxisIp_kernel(
        R,
        Z,
        psi,
        prediction.psi_axis.astype(psi.dtype),
        prediction.psi_bndry.astype(psi.dtype),
        prediction.core_mask,
        jnp.asarray(paxis, dtype=psi.dtype),
        jnp.asarray(Ip, dtype=psi.dtype),
        jnp.asarray(Raxis, dtype=psi.dtype),
        jnp.asarray(alpha_m, dtype=psi.dtype),
        jnp.asarray(alpha_n, dtype=psi.dtype),
    )
    return jtor, jtorshape, prediction


@eqx.filter_jit
def build_neural_jtor_jit(
    model: XPlimnet,
    limiter_mask: jax.Array,
    R: jax.Array,
    Z: jax.Array,
    psi: jax.Array,
    paxis: jax.Array | float,
    Ip: jax.Array | float,
    Raxis: jax.Array | float,
    alpha_m: jax.Array | float,
    alpha_n: jax.Array | float,
) -> jax.Array:
    """Compiled direct neural-current path for eager simulation call sites.

    ``F_function`` is Python-level, so calling :func:`build_neural_jtor`
    there would otherwise dispatch each network layer separately.  The
    Newton--Krylov residual is already compiled as a whole and should keep
    calling the undecorated helper.
    """
    return build_neural_jtor(
        model,
        limiter_mask,
        R,
        Z,
        psi,
        paxis=paxis,
        Ip=Ip,
        Raxis=Raxis,
        alpha_m=alpha_m,
        alpha_n=alpha_n,
    )[0]
