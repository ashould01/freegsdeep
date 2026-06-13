from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import maximum_filter
from skimage import measure
from matplotlib.path import Path
from jax.scipy.special import beta
from freegsnke.freegsnke.limiter_func import Limiter_handler
from freegs4e.gradshafranov import Greens

_NUM_DUPLICATION_STEPS = 6

@partial(jax.jit, inline=True)
def _carlson_duplication_step(
    state: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], _: None
) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], None]:
    """
    Perform one iteration of Carlson's duplication algorithm for R_F and R_D.

    Args:
        state: A tuple of five arrays:
            - X, Y, Z: Current values of the integral arguments.
            - rd_accumulator: Accumulator for R_D.
            - scale_factor: Current scale factor in the iteration.
        _: Unused placeholder for compatibility with `jax.lax.scan`.

    Returns:
        new_state: Updated state after one duplication step.
        None: Placeholder for scan output.
    """
    X, Y, Z, rd_acc, scale = state

    sqrt_X, sqrt_Y, sqrt_Z = jnp.sqrt(X), jnp.sqrt(Y), jnp.sqrt(Z)
    lambda_val = sqrt_X * sqrt_Y + sqrt_Y * sqrt_Z + sqrt_Z * sqrt_X

    rd_acc += scale / (sqrt_Z * (Z + lambda_val))
    scale *= 0.25

    X = 0.25 * (X + lambda_val)
    Y = 0.25 * (Y + lambda_val)
    Z = 0.25 * (Z + lambda_val)

    return (X, Y, Z, rd_acc, scale), None


@jax.jit
def _compute_rf_rd(
    x0: jax.Array, y0: jax.Array, z0: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """
    Compute Carlson symmetric integrals R_F(x0, y0, z0) and R_D(x0, y0, z0)
    using iterative duplication and series correction.

    Args:
        x0, y0, z0: Non-negative input arrays representing the arguments of the integrals.

    Returns:
        rf: Value of R_F(x0, y0, z0).
        rd: Value of R_D(x0, y0, z0).
    """
    init_state = (x0, y0, z0, jnp.zeros_like(x0), jnp.ones_like(x0))

    (X, Y, Z, rd_acc, scale), _ = jax.lax.scan(
        _carlson_duplication_step, init_state, None, length=_NUM_DUPLICATION_STEPS
    )

    mu = (X + Y + Z) / 3.0
    dx = (mu - X) / mu
    dy = (mu - Y) / mu
    dz = (mu - Z) / mu
    # After computing dx, dy, dz and:
    e2 = dx * dy - dz * dz
    e3 = dx * dy * dz
    e2_sq, e2_cu, e3_sq = e2 * e2, e2 * e2 * e2, e3 * e3

    rf = (
        1.0
        - e2 / 10.0
        + e3 / 14.0
        + e2_sq / 24.0
        - 3.0 * e2 * e3 / 44.0
        - 5.0 * e2_cu / 208.0
        + 3.0 * e3_sq / 104.0
    ) / jnp.sqrt(mu)

    ave = (X + Y + 3.0 * Z) / 5.0
    dx2 = (ave - X) / ave
    dy2 = (ave - Y) / ave
    dz2 = (ave - Z) / ave

    C1 = 3.0 / 14.0
    C2 = 1.0 / 6.0
    C3 = 9.0 / 22.0
    C4 = 3.0 / 26.0

    ea2 = dx2 * dy2
    eb2 = dz2 * dz2
    ec2 = ea2 - eb2
    ed2 = ea2 - 6.0 * eb2
    ef2 = ed2 + 2.0 * ec2

    s1 = ed2 * (-C1 + 0.25 * C3 * ed2 - 1.5 * C4 * dz2 * ef2)
    s2 = dz2 * (C2 * ef2 + dz2 * (-C3 * ec2 + dz2 * C4 * ea2))

    norm = ave * jnp.sqrt(ave)
    rd = 3.0 * rd_acc + (scale / norm) * (1.0 + s1 + s2)

    return rf, rd


@jax.jit
def ellipk(m: jax.Array) -> jax.Array:
    """
    Compute the complete elliptic integral of the first kind, K(m).

    Uses the Carlson R_F form: K(m) = R_F(0, 1 - m, 1), valid for 0 <= m <= 1.

    Args:
        m: Input array of modulus values.

    Returns:
        K(m) array with:
            - K(1) = +inf
            - NaN for m > 1
    """
    rf, _ = _compute_rf_rd(jnp.zeros_like(m), 1.0 - m, jnp.ones_like(m))
    inf_mask = m == 1.0
    nan_mask = m > 1.0
    result = jnp.where(inf_mask, jnp.inf, rf)
    return jnp.where(nan_mask, jnp.nan, result)


@jax.jit
def ellipkm1(x: jax.Array) -> jax.Array:
    """
    Compute K(1 - x) with improved accuracy for small x.

    Uses a logarithmic series approximation for x < 1e-3:
        K(1 - x) ≈ (1/2) * ln(16/x) + (x/16) * (ln(16/x) - 1)
    Falls back to ellipk(1 - x) otherwise.

    Args:
        x: Input array representing 1 - m.

    Returns:
        K(1 - x) array with high accuracy near x = 0.
    """
    use_series = x < 1e-3
    log_term = jnp.log(16.0 / x)
    series_approx = 0.5 * log_term + (x / 16.0) * (log_term - 1.0)

    return jnp.where(use_series, series_approx, ellipk(1.0 - x))


@jax.jit
def ellipe(m: jax.Array) -> jax.Array:
    """
    Compute the complete elliptic integral of the second kind, E(m).

    Uses the Carlson forms: E(m) = R_F(0, 1 - m, 1) - (m / 3) * R_D(0, 1 - m, 1).

    Args:
        m: Input array of modulus values.

    Returns:
        E(m) array with:
            - E(1) = 1
            - NaN for m > 1
    """
    rf, rd = _compute_rf_rd(jnp.zeros_like(m), 1.0 - m, jnp.ones_like(m))
    val = rf - (m / 3.0) * rd
    # first force the known endpoint
    val = jnp.where(m == 1.0, 1.0, val)
    return jnp.where(m > 1.0, jnp.nan, val)

def Greens_jax(Rc, Zc, R, Z, mu0=4.0 * jnp.pi * 1e-7):
    k2 = 4.0 * R * Rc / ((R + Rc) ** 2 + (Z - Zc) ** 2)

    k2 = jnp.clip(k2, 1e-10, 1.0 - 1e-10)
    k = jnp.sqrt(k2)

    # note definition of ellipk, ellipe in scipy is K(k^2), E(k^2)
    return (
        (mu0 / (2.0 * jnp.pi))
        * jnp.sqrt(R * Rc)
        * ((2.0 - k2) * ellipk(k2) - 2.0 * ellipe(k2))
        / k
    )

from jax import lax

def break_if_nan(x):
    nan_mask = jnp.isnan(x)
    nan_any = jnp.any(nan_mask)
    nan_cnt = jnp.sum(nan_mask)
    B = x.shape[0]
    per_b = jnp.any(nan_mask.reshape(B, -1), axis=1)
    bad_b = jnp.where(per_b, size=min(16, B), fill_value=-1)[0]

    # cheap summary (won't print arrays)
    jax.debug.print(
        "[NaN-check] any={a}, count={c}, shape={s}, bad_b={b}",
        a=nan_any, c=nan_cnt, s=x.shape, b=bad_b
        )

    # Break only when NaN occurs
    def _do_break(_):
        jax.debug.breakpoint()
        return 0

    def _no_break(_):
        return 0

    _ = lax.cond(nan_any, _do_break, _no_break, operand=0)
    return x

class vectorized_nksolver:
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
        self.dummy_hessenberg_residual = jnp.zeros(problem_dimension)
        self.dummy_hessenberg_residual = self.dummy_hessenberg_residual.at[0].set(1.0)
        self.verbose = verbose
        self.set_regularization(l2_reg, collinearity_reg)
        # self.force_sign_alignment = force_sign_alignment

    def Arnoldi_unit(
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
        res_calculated = False
        dx1 = np.copy(np.asarray(dx))
        while res_calculated is False:
            try:
                candidate_x = x0 + dx1
                R_dx = F_function(candidate_x[:, None], *args).squeeze(1)
                res_calculated = True
            except:
                dx1 *= 0.75
                self.Q = self.Q.at[:, self.n_it].set(self.Q[:, self.n_it] * 0.75)
        useful_residual = jnp.asarray(R_dx - R0)
        # dot_product = np.dot(useful_residual, R0)

        # if self.force_sign_alignment and (dot_product > 0):
        #     # need sign reversal!
        #     print(f"term {self.n_it} being reversed")
        #     res_calculated = False
        #     dx1 = -np.copy(dx)
        #     self.Qn[:, self.n_it] *= -1
        #     self.Q[:, self.n_it] *= -1
        #     while res_calculated is False:
        #         try:
        #             candidate_x = x0 + dx1
        #             R_dx = F_function(candidate_x, *args)
        #             res_calculated = True
        #         except:
        #             dx1 *= 0.75
        #             self.Q[:, self.n_it] *= 0.75
        #     useful_residual = R_dx - R0

        self.n_G = self.n_G.at[self.n_it].set(jnp.linalg.norm(useful_residual))
        self.G = self.G.at[:, self.n_it].set(useful_residual)
        self.Gn = self.Gn.at[:, self.n_it].set(useful_residual / self.n_G[self.n_it])
        self.collinearity = self.collinearity.at[: self.n_it, self.n_it].set(
            jnp.sum(self.Gn[:, self.n_it, None] * self.Gn[:, : self.n_it], axis=0)
        )
        # print('coll', self.n_it, self.collinearity[:self.n_it, self.n_it])

        if build_next:
            # append to Hessenberg matrix
            self.Hm = self.Hm.at[: self.n_it + 1, self.n_it].set(
                jnp.sum(self.Qn[:, : self.n_it + 1] * useful_residual[:, None], axis=0)
            )

            # ortogonalise wrt previous directions
            next_candidate = useful_residual - jnp.sum(
                self.Qn[:, : self.n_it + 1]
                * self.Hm[: self.n_it + 1, self.n_it][None, :],
                axis=1,
            )

            # append to Hessenberg matrix and normalize
            self.Hm = self.Hm.at[self.n_it + 1, self.n_it].set(jnp.linalg.norm(next_candidate))
            # normalise the candidate direction for next iteration
            next_candidate /= self.Hm[self.n_it + 1, self.n_it]

            # # build the relevant Givens rotation
            # givrot = np.eye(self.n_it + 2)
            # rho = np.dot(self.Omega[self.n_it], self.Hm[: self.n_it + 1, self.n_it])
            # rr = (rho**2 + self.Hm[self.n_it + 1, self.n_it] ** 2) ** 0.5
            # givrot[-2, -2] = givrot[-1, -1] = rho / rr
            # givrot[-2, -1] = self.Hm[self.n_it + 1, self.n_it] / rr
            # givrot[-1, -2] = -1.0 * givrot[-2, -1]
            # # update Omega matrix
            # Omega = np.eye(self.n_it + 2)
            # Omega[:-1, :-1] = 1.0 * self.Omega
            # self.Omega = np.matmul(givrot, Omega)
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
    
    def starting_G(self, F_function, x0, R0, args):
        self.G = jnp.zeros((self.problem_dimension, self.max_dim))
        # orthonormal basis in residual space
        self.Gn = jnp.zeros((self.problem_dimension, self.max_dim))
        # norms of residual vectors
        self.n_G = jnp.zeros(self.max_dim)
        self.collinearity = jnp.zeros((self.max_dim, self.max_dim))
        res_calculated = False
        while res_calculated is False:
            candidate_x = x0[:, None] + self.Q[:, :self.n_it] 
            R_dx = F_function(candidate_x, *args)
            try:
                res_calculated = True
            except:
                dx1 *= 0.75
                self.Q = self.Q.at[:, self.n_it].set(self.Q[:, self.n_it] * 0.75)
        useful_residual = R_dx - R0[:, None]
        self.G = self.G.at[:, :self.n_it].set(useful_residual)
        self.n_G = self.n_G.at[:self.n_it].set(jnp.linalg.norm(useful_residual, axis=0))
        self.Gn = self.Gn.at[:, :self.n_it].set(useful_residual / self.n_G[:self.n_it][None, :])
        self.collinearity = self.collinearity.at[: self.n_it, :self.n_it].set(
            jnp.einsum('ij,ik->jk', self.Gn[:, :self.n_it], self.Gn[:, :self.n_it])
            )
        self.collinearity = jnp.triu(self.collinearity, 1)
        self.Hm = jnp.zeros((self.max_dim + 1, self.max_dim))
        self.Hm = self.Hm.at[: self.n_it, :self.n_it].set(
            jnp.einsum('ij,ik->jk', self.Qn[:, :self.n_it], self.G[:, :self.n_it])
        )
        self.Hm = jnp.triu(self.Hm)
        rows = jnp.arange(1, self.n_it + 1)
        cols = jnp.arange(self.n_it)
        self.Hm = self.Hm.at[rows, cols].set(
            jnp.linalg.norm(self.G[:, :self.n_it] - jnp.matmul(
                self.Qn[:, :self.n_it], self.Hm[: self.n_it, :self.n_it]
                ), axis=0)
        )

    def Arnoldi_iteration(
        self,
        x0,
        Q_pred, 
        R0,
        F_function,
        args,
        step_size,
        scaling_with_n,
        target_relative_unexplained_residual,
        max_n_directions,
        clip,
        true_solver=None,
        vectorized_solver=None,
        # l2_reg=1e-5,
        # collinearity_reg=1e-6,
    ):
        """Performs the iteration of the NK solution method:
        1) explores direction dx
        2) checks what fraction of the residual can be (linearly) canceled
        3) restarts if not satisfied
        The best candidate step combining all explored directions is stored at self.dx

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
        step_size : float
            l2 norm of proposed step in units of the residual norm
        scaling_with_n : float
            allows to further scale dx candidate steps as a function of the iteration number n_it, by a factor
            (1 + self.n_it)**scaling_with_n
        target_relative_explained_residual : float between 0 and 1
            terminates iteration when such a fraction of the initial residual R0
            can be (linearly) cancelled
        max_n_directions : int
            terminates iteration even though condition on
            explained residual is not met
        clip : float
            maximum step size for each explored direction, in units
            of exploratory step dx_i
        """
        self.n_it = Q_pred.shape[1]
        self.x0 = jnp.copy(x0)
        self.R0 = jnp.copy(R0)

        self.relative_unexplained_residuals = []
        nR0 = jnp.linalg.norm(R0)
        self.nR0 = 1.0 * nR0
        self.max_dim = int(max_n_directions + 1)

        # orthogonal basis in x space
        self.Q = jnp.zeros((self.problem_dimension, self.max_dim))
        # orthonormal basis in x space
        self.Qn = jnp.zeros((self.problem_dimension, self.max_dim))
        self.Qn = self.Qn.at[:, :self.n_it].set(Q_pred)
        adjusted_step_size = step_size * nR0
        this_step_size = adjusted_step_size * (
            (1 + jnp.arange(0, self.n_it)) ** scaling_with_n
        )
        self.Q = self.Q.at[:, :self.n_it].set(
            self.Qn[:, :self.n_it] * this_step_size[None, :]
            )

        # basis in residual space
        self.solver_true = true_solver
        self.vectorized_solver = vectorized_solver
        self.starting_G(F_function, x0, R0, args)

        self.n_it -= 1
        dx = jnp.copy(self.Q[:, self.n_it])

        explore = 1 
        while explore:
            # build Arnoldi update
            dx = self.Arnoldi_unit(x0, dx, R0, F_function, args)

            # prepare to calculate explained residual
            collinearity_penalty = jnp.diag(
                jnp.max(
                    1
                    / (1 - jnp.abs(self.collinearity[: self.n_it + 1, : self.n_it + 1]))
                    ** 2,
                    axis=0,
                )
                - 1
            )
            collinear_aware_regulariz = (
                jnp.eye(self.n_it + 1) * self.l2_reg
                + collinearity_penalty * self.collinearity_reg
            )
            self.collinear_aware_regulariz = collinear_aware_regulariz * nR0**2

            # solve the regularised least sq problem
            coeffs = jnp.dot(
                jnp.linalg.inv(
                    self.G[:, : self.n_it + 1].T @ self.G[:, : self.n_it + 1] \
                    + self.collinear_aware_regulariz
                ),
                jnp.dot(self.G[:, : self.n_it + 1].T, -R0),
            )
            coeffs = jnp.clip(coeffs, -clip, clip)
            # calculare the corresponding fraction of residual that is currently explained
            expl_res = jnp.sum(
                self.G[:, : self.n_it + 1] * coeffs[jnp.newaxis, :], axis=1
            )
            self.relative_unexplained_residuals.append(
                jnp.linalg.norm(R0 + expl_res) / nR0
            )

            explore = self.n_it < max_n_directions
            explore *= (
                self.relative_unexplained_residuals[-1]
                > target_relative_unexplained_residual
            )
            # explore = self.n_it < 6

            # prepare for next step
            if explore:
                self.n_it += 1
                # # new addition
                # if clip_quantiles is not None:
                #     q1, q2 = np.quantile(dx, clip_quantiles)
                #     dx = np.clip(dx, q1, q2)
                self.Qn = self.Qn.at[:, self.n_it].set(jnp.copy(dx))
                this_step_size = adjusted_step_size * (
                    (1 + self.n_it) ** scaling_with_n
                )
                dx *= this_step_size
                self.Q = self.Q.at[:, self.n_it].set(jnp.copy(dx))

        # self.coeffs = -nR0 * np.dot(
        #     np.linalg.inv(self.Omega[:-1] @ self.Hm[: self.n_it + 2, : self.n_it + 1]),
        #     self.Omega[:-1, 0],
        # )

        # collinearity = np.sum(self.G[:,np.newaxis,:self.n_it + 1]*self.G[:,:self.n_it + 1,np.newaxis],axis=0)
        # d_collinearity = np.diag(collinearity)**.5
        # collinearity /= (d_collinearity[:, np.newaxis] * d_collinearity[np.newaxis, :])
        # self.collinearity = np.abs(np.triu(collinearity, 1))
        # d_collinearity = np.diag(np.max(1/(1-np.abs(self.collinearity))**2, axis=0))

        # collinear_aware_regulariz = np.eye(self.n_it + 1)*1e-4
        # collinear_aware_regulariz += d_collinearity*1e-4
        # self.collinear_aware_regulariz = collinear_aware_regulariz * nR0**2

        # Hm_ = np.copy(self.Hm[: self.n_it + 2, : self.n_it + 1])
        # self.coeffs = -self.sign * nR0 * np.dot(np.linalg.inv(Hm_.T@Hm_ + self.collinear_aware_regulariz), Hm_[0])
        # self.vanilla_coeffs = np.dot(np.linalg.inv(self.G[:, : self.n_it + 1].T@self.G[:, : self.n_it + 1] + self.collinear_aware_regulariz), np.dot(self.G[:, : self.n_it + 1].T, -R0))
        collinearity_penalty = jnp.diag(
            jnp.max(
                1
                / (1 - jnp.abs(self.collinearity[: self.n_it + 1, : self.n_it + 1]))
                ** 2,
                axis=0,
            )
            - 1
        )
        collinear_aware_regulariz = (
            jnp.eye(self.n_it + 1) * self.l2_reg
            + collinearity_penalty * self.collinearity_reg
        )
        self.collinear_aware_regulariz = collinear_aware_regulariz * nR0**2

        # solve the regularised least sq problem
        coeffs = jnp.dot(
            jnp.linalg.inv(
                self.G[:, : self.n_it + 1].T @ self.G[:, : self.n_it + 1] \
                + self.collinear_aware_regulariz
            ),
            jnp.dot(self.G[:, : self.n_it + 1].T, -R0),
        )
        coeffs = jnp.clip(coeffs, -clip, clip)

        self.coeffs = jnp.copy(coeffs)
        self.dx = jnp.sum(self.Q[:, : self.n_it + 1] * coeffs[jnp.newaxis, :], axis=1)

    # def review_Arnoldi_iteration(
    #     self,
    #     F_function,
    #     args,
    #     target_relative_unexplained_residual,
    #     clip,
    #     l2_reg=1e-4,
    #     collinearity_reg=1e-4,
    #     threshold=0.1,
    # ):

    #     # resize the directions in x space
    #     self.Q = self.Q[:, : self.n_it + 1] * self.coeffs[np.newaxis, :]

    #     # # select those that's worth analyzing
    #     # mask = (self.n_G[:self.n_it + 1] * self.coeffs) > threshold*self.nR0
    #     # max_n_directions = np.sum(mask.astype(float))
    #     # # apply the selection
    #     # self.Q = self.Q[:, mask]
    #     max_n_directions = 1.0 * self.n_it

    #     self.relative_unexplained_residuals_review = []
    #     self.n_it = 0
    #     explore = 1

    #     while explore:
    #         self.Arnoldi_unit(
    #             self.x0, self.Q[:, self.n_it], self.R0, self.nR0, F_function, args
    #         )

    #         # prepare to calculate explained residual
    #         collinearity_penalty = np.diag(
    #             np.max(
    #                 1
    #                 / (1 - np.abs(self.collinearity[: self.n_it + 1, : self.n_it + 1]))
    #                 ** 2,
    #                 axis=0,
    #             )
    #             - 1
    #         )
    #         collinear_aware_regulariz = (
    #             np.eye(self.n_it + 1) * l2_reg + collinearity_penalty * collinearity_reg
    #         )
    #         self.collinear_aware_regulariz = collinear_aware_regulariz * self.nR0**2

    #         # solve the regularised least sq problem
    #         coeffs = np.dot(
    #             np.linalg.inv(
    #                 self.G[:, : self.n_it + 1].T @ self.G[:, : self.n_it + 1]
    #                 + self.collinear_aware_regulariz
    #             ),
    #             np.dot(self.G[:, : self.n_it + 1].T, -self.R0),
    #         )
    #         coeffs = np.clip(coeffs, -clip, clip)
    #         # calculare the corresponding fraction of residual that is currently explained
    #         expl_res = np.sum(
    #             self.G[:, : self.n_it + 1] * coeffs[np.newaxis, :], axis=1
    #         )
    #         self.relative_unexplained_residuals_review.append(
    #             np.linalg.norm(self.R0 + expl_res) / self.nR0
    #         )

    #         explore = self.n_it < max_n_directions
    #         explore *= (
    #             self.relative_unexplained_residuals_review[-1]
    #             > target_relative_unexplained_residual
    #         )
    #         self.n_it += 1

    #     self.coeffs_review = np.copy(coeffs)
    #     self.dx_review = np.sum(
    #         self.Q[:, : self.n_it + 1] * coeffs[np.newaxis, :], axis=1
    #     )

from freegs4e.gradshafranov import GSsparse4thOrder

class vectorized_solver:
    
    def __init__(self, eq, limiter):
        mu0 = 4e-7 * jnp.pi
        self.limiter_handler = vectorized_Limiter_handler(eq, limiter)
        self.R = eq.R
        self.Z = eq.Z
        self.rhs_before_jtor = -mu0 * self.R
        R_1D = self.R[:, 0]
        Z_1D = self.Z[0, :]
        self.dR = R_1D[1] - R_1D[0]
        self.dZ = Z_1D[1] - Z_1D[0]
        self.dR_dZ = np.array([self.dR, self.dZ])
        self.R0Z0 = np.array([R_1D[0], Z_1D[0]])
        self.dRdZ = self.dR * self.dZ
        nx = eq.nx
        ny = eq.ny
        self.linear_GS_solver = vectorized_createVcycle(
            nx, ny, 
            GSsparse4thOrder(self.R[0, 0], self.R[-1, 0], self.Z[0, 0], self.Z[0, -1]),
            nlevels=1, ncycle=1, niter=2, direct=True
            )

        bndry_indices = np.concatenate(
            [
                [(x, 0) for x in range(nx)],
                [(x, ny - 1) for x in range(nx)],
                [(0, y) for y in np.arange(1, ny - 1)],
                [(nx - 1, y) for y in np.arange(1, ny - 1)],
            ]
        )
        self.bndry_indices = bndry_indices

        # matrices of responses of boundary locations to each grid positions

        greenfunc = Greens(
            self.R[np.newaxis, :, :],
            self.Z[np.newaxis, :, :],
            R_1D[bndry_indices[:, 0]][:, np.newaxis, np.newaxis],
            Z_1D[bndry_indices[:, 1]][:, np.newaxis, np.newaxis],
        )
        greenfunc = jnp.asarray(greenfunc)
        # Prevent infinity/nan by removing Greens(x,y;x,y)
        zeros = np.ones_like(greenfunc)
        zeros[
            np.arange(len(bndry_indices)), bndry_indices[:, 0], bndry_indices[:, 1]
        ] = 0
        self.greenfunc = greenfunc * zeros * self.dRdZ

        # for reshaping
        nx, ny = np.shape(self.R)
        self.nx = nx
        self.ny = ny

        # for integration
        self.grid_points = np.concatenate(
            (eq.R[:, :, np.newaxis], eq.Z[:, :, np.newaxis]), axis=-1
        )
        self.nx, self.ny = np.shape(eq.R)
        self.eqRidx = np.tile(np.arange(self.nx)[:, np.newaxis], (1, self.ny))
        self.eqZidx = np.tile(np.arange(self.ny)[:, np.newaxis], (1, self.nx)).T
        self.idx_grid_points = np.concatenate(
            (self.eqRidx[:, :, np.newaxis], self.eqZidx[:, :, np.newaxis]), axis=-1
        ).reshape(-1, 2)

        self.mask_inside_limiter = self.limiter_handler.mask_inside_limiter
        mask_outside_limiter = np.logical_not(self.mask_inside_limiter)
        # Note the factor 2 is not a typo: used in critical.inside_mask
        self.mask_outside_limiter = (2 * mask_outside_limiter).astype(float)

    def F_function(self, plasma_psi, tokamak_psi, profiles):
        self.update_profiles(profiles)
        self.jtor = self.Jtor(
            self.R, self.Z,
            (tokamak_psi[:, None] + plasma_psi).reshape(self.nx, self.ny, -1)
            )
        self.rhs = self.rhs_before_jtor[:, :, None] * self.jtor

        # calculates and imposes the boundary conditions
        self.psi_boundary = jnp.zeros_like(self.jtor)
        psi_bnd = jnp.tensordot(self.greenfunc, self.jtor, axes=([1, 2], [0, 1]))

        self.psi_boundary = self.psi_boundary.at[:, 0].set(psi_bnd[: self.nx])
        self.psi_boundary = self.psi_boundary.at[:, -1].set(psi_bnd[self.nx : 2 * self.nx])
        self.psi_boundary = self.psi_boundary.at[0, 1 : self.ny - 1].set(psi_bnd[
            2 * self.nx : 2 * self.nx + self.ny - 2
        ])
        self.psi_boundary = self.psi_boundary.at[-1, 1 : self.ny - 1].set(psi_bnd[2 * self.nx + self.ny - 2 :])

        self.rhs = self.rhs.at[0, :].set(self.psi_boundary[0, :])
        self.rhs = self.rhs.at[:, 0].set(self.psi_boundary[:, 0])
        self.rhs = self.rhs.at[-1, :].set(self.psi_boundary[-1, :])
        self.rhs = self.rhs.at[:, -1].set(self.psi_boundary[:, -1])
        residual = jnp.zeros_like(plasma_psi)
        residual = plasma_psi - self.linear_GS_solver(
            self.psi_boundary, self.rhs
        ).reshape(-1, plasma_psi.shape[1])

        return residual
    
    def update_profiles(self, profiles):
        self.alpha_m = profiles.alpha_m
        self.alpha_n = profiles.alpha_n
        self.paxis = profiles.paxis
        self.Ip = profiles.Ip
        self.Raxis = profiles.Raxis
    
    def Jtor(self, *args, **kwargs):
        return self.Jtor_unrefined(*args, **kwargs)
    
    def Jtor_unrefined(self, R, Z, psi, psi_bndry=None):
        (
            self.jtor,
            self.opt,
            self.xpt,
            self.psi_bndry,
            self.diverted_core_mask,
            self.limiter_core_mask,
            self.flag_limiter,
        ) = self.Jtor_build(
            self.diverted_critical_complete,
            # self.Jtor_part1,
            self.Jtor_part2,
            self.limiter_handler.core_mask_limiter,
            # self.core_mask_limiter,
            R,
            Z,
            psi,
            psi_bndry,
            self.mask_outside_limiter,
            self.limiter_handler.limiter_mask_out,
        )
        return self.jtor
    
    def Jtor_build(
        self,
        Jtor_part1,
        Jtor_part2,
        core_mask_limiter,
        R,
        Z,
        psi,
        psi_bndry,
        mask_outside_limiter,
        limiter_mask_out,
    ):
        """Universal function that calculates the plasma current distribution,
        common to all of the different types of profile parametrizations used in FreeGSNKE.

        Parameters
        ----------
        Jtor_part1 : method
            method from the freegs4e Profile class
            returns opt, xpt, diverted_core_mask
        Jtor_part2 : method
            method from each individual profile class
            returns jtor itself
        core_mask_limiter : method
            method of the limiter_handler class
            returns the refined core_mask where jtor>0 accounting for the limiter
        R : np.ndarray
            R coordinates of the domain grid points
        Z : np.ndarray
            Z coordinates of the domain grid points
        psi : np.ndarray
            Poloidal field flux / 2*pi at each grid points (for example as returned by Equilibrium.psi())
        psi_bndry : float, optional
            Value of the poloidal field flux at the boundary of the plasma (last closed flux surface), by default None
        mask_outside_limiter : np.ndarray
            Mask of points outside the limiter, if any, optional
        limiter_mask_out : np.ndarray
            The mask identifying the border of the limiter, including points just inside it, the 'last' accessible to the plasma.
            Same size as psi.
        """

        assert len(psi.shape) == 3, "If psi is 1D, then use freegsnke solver"
        len_vec = psi.shape[2]
        vec_opt = jnp.zeros((3, len_vec))
        vec_xpt = jnp.zeros((3, len_vec))
        vec_diverted_core_mask = jnp.zeros((self.nx, self.ny, len_vec))
        self.vec_diverted_psi_bndry = jnp.zeros((len_vec,))
        for i in range(psi.shape[2]):
            opt, xpt, diverted_core_mask, self.diverted_psi_bndry = Jtor_part1(
                R, Z, psi[:, :, i], psi_bndry, mask_outside_limiter
            )
            vec_opt = vec_opt.at[:, i].set(opt[0])
            vec_xpt = vec_xpt.at[:, i].set(xpt[0])
            vec_diverted_core_mask = vec_diverted_core_mask.at[:, :, i].set(diverted_core_mask)
            self.vec_diverted_psi_bndry = self.vec_diverted_psi_bndry.at[i].set(self.diverted_psi_bndry)

        if diverted_core_mask is None:
            # print('no xpt')
            psi_bndry, limiter_core_mask, flag_limiter = (
                self.diverted_psi_bndry,
                None,
                False,
            )
            # psi_bndry = np.amin(psi[self.limiter_mask_out])
            # diverted_core_mask = np.copy(self.mask_inside_limiter)

        else:
            psi_bndry, limiter_core_mask, flag_limiter = core_mask_limiter(
                psi,
                self.vec_diverted_psi_bndry,
                vec_diverted_core_mask * self.mask_inside_limiter[:, :, None],
                limiter_mask_out,
            )
            limiter_core_mask = jnp.where((
                jnp.sum(
                    limiter_core_mask * self.mask_inside_limiter[:, :, None],
                    axis=(0, 1)
                    ) == 0)[None, None, :],
                (diverted_core_mask * self.mask_inside_limiter)[:, :, None],
                limiter_core_mask
                )
            psi_bndry = jnp.where((
                jnp.sum(
                    limiter_core_mask * self.mask_inside_limiter[:, :, None],
                    axis=(0, 1)
                    ) == 0),
                self.vec_diverted_psi_bndry,
                psi_bndry
            )

        self.inputs = [vec_opt[2, :], psi_bndry, limiter_core_mask]

        jtor = Jtor_part2(R, Z, psi, vec_opt[2, :], psi_bndry, limiter_core_mask)
        return (
            jtor,
            vec_opt,
            vec_xpt,
            psi_bndry,
            vec_diverted_core_mask,
            limiter_core_mask,
            flag_limiter,
        )
    
    def diverted_critical_complete(
        self,
        R,
        Z,
        psi,
        psi_bndry=None,
        mask_outside_limiter=None,
        rel_tolerance_xpt=1e-4,
        starting_dx=0.05,
    ):
        # try:
        #     opt, xpt, diverted_core_mask, psi_bndry = self.Jtor_part1(
        #         R, Z, psi, psi_bndry, mask_outside_limiter
        #     )
        # except:
        #     opt, xpt, diverted_core_mask, psi_bndry = self.diverted_critical(
        #         R,
        #         Z,
        #         psi,
        #         psi_bndry,
        #         mask_outside_limiter,
        #         rel_tolerance_xpt,
        #         starting_dx,
        #     )
        opt, xpt, diverted_core_mask, psi_bndry = self.diverted_critical(
            R,
            Z,
            psi,
            psi_bndry,
            mask_outside_limiter,
            rel_tolerance_xpt,
            starting_dx,
        )

        return opt, xpt, diverted_core_mask, psi_bndry

    def diverted_critical(
        self,
        R,
        Z,
        psi,
        psi_bndry=None,
        mask_outside_limiter=None,
        rel_tolerance_xpt=1e-10,
        starting_dx=0.05,
    ):
        """
        Replaces Jtor_part1 when that fails. Implements a new algorithm to define the LCFS.
        This is considerably more time consuming, but essential when the default routines in
        critical fail, as for example when the Xpt is not correctly identified.


        Parameters
        ----------
        R : np.ndarray
            Radial coordinates of the grid points.
        Z : np.ndarray
            Vertical coordinates of the grid points.
        psi : np.ndarray
            Total poloidal field flux at each grid point [Webers/2pi].
        psi_bndry : float, optional
            Value of the poloidal field flux at the boundary of the plasma (last closed
            flux surface).
        mask_outside_limiter : np.ndarray
            Mask of points outside the limiter, if any.

        Returns
        -------
        np.array
            Each row represents an O-point of the form [R, Z, ψ(R,Z)] [m, m, Webers/2pi].
        np.array
            Each row represents an X-point of the form [R, Z, ψ(R,Z)] [m, m, Webers/2pi].
        np.bool
            An array, the same shape as the computational grid, indicating the locations
            at which the core plasma resides (True) and where it does not (False).
        float
            Value of the poloidal field flux at the boundary of the plasma (last closed
            flux surface).
        """

        # prepare psi_map to use
        psi_map = np.copy(psi)
        self.psi_map = psi_map
        min_psi = np.amin(psi_map)
        psi_map[:, 0] = psi_map[0, :] = psi_map[-1, :] = psi_map[:, -1] = min_psi
        del_psi = np.amax(psi_map) - min_psi
        psi_map /= del_psi

        # find all the local maxima

        maxima_psi_mask = (maximum_filter(psi_map, size=3)) == psi_map
        # select those inside the limiter region
        maxima_psi_mask_in = maxima_psi_mask * self.mask_inside_limiter
        if np.sum(maxima_psi_mask_in) < 1:
            raise ValueError(
                "No O-point in the limiter region. Guess psi_plasma is likely inappropriate."
            )

        # identify the location of the local maximum inside the limiter
        valid_max_psi = np.amax(psi_map[maxima_psi_mask_in])
        mask = psi_map * maxima_psi_mask_in == valid_max_psi
        idx_valid_max = np.array([self.eqRidx[mask][0], self.eqZidx[mask][0]])

        # select the local maxima outside the limiter region
        maxima_psi_mask_out = maxima_psi_mask * mask_outside_limiter
        # include the edges of the map to the excluded region
        maxima_psi_mask_out[1, :] = maxima_psi_mask_out[:, 1] = maxima_psi_mask_out[
            -1, :
        ] = maxima_psi_mask_out[:, -1] = True
        maxima_psi_mask_out = maxima_psi_mask_out.astype(bool)
        idx_excluded_max = np.array(
            [self.eqRidx[maxima_psi_mask_out], self.eqZidx[maxima_psi_mask_out]]
        ).T

        # start root finding for the xpoint flux value
        increment = -starting_dx
        desired_check_larger = True
        current_psi_level = valid_max_psi + increment
        self.record_xpt = [valid_max_psi, current_psi_level]

        while abs(increment) > rel_tolerance_xpt or desired_check_larger is False:
            # design regions
            all_regions = measure.find_contours(psi_map, current_psi_level)
            # sort them by distance to the valid maximum
            mean_dist = [
                np.linalg.norm(np.mean(region, axis=0) - idx_valid_max)
                for region in all_regions
            ]
            regions_order = np.argsort(mean_dist)
            # identify the region containing the valid local maximum
            region_found = False
            idx = -1
            while region_found is False:
                idx += 1
                path = Path(all_regions[regions_order[idx]])
                region_found = path.contains_point(idx_valid_max)
            # check if any excluded points have been included
            check_larger = np.any(path.contains_points(idx_excluded_max.astype(float)))
            if check_larger == desired_check_larger:
                # invert sign and decrease size
                desired_check_larger = np.logical_not(desired_check_larger)
                increment *= -0.5
            # else:
            # keep exploring in the same direction
            # so no action needed
            current_psi_level += increment
            self.record_xpt.append(current_psi_level)

        # build opt, xpt and diverted core mask accordingly
        self.lcfs = all_regions[regions_order[idx]][:-1]
        self.lcfs_grid = self.lcfs
        self.lcfs = self.lcfs * self.dR_dZ[np.newaxis] + self.R0Z0[np.newaxis]
        # build xpt
        psi_bndry = current_psi_level * del_psi
        dist = np.linalg.norm(
            self.lcfs[:, np.newaxis] - self.lcfs[np.newaxis, :], axis=-1
        ) + 10 * np.eye(len(self.lcfs))
        mask = dist == np.amin(dist)
        xpt_coords = (
            np.mean(self.lcfs[np.any(mask, axis=0)], axis=0)
        )
        self.xpt = np.concatenate((xpt_coords, [psi_bndry]))[np.newaxis]
        # build opt
        self.opt = np.concatenate(
            (idx_valid_max * self.dR_dZ + self.R0Z0, [valid_max_psi * del_psi])
        )[np.newaxis]
        # build diverted_core_mask
        diverted_core_mask = path.contains_points(self.idx_grid_points).reshape(
            (self.nx, self.ny)
        )

        return self.opt, self.xpt, diverted_core_mask, psi_bndry

    def Jtor_part2(self, R, Z, psi, psi_axis, psi_bndry, mask):
        """
        Second part of the calculation that will use the explicit
        parameterisation of the chosen profile function to calculate Jtor.

        This is given by:
            Jtor(ψ, R, Z) = L * [ (beta0 * R / Raxis) + ((1 - Beta0) * Raxis / R) ] * (1 - ψ^alpha_m)^alpha_n.

        Parameters
        ----------
        R : np.ndarray
            Radial coordinates of the grid points.
        Z : np.ndarray
            Vertical coordinates of the grid points.
        psi : np.ndarray
            Total poloidal field flux at each grid point [Webers/2pi].
        psi_axis : float
            Value of the poloidal field flux at the magnetic axis of the plasma.
        psi_bndry : float
            Value of the poloidal field flux at the boundary of the plasma (last closed
            flux surface).
        mask : np.ndarray
            Mask of points inside the last closed flux surface.

        Returns
        -------
        np.array
            Toroidal current density on the computational grid [A/m^2].
        """

        # set flux on boundary
        if psi_bndry is None:
            psi_bndry = psi[0, 0]
        self.psi_bndry = psi_bndry
        self.psi_axis = psi_axis

        # grid sizes
        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]

        # calculate normalised psi
        self.psi_norm = np.clip((psi - psi_axis) / (psi_bndry - psi_axis), 0.0, 1.0)

        # shape function
        jtorshape = (
            1.0 - self.psi_norm ** self.alpha_m
        ) ** self.alpha_n

        # if there is a masking function, use it
        if mask is not None:
            jtorshape *= mask
            self.mask = mask

        # now apply constraints to define constants
        self.shapeintegral = (
            beta(1.0 / self.alpha_m, 1.0 + self.alpha_n) / self.alpha_m
        )
        self.shapeintegral *= psi_bndry - psi_axis

        # integrate current density components
        self.IR = (
            jnp.sum(jtorshape * R[:, :, None] / self.Raxis, axis=(0, 1)) * dR * dZ
        )  # romb(romb(jtorshape * R / self.Raxis)) * dR * dZ
        self.I_R = (
            jnp.sum(jtorshape * self.Raxis / R[:, :, None], axis=(0, 1)) * dR * dZ
        )  # romb(romb(jtorshape * self.Raxis / R)) * dR * dZ


        # find L scaling parameter and scaled beta
        self.LBeta0 = -self.paxis * self.Raxis / self.shapeintegral
        self.L = self.Ip / self.I_R - self.LBeta0 * (self.IR / self.I_R - 1)
        self.Beta0 = self.LBeta0 / self.L

        # calculate final toroidal current density
        Jtor = self.L[None, None, :] * (
            self.Beta0[None, None, :] * R[:, :, None] / self.Raxis + \
                (1 - self.Beta0)[None, None, :] * self.Raxis / R[:, :, None]
            ) * jtorshape

        # store parameters
        self.jtor = Jtor
        self.jtorshape = jtorshape
        return Jtor

class vectorized_Limiter_handler(Limiter_handler):
    
    def __init__(self, eq, limiter):
        super().__init__(eq, limiter)
    
    def move_np_to_jnp(self):
        ...

    def core_mask_limiter(
        self,
        psi,
        psi_bndry,
        core_mask,
        limiter_mask_out,
        #   limiter_mask_in,
        #   linear_coeff=.5
    ):
        """Checks if plasma is in a limiter configuration rather than a diverted configuration.
        This is obtained by checking whether the core mask deriving from the assumption of a diverted configuration
        implies an overlap with the limiter. If so, an interpolation of psi on the limiter boundary points
        is called to determine the value of psi_boundary and to recalculate the core_mask accordingly.

        Parameters
        ----------
        psi : jnp.array
            The flux function, including both plasma and metal components.
            np.shape(psi) = (eq.nx, eq.ny)
        psi_bndry : float
            The value of the flux function at the boundary.
            This is xpt[0][2] for a diverted configuration, where xpt is the output of critical.find_critical
        core_mask : np.array
            The mask identifying the plasma region under the assumption of a diverted configuration.
            This is the result of FreeGS4E's critical.core_mask
            Same size as psi.
        limiter_mask_out : np.array
            The mask identifying the border of the limiter, including points just inside it, the 'last' accessible to the plasma.
            Same size as psi.
        Returns
        -------
        psi_bndry : float
            The value of the flux function at the boundary.
        core_mask : np.array
            The core mask after correction
        flag_limiter : bool
            Flag to identify if the plasma is in a diverted or limiter configuration.

        """
        assert (len(psi.shape) == 3) & (len(psi_bndry.shape) == 1) & (len(core_mask.shape) == 3), \
            "This function is meant to be used with vectorized inputs."

        core_mask = core_mask.astype(float)
        # identify the grid points just left-below of points on the limiter that need checking
        offending_mask_adj = (
            core_mask[:-1, :-1]
            + core_mask[1:, :-1]
            + core_mask[:-1, 1:]
            + core_mask[1:, 1:]
        )
        offending_mask_adj = (offending_mask_adj > 0) * (offending_mask_adj < 4)
        offending_mask = jnp.tile(
            self.offending_mask[:, :, None], (1, 1, psi.shape[2])
            )
        offending_mask = offending_mask.at[:-1, :-1].set(offending_mask_adj)
        offending_mask *= self.mask_limiter_cells[:, :, None]
        # self.offending_mask = self.offending_mask.astype(bool)

        self.flag_limiter = jnp.array([False] * psi.shape[2])

        off_rows, off_cols, off_idx = jnp.where(offending_mask)

        offending_cells_id_R = [[self.eqRidx[i, j], k] for i, j, k in zip(off_rows, off_cols, off_idx)]
        offending_cells_id_Z = [[self.eqZidx[i, j], k] for i, j, k in zip(off_rows, off_cols, off_idx)]

        interpolated_on_limiter = []
        for i, j in zip(offending_cells_id_R, offending_cells_id_Z):
            assert i[1] == j[1], f"Indices do not match: {i[1]} != {j[1]}"
            vals_, idxs_ = self.interp_on_limiter_points_cell(
                i[0], j[0], psi[:, :, i[1]]
            )
            interpolated_on_limiter.append([[v, i[1]] for v in vals_])
        psi_bndry = jnp.zeros(psi.shape[2])
        core_mask_out = jnp.zeros_like(core_mask)

        if len(interpolated_on_limiter):
            self.interpolated_on_limiter = jnp.asarray(np.concatenate(interpolated_on_limiter))
            for i in jnp.unique(self.interpolated_on_limiter[:, 1]):
                psi_on_limiter = jnp.amax(
                    self.interpolated_on_limiter[self.interpolated_on_limiter[:, 1] == i][:, 0]
                    )
                i = int(i)
                if psi_on_limiter > psi_bndry[i]:
                    self.flag_limiter = self.flag_limiter.at[i].set(True)
                    psi_bndry = psi_bndry.at[i].set(1.0 * psi_on_limiter)
                    core_mask_out = core_mask_out.at[:, :, i].set(
                        (psi[:, :, i] > psi_bndry[i]) * core_mask[:, :, i]
                        )

        return psi_bndry, core_mask_out, self.flag_limiter

    def interp_on_limiter_points_cell(self, id_R, id_Z, psi):
        """Calculates a bilinear interpolation of the flux function psi in the solver's grid
        cell [eq.R[id_R], eq.R[id_R + 1]] x [eq.Z[id_Z], eq.Z[id_Z + 1]]. The interpolation is returned directly for
        the refined points on the limiter boundary that fall in that grid cell, as assigned
        through the self.fine_point_per_cell objects.


        Parameters
        ----------
        id_R : int
            index of the R coordinate for the relevant grid cell
        id_Z : int
            index of the Z coordinate for the relevant grid cell
        psi : np.array on the solver's grid
            Vaules of the total flux function ofn the solver's grid.

        Returns
        -------
        vals : np.array
            Collection of floating point interpolated values of the flux function
            at the self.fine_point_per_cell[id_R, id_Z] locations.
        """
        id_R, id_Z = int(id_R), int(id_Z)
        if (id_R, id_Z) in self.fine_point_per_cell_Z.keys():
            ker = psi[id_R : id_R + 2, id_Z : id_Z + 2][np.newaxis, :, :]
            # ker *= self.ker_signs
            vals = jnp.sum(ker * self.fine_point_per_cell_Z[id_R, id_Z], axis=-1)
            vals = jnp.sum(vals * self.fine_point_per_cell_R[id_R, id_Z], axis=-1)
            vals /= self.dRdZ
            idxs = self.fine_point_per_cell[id_R, id_Z]
        else:
            vals = []
            idxs = []
        return vals, idxs

from freegs4e.gradshafranov import GSsparse4thOrder
from jax.experimental.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized

class MGDirect:
    def __init__(self, A):
        # self.A = A
        self.data = jnp.array(A.data)
        self.indices = jnp.array(A.indices)
        self.indptr = jnp.array(A.indptr)

    def __call__(self, x, b):
        b1d = b.reshape(-1, b.shape[-1])  # 1D view

        # x = self.solve(b1d)
        x_T = jax.lax.map(lambda b: spsolve(
            self.data, self.indices, self.indptr, b
        ), b1d.T)
        x = x_T.T

        return jnp.reshape(x, b.shape)


class MGJacobi:
    def __init__(self, A, ncycle=4, niter=10, subsolver=None):
        """
        Initialise solver

        A   - The matrix to solve
        subsolver - An operator at lower resolution
        ncycle - Number of V-cycles
        niter - Number of Jacobi iterations

        """
        self.A = A
        self.diag = A.diagonal()
        self.subsolver = subsolver
        self.niter = niter
        self.ncycle = ncycle

        self.sub_b = None
        self.xupdate = None

    def __call__(self, xi, bi, ncycle=None, niter=None):
        """
        Solve Ax = b, given initial guess for x

        ncycle - Optional number of cycles

        """

        # Need to reshape x and b into 1D arrays
        x = jnp.reshape(xi, -1)
        b = jnp.reshape(bi, -1)

        if ncycle is None:
            ncycle = self.ncycle
        if niter is None:
            niter = self.niter

        for c in range(ncycle):
            # Jacobi smoothing
            for i in range(niter):
                x += (b - self.A.dot(x)) / self.diag

            if self.subsolver:
                # Calculate the error
                error = b - self.A.dot(x)

                # Restrict error onto coarser mesh
                self.sub_b = restrict(jnp.reshape(error, xi.shape))

                # smooth this error
                sub_x = jnp.zeros(self.sub_b.shape)
                sub_x = self.subsolver(sub_x, self.sub_b)

                # Prolong the solution
                self.xupdate = interpolate(sub_x)

                x += jnp.reshape(self.xupdate, -1)

            # Jacobi smoothing
            for i in range(niter):
                x += (b - self.A.dot(x)) / self.diag

        return x.reshape(xi.shape)


def vectorized_createVcycle(
    nx, ny, generator, nlevels=4, ncycle=1, niter=10, direct=True
):
    """
    Create a hierarchy of solvers in a multigrid V-cycle

    nx, ny - The highest resolution
    generator(nx,ny) - Returns a sparse matrix, given resolution
    nlevels - Number of multigrid levels
    direct - Lowest level uses direct solver
    ncycle - Number of V cycles. This is only passed to the top level MGJacobi object
    niter - Number of Jacobi iterations per level

    """

    if (nx - 1) % 2 == 1 or (ny - 1) % 2 == 1:
        # Can't divide any further
        nlevels = 1

    if nlevels > 1:
        # Create the solver at lower resolution

        nxsub = (nx - 1) // 2 + 1
        nysub = (ny - 1) // 2 + 1

        subsolver = vectorized_createVcycle(
            nxsub, nysub, generator, nlevels - 1, niter=niter, direct=direct
        )

        # Create the sparse matrix
        A = generator(nx, ny)
        # Create the solver
        return MGJacobi(A, niter=niter, subsolver=subsolver, ncycle=ncycle)

    # At lowest level

    # Create the sparse matrix
    A = generator(nx, ny)
    if direct:
        return MGDirect(A)
    return MGJacobi(A, niter=niter, ncycle=ncycle, subsolver=None)


def smoothJacobi(A, x, b, dx, dy):
    """
    Smooth the solution using Jacobi method
    """

    if b.shape != x.shape:
        raise ValueError("b and x have different shapes")

    smooth = x + (b - A(x, dx, dy)) / A.diag(dx, dy)

    return smooth


def restrict(orig, out=None, avg=False):
    """
    Coarsen the original onto a coarser mesh

    Inputs
    ------

    orig[nx,ny] - A 2D numpy array. Each dimension must have
                  a size (2^n + 1) though nx != ny is possible

    Returns
    -------

    A 2D numpy array of size [(nx-1)/2+1, (ny-1)/2+1]
    """

    nx = orig.shape[0]
    ny = orig.shape[1]

    if (nx - 1) % 2 == 1 or (ny - 1) % 2 == 1:
        # Can't divide any further
        if out is None:
            return orig
        out.resize(orig.shape)
        out[:, :] = orig
        return

    # Dividing x and y in 2
    nx = (nx - 1) // 2 + 1
    ny = (ny - 1) // 2 + 1

    if out is None:
        out = jnp.zeros([nx, ny])
    else:
        out = jnp.resize(out, [nx, ny])

    for x in range(1, nx - 1):
        for y in range(1, ny - 1):
            x0 = 2 * x
            y0 = 2 * y
            out[x, y] = orig[x0, y0] / 4.0
            +(
                orig[x0 + 1, y0]
                + orig[x0 - 1, y0]
                + orig[x0, y0 + 1]
                + orig[x0, y0 - 1]
            ) / 8.0
            +(
                orig[x0 - 1, y0 - 1]
                + orig[x0 - 1, y0 + 1]
                + orig[x0 + 1, y0 - 1]
                + orig[x0 + 1, y0 + 1]
            ) / 16.0
    if not avg:
        out *= 4.0

    return out


def interpolate(orig, out=None):
    """
    Interpolate a solution onto a finer mesh
    """
    nx = orig.shape[0]
    ny = orig.shape[1]

    nx2 = 2 * (nx - 1) + 1
    ny2 = 2 * (ny - 1) + 1

    if out is None:
        out = jnp.zeros([nx2, ny2])
    else:
        out[:, :] = 0.0

    for x in range(1, nx - 1):
        for y in range(1, ny - 1):
            x0 = 2 * x
            y0 = 2 * y

            out[x0 - 1, y0 - 1] += 0.25 * orig[x, y]
            out[x0 - 1, y0] += 0.5 * orig[x, y]
            out[x0 - 1, y0 + 1] += 0.25 * orig[x, y]

            out[x0, y0 - 1] += 0.5 * orig[x, y]
            out[x0, y0] = orig[x, y]
            out[x0, y0 + 1] += 0.5 * orig[x, y]

            out[x0 + 1, y0 - 1] += 0.25 * orig[x, y]
            out[x0 + 1, y0] += 0.5 * orig[x, y]
            out[x0 + 1, y0 + 1] += 0.25 * orig[x, y]

    return out


def smoothVcycle(A, x, b, dx, dy, niter=10, sublevels=0, direct=True):
    """
    Perform smoothing using multigrid


    """

    # Smooth
    for i in range(niter):
        x = smoothJacobi(A, x, b, dx, dy)

    if sublevels > 0:
        # Calculate the error
        error = b - A(x, dx, dy)

        # Restrict error onto coarser mesh
        Cerror = restrict(error)

        # smooth this error
        Cx = jnp.zeros(Cerror.shape)
        Cx = smoothVcycle(
            A, Cx, Cerror, dx * 2.0, dy * 2.0, niter, sublevels - 1
        )

        # Prolong the solution
        xupdate = interpolate(Cx)

        x = x + xupdate

    # Smooth
    for i in range(niter):
        x = smoothJacobi(A, x, b, dx, dy)

    return x


def smoothMG(A, x, b, dx, dy, niter=10, sublevels=1, ncycle=2):
    error = b - A(x, dx, dy)
    print("Starting max residual: %e" % (max(abs(error)),))

    for c in range(ncycle):
        x = smoothVcycle(A, x, b, dx, dy, niter, sublevels)

        error = b - A(x, dx, dy)
        print(
            "Cycle %d : %e"
            % (
                c,
                max(abs(error)),
            )
        )
    return x


class LaplacianOp:
    """
    Implements a simple Laplacian operator
    for use with the multigrid solver
    """

    def __call__(self, f, dx, dy):
        nx = f.shape[0]
        ny = f.shape[1]

        b = jnp.zeros([nx, ny])

        for x in range(1, nx - 1):
            for y in range(1, ny - 1):
                # Loop over points in the domain

                b[x, y] = (f[x - 1, y] - 2 * f[x, y] + f[x + 1, y]) / dx**2 + (
                    f[x, y - 1] - 2 * f[x, y] + f[x, y + 1]
                ) / dy**2

        return b

    def diag(self, dx, dy):
        return -2.0 / dx**2 - 2.0 / dy**2


class LaplaceSparse:
    def __init__(self, Lx, Ly):
        self.Lx = Lx
        self.Ly = Ly

    def __call__(self, nx, ny):
        dx = self.Lx / (nx - 1)
        dy = self.Ly / (ny - 1)

        # Create a linked list sparse matrix
        N = nx * ny
        A = jnp.eye(N)
        for x in range(1, nx - 1):
            for y in range(1, ny - 1):
                row = x * ny + y
                A = A.at[row, row].set(-2.0 / dx**2 - 2.0 / dy**2)

                # y-1
                A = A.at[row, row - 1].set(1.0 / dy**2)

                # y+1
                A = A.at[row, row + 1].set(1.0 / dy**2)

                # x-1
                A = A.at[row, row - ny].set(1.0 / dx**2)

                # x+1
                A = A.at[row, row + ny].set(1.0 / dx**2)
        # Convert to Compressed Sparse Row (CSR) format
        return A.tocsr()

from freegsnke import nk_solver_H

class PCA_preconditioner(nk_solver_H.nksolver):

    def __init__(self, problem_dimension, P, Q, F_mean):
        super().__init__(problem_dimension)
        self.P_PCA = P
        self.Q_PCA = Q
        self.F_mean = F_mean
    
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
        # l2_reg=1e-5,
        # collinearity_reg=1e-6,
    ):

        nR0 = np.linalg.norm(R0)
        self.max_dim = int(max_n_directions + 1)

        # orthogonal basis in x space
        self.Q = np.zeros((self.problem_dimension, self.max_dim))
        # orthonormal basis in x space
        self.Qn = np.zeros((self.problem_dimension, self.max_dim))

        # basis in residual space
        self.G = np.zeros((self.problem_dimension, self.max_dim))
        # orthonormal basis in residual space
        self.Gn = np.zeros((self.problem_dimension, self.max_dim))
        # norms of residual vectors
        self.n_G = np.zeros(self.max_dim)

        self.collinearity = np.zeros((self.max_dim, self.max_dim))

        # QR decomposition of Hm: Hm = T@R
        # self.Omega = np.array([[1]])

        # Hessenberg matrix
        self.Hm = np.zeros((self.max_dim + 1, self.max_dim))

        # resize step based on residual
        adjusted_step_size = step_size * nR0

        # prepare for first direction exploration
        self.n_it = 0
        self.n_it_tot = 0
        this_step_size = adjusted_step_size * ((1 + self.n_it) ** scaling_with_n)

        dx /= np.linalg.norm(dx)
        # # new addition
        # if clip_quantiles is not None:
        #     q1, q2 = np.quantile(dx, clip_quantiles)
        #     dx = np.clip(dx, q1, q2)

        self.Qn[:, self.n_it] = np.copy(dx)
        dx *= this_step_size
        self.Q[:, self.n_it] = np.copy(dx)

        explore = 1
        while explore:
            # build Arnoldi update
            dx = self.Arnoldi_unit(x0, dx, R0, F_function, args)

            # prepare to calculate explained residual
            collinearity_penalty = np.diag(
                np.max(
                    1
                    / (1 - np.abs(self.collinearity[: self.n_it + 1, : self.n_it + 1]))
                    ** 2,
                    axis=0,
                )
                - 1
            )
            collinear_aware_regulariz = (
                np.eye(self.n_it + 1) * self.l2_reg
                + collinearity_penalty * self.collinearity_reg
            )
            self.collinear_aware_regulariz = collinear_aware_regulariz * nR0**2

            # solve the regularised least sq problem
            coeffs = np.dot(
                np.linalg.inv(
                    self.G[:, : self.n_it + 1].T @ self.G[:, : self.n_it + 1]
                    + self.collinear_aware_regulariz
                ),
                np.dot(self.G[:, : self.n_it + 1].T, -R0),
            )
            coeffs = np.clip(coeffs, -clip, clip)
            # calculare the corresponding fraction of residual that is currently explained
            expl_res = np.sum(
                self.G[:, : self.n_it + 1] * coeffs[np.newaxis, :], axis=1
            )
            self.relative_unexplained_residuals.append(
                np.linalg.norm(R0 + expl_res) / nR0
            )

            explore = self.n_it < max_n_directions
            explore *= ( 
                self.relative_unexplained_residuals[-1]
                > target_relative_unexplained_residual
            )

            # prepare for next step
            if explore:
                self.n_it += 1
                # # new addition
                # if clip_quantiles is not None:
                #     q1, q2 = np.quantile(dx, clip_quantiles)
                #     dx = np.clip(dx, q1, q2)
                self.Qn[:, self.n_it] = np.copy(dx)
                this_step_size = adjusted_step_size * (
                    (1 + self.n_it) ** scaling_with_n
                )
                dx *= this_step_size
                self.Q[:, self.n_it] = np.copy(dx)

        # self.coeffs = -nR0 * np.dot(
        #     np.linalg.inv(self.Omega[:-1] @ self.Hm[: self.n_it + 2, : self.n_it + 1]),
        #     self.Omega[:-1, 0],
        # )

        # collinearity = np.sum(self.G[:,np.newaxis,:self.n_it + 1]*self.G[:,:self.n_it + 1,np.newaxis],axis=0)
        # d_collinearity = np.diag(collinearity)**.5
        # collinearity /= (d_collinearity[:, np.newaxis] * d_collinearity[np.newaxis, :])
        # self.collinearity = np.abs(np.triu(collinearity, 1))
        # d_collinearity = np.diag(np.max(1/(1-np.abs(self.collinearity))**2, axis=0))

        # collinear_aware_regulariz = np.eye(self.n_it + 1)*1e-4
        # collinear_aware_regulariz += d_collinearity*1e-4
        # self.collinear_aware_regulariz = collinear_aware_regulariz * nR0**2

        # Hm_ = np.copy(self.Hm[: self.n_it + 2, : self.n_it + 1])
        # self.coeffs = -self.sign * nR0 * np.dot(np.linalg.inv(Hm_.T@Hm_ + self.collinear_aware_regulariz), Hm_[0])
        # self.vanilla_coeffs = np.dot(np.linalg.inv(self.G[:, : self.n_it + 1].T@self.G[:, : self.n_it + 1] + self.collinear_aware_regulariz), np.dot(self.G[:, : self.n_it + 1].T, -R0))

        self.coeffs = np.copy(coeffs)
        self.dx = np.sum(self.Q[:, : self.n_it + 1] * coeffs[np.newaxis, :], axis=1)

def PCA_preconditioner(P, Q, F_data_mean, x0, R0, F_function, args):

    R0 = P @ P.T @ (R0 - F_data_mean) + F_data_mean
    nR0 = np.linalg.norm(R0)
    step_size = nR0 * 2.5
    index = 0
    PR0 = P.T @ R0

    while True:
        approx_jvp = np.zeros_like(Q)
        for i in range(Q.shape[1]):
            approx_jvp[:, i] = F_function(x0 + step_size * Q[:, i], *args) - R0
        PJ = P.T @ approx_jvp
        Sp = np.linalg.solve(PJ, -PR0)


        dx = step_size * Q @ Sp
        R0 = P @ P.T @ (F_function(x0 + dx, *args) - F_data_mean) + F_data_mean
        if (
            np.linalg.norm(R0) < 0.2 * nR0
            ) | (index >= 30):
            
            print('index : ', index)
            break

        PR0 = P.T @ R0
        index += 1

        
    return x0 + step_size * Q @ Sp


    

