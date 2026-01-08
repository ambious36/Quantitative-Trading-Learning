import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
class GaussPlusModel:
    """
    Gauss+ (Gaussian Cascade) Term Structure Model
    =============================================
    
    This class implements the Gauss+ model described in Appendix A9 of *FIXED INCOME SECURITIES Tools for Today's Markets
    FOURTH EDITION by BRUCE TUCKMAN ANGEL SERRAT*. 
    The model is a three-factor Gaussian affine term structure model written in cascade form.

    Factors (cascade form)
    ----------------------
    x_t = (r_t, m_t, l_t)
    r_t : short-rate factor (observed policy rate)
    m_t : medium-term factor (monetary policy expectations)
    l_t : long-term factor (long-run rate expectations)

    Parameters
    ----------
    alpha = (alpha_r, alpha_m, alpha_l) : Mean-reversion speeds of the three factors.
    sigma = (sigma_l, sigma_m, rho) : Volatility of long factor, volatility of medium factor, and correlation between medium and long shocks.
    mu : float : Long-run mean level of the short rate under Q.
    """

    def __init__(self, alpha, sigma, mu):
        """
        Initialize the Gauss+ model.

        Parameters
        ----------
        alpha : iterable of length 3 (alpha_r, alpha_m, alpha_l)
        sigma : iterable of length 3 (sigma_l, sigma_m, rho)
        mu : float : Long-run mean of the short rate
        """
        self.alpha = np.array(alpha, dtype=float)
        self.sigma = np.array(sigma, dtype=float)
        self.mu = float(mu)

        # Precompute matrices that depend only on the parameters
        self.Ainv = self._A_inv()
        self.Sigma = self._Sigma()

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------
    @staticmethod
    def B(tau, alpha):
        """ Basis function B_i(tau) used in affine yield formulas. """
        return (1.0 - np.exp(-alpha * tau)) / (alpha * tau)

    def _A_inv(self):
        """ Compute A(alpha)^{-1}, the inverse of the cascade-to-reduced-form transformation matrix. """
        ar, am, al = self.alpha
        return np.array([
            [1, -ar/(ar-am), ar*am/((ar-am)*(ar-al))],
            [0.0, ar/(ar-am), ar*am/((ar-am)*(am-al))],
            [0.0, 0.0, ar*am/((ar-al)*(am-al))]
        ])

    def _Omega(self):
        """ Instantaneous volatility matrix Omega(sigma). """
        sig_l, sig_m, rho = self.sigma
        return np.array([
            [0.0, 0.0, 0.0],
            [rho*sig_m, np.sqrt(1-rho**2)*sig_m, 0.0],
            [sig_l, 0.0, 0.0]
        ])

    def _Sigma(self):
        """ Covariance matrix of reduced-form factor innovations. """
        Om = self._Omega()
        return self.Ainv @ Om @ Om.T @ self.Ainv

    # ------------------------------------------------------------------
    # Yield and forward loadings
    # ------------------------------------------------------------------
    def upsilon(self, tau):
        """ Yield loadings Upsilon(tau). """
        Bvec = self.B(tau, self.alpha)
        return Bvec @ self.Ainv

    def upsilon_forward(self, tau, tau_p):
        """ Forward-rate loadings Upsilon'(tau, tau'). """
        B1 = self.B(tau + tau_p, self.alpha)
        B0 = self.B(tau, self.alpha)
        return (B1 - B0) @ self.Ainv / tau_p

    # ------------------------------------------------------------------
    # Convexity corrections
    # ------------------------------------------------------------------
    def C(self, tau):
        """ Convexity correction C(tau) for zero-coupon yields. """
        Cval = 0.0
        for i in range(3):
            for j in range(3):
                ai, aj = self.alpha[i], self.alpha[j]
                Bij = (1 - self.B(tau, ai) - self.B(tau, aj))
                exp_term = (1 - np.exp(-(ai+aj)*tau)) / ((ai+aj)*tau)
                Cval += (self.Sigma[i, j]/(2*ai*aj)) * (Bij - exp_term)
        return Cval

    def C_forward(self, tau, tau_p):
        """ Convexity correction for forward rates. """
        return (self.C(tau + tau_p) - self.C(tau)) / tau_p

    # ------------------------------------------------------------------
    # Pricing formulas
    # ------------------------------------------------------------------
    def zero_yield(self, tau, x):
        """ Zero-coupon yield y_t(tau). """
        ups = self.upsilon(tau)
        return self.mu * (1.0 - ups.sum()) - self.C(tau) + ups @ x

    def forward_rate(self, tau, tau_p, x):
        """ Continuously compounded forward rate f_t(tau, tau'). """
        ups_p = self.upsilon_forward(tau, tau_p)
        return self.mu * (1.0 - ups_p.sum()) + ups_p @ x - self.C_forward(tau, tau_p)

    # ------------------------------------------------------------------
    # Factor extraction
    # ------------------------------------------------------------------
    def extract_factors(self, r_t, y2, y10):
        """ Recover the medium- and long-term factors by exact fit. """
        taus = np.array([2.0, 10.0])
        Y = np.array([y2, y10])
        U = np.vstack([self.upsilon(tau) for tau in taus])
        Cvec = np.array([self.C(tau) for tau in taus])
        rhs = Y - (self.mu * (1 - U.sum(axis=1)) - Cvec + U[:, 0] * r_t)
        ml = np.linalg.solve(U[:, 1:], rhs)
        return np.array([r_t, ml[0], ml[1]])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def yield_curve(self, taus, x):
        return np.array([self.zero_yield(tau, x) for tau in taus])

    def forward_curve(self, taus, tau_p, x):
        return np.array([self.forward_rate(tau, tau_p, x) for tau in taus])

    # ------------------------------------------------------------------
    # Parameter calibration and fitting
    # ------------------------------------------------------------------
    def calibrate_alpha(
            self,
            delta_y,      # T × N yield changes (all maturities)
            delta_yb,     # T × 2 benchmark yield changes (2y, 10y)
            maturities,   # length-N list
            tau_bench=(2.0, 10.0)
        ):

            # ---------- OLS slopes (data) ----------
            # beta_hat: 2 × N
            beta_hat = (
                np.linalg.inv(delta_yb.T @ delta_yb)
                @ delta_yb.T
                @ delta_y
            )

            def objective(alpha_vec):
                # Update alpha and dependent matrices
                self.alpha = alpha_vec
                self.Ainv = self._A_inv()

                # ----- Net out short-rate contribution -----
                # Build reduced loadings (drop Υs)
                Ups_all = []
                Ups_b = []

                for tau in maturities:
                    U = self.upsilon(tau)      # (Υs, Υm, Υl)
                    Ups_all.append(U[1:])      # (Ym, Yl)

                for tau in tau_bench:
                    U = self.upsilon(tau)
                    Ups_b.append(U[1:])

                Ups_all = np.asarray(Ups_all)  # N × 2
                Ups_b = np.asarray(Ups_b)      # 2 × 2

                # Invert benchmark loadings
                Ups_b_inv = np.linalg.inv(Ups_b)

                # Model-implied slopes: N × 2
                model_slopes = Ups_all @ Ups_b_inv

                # Match equation (A9.20)
                return np.linalg.norm(model_slopes.T - beta_hat)

            # ---------- Optimization ----------
            res = minimize(
                objective,
                self.alpha,
                method="Nelder-Mead"
            )

            self.alpha = res.x
            self.Ainv = self._A_inv()

            return self.alpha


    def calibrate_sigma(self, delta_yb):
        """ Estimate the sigma parameters by matching model-implied variances. """
        def obj(sigma):
            self.sigma = sigma
            Om = self._Omega()
            ups_b = np.vstack([self.upsilon(tau) for tau in [2,10]])
            model_implied_yield = ups_b @ Om @ Om.T @ ups_b.T
            realized_volatilities_yields = np.diag(delta_yb.T @ delta_yb * 252 / delta_yb.shape[0])
            return np.linalg.norm(model_implied_yield - realized_volatilities_yields)

        bounds = [(1e-6, None), (1e-6, None), (-0.999, 0.999)]
        res = minimize(obj, self.sigma, bounds=bounds, method="L-BFGS-B")
        self.sigma = res.x
        self.Sigma = self._Sigma()
        return self.sigma

    def calibrate_mu(self, Y, X, maturities):
            Y = np.array(Y)
            def obj(mu):
                self.mu = mu[0]
                model_yields = np.vstack([self.yield_curve(maturities, x) for x in X])
                errors = Y - model_yields
                return np.sum(errors**2)
            res = minimize(obj, np.array([self.mu]), method='L-BFGS-B', bounds=[(1e-6, None)])
            self.mu = round(res.x[0], 4)
            return self.mu