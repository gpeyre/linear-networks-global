"""Fast consistency checks for the numerical experiments, not proofs of theorems."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
from scipy.integrate import solve_ivp
from numpy.testing import assert_allclose

from . import certificates, first_order, geometry, momentum, smooth_lasso
from .common import check_times, load_data, save_data, validate_arrays


class FigureChecks(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(17)

    def test_covariance_closure_and_gradient_sign(self):
        for module in (first_order, smooth_lasso):
            width = 32
            w = self.rng.standard_normal((module.M, width))
            u, v = w[:module.DU], w[module.DU:]
            teacher = self.rng.standard_normal((module.DV, module.DU))
            predictor = v @ u.T / width
            gradient = (predictor - teacher if module is first_order
                        else module.gradient(predictor, teacher))
            velocity = np.vstack([-gradient.T @ v, -gradient @ u])
            sigma = w @ w.T / width
            g = module.force(gradient)
            derivative = (velocity @ w.T + w @ velocity.T) / width
            assert_allclose(derivative, -g @ sigma - sigma @ g, atol=2e-13)
            predictor_speed = derivative[module.DU:, :module.DU]
            assert_allclose(np.sum(gradient * predictor_speed),
                            -np.sum(velocity**2) / width, rtol=2e-14)

    def test_factor_covariance_trajectory_agreement(self):
        times = np.linspace(0, 0.1, 9)
        width, trial = 32, 0
        w = np.random.default_rng(10000 + 1000 * width + trial).standard_normal(
            (first_order.M, width))
        sigma = w @ w.T / width
        teacher = self.rng.standard_normal((first_order.DV, first_order.DU)) / 4
        actual, _ = first_order.factor_flow(width, trial, teacher, times)
        expected = first_order.covariance_flow(sigma, teacher, times)
        assert_allclose(actual, expected, rtol=2e-7, atol=1e-9)
        optimum = smooth_lasso.minimizer(teacher)
        actual, _ = smooth_lasso.factor_flow(width, trial, teacher, optimum, times)
        expected = smooth_lasso.covariance_flow(sigma, teacher, optimum, times)
        assert_allclose(actual, expected, rtol=2e-7, atol=1e-9)

    def test_geometry_leaf_and_convergence(self):
        for alpha, b, r0 in [(0.24, 0, -1.05), (0.30, -0.64, -1.05),
                              (0.64, 0, -0.76), (0.30, 0.64, -1.05)]:
            tr = geometry.trajectory(alpha, b, r0, t_final=4.25)
            assert_allclose(tr["p"] * tr["q"] - tr["r"]**2, alpha**2, atol=2e-14)
            self.assertTrue(np.all(tr["p"] > 0) and np.all(tr["q"] > 0))
            residual_bound = abs(r0-1) * np.exp(-2 * tr["h"] * tr["t"][-1])
            self.assertLessEqual(abs(tr["r"][-1]-1), residual_bound + 1e-8)
            self.assertTrue(np.all(np.diff((tr["r"]-1)**2) <= 1e-12))

    def test_geometry_rk4_against_adaptive_solver(self):
        t, theta = geometry.rk4_theta(-1., 0.8, t_final=4.2)
        result = solve_ivp(lambda _t, x: 2 * (1 - 0.8 * np.sinh(x)),
                           (0, t[-1]), [-1.], t_eval=t, rtol=1e-12, atol=1e-14)
        self.assertTrue(result.success)
        assert_allclose(theta, result.y[0], atol=1e-8, rtol=1e-8)

    def test_smooth_lasso_gradient_and_minimizer(self):
        teacher = self.rng.normal(size=(smooth_lasso.DV, smooth_lasso.DU)) / 4
        optimum = smooth_lasso.minimizer(teacher)
        self.assertLess(np.linalg.norm(smooth_lasso.gradient(optimum, teacher)), 1e-12)
        x = self.rng.normal(size=teacher.shape) / 4
        direction = self.rng.normal(size=teacher.shape)
        direction /= np.linalg.norm(direction)
        h = 1e-5
        finite_difference = (smooth_lasso.objective(x + h * direction, teacher)
                             - smooth_lasso.objective(x - h * direction, teacher)) / (2*h)
        self.assertAlmostEqual(finite_difference,
                               np.sum(smooth_lasso.gradient(x, teacher) * direction), places=7)
        direct_gap = smooth_lasso.objective(x, teacher) - smooth_lasso.objective(optimum, teacher)
        assert_allclose(smooth_lasso.loss(x, optimum), direct_gap, rtol=1e-13)
        for h in (1e-4, 1e-8, 1e-12):
            near = optimum + h * direction
            squared_distance = np.sum((near - optimum)**2)
            gap = smooth_lasso.loss(near, optimum)
            self.assertGreaterEqual(gap, 0.5 * squared_distance)
            self.assertLessEqual(gap, 0.5 * smooth_lasso.SMOOTHNESS * squared_distance)

    def test_phase_closure_and_energy_dissipation(self):
        m, du, gamma = momentum.M, momentum.DU, 30.
        z = self.rng.normal(size=(2*m, 12))
        sigma = z @ z.T / 12
        w, speed = z[:m], z[m:]
        error = sigma[du:m, :du] - momentum.TEACHER
        g = np.block([[np.zeros((du, du)), error.T],
                      [error, np.zeros((m-du, m-du))]])
        dz = np.vstack([speed, -g @ w - gamma * speed])
        actual = momentum.phase_rhs(0, sigma.ravel(), gamma).reshape(2*m, 2*m)
        assert_allclose(actual, (dz @ z.T + z @ dz.T) / 12, atol=1e-13)
        energy_derivative = np.sum(error * actual[du:m, :du]) + 0.5*np.trace(actual[m:, m:])
        assert_allclose(energy_derivative, -gamma*np.trace(sigma[m:, m:]), atol=1e-12)

    def test_dimension_free_bootstrap_formula(self):
        s = np.geomspace(1e-4, 1, 37)[:, None]
        gamma = np.geomspace(10, 1e4, 29)[None, :]
        sigma = 200.
        a, b = sigma**2 * np.exp(-2*s), sigma**2 * np.exp(2*s)
        rho = 4*b*certificates.SMOOTHNESS + 2*np.sqrt(
            2*certificates.SMOOTHNESS*certificates.ENERGY)
        epsilon = np.minimum(1 / np.sqrt(8*b*certificates.SMOOTHNESS),
                             gamma / (2*(rho + 2*gamma**2)))
        original_rate = (2/3) * np.minimum(3*epsilon*a*certificates.KAPPA, gamma)
        gap, rate = certificates.quantities(s, sigma, gamma)
        assert_allclose(rate, original_rate, rtol=1e-14)
        polynomial = (2*gamma**2 - s*certificates.KAPPA*sigma**3*np.exp(-3*s)
                      * gamma/(4*np.sqrt(3*certificates.ENERGY/2)) + rho)
        np.testing.assert_array_equal(gap < 0, polynomial < 0)

    def test_all_plotted_momentum_certificates_are_strict(self):
        for sigma in momentum.SCALES:
            for gamma in momentum.DAMPING:
                s, gap = certificates.strict_tube(sigma, gamma)
                self.assertLess(gap, 0)
                self.assertGreater(s, 0)
                a, b = sigma**2*np.exp(-2*s), sigma**2*np.exp(2*s)
                rate, q = certificates.tightened_rate(a, b, gamma)
                self.assertGreater(rate, 0)
                self.assertTrue(0 < q < 1)

    def test_tightened_rate_formula(self):
        a, b, gamma = 39000., 41000., 100.
        k, l = 4*a*certificates.KAPPA, 4*b*certificates.SMOOTHNESS
        rho = l + 2*np.sqrt(2*certificates.SMOOTHNESS*certificates.ENERGY)
        eta = gamma / (gamma + np.sqrt(gamma**2 + 2*rho))
        aa, bb = (1-eta)*k, rho + gamma**2/(2*eta)
        epsilon = 2*gamma/(aa+bb)
        q = epsilon*np.sqrt(l/2)
        original_rate = min(epsilon*aa, 2*gamma-epsilon*bb)/(1+q)
        assert_allclose(certificates.tightened_rate(a, b, gamma), [original_rate, q], rtol=1e-14)

    def test_numeric_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            arrays = {"x": np.arange(5.), "seed": np.array(123)}
            save_data(directory, "test", **arrays)
            cached = load_data(Path(directory) / "data", "test")
            validate_arrays(cached, {"x": (5,), "seed": ()})
            for key in arrays:
                np.testing.assert_array_equal(arrays[key], cached[key])
            with self.assertRaises(ValueError):
                save_data(directory, "invalid", x=np.array([{}], dtype=object))

    def test_corrupted_arrays_are_rejected(self):
        for data in ({}, {"x": np.array([np.nan])}, {"x": np.ones(2)},
                     {"x": np.array(["invalid"])}, {"x": np.array([np.inf])}):
            with self.assertRaises(ValueError):
                validate_arrays(data, {"x": (1,)})

    def test_invalid_time_grids_are_rejected(self):
        for times in ([0], [1, 2], [0, 0], [0, np.nan], [0, -1]):
            with self.assertRaises(ValueError):
                check_times(times)


if __name__ == "__main__":
    unittest.main()
