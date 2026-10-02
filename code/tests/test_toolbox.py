"""Independent structural tests of the public computational API."""
import subprocess
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import solve_ivp

import linear_networks as ln
from linear_networks.certificates import white_bootstrap, tightened_momentum_rate
from linear_networks.losses import smooth_lasso_gap


class ToolboxTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.loss = ln.QuadraticLoss(self.rng.normal(size=(2, 3)))
        self.w = self.rng.normal(size=(5, 8))
        self.sigma = ln.factor_covariance(self.w)
        self.times = np.linspace(0, 0.4, 15)

    def test_import_does_not_select_a_plotting_backend(self):
        code = "import sys, linear_networks; assert 'matplotlib' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_rectangular_factor_covariance_closure(self):
        for du, dv in ((1, 1), (2, 4), (4, 2)):
            w = self.rng.normal(size=(du+dv, 7))
            loss = ln.QuadraticLoss(self.rng.normal(size=(dv, du)))
            dw = ln.factor_rhs(w, loss)
            sigma = ln.factor_covariance(w)
            assert_allclose((dw@w.T + w@dw.T)/7, ln.covariance_rhs(sigma, loss), atol=3e-14)
            u, v, vu = ln.covariance_blocks(sigma, du)
            assert_allclose(vu, w[du:]@w[:du].T/7)
            self.assertEqual(u.shape, (du,du))
            self.assertEqual(v.shape, (dv,dv))

    def test_loss_gradient_finite_differences(self):
        for loss in (self.loss, ln.SmoothedLassoLoss(self.loss.target),
                     ln.QuadraticLoss(self.loss.target, [[0, 0.2, 2], [1, 0, 3]])):
            x = self.rng.normal(size=(2, 3))
            direction = self.rng.normal(size=x.shape)
            h = 1e-6
            fd = (loss.value(x+h*direction)-loss.value(x-h*direction))/(2*h)
            assert_allclose(fd, np.sum(loss.gradient(x)*direction), rtol=2e-7)

    def test_nonunique_quadratic_minimum(self):
        loss = ln.QuadraticLoss([[1, 7]], [[2, 0]])
        self.assertEqual(loss.kappa, 2)
        self.assertEqual(loss.smoothness, 2)
        self.assertEqual(loss.gap([[1, -900]]), 0)
        result = ln.integrate_covariance(np.eye(3), loss, np.linspace(0, 8, 80))
        self.assertLess(result.loss_gaps[-1], 1e-14)
        assert_allclose(result.predictors[-1], [[1, 0]], atol=1e-7)
        zero = ln.QuadraticLoss([[1]], [[0]])
        self.assertEqual(zero.kappa, 0)
        self.assertEqual(zero.smoothness, 0)

    def test_singular_support_margins_include_zeros(self):
        margins = ln.support_margins(np.diag([2, 2, 0, 0, 0]), 2)
        assert_allclose([margins.a_plus, margins.a_minus, margins.gap], [2, 0, 2])
        # Nonzero signed eigenvalues alone would give the wrong positive gap.
        self.assertEqual(ln.support_margins(np.diag([1, 0, 1, 0]), 2).gap, 0)
        self.assertEqual(ln.support_margins(np.zeros((4,4)), 2).gap, 0)
        self.assertEqual(ln.support_margins(np.ones((2,2)), 1).gap, 0)

    def test_signed_margins_depend_on_cross_covariance(self):
        margins = ln.support_margins([[1, 0.6], [0.6, 1]], 1)
        assert_allclose([margins.a_plus, margins.a_minus], [0.8, 0.8])
        for sigma in (self.sigma, np.diag([2,2,2,0,0])):
            margins = ln.support_margins(sigma, 3)
            j = ln.signature_matrix(3, 2)
            self.assertGreaterEqual(np.linalg.eigvalsh(sigma-margins.a_plus*j).min(), -1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(sigma+margins.a_minus*j).min(), -1e-12)

    def test_factor_gd_contains_quadratic_covariance_term(self):
        tau = 0.01
        before = self.w.copy()
        stepped = ln.factor_step(self.w, self.loss, tau)
        expected = ln.covariance_step(self.sigma, self.loss, tau)
        assert_allclose(ln.factor_covariance(stepped), expected, atol=1e-14)
        g = ln.force_matrix(self.loss.gradient(self.sigma[3:, :3]))
        euler = self.sigma + tau*ln.covariance_rhs(self.sigma, self.loss)
        assert_allclose(expected-euler, tau*tau*g@self.sigma@g, atol=1e-14)
        self.assertGreater(np.linalg.norm(expected-euler), 1e-6)
        assert_array_equal(self.w, before)

    def test_factor_and_covariance_trajectories(self):
        factors = ln.integrate_factors(self.w, self.loss, self.times)
        covariance = ln.integrate_covariance(self.sigma, self.loss, self.times)
        assert_allclose(factors.covariances, covariance.covariances, atol=1e-8, rtol=2e-8)
        spectrum = ln.signed_spectrum(self.sigma, 3)
        for sigma in covariance.covariances:
            assert_allclose(ln.signed_spectrum(sigma, 3), spectrum, atol=2e-8)
        self.assertEqual(covariance.predictors.shape, (15, 2, 3))

    def test_gf_support_gap_envelope(self):
        result = ln.integrate_covariance(self.sigma, self.loss, self.times)
        bound = result.loss_gaps[0]*np.exp(-2*self.loss.kappa*ln.support_margins(self.sigma,3).gap*self.times)
        self.assertLessEqual(np.max(result.loss_gaps-bound), 2e-9)

    def test_true_gd_and_stationary_run(self):
        result = ln.gradient_descent(self.w, self.loss, 0.005, 300)
        self.assertTrue(np.all(np.diff(result.loss_gaps) <= 1e-12))
        self.assertLess(result.loss_gaps[-1], result.loss_gaps[0]/100)
        self.assertEqual(result.factors.shape, (301,5,8))
        empty = ln.gradient_descent(self.w, self.loss, 0.1, 0)
        assert_array_equal(empty.factors[0], self.w)
        self.assertEqual(len(empty.times),1)

    def test_momentum_energy_and_conformal_law(self):
        z = self.rng.normal(size=(10, 15))
        initial = ln.factor_covariance(z)
        gamma = 0.7
        result = ln.integrate_momentum(initial, self.loss, self.times, gamma)
        self.assertTrue(np.all(np.diff(result.energies) <= 1e-8))
        omega = np.block([[np.zeros((5,5)), np.eye(5)], [-np.eye(5), np.zeros((5,5))]])
        law = np.array([np.trace((omega@s)@(omega@s)) for s in result.phase_covariances])
        assert_allclose(law, law[0]*np.exp(-2*gamma*self.times), atol=1e-8, rtol=1e-8)
        w, velocity = z[:5], z[5:]
        def rhs(_t, flat):
            positions, velocities = flat.reshape(2,5,15)
            return np.stack([velocities, ln.factor_rhs(positions,self.loss)-gamma*velocities]).ravel()
        direct = solve_ivp(rhs, (0,self.times[-1]), np.stack([w,velocity]).ravel(),
                           t_eval=self.times, rtol=1e-10, atol=1e-12)
        self.assertTrue(direct.success)
        states = direct.y.T.reshape(-1,10,15)
        assert_allclose(result.phase_covariances, states@states.transpose(0,2,1)/15, atol=1e-8, rtol=1e-8)

    def test_zero_velocity_and_zero_damping(self):
        initial = ln.white_phase_covariance(3,2,1,0)
        result = ln.integrate_momentum(initial,self.loss,self.times,0)
        assert_allclose(result.energies,result.energies[0],atol=1e-8)

    def test_smooth_lasso_gap_and_stationarity(self):
        loss = ln.SmoothedLassoLoss(self.loss.target)
        assert_allclose(loss.gradient(loss.minimizer),0,atol=1e-14)
        x = self.rng.normal(size=loss.shape)
        assert_allclose(loss.gap(x), loss.value(x)-loss.minimum, atol=1e-13)
        for eps in (1e-5,1e-9,1e-12):
            nearby = loss.minimizer+eps
            squared = np.sum((nearby-loss.minimizer)**2)
            self.assertGreaterEqual(loss.gap(nearby),0.5*squared)
            self.assertLessEqual(loss.gap(nearby),0.5*loss.smoothness*squared)
        # This isolates cancellation in h(x)h(z)+xz for opposite large entries.
        x,z = np.array([[-1e8]]),np.array([[1e8]])
        h = np.hypot(z,0.01)
        expected = 0.5*np.sum((x-z)**2)+np.sum(np.hypot(x,0.01)-h-z/h*(x-z))
        assert_allclose(smooth_lasso_gap(x,z,1,0.01),expected,rtol=2e-16)

    def test_reproducible_sampling_and_no_input_mutation(self):
        before = self.sigma.copy()
        assert_array_equal(ln.sample_factors(self.sigma,9,seed=4),ln.sample_factors(self.sigma,9,seed=4))
        singular = ln.sample_factors(np.diag([1,0,2]),5,seed=2)
        assert_array_equal(singular[1],0)
        root = ln.psd_square_root(self.sigma)
        assert_allclose(root@root,self.sigma,atol=1e-14)
        ln.integrate_covariance(self.sigma,self.loss,self.times)
        assert_array_equal(self.sigma,before)

    def test_custom_loss_needs_no_known_optimum(self):
        class CustomLoss:
            shape=(1,1)
            def value(self,s): return float(0.5*(s[0,0]-1)**2)
            def gradient(self,s): return s-1
        result=ln.integrate_covariance(np.eye(2),CustomLoss(),self.times)
        self.assertIsNone(result.loss_gaps)
        self.assertLess(result.objectives[-1],result.objectives[0])
        with self.assertRaises(ValueError):
            _=result.energies

    def test_invalid_inputs_raise_clear_errors(self):
        calls = [
            lambda: ln.QuadraticLoss([[1]], [[-1]]),
            lambda: ln.SmoothedLassoLoss([[1]], smoothing=0),
            lambda: ln.QuadraticLoss([[1j]]),
            lambda: ln.integrate_covariance(np.eye(4),self.loss,self.times),
            lambda: ln.integrate_covariance(-np.eye(5),self.loss,self.times),
            lambda: ln.psd_square_root([[1,2],[0,1]]),
            lambda: ln.integrate_covariance(np.eye(5),self.loss,[0,0]),
            lambda: ln.factor_step(self.w,self.loss,0),
            lambda: ln.gradient_descent(self.w,self.loss,0.1,1.2),
            lambda: ln.integrate_factors(np.ones((4,8)),self.loss,self.times),
            lambda: ln.integrate_momentum(np.eye(10),self.loss,self.times,-1),
            lambda: ln.sample_factors(np.eye(2),0),
            lambda: ln.support_margins(np.eye(2),2),
            lambda: ln.support_margins(np.eye(2),1,rtol=-1),
        ]
        for call in calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()

    def test_white_bootstrap_broadcast_and_scope(self):
        s=np.array([0.01,0.1])[:,None]
        gamma=np.array([30,100,300])[None,:]
        gap,rate=white_bootstrap(s,200,gamma,kappa=1,smoothness=1,energy=3.5)
        self.assertEqual(gap.shape,(2,3))
        self.assertTrue(np.all(rate>0))
        self.assertTrue(np.any(gap<0))
        rate,q=tightened_momentum_rate(1,2,3,kappa=1,smoothness=2,energy=1)
        self.assertGreater(rate,0)
        self.assertTrue(0<q<1)
        with self.assertRaises(ValueError):
            white_bootstrap(0.1,1,0,kappa=1,smoothness=1,energy=1)
        with self.assertRaises(ValueError):
            tightened_momentum_rate(2,1,3,kappa=1,smoothness=1,energy=1)


if __name__ == "__main__":
    unittest.main()
