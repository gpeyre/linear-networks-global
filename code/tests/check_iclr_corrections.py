"""Focused mathematical regression checks for the ICLR correction pass."""

import __future__
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose


RNG = np.random.default_rng(9082)


def root(matrix):
    values, vectors = np.linalg.eigh(matrix)
    assert values.min() >= -1e-9
    return (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T


def covariance_blocks(u, v):
    n = u.shape[1]
    su, sv, predictor = u @ u.T / n, v @ v.T / n, v @ u.T / n
    return su, sv, predictor, np.block([[su, predictor.T], [predictor, sv]])


def check_relaxed_balance():
    for beta in (-2.0, 0.0, 3.0):
        for index in range(60):
            n = index % 3 + 1
            du, dv = n + index % 2, n + index % 4
            raw = RNG.normal(size=(n, n))
            gram = raw @ raw.T + (abs(beta) + 1) * np.eye(n)
            left, _ = np.linalg.qr(RNG.normal(size=(du, n)))
            right, _ = np.linalg.qr(RNG.normal(size=(dv, n)))
            u = np.sqrt(n) * left @ root(gram)
            v = np.sqrt(n) * right @ root(gram + beta * np.eye(n))
            su, sv, predictor, sigma = covariance_blocks(u, v)
            pu = np.linalg.pinv(predictor) @ predictor
            pv = predictor @ np.linalg.pinv(predictor)
            recovered_u = pu @ (-beta / 2 * np.eye(du)
                                 + root(predictor.T @ predictor + beta**2 / 4 * np.eye(du)))
            recovered_v = pv @ (beta / 2 * np.eye(dv)
                                 + root(predictor @ predictor.T + beta**2 / 4 * np.eye(dv)))
            assert_allclose(su, recovered_u, atol=2e-6)
            assert_allclose(sv, recovered_v, atol=2e-6)
            assert_allclose((v.T @ v - u.T @ u) / n, beta * np.eye(n), atol=1e-10)
            error = RNG.normal(size=(dv, du))
            assert_allclose(-error @ su - sv @ error,
                            -error @ recovered_u - recovered_v @ error, atol=1e-5)
            signature = np.diag([1.0] * du + [-1.0] * dv)
            assert_allclose(sigma @ signature @ sigma, -beta * sigma, atol=1e-9)

    # Full-column rank does not remove the ambient support projectors.
    u, v = np.array([[1.0], [0.0]]), np.array([[2.0], [0.0]])
    su, sv, predictor, _ = covariance_blocks(u, v)
    unprojected_v = 1.5 * np.eye(2) + root(predictor @ predictor.T + 2.25 * np.eye(2))
    assert_allclose(unprojected_v, np.diag([4.0, 3.0]))
    assert_allclose(sv, np.diag([4.0, 0.0]))
    for vector in np.eye(2):
        su, sv, predictor, _ = covariance_blocks(np.zeros((2, 1)), vector[:, None])
        assert_allclose(predictor, 0)
        assert_allclose(-su - sv, -np.outer(vector, vector))

    # The beta=0 square-root identities do not require full column rank.
    u = RNG.normal(size=(2, 5))
    orthogonal, _ = np.linalg.qr(RNG.normal(size=(2, 2)))
    v = orthogonal @ u
    su, sv, predictor, sigma = covariance_blocks(u, v)
    assert np.linalg.matrix_rank(sigma) == 2 < u.shape[1]
    assert_allclose(su, root(predictor.T @ predictor), atol=1e-9)
    assert_allclose(sv, root(predictor @ predictor.T), atol=1e-9)
    assert_allclose(sigma @ np.diag([1, 1, -1, -1]) @ sigma, 0, atol=1e-9)


def check_gaussian_boundary():
    times = np.linspace(0, 10, 100)
    h = 1 / (2 * np.exp(2 * times) - 1)
    dh = -4 * np.exp(2 * times) / (2 * np.exp(2 * times) - 1)**2
    assert_allclose(dh, -2 * h * (h + 1))
    assert_allclose(-dh, -2 * h * (-h - 1))
    assert h[-1] < 1e-8
    for initial in (-2.0, 0.0, 3.0):
        r = initial / (1 + 2 * abs(initial) * times)
        dr = -2 * abs(initial) * initial / (1 + 2 * abs(initial) * times)**2
        assert_allclose(dr, -2 * abs(r) * r)


def check_hi_spectrum_and_moments():
    for du, dv in [(2, 2), (2, 4), (4, 2)]:
        for rank in range(min(du, dv) + 1):
            alpha = 0.7
            predictor = np.zeros((dv, du))
            singular = np.arange(1, rank + 1) / 3
            predictor[np.arange(rank), np.arange(rank)] = singular
            su = root(predictor.T @ predictor + alpha**2 * np.eye(du))
            sv = root(predictor @ predictor.T + alpha**2 * np.eye(dv))
            sigma = np.block([[su, predictor.T], [predictor, sv]])
            expected = np.concatenate([np.hypot(singular, alpha) + singular,
                                       np.hypot(singular, alpha) - singular,
                                       np.full(du + dv - 2 * rank, alpha)])
            assert_allclose(np.linalg.eigvalsh(sigma), np.sort(expected), atol=1e-12)
            m = du + dv
            factors = np.sqrt(m) * root(sigma)
            error = RNG.normal(size=(dv, du))
            force = np.block([[np.zeros((du, du)), error.T],
                              [error, np.zeros((dv, dv))]])
            speed = -force @ factors
            dcovariance = (speed @ factors.T + factors @ speed.T) / m
            assert_allclose(dcovariance, -force @ sigma - sigma @ force, atol=1e-11)


def check_notebook_radius():
    path = Path(__file__).resolve().parents[1] / "notebooks/research/bures_mean_field_convergence.ipynb"
    notebook = json.loads(path.read_text())
    names = {"split_blocks", "sublevel_radius_bound", "coarse_ellipticity_constant"}
    functions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            functions.extend(node for node in tree.body
                             if isinstance(node, ast.FunctionDef) and node.name in names)
    assert {node.name for node in functions} == names
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {"np": np}
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    radius = namespace["sublevel_radius_bound"]
    du, dv = 2, 3
    teacher = RNG.normal(size=(dv, du))
    sigma = np.eye(du + dv)
    result = SimpleNamespace(sigma=np.array([sigma]), loss=np.array([np.sum(teacher**2) / 2]))
    bound = radius(result, du, dv)
    assert_allclose(bound, 2 * np.linalg.norm(teacher))
    boundary_point = 2 * teacher
    assert_allclose(np.linalg.norm(boundary_point), bound)
    assert_allclose(np.sum((boundary_point - teacher)**2) / 2, result.loss[0])

    for _ in range(100):
        initial = RNG.normal(size=(dv, du))
        sigma = (1 + np.linalg.norm(initial, 2)) * np.eye(du + dv)
        sigma[du:, :du], sigma[:du, du:] = initial, initial.T
        result = SimpleNamespace(sigma=np.array([sigma]),
                                 loss=np.array([np.sum((initial - teacher)**2) / 2]))
        direction = RNG.normal(size=(dv, du))
        direction /= np.linalg.norm(direction)
        point = teacher + RNG.uniform() * np.linalg.norm(initial - teacher) * direction
        assert np.linalg.norm(point) <= radius(result, du, dv) + 1e-10


if __name__ == "__main__":
    for check in [check_relaxed_balance, check_gaussian_boundary,
                  check_hi_spectrum_and_moments, check_notebook_radius]:
        check()
        print(f"PASS {check.__name__}")
    print("All ICLR correction checks passed; these supplement, not replace, the proofs.")
