"""Check manuscript structure and the general/special covariance identities."""
from collections import Counter
from pathlib import Path
import json
import re
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(20260910)


def spd(n):
    a = RNG.normal(size=(n, n))
    return a @ a.T + np.eye(n)


def root(a):
    values, vectors = np.linalg.eigh(a)
    return (vectors * np.sqrt(values)) @ vectors.T


def generator(e):
    dv, du = e.shape
    return np.block([[np.zeros((du, du)), e.T],
                     [e, np.zeros((dv, dv))]])


def support_margins(covariance, du):
    values, vectors = np.linalg.eigh(covariance)
    square_root = (vectors * np.sqrt(np.maximum(values, 0.))) @ vectors.T
    dv = len(covariance)-du
    signature = np.diag([1.] * du + [-1.] * dv)
    signed = np.linalg.eigvalsh(square_root @ signature @ square_root)
    return max(0., signed[dv]), max(0., -signed[dv-1])


def structure():
    sources = {p: p.read_text() for p in (ROOT / "sections").glob("*.tex")}
    text = "\n".join(sources.values())
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    refs = re.findall(r"\\(?:ref|eqref|pageref)\{([^}]+)\}", text)
    duplicates = [x for x, n in Counter(labels).items() if n > 1]
    assert not duplicates, duplicates
    assert not set(refs).difference(labels), sorted(set(refs).difference(labels))
    forbidden = re.compile(r"\bHI\b|\bCSI\b|hyperbolic isotrop|isotropy condition", re.I)
    for p, source in sources.items():
        if p.name != "intrinsic.tex":
            visible = re.sub(r"%[^\n]*|\\label\{[^}]*\}", "", source)
            assert not forbidden.search(visible), (p, forbidden.search(visible))
        assert not re.search(r"\\(?:Su|Sv|Apos|Bvel|Ezero)(?![A-Za-z])", source), p
    bib = (ROOT / "iclr2027_conference.bib").read_text()
    keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
    citations = re.findall(r"\\cite[tp]?(?:\[[^\]]*\]){0,2}\{([^}]+)\}", text)
    assert not {k.strip() for group in citations for k in group.split(",")}.difference(keys)
    main = (ROOT / "paper.tex").read_text()
    for source in re.findall(r"\\input\{([^}]+)\}", main):
        assert (ROOT / (source + ".tex")).exists(), source
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        assert (ROOT / "figures" / name).exists(), name
    for name in re.findall(r"([A-Za-z0-9_-]+)\.pgf", text):
        assert (ROOT.parent / "code/figures" / (name + ".pgf")).exists(), name
    return {"labels": len(labels), "references": len(refs),
            "citations": len(citations), "intrinsic_terms_isolated": True}


def historical_presentation_alignment():
    """Earlier layout comparison, retained for the archived manuscript."""
    original = ROOT.parent / "legacy" / "paper_iclr_old_(with_hi)"
    def spine(path):
        main = (path / ("paper.tex" if path == ROOT else "iclr2026_conference.tex")).read_text().split(r"\appendix")[0]
        return [p for p in re.findall(r"\\input\{(sections/[^}]+)\}", main)
                if p != "sections/abstract"]
    assert spine(ROOT) == spine(original)
    def ordered_source(path, files):
        return "\n".join((path / (name + ".tex")).read_text() for name in files)
    original_body = ordered_source(original, spine(original))
    new_body = ordered_source(ROOT, spine(ROOT))
    old_titles = re.findall(r"^\\section\{([^}]+)\}", original_body, re.M)
    new_titles = re.findall(r"^\\section\{([^}]+)\}", new_body, re.M)
    renamed_sections = {"Conservation Laws and Hyperbolic Isotropy":
                        "Mean Field Conservation Laws",
                        "Linear Convergence": "Linear Convergence of Gradient Flow",
                        "Convergence under Discretization":
                        "Linear Convergence of Gradient Descent"}
    assert new_titles == [renamed_sections.get(title, title)
                          for title in old_titles], (old_titles, new_titles)
    model_source = (ROOT / "sections/model.tex").read_text()
    intro_source = (ROOT / "sections/introduction.tex").read_text()
    contributions = intro_source.split(r"\paragraph{Our contributions}", 1)[1]
    contributions = contributions.split("%%% Local Variables:", 1)[0].strip()
    assert not re.search(r"\n\s*\n", contributions)
    for label in ("sec:laws", "sec:convergence", "sec:gd", "sec:resnets", "sec:momentum"):
        assert r"Section~\ref{"+label+"}" in contributions
    for label in ("prop:covariance-closure-laws", "lem:margins", "thm:gf",
                  "cor:finite-width", "thm:gd", "thm:resnet-finite",
                  "thm:resnet-continuous", "prop:phase-laws", "thm:momentum",
                  "cor:two-scale-white-window"):
        assert r"\ref{"+label+"}" in contributions
    law_source = (ROOT / "sections/laws.tex").read_text()
    convergence_source = (ROOT / "sections/convergence.tex").read_text()
    assert re.findall(r"\\paragraph\{([^}]+)\}", law_source)[0] == (
        "Gradient flow and covariance closure")
    for source in (intro_source, model_source):
        assert "gradient flow" not in source.lower()
        assert "eq:mean-field-gradient-flow" not in source
        assert not re.search(r"\\(?:dot|ddot|partial_t)\b|\b[EG]_t\b", source)
    measure_source = (ROOT / "sections/mean-field.tex").read_text()
    assert not re.search(r"\\mu\b|\\widehat\s*\\mu", new_body)
    assert not re.search(r"\\nu\b", new_body)
    assert r"\label{eq:mean-field-gradient-flow}" in law_source
    assert r"-n\nabla_{(U,V)}\cL(U_t,V_t)" in law_source
    assert r"\cL(U,V)=\ell" in model_source
    assert r"\mathcal L_n" not in new_body and r"\cL_n" not in new_body
    closure_statement = law_source.split(r"\begin{proposition}", 1)[1].split(
        r"\end{proposition}", 1)[0]
    for label in ("eq:bures-G", "eq:lax-spectral-laws"):
        assert r"\label{"+label+"}" in closure_statement
    for label in ("eq:wasserstein-pde", "eq:split-wasserstein-pde",
                  "def:mean-field-law", "eq:gram-mean-field-law",
                  "prop:moment-closure", "eq:finite-spectrum-gamma"):
        assert r"\label{"+label+"}" in measure_source, label
    for path in (ROOT / "sections").glob("*.tex"):
        source = path.read_text()
        assert r"\cU" not in source and r"\mathcal U" not in source
        assert not re.search(r"\\widehat\s*\\Sigma|\\hS(?:uu|uv|vu|vv)\b", source), path
        if path.name != "mean-field.tex":
            assert not re.search(r"\\partial_t\s*\\mu", source), path
    assert "eq:quadratic-growth" not in model_source
    assert "quadratic-growth" not in model_source
    assert convergence_source.count(r"\begin{theorem}") == 1
    assert r"\begin{proposition}" not in convergence_source
    assert r"\paragraph{Proof sketch}" in convergence_source
    assert "eq:flow-dissipation" not in convergence_source
    assert "thm:uniform-nondegeneracy-convergence" not in convergence_source
    assert r"\begin{definition}[Covariance support margins and gap]" in law_source
    assert r"\label{def:support-gap}" in law_source
    assert r"\label{eq:gap-spectrum}" in law_source
    assert "including zeros" in law_source
    gf_statement = convergence_source.split(r"\begin{theorem}", 1)[1].split(
        r"\end{theorem}", 1)[0]
    assert r"\Sigma_0\succeq0" in gf_statement
    assert "possibly" in gf_statement and "singular covariance" in gf_statement
    assert r"\Sigma_0\succ0" not in gf_statement
    assert r"\gap{\Sigma_0}>0" in gf_statement
    assert r"\rankop\Sigma_\infty=\rankop\Sigma_0" in gf_statement
    old_figures = re.findall(r"([A-Za-z0-9_-]+)\.pgf", original_body)
    new_figures = re.findall(r"([A-Za-z0-9_-]+)\.pgf", new_body)
    assert new_figures == old_figures, (old_figures, new_figures)
    assert re.findall(r"\\begin\{wrapfigure\}\{[^}]+\}\{([^}]+)\}", new_body) == [
        "0.32\\textwidth", "0.34\\linewidth"]
    def appendix_spine(path):
        appendix = (path / ("paper.tex" if path == ROOT else "iclr2026_conference.tex")).read_text().split(r"\appendix")[1]
        return re.findall(r"\\input\{(sections/[^}]+)\}", appendix)
    assert appendix_spine(ROOT) == (["sections/mean-field"] +
                                   appendix_spine(original) + ["sections/intrinsic"])
    triptych = "bures_convergence_triptych.pgf"
    original_art = (ROOT.parent / "code/figures" / triptych).read_text()
    adapted_art = (ROOT / "figures" / triptych).read_text()
    assert adapted_art == original_art.replace("(b) Broken isotropy",
                                                "(b) Covariance conditioning")
    proof_source = (ROOT / "sections/appendix.tex").read_text()
    retained_bounds = ["eq:uniform-nondegeneracy-covariance-bounds", "eq:gf-tube",
                       "eq:gf-tail", "eq:gf-predictor-tail", "eq:gd-tails",
                       "eq:sampling-event", "eq:resnet-length",
                       "eq:resnet-integral", "eq:momentum-tail",
                       "eq:quadratic-growth", "eq:flow-dissipation"]
    for label in retained_bounds:
        assert r"\label{"+label+"}" in proof_source, label
    relocated_details = ["eq:sampling-budgets", "eq:budget-width",
                         "thm:sampling-budgets", "eq:gd-signature",
                         "eq:gd-predictor-update", "rem:gd-local",
                         "eq:resnet-layer-gradient", "eq:single-particle-momentum-main",
                         "eq:phase-block-ode", "eq:predictor-speed",
                         "eq:regression-bound-main"]
    for label in relocated_details:
        assert r"\label{"+label+"}" in proof_source, label
        assert r"\label{"+label+"}" not in new_body, label
    result_pattern = re.compile(
        r"\\begin\{(theorem|proposition|corollary)\}.*?\\end\{\1\}", re.S)
    sketched_results = {}
    for name in spine(ROOT):
        source = (ROOT / (name + ".tex")).read_text()
        results = list(result_pattern.finditer(source))
        for index, result in enumerate(results):
            end = results[index+1].start() if index+1 < len(results) else len(source)
            sketch = source[result.end():end]
            assert "Proof sketch" in sketch, (name, result.group())
            assert len(re.findall(r"\\qed\b", sketch.split("Proof sketch", 1)[1])) == 1, (
                name, "Each proof sketch must end with exactly one QED symbol")
        if results:
            sketched_results[name] = len(results)
    gd_source = (ROOT / "sections/sampling.tex").read_text()
    momentum_source = (ROOT / "sections/momentum.tex").read_text()
    assert r"\Xi=W\dot W^\top/n" in momentum_source
    assert r"\Spos=WW^\top/n" in momentum_source
    assert r"\Svel=\dot W\dot W^\top/n" in momentum_source
    assert r"G(\Svu):=" in momentum_source
    assert r"\cE_t=\Delta_t+\mathcal T_t" in momentum_source
    assert r"\nu_t\in\cP_2(\R^{2m})" in proof_source
    equations = re.findall(r"\\begin\{equation\}(.*?)\\end\{equation\}",
                           momentum_source, re.S)
    phase_equation = next(eq for eq in equations if r"\label{eq:phase}" in eq)
    assert r"\dot\Phase_t=\Mob_t\Phase_t+\Phase_t\Mob_t^\top" in phase_equation
    phase_line = re.sub(r"\\begin\{psmallmatrix\}.*?\\end\{psmallmatrix\}",
                        "", phase_equation, flags=re.S)
    assert r"\\" not in phase_line
    lower_bounds = law_source.split(r"\label{eq:block-lower}", 1)[1].split(
        r"\end{equation}", 1)[0]
    assert r"\\" not in lower_bounds
    sampling_heading = r"\paragraph{Convergence under Finite Width Sampling}"
    assert convergence_source.index(sampling_heading) > convergence_source.index(
        r"\label{fig:bures-finite-width-comparison}")
    assert sampling_heading not in gd_source
    assert r"\label{cor:finite-width}" in convergence_source
    assert r"\label{eq:width}" in convergence_source
    assert r"r_0=\rankop\tSigma_0" in convergence_source
    assert r"\emph{subgaussian}" in convergence_source
    assert r"\Sigma_0=W_0W_0^\top/n" in convergence_source
    assert r"\gap{\Sigma_0}\ge(1-\varepsilon)\gap{\tSigma_0}>0" in convergence_source
    assert r"\sqrt{\widetilde\Delta_0}" in proof_source
    assert r"(1-\varepsilon)\tSigma_0\preceq\Sigma_0" in proof_source
    assert r"\bigl(r_0+\log(2/\delta)\bigr)" in convergence_source
    assert "This positive empirical gap, not positive definiteness" in convergence_source
    assert r"\label{rem:rank-adaptive-width}" in convergence_source
    assert r"\label{ex:white-small-width}" in proof_source
    assert r"\bigl(r_0+\log(2/\delta)\bigr)" in proof_source
    for label in ("sec:sampling", "sec:finite-width"):
        assert r"\label{"+label+"}" in convergence_source
        assert r"\label{"+label+"}" not in gd_source
    assert r"\label{eq:gd-step}" in gd_source
    assert r"\label{sec:gd}" in gd_source
    assert r"\paragraph{Gradient Descent on the factors}" in gd_source
    assert r"\paragraph{Linear convergence.}" in gd_source
    for proof_only in (r"\alpha_\star", r"M_{\nabla}", r"B_\star"):
        assert proof_only not in gd_source
        assert proof_only in proof_source
    assert "I_\\star" not in gd_source.split(r"\begin{theorem}", 1)[0]
    gd_statement = gd_source.split(r"\begin{theorem}", 1)[1].split(
        r"\end{theorem}", 1)[0]
    assert r"I_\star=4\sqrt{\Delta_0/\kappa}/\gap{\Sigma_0}" in gd_statement
    assert r"\gap{\Sigma_0}>0" in gd_statement
    assert r"\rankop\Sigma_\infty=\rankop\Sigma_0" in gd_statement
    resnet_source = (ROOT / "sections/resnets.tex").read_text()
    for label in ("thm:resnet-finite", "thm:resnet-continuous"):
        result = next(result.group() for result in result_pattern.finditer(resnet_source)
                      if r"\label{"+label+"}" in result.group())
        assert result.startswith(r"\begin{theorem}"), label
        assert r"Proof of Theorem~\ref{"+label+"}" in proof_source
    assert r"\begin{remark}[Sharpness of the boundedness hypothesis]" in resnet_source
    assert r"\label{rem:resnet-path-escape}" in resnet_source
    assert r"\paragraph{Orientation and the path bounds.}" not in resnet_source
    assert r"Verification of Remark~\ref{rem:resnet-path-escape}" in proof_source
    assert r"p_\star=-1" in resnet_source
    assert "orientation-reversing target" in resnet_source
    for label in ("eq:resnet-escape-finite", "eq:resnet-escape-continuous"):
        assert r"\label{"+label+"}" in proof_source
        assert r"\label{"+label+"}" not in resnet_source
    assert r"\label{eq:certificate}" in momentum_source
    assert r"\label{eq:white-direct}" in momentum_source
    heading_pattern = re.compile(r"\\(?:section|subsection|paragraph)\{[^}]*\}")
    headings = list(heading_pattern.finditer(new_body))
    prose_openings = 0
    for index, heading in enumerate(headings):
        end = headings[index+1].start() if index+1 < len(headings) else len(new_body)
        opening = re.sub(r"%[^\n]*|\\label\{[^}]*\}", "",
                         new_body[heading.end():end]).strip()
        if not opening:
            continue
        assert not re.match(
            r"(?:Let|Assume|Set|Define|Write)\b|(?:For|With)\s*(?:\$|\\\()|"
            r"\\\[|\\begin\{(?:equation|align|gather|wrapfigure)", opening), (
                heading.group(), opening[:150])
        prose_openings += 1
    display_pattern = r"\\begin\{(?:equation|align|gather)\*?\}|\\\["
    return {"original_section_spine_preserved": True,
            "factor_closure_and_spectral_laws_open_section_two": True,
            "measure_formalism_and_pde_in_dedicated_appendix": True,
            "unhatted_empirical_and_tilded_population_covariances": True,
            "single_main_convergence_theorem_with_proof_sketch": True,
            "finite_width_sampling_closes_gradient_flow_section": True,
            "semidefinite_support_gap_definition_and_rank_conclusions": True,
            "rank_adaptive_width_and_positive_empirical_gap_sketch": True,
            "main_sections": new_titles, "original_main_figures": new_figures,
            "original_appendices_follow_measure_appendix": True,
            "triptych_only_title_changed": True,
            "quantitative_bounds_retained_in_appendix": retained_bounds,
            "supporting_calculations_relocated": relocated_details,
            "main_results_with_proof_sketches": sketched_results,
            "explicit_width_stepsize_and_damping_conditions_retained": True,
            "GD_auxiliary_constants_restricted_to_proof_appendix": True,
            "single_contribution_paragraph_with_section_and_result_references": True,
            "factor_first_momentum_and_unified_loss_gap_notation": True,
            "one_line_support_bounds_and_phase_closure": True,
            "main_heading_openings_checked": prose_openings,
            "main_display_blocks": len(re.findall(display_pattern, new_body))}


def presentation_alignment():
    main = (ROOT / "paper.tex").read_text()
    body, appendix = main.split(r"\appendix", 1)
    body_files = re.findall(r"\\input\{(sections/[^}]+)\}", body)
    appendix_files = re.findall(r"\\input\{(sections/[^}]+)\}", appendix)
    expected_body = ["introduction", "model", "laws", "convergence", "sampling",
                     "resnets", "momentum", "conclusion"]
    expected_appendix = ["notation", "mean-field", "appendix", "appendix-momentum",
                         "appendix-laws", "intrinsic", "appendix-smooth-lasso"]
    assert body_files[1:] == [f"sections/{name}" for name in expected_body], body_files
    assert appendix_files == [f"sections/{name}" for name in expected_appendix], appendix_files
    titles = [re.search(r"\\section\{([^}]+)\}", (ROOT / f"{name}.tex").read_text())
              for name in body_files[1:] if name not in {"sections/model", "sections/sampling"}]
    assert all(title is not None for title in titles)
    assert [title.group(1) for title in titles] == [
        "Introduction", "Mean Field Conservation Laws",
        "Linear Convergence of Training Dynamics", "Deep Linear Residual Networks",
        "Heavy-ball Momentum Convergence", "Conclusion"]
    return {"main_sections": len(titles), "appendix_sources": len(appendix_files)}


def factor_covariance_closure():
    rng = np.random.default_rng(20260911)
    phase_rng = np.random.default_rng(20260915)
    for _ in range(80):
        du, dv = rng.integers(1, 5, size=2)
        m = du + dv
        n = int(rng.integers(1, 2*m+1))
        w = rng.normal(size=(m, n))
        target = rng.normal(size=(dv, du))
        direction = rng.normal(size=w.shape)

        def loss(factors):
            predictor = factors[du:] @ factors[:du].T / n
            return .5*np.linalg.norm(predictor-target)**2

        covariance = w @ w.T/n
        e = covariance[du:, :du]-target
        g = generator(e)
        velocity = -g @ w
        step = 1e-5
        difference = (loss(w+step*direction)-loss(w-step*direction))/(2*step)
        derivative = np.sum((g @ w/n)*direction)
        assert_allclose(difference, derivative, rtol=2e-7, atol=2e-7)
        covariance_velocity = (velocity @ w.T+w @ velocity.T)/n
        assert_allclose(covariance_velocity, -g @ covariance-covariance @ g,
                        atol=2e-12)
        signature = np.diag([1.]*du+[-1.]*dv)
        signed = signature @ covariance
        signed_velocity = signature @ covariance_velocity
        assert_allclose(signed_velocity, g @ signed-signed @ g, atol=2e-12)
        assert_allclose(velocity.T @ signature @ w+w.T @ signature @ velocity,
                        0., atol=2e-12)
        for k in range(1, m+1):
            term = np.linalg.matrix_power(signed, k-1) @ signed_velocity
            assert abs(k*np.trace(term)) <= 1e-10*max(1., k*np.linalg.norm(term))
        tau = .01/(1+np.linalg.norm(g, 2))
        updated = w+tau*velocity
        exact = covariance+tau*covariance_velocity+tau**2*g @ covariance @ g
        assert_allclose(updated @ updated.T/n, exact, atol=2e-12)
        speed = phase_rng.normal(size=w.shape)
        damping = phase_rng.uniform(.1, 3.)
        acceleration = -g @ w-damping*speed
        phase_factors = np.vstack([w, speed])
        phase_velocity = np.vstack([speed, acceleration])
        phase = phase_factors @ phase_factors.T/n
        drift = np.block([[np.zeros((m, m)), np.eye(m)],
                          [-g, -damping*np.eye(m)]])
        phase_derivative = (phase_velocity @ phase_factors.T
                            +phase_factors @ phase_velocity.T)/n
        assert_allclose(phase_derivative, drift @ phase+phase @ drift.T, atol=2e-12)
        assert_allclose(phase[:m, m:], w @ speed.T/n, atol=2e-12)
        loss_derivative = np.sum(e*((speed[du:] @ w[:du].T
                                    +w[du:] @ speed[:du].T)/n))
        kinetic = np.sum(speed**2)/(2*n)
        kinetic_derivative = np.sum(speed*acceleration)/n
        assert_allclose(loss_derivative+kinetic_derivative,
                        -2*damping*kinetic, atol=2e-12)
    return {"factor_loss_normalization_and_covariance_closure_cases": 80,
            "gram_and_covariance_spectral_laws": True,
            "exact_factor_GD_quadratic_covariance_increment": True,
            "finite_factor_phase_closure_and_mechanical_energy_cases": 80}


def phase_laws():
    for _ in range(100):
        du, dv = RNG.integers(1, 4, size=2)
        m = du + dv
        canonical = np.block([[np.zeros((m, m)), np.eye(m)],
                              [-np.eye(m), np.zeros((m, m))]])
        phase0 = spd(2*m)
        fundamental = np.eye(2*m)
        damping = RNG.uniform(0.1, 2.)
        elapsed = 0.
        for _ in range(5):
            e = RNG.normal(size=(dv, du))
            drift = np.block([[np.zeros((m, m)), np.eye(m)],
                              [-generator(e), -damping*np.eye(m)]])
            assert_allclose(drift.T @ canonical + canonical @ drift,
                            -damping * canonical, atol=1e-12)
            dt = RNG.uniform(0.01, 0.08)
            fundamental = expm(dt * drift) @ fundamental
            elapsed += dt
        scale = np.exp(-damping * elapsed)
        assert_allclose(fundamental.T @ canonical @ fundamental,
                        scale * canonical, atol=1e-10)
        phase = fundamental @ phase0 @ fundamental.T
        expected = np.linalg.solve(fundamental.T, canonical @ phase0) @ fundamental.T
        assert_allclose(canonical @ phase / scale, expected, atol=1e-9)
    return {"noncommuting_phase_paths": 100, "steps_per_path": 5}


def intrinsic_identities():
    for index in range(150):
        du, dv = RNG.integers(1, 4, size=2)
        m = du + dv
        predictor = RNG.normal(size=(dv, du))
        alpha = RNG.uniform(0.2, 2.)
        su = root(predictor.T @ predictor + alpha**2 * np.eye(du))
        sv = root(predictor @ predictor.T + alpha**2 * np.eye(dv))
        covariance = np.block([[su, predictor.T], [predictor, sv]])
        signature = np.diag([1.] * du + [-1.] * dv)
        assert_allclose(covariance @ signature @ covariance,
                        alpha**2 * signature, atol=1e-9)
        assert np.linalg.eigvalsh(covariance).min() > 0
        eigenvalues = np.linalg.eigvalsh(root(covariance) @ signature @ root(covariance))
        assert_allclose(np.abs(eigenvalues), alpha, atol=1e-9)

        position = spd(m)
        c = RNG.normal(size=(m, m))
        c = (c + c.T) / 2
        theta = 0. if index % 3 == 0 else RNG.uniform(0.1, 2.)
        mixed = position @ c
        velocity = c @ position @ c + theta**2 * np.linalg.inv(position)
        phase = np.block([[position, mixed], [mixed.T, velocity]])
        canonical = np.block([[np.zeros((m, m)), np.eye(m)],
                              [-np.eye(m), np.zeros((m, m))]])
        assert_allclose(phase @ canonical @ phase, theta**2 * canonical, atol=1e-8)
        force = generator(RNG.normal(size=(dv, du)))
        damping = 0.7
        da = c @ position + position @ c
        dc = -force - damping*c - c@c + theta**2 * np.linalg.matrix_power(position, -2)
        mixed_dot = da @ c + position @ dc
        assert_allclose(mixed_dot, velocity-position@force-damping*mixed, atol=1e-8)
    return {"rectangular_position_cases": 150, "phase_cases": 150,
            "zero_scale_phase_cases": 50}


def general_regression():
    for _ in range(200):
        du, dv = 2, 3
        m = du + dv
        a = spd(m)
        d = RNG.normal(size=(m, m))
        q = spd(m)
        x = a @ d.T
        b = d @ a @ d.T + q
        force = generator(RNG.normal(size=(dv, du)))
        damping = 0.9
        da = d @ a + a @ d.T
        dx = b-a@force-damping*x
        db = -force@x-x.T@force-2*damping*b
        inverse = np.linalg.inv(a)
        dd = dx.T @ inverse-d@da@inverse
        dq = db-dd@a@d.T-d@da@d.T-d@a@dd.T
        assert_allclose(dd, -force-damping*d-d@d+q@inverse, atol=1e-8)
        assert_allclose(dq, -2*damping*q-d@q-q@d.T, atol=1e-8)
        assert np.linalg.norm(d, 2)**2 <= np.trace(b)/np.linalg.eigvalsh(a).min()+1e-9
    return {"nonsymmetric_regression_cases": 200}


def spectral_support_and_lie_algebra():
    for _ in range(200):
        du, dv = RNG.integers(1, 4, size=2)
        m = du + dv
        covariance = spd(m)
        signature = np.diag([1.] * du + [-1.] * dv)
        signed = np.linalg.eigvalsh(root(covariance) @ signature @ root(covariance))
        plus, minus = signed[signed > 0].min(), -signed[signed < 0].max()
        assert np.linalg.eigvalsh(covariance-plus*signature).min() >= -1e-9
        assert np.linalg.eigvalsh(covariance+minus*signature).min() >= -1e-9
        assert np.linalg.eigvalsh(covariance-1.0001*plus*signature).min() < 0
        assert np.linalg.eigvalsh(covariance+1.0001*minus*signature).min() < 0
        transport = expm(-.1*generator(RNG.normal(size=(dv, du))))
        moved = transport @ covariance @ transport.T
        assert np.linalg.eigvalsh(moved[:du, :du]).min() >= plus-1e-9
        assert np.linalg.eigvalsh(moved[du:, du:]).min() >= minus-1e-9

    for du in range(1, 4):
        for dv in range(1, 4):
            m = du+dv
            basis = [generator(e.reshape(dv, du)) for e in np.eye(du*dv)]
            brackets = [a@b-b@a for a in basis for b in basis]
            span = np.stack([a.ravel() for a in basis+brackets])
            assert np.linalg.matrix_rank(span) == m*(m-1)//2
            a, b = basis[0], basis[-1]
            covariance = spd(m)
            def field(a, c):
                return -a@c-c@a.T
            bracket = field(b, field(a, covariance))-field(a, field(b, covariance))
            assert_allclose(bracket, field(a@b-b@a, covariance), atol=1e-10)
            eigenvalues = np.r_[np.linspace(.8, 1.8, du), -np.linspace(.9, 1.9, dv)]
            derivative = np.stack([k*eigenvalues**(k-1) for k in range(1, m+1)])
            assert np.linalg.matrix_rank(derivative) == m
    return {"optimal_support_cases": 200, "lie_dimension_and_law_rank_cases": 9}


def semidefinite_support_gap():
    rng = np.random.default_rng(20260912)
    for _ in range(180):
        du, dv = rng.integers(1, 5, size=2)
        m = du+dv
        rank = int(rng.integers(1, m+1))
        factor = rng.normal(size=(m, rank))
        covariance = factor @ factor.T
        signature = np.diag([1.] * du + [-1.] * dv)
        plus, minus = support_margins(covariance, du)
        scale = max(1., np.linalg.norm(covariance, 2))
        for margin, sign in [(plus, 1.), (minus, -1.)]:
            assert np.linalg.eigvalsh(covariance-sign*margin*signature).min() >= -1e-7*scale
            outside = covariance-sign*(margin+1e-3*scale)*signature
            assert np.linalg.eigvalsh(outside).min() < -1e-10*scale
        assert plus <= np.linalg.eigvalsh(covariance[:du, :du]).min()+1e-7*scale
        assert minus <= np.linalg.eigvalsh(covariance[du:, du:]).min()+1e-7*scale
        transport = expm(-.1*generator(rng.normal(size=(dv, du))))
        moved = transport @ covariance @ transport.T
        assert_allclose(support_margins(moved, du), (plus, minus), atol=2e-7*scale)
        eps = 1e-5
        regularized = support_margins(covariance+eps*np.eye(m), du)
        assert np.all(np.array(regularized) >= np.array([plus, minus])-1e-7*scale)

    for correlation in [0., .3, .9, 1.]:
        covariance = np.array([[1., correlation], [correlation, 1.]])
        assert_allclose(support_margins(covariance, 1),
                        np.repeat(np.sqrt(1-correlation**2), 2), atol=1e-10)
    for du, dv in [(1, 3), (3, 1), (2, 2)]:
        covariance = np.diag([1.] * du + [0.] * dv)
        assert_allclose(support_margins(covariance, du), (1., 0.))
        covariance = np.diag([0.] * du + [1.] * dv)
        assert_allclose(support_margins(covariance, du), (0., 1.))

    # Nonzero signed eigenvalues cannot compensate for missing predictor directions.
    covariance = np.diag([1., 0., 1., 0.])
    target = np.diag([0., 1.])
    force = generator(-target)
    assert_allclose(support_margins(covariance, 2), (0., 0.))
    assert_allclose(force @ covariance+covariance @ force, 0.)
    assert_allclose((np.eye(4)-.1*force) @ covariance @ (np.eye(4)-.1*force), covariance)
    return {"semidefinite_support_and_transport_cases": 180,
            "fixed_hidden_block_examples": 4, "one_sided_block_examples": 6,
            "zero_gap_stationary_counterexample": True}


def rank_adaptive_sampling():
    rng = np.random.default_rng(20260914)
    for index in range(80):
        du, dv = rng.integers(1, 5, size=2)
        m = du+dv
        selected = np.r_[np.arange(du), du+np.arange(rng.integers(0, dv+1))]
        transport = expm(-.15*generator(rng.normal(size=(dv, du))))
        factor = transport[:, selected]*rng.uniform(.7, 1.3, size=len(selected))
        basis, triangular = np.linalg.qr(factor, mode="reduced")
        reduced_covariance = triangular @ triangular.T
        population = factor @ factor.T
        rank = len(selected)
        n = 128*rank
        xi = (rng.normal(size=(m, n)) if index % 2 else
              rng.choice([-1., 1.], size=(m, n)))
        eta = basis.T @ xi
        samples = basis @ root(reduced_covariance) @ eta
        empirical = samples @ samples.T/n
        relative_error = np.linalg.norm(eta @ eta.T/n-np.eye(rank), 2)
        assert relative_error < .6
        assert_allclose(population, basis @ reduced_covariance @ basis.T, atol=1e-12)
        for difference in (empirical-(1-relative_error)*population,
                           (1+relative_error)*population-empirical):
            assert np.linalg.eigvalsh(difference).min() >= -1e-10
        initial_margins = np.array(support_margins(population, du))
        sampled_margins = np.array(support_margins(empirical, du))
        assert initial_margins.sum() > 0
        assert np.all(sampled_margins >= (1-relative_error)*initial_margins-1e-7)
        assert np.linalg.matrix_rank(empirical) == rank

    # One-sided sampling needs only the smaller block to have full row rank.
    for du, dv in [(2, 41), (41, 2)]:
        rank = min(du, dv)
        for n in [rank-1, rank, 16*rank]:
            active = np.sqrt(2)*rng.normal(size=(rank, n))
            samples = np.zeros((du+dv, n))
            samples[:du] = active if du <= dv else 0.
            if dv < du:
                samples[du:] = active
            covariance = samples @ samples.T/n
            gap = sum(support_margins(covariance, du))
            assert_allclose(covariance[du:, :du], 0.)
            assert np.linalg.matrix_rank(covariance) == min(n, rank)
            if n < rank:
                assert gap < 1e-7
            else:
                expected = np.linalg.eigvalsh(active @ active.T/n)[0]
                assert_allclose(gap, expected, atol=1e-7)
                assert gap > 0 and n < du+dv

    # Verify the deterministic implication on the Gaussian concentration event.
    for dimension in [64, 128]:
        n = dimension//16
        for _ in range(4):
            x = rng.normal(size=(dimension, n))
            y = rng.normal(size=n)
            assert np.linalg.svd(x, compute_uv=False)[-1] >= 5*np.sqrt(dimension)/8
            assert np.linalg.norm(y) <= 3*np.sqrt(dimension)/8
            gram = x.T @ x-np.outer(y, y)
            assert np.linalg.eigvalsh(gram)[0] >= dimension/4
            samples = np.vstack([x, y])
            covariance = samples @ samples.T/n
            assert_allclose(support_margins(covariance, dimension), (0., 0.), atol=1e-6)
    return {"projected_gaussian_and_bernoulli_sampling_cases": 80,
            "one_sided_width_and_zero_predictor_cases": 6,
            "full_white_zero_gap_obstruction_cases": 8}


def rank_deficient_training():
    rng = np.random.default_rng(20260913)
    cases = 0
    for du, dv in [(1, 1), (2, 3), (3, 2), (2, 2)]:
        for positive_side in [True, False]:
            m = du+dv
            selected = np.arange(du) if positive_side else np.arange(du, m)
            rank = len(selected)
            transport = expm(-.04*generator(rng.normal(size=(dv, du))))
            factor0 = np.sqrt(1.5)*transport[:, selected]
            covariance0 = factor0 @ factor0.T
            gap = sum(support_margins(covariance0, du))
            assert_allclose(gap, 1.5, atol=1e-7)
            target = .12*rng.normal(size=(dv, du))
            assert np.linalg.matrix_rank(target) == min(du, dv)
            residual0 = covariance0[du:, :du]-target
            loss0 = np.sum(residual0**2)/2
            upper0 = np.linalg.norm(covariance0, 2)
            length = np.sqrt(2*loss0)/gap
            upper = upper0*np.exp(2*length)
            minimum_positive = np.linalg.svd(factor0, compute_uv=False)[-1]**2

            def ode(_, state):
                factor = state[:-1].reshape(m, rank)
                covariance = factor @ factor.T
                residual = covariance[du:, :du]-target
                velocity = -generator(residual) @ factor
                return np.r_[velocity.ravel(), np.linalg.norm(residual)]

            times = np.linspace(0., 10/gap, 101)
            solution = solve_ivp(ode, (0., times[-1]), np.r_[factor0.ravel(), 0.],
                                 t_eval=times, method="DOP853", rtol=2e-10, atol=2e-12)
            assert solution.success
            for index, time in enumerate(times):
                factor = solution.y[:-1, index].reshape(m, rank)
                covariance = factor @ factor.T
                loss = np.sum((covariance[du:, :du]-target)**2)/2
                assert loss <= loss0*np.exp(-2*gap*time)+2e-10
                assert_allclose(support_margins(covariance, du),
                                support_margins(covariance0, du), atol=2e-7)
                assert np.linalg.norm(covariance, 2) <= upper+1e-8
                assert np.linalg.svd(factor, compute_uv=False)[-1]**2 >= (
                    minimum_positive*np.exp(-2*length)-1e-8)
            assert solution.y[-1, -1] <= length+1e-8
            final_factor = solution.y[:-1, -1].reshape(m, rank)
            assert np.linalg.matrix_rank(final_factor @ final_factor.T) == rank
            assert np.linalg.norm((final_factor @ final_factor.T)[du:, :du]-target) < 2e-5

            # Use exactly the manuscript's unchanged stepsize and rank certificate.
            alpha = gap/(2*np.sqrt(2))
            gradient_bound = np.sqrt(2*loss0)
            length_gd = np.sqrt(2*loss0)/alpha
            upper_gd = upper0*np.exp(2*length_gd)
            step = .8*min(alpha/(upper_gd*gradient_bound+4.5*upper_gd**2),
                          (1-1/np.sqrt(2))*alpha/(np.exp(2*length_gd)*loss0))
            contraction = 1-2*alpha*step
            iterations = int(np.ceil(14/(-np.log(contraction))))
            factor = factor0.copy()
            accumulated = 0.
            inverse_floor = np.exp(-length_gd/(1-step*gradient_bound))
            for k in range(iterations):
                covariance = factor @ factor.T
                residual = covariance[du:, :du]-target
                accumulated += step*np.linalg.norm(residual)
                factor -= step*generator(residual) @ factor
                if k % 100 == 0 or k+1 == iterations:
                    covariance = factor @ factor.T
                    loss = np.sum((covariance[du:, :du]-target)**2)/2
                    assert loss <= loss0*contraction**(k+1)+1e-11
                    assert np.linalg.norm(covariance, 2) <= upper_gd+1e-8
                    assert np.linalg.svd(factor, compute_uv=False)[-1]**2 >= (
                        minimum_positive*inverse_floor**2-1e-8)
            assert accumulated <= length_gd+1e-8
            assert np.linalg.matrix_rank(covariance) == rank
            cases += 1
    return {"one_sided_singular_gradient_flows": cases,
            "one_sided_singular_factor_GD_paths": cases,
            "full_rank_targets_and_unchanged_rate_certificates": True}


def phase_energy_estimates():
    for _ in range(200):
        du, dv = 2, 3
        m, size = du+dv, du*dv
        phase = spd(2*m)
        a, mixed, b = phase[:m, :m], phase[:m, m:], phase[m:, m:]
        h = spd(size)
        h_values = np.linalg.eigvalsh(h)
        kappa, smoothness = h_values[0], h_values[-1]
        residual = a[du:, :du]-RNG.normal(size=(dv, du))
        error = (h@residual.ravel()).reshape(dv, du)
        potential = residual.ravel()@h@residual.ravel()/2
        kinetic = np.trace(b)/2
        energy = potential+kinetic
        force = generator(error)
        speed = (mixed+mixed.T)[du:, :du]
        damping = RNG.uniform(.1, 5.)
        velocity_dot = -force@mixed-mixed.T@force-2*damping*b
        potential_dot = np.sum(error*speed)
        assert_allclose(potential_dot+np.trace(velocity_dot)/2,
                        -2*damping*kinetic, atol=1e-8)
        predictor_acc = (2*b[du:, :du]-error@a[:du, :du]
                         -a[du:, du:]@error-damping*speed)
        potential_ddot = speed.ravel()@h@speed.ravel()+np.sum(error*predictor_acc)
        squared_force = np.trace(force@a@force)
        amin, bmax = np.linalg.eigvalsh(a)[[0, -1]]
        assert 2*amin*np.sum(error**2) <= squared_force+1e-7
        assert squared_force <= 2*bmax*np.sum(error**2)+1e-7
        assert potential_dot**2 <= 2*kinetic*squared_force+1e-7
        rho = 4*bmax*smoothness+2*np.sqrt(2*smoothness*energy)
        assert potential_ddot+squared_force+damping*potential_dot <= rho*kinetic+1e-7
        correction = damping/(2*(rho+2*damping**2))
        lyapunov = energy+correction*potential_dot
        rate = amin*kappa*damping/(rho+2*damping**2)
        lyapunov_dot = -2*damping*kinetic+correction*potential_ddot
        assert energy/2 <= lyapunov <= 3*energy/2
        assert lyapunov_dot <= -rate*lyapunov+1e-7
    return {"energy_identity_and_conditional_Lyapunov_cases": 200}


def local_gd_and_force_tails():
    rng = np.random.default_rng(20260913)
    for _ in range(200):
        covariance_norm, kappa = 10**rng.uniform(-2, 2, size=2)
        smoothness = kappa*10**rng.uniform(0, 2)
        support_gap = 2*covariance_norm*rng.uniform(.01, 1)
        gap = kappa*(support_gap*rng.uniform(.001, 2)/4)**2
        alpha = support_gap/(2*np.sqrt(2))
        force = np.sqrt(2*smoothness*gap)
        old_length = np.sqrt(2*gap/kappa)/alpha
        length = 4*np.sqrt(gap/kappa)/support_gap
        upper = covariance_norm*np.exp(2*old_length)
        old_terms = [alpha/(upper*force+4.5*smoothness*upper**2),
                     (1-1/np.sqrt(2))*alpha/(np.exp(2*old_length)*gap)]
        terms = support_gap*np.exp(-2*length)*np.array([
            1/(4*covariance_norm*np.sqrt(smoothness*gap)
               +9*np.sqrt(2)*smoothness*covariance_norm**2*np.exp(2*length)),
            (np.sqrt(2)-1)/(4*gap)])
        assert_allclose(length, old_length, rtol=1e-13, atol=0)
        assert_allclose(terms, old_terms, rtol=1e-13, atol=0)
        step = min(terms)
        q = 1-kappa*step*support_gap/np.sqrt(2)
        assert_allclose(q, 1-2*alpha*kappa*step, rtol=1e-14)
        assert 7/9 <= q < 1
    for _ in range(100):
        du, dv = 2, 2
        sigma = RNG.uniform(.4, 2.)
        h = spd(du*dv)
        eigenvalues = np.linalg.eigvalsh(h)
        smoothness, kappa = eigenvalues[-1], eigenvalues[0]
        perturbation = RNG.normal(size=(dv, du))
        step = .8/(sigma**2*smoothness)
        expected = perturbation-2*step*sigma**2*(h@perturbation.ravel()).reshape(dv, du)
        eps = 1e-5
        covariance = sigma**2*np.eye(du+dv)
        covariance[du:, :du] = eps*perturbation
        covariance[:du, du:] = eps*perturbation.T
        error = (h@covariance[du:, :du].ravel()).reshape(dv, du)
        update = np.eye(du+dv)-step*generator(error)
        after = update@covariance@update.T
        assert_allclose(after[du:, :du]/eps, expected, rtol=1e-7, atol=1e-7)
        assert np.max(np.abs(1-2*step*sigma**2*eigenvalues)) < 1
        gap = 1e-20
        alpha = sigma**2/np.sqrt(2)
        length = np.sqrt(2*gap/kappa)/alpha
        upper = sigma**2*np.exp(2*length)
        threshold = min(alpha/(upper*np.sqrt(2*smoothness*gap)+4.5*smoothness*upper**2),
                        (1-1/np.sqrt(2))*alpha/(np.exp(2*length)*gap))
        assert_allclose(threshold, np.sqrt(2)/(9*smoothness*sigma**2), rtol=1e-7)
        covariance = spd(du+dv)
        error = RNG.normal(size=(dv, du))
        squared_speed = np.trace(generator(error)@covariance@generator(error))
        assert squared_speed <= 2*np.linalg.norm(covariance, 2)*np.sum(error**2)+1e-9
    return {"equivalent_explicit_GD_stepsizes_and_rates": 200,
            "local_GD_linearizations_and_dimension_free_force_bounds": 100}


def first_order_length_and_predictor_tails():
    du, dv = 2, 3
    m = du+dv
    signature = np.diag([1.]*du+[-1.]*dv)
    kappa, smoothness = .3, 1.8
    for _ in range(12):
        initial = spd(m)/m+.8*np.eye(m)
        orthogonal, _ = np.linalg.qr(RNG.normal(size=(du*dv, du*dv)))
        curvature = np.array([0., .3, .6, .9, 1.2, 1.8])
        teacher = initial[du:, :du]+.15*RNG.normal(size=(dv, du))

        def gap_and_force(covariance):
            residual = (covariance[du:, :du]-teacher).ravel()
            # Sum squares in the eigenbasis to avoid cancellation in the kernel.
            coordinates = orthogonal.T@residual
            error = (orthogonal@(curvature*coordinates)).reshape(dv, du)
            return np.sum(curvature*coordinates**2)/2, error

        signed = np.linalg.eigvalsh(root(initial)@signature@root(initial))
        positive, negative = signed[signed > 0].min(), -signed[signed < 0].max()
        margin, hidden_floor = positive+negative, min(positive, negative)
        assert np.linalg.eigvalsh(initial).min() <= hidden_floor+1e-12
        assert max(positive, negative) <= np.linalg.norm(initial, 2)+1e-12
        gap0, _ = gap_and_force(initial)
        length = np.sqrt(2*gap0/kappa)/margin
        upper = np.linalg.norm(initial, 2)*np.exp(2*length)
        new_exponent = np.sqrt(2*gap0/kappa)/hidden_floor
        old_exponent = np.sqrt(2*smoothness*gap0)/(kappa*hidden_floor)
        assert_allclose(new_exponent/old_exponent, np.sqrt(kappa/smoothness))

        def rhs(_, state):
            covariance = state[:-1].reshape(m, m)
            _, error = gap_and_force(covariance)
            force = generator(error)
            derivative = -force@covariance-covariance@force
            assert np.linalg.norm(derivative[du:, :du]) <= (
                2*np.linalg.norm(covariance, 2)*np.linalg.norm(error)+1e-10)
            return np.r_[derivative.ravel(), np.linalg.norm(error)]

        times = np.linspace(0, 10/(kappa*margin), 101)
        solution = solve_ivp(rhs, (0, times[-1]), np.r_[initial.ravel(), 0.],
                             t_eval=times, rtol=1e-10, atol=1e-12)
        assert solution.success, solution.message
        states = solution.y[:-1].T.reshape(-1, m, m)
        gaps = np.array([gap_and_force(c)[0] for c in states])
        remaining_length = solution.y[-1, -1]-solution.y[-1]
        finite_bound = np.sqrt(2/kappa)/margin*(np.sqrt(gaps)-np.sqrt(gaps[-1]))
        assert np.all(remaining_length <= finite_bound+1e-8)
        predictor_tail = np.linalg.norm(states[:, du:, :du]-states[-1, du:, :du],
                                        axis=(1, 2))
        assert np.all(predictor_tail <= 2*upper*length*np.exp(-kappa*margin*times)+1e-8)
        assert np.all(gaps <= gap0*np.exp(-2*kappa*margin*times)+1e-9)
    return {"rectangular_quadratic_kernel_trajectories": 12,
            "sharper_length_and_selected_predictor_bounds": True}


def resnet_residual_path_escape():
    times = np.linspace(0, 20, 81)
    factor_cases = 0
    for target in (0., -1.):
        for depth in (1, 2, 4, 8):
            if target == -1 and depth == 1:
                continue
            initial = np.tile(np.sqrt(2)*np.eye(2), (depth, 1, 1))

            def factor_rhs(_, state):
                factors = state.reshape(depth, 2, 2)
                residuals = np.sum(factors[:, 0]*factors[:, 1], axis=1)/2
                layers = 1+residuals/depth
                prefix = np.r_[1., np.cumprod(layers[:-1])]
                suffix = np.r_[np.cumprod(layers[:0:-1])[::-1], 1.]
                forces = (np.prod(layers)-target)*prefix*suffix
                return (-forces[:, None, None]*factors[:, ::-1]).ravel()

            def scalar_rhs(_, residual):
                layer = 1+residual/depth
                return -2*np.sqrt(1+residual**2)*(layer**depth-target)*layer**(depth-1)

            factors = solve_ivp(factor_rhs, (0, times[-1]), initial.ravel(),
                                t_eval=times, rtol=2e-11, atol=2e-12)
            scalar = solve_ivp(scalar_rhs, (0, times[-1]), [0.],
                               t_eval=times, rtol=2e-11, atol=2e-12)
            assert factors.success and scalar.success
            for state, residual in zip(factors.y.T, scalar.y[0]):
                w = state.reshape(depth, 2, 2)
                covariance = w@w.transpose(0, 2, 1)/2
                diagonal = np.sqrt(1+residual**2)
                expected = np.tile([[diagonal, residual], [residual, diagonal]],
                                   (depth, 1, 1))
                assert_allclose(covariance, expected, rtol=2e-8, atol=2e-9)
                assert np.linalg.eigvalsh(covariance).min() > 0
                assert_allclose(sum(support_margins(covariance[0], 1)), 2., rtol=2e-8)
            assert np.all(np.diff(scalar.y[0]) < 1e-10)
            assert scalar.y[0].min() > -depth-1e-10
            if target == -1:
                endpoints = (1+scalar.y[0]/depth)**depth
                assert np.all(endpoints > 0)
                # Evaluate loss minus 1/2 without cancellation near the limit.
                assert np.all(endpoints*(1+.5*endpoints) > 0)
                assert np.max(np.abs(scalar_rhs(0, scalar.y[0]))) <= 4+1e-10
            factor_cases += 1

    depth_limit_cases = 0
    for target in (0., -1.):
        def continuum_rhs(_, residual):
            return -2*np.sqrt(1+residual**2)*(np.exp(residual)-target)*np.exp(residual)

        continuum = solve_ivp(continuum_rhs, (0, times[-1]), [0.],
                              t_eval=times, rtol=2e-12, atol=2e-13)
        assert continuum.success
        assert np.max(np.abs(continuum_rhs(0, continuum.y[0]))) <= 4+1e-10
        errors = []
        for depth in (8, 16, 32, 64, 128):
            def finite_rhs(_, residual):
                layer = 1+residual/depth
                return -2*np.sqrt(1+residual**2)*(layer**depth-target)*layer**(depth-1)

            finite = solve_ivp(finite_rhs, (0, times[-1]), [0.], t_eval=times,
                               rtol=2e-12, atol=2e-13)
            assert finite.success
            errors.append(np.max(np.abs(finite.y-continuum.y)))
            depth_limit_cases += 1
        assert np.all(np.diff(errors) < 0)
        assert errors[-1] < errors[0]/10
    return {"full_factor_vs_scalar_trajectories": factor_cases,
            "positive_covariances_and_constant_support_gap": True,
            "negative_target_orientation_and_speed_bounds": True,
            "compact_time_depth_limit_checks": depth_limit_cases}


def tightened_closed_form_optimizer():
    for _ in range(400):
        force_smoothness = 10.**RNG.uniform(-3, 3)
        force_pl = force_smoothness*10.**RNG.uniform(-4, 0)
        rho = force_smoothness*(1+10.**RNG.uniform(-4, 4))
        damping = np.sqrt(rho)*10.**RNG.uniform(-3, 3)
        discriminant = damping**2+2*rho+2*damping*np.sqrt(2*force_smoothness)
        eta_star = damping/(damping+np.sqrt(discriminant))
        exact = 2*damping*force_pl/(force_pl+.5*(damping+np.sqrt(discriminant))**2)

        def rate(eta):
            a = (1-eta)*force_pl
            b = rho+damping**2/(2*eta)
            epsilon = 2*damping/(a+b)
            q = epsilon*np.sqrt(force_smoothness/2)
            assert 0 < q < 1
            simplified = 2*damping*a/(a+b+damping*np.sqrt(2*force_smoothness))
            assert_allclose(simplified, epsilon*a/(1+q), rtol=1e-12)
            return epsilon*a/(1+q)

        assert 0 < eta_star < .5
        assert_allclose(rate(eta_star), exact, rtol=1e-12)
        optimum = minimize_scalar(lambda eta: -rate(eta)/exact,
                                  bounds=(1e-10, .5), method="bounded",
                                  options={"xatol": 1e-13})
        assert optimum.success
        assert_allclose(-optimum.fun, 1., rtol=1e-8)
        eta_approx = damping/(damping+np.sqrt(damping**2+2*rho))
        assert rate(eta_approx) <= exact*(1+1e-12)
        basic = force_pl*damping/(4*(rho+2*damping**2))
        assert basic <= exact*(1+1e-12)
        for eta in [.7*eta_star, 1.3*eta_star]:
            derivative_sign = ((rho+damping*np.sqrt(2*force_smoothness))*eta**2
                               +damping**2*eta-damping**2/2)
            assert np.sign(derivative_sign) == np.sign(eta-eta_star)
    return {"closed_form_vs_scalar_optimization_cases": 400,
            "compact_fixed_eta_formula_verified": True,
            "original_explicit_and_basic_rates_recovered": True}


if __name__ == "__main__":
    results = {}
    for check in [structure, presentation_alignment, factor_covariance_closure,
                  phase_laws, intrinsic_identities, general_regression,
                  spectral_support_and_lie_algebra, semidefinite_support_gap,
                  rank_adaptive_sampling, rank_deficient_training, phase_energy_estimates,
                  local_gd_and_force_tails, first_order_length_and_predictor_tails,
                  resnet_residual_path_escape, tightened_closed_form_optimizer]:
        results[check.__name__] = check()
        print("PASS", check.__name__, results[check.__name__], flush=True)
    (ROOT / "numerics/manuscript_checks.json").write_text(
        json.dumps({"seed": 20260910, "checks": results}, indent=2) + "\n")
