# Response to Reviewer 8YXk

Thank you for identifying these missed references. We were not aware of the relevant results in Min et al. (2021, 2022, 2023) and Xu et al. (2025) when formulating our contribution claims, and apologize for this oversight. We acknowledge that the basic gradient-flow and small-step GD convergence guarantees are not new. We will refocus the paper on the covariance/mean-field formulation, the quantitative sampling theorem, and, most importantly, the momentum and residual-network results, which we believe are new relative to the cited analyses.

## 1. Existing gradient-flow guarantee and normalization

> [...] central result coincides [...]

**We agree: the loss-decay statement of Theorem 3.1 recovers an existing guarantee, and mean-field normalization does not make it a different result.** With $\bar U=U/\sqrt n$, $\bar V=V/\sqrt n$, and $D=\bar U^\top\bar U-\bar V^\top\bar V=-\Gamma/n$, our equations become ordinary factor gradient flow for $\ell(\bar V\bar U^\top)$ in the same time variable. The shared nonzero spectra of $D$ and $J\Sigma$ give

$$
g(\Sigma_0)=[\lambda_{d_u}(D_0)]_+ +[\lambda_{d_v}(-D_0)]_+,
$$

with decreasing eigenvalue order and zero padding. This is the effective imbalance in [Min et al. (2021), Theorem 1](https://proceedings.mlr.press/v139/min21c/min21c.pdf). Moreover, [Min et al. (2023), Theorem 1(1)](https://proceedings.mlr.press/v202/min23d/min23d.pdf) already covers PL losses with exactly our exponent $2\kappa g(\Sigma_0)$. Factor convergence also appears in the earlier quadratic analysis. Neither PL nor our state/rank refinements justify claiming a new basic guarantee.

**Revision.** We will attribute the first-order guarantees, put this correspondence beside Theorem 3.1, and revise the abstract and contributions. Theorem 3.6's smooth-PL scope will likewise be described as a refinement of existing GD theory, not our principal novelty claim.

## 2. Stronger margin-dependent rates

> [...] margin contribution [...]

**Yes, the covariance formulation can express the earlier margin improvement, which is due to Min et al., not our submitted theorem.** For $d_u=d_v=d$, write $S_t=\Sigma_t^{vu}$ and take $\ell(S)=\|S-S_\star\|_F^2/2$. Define $g_0=g(\Sigma_0)$, $\delta_\pm=[\lambda_{\max}(\pm D_0)]_+-a_\pm(\Sigma_0)$, and $\eta=[\sigma_{\min}(S_\star)-\|S_0-S_\star\|_F]_+$. The bound of [Min et al. (2022), Proposition 1 and Theorem 3](https://arxiv.org/pdf/2105.06351) gives

$$
c_\eta=\frac{\sqrt{(\delta_++g_0)^2+4\eta^2}-\delta_+
+\sqrt{(\delta_-+g_0)^2+4\eta^2}-\delta_-}{2},
\qquad \Delta_t\le\Delta_0e^{-2c_\eta t}.
$$

Here $c_\eta=g_0$ when $\eta=0$ and $c_\eta>g_0$ when $\eta>0$. The covariance dissipation formula admits this sharper bound on $\lambda_{\min}(\Sigma^u_t)+\lambda_{\min}(\Sigma^v_t)$; our theorem uses only the conserved part. The earlier result can therefore certify cases with $g_0=0$ that ours does not.

**Revision.** We will add this attributed comparison, distinguish the conserved support gap from the prediction margin, and remove any suggestion that our first-order rate dominates the previous rates.

## 3. Covariance closure versus the earlier lifted analysis

> [...] lifted block-matrix analysis [...]

**The distinction is a width-independent covariance and measure formulation, not new Gram conservation or stacked-factor dynamics.** We cited [Tarmoun et al. (2021)](https://proceedings.mlr.press/v139/tarmoun21a/tarmoun21a.pdf), but insufficiently explained its lifted dynamics and Riccati-type reductions. Our fixed-size state $\Sigma$ describes empirical and population neuron laws through the same exact equation for general split losses. Its spectral laws are covariance versions of the known Gram invariants.

This identifies initialization information independent of neuron labels, connects training to Wasserstein/Bures geometry, and supports population-to-sampling comparisons. Appendix F also characterizes the locally complete smooth covariance laws common to all split losses on the positive-definite simple-spectrum stratum, not all measure-level invariants.

**Revision.** We will explain the relationship to Tarmoun et al. and distinguish the geometric interpretation from new convergence claims in the main text.

## 4. Sampling guarantee and width dependence

> [...] population covariance rank [...]

**We believe Corollary 3.4 is a new quantitative population-to-finite-width rate-transfer statement, although its concentration ingredient is standard.** Let $\widetilde\Sigma_0$ have positive support gap and rank $r_0$, and sample neurons from its subgaussian law, with whitened subgaussian bound $K$. For

$$
n\ge CK^4\varepsilon^{-2}\bigl(r_0+\log(2/\delta)\bigr),
$$

the empirical covariance satisfies $g(\Sigma_0)\ge(1-\varepsilon)g(\widetilde\Sigma_0)$ with probability at least $1-\delta$, and hence $\Delta_t\le\Delta_0\exp[-2\kappa(1-\varepsilon)g(\widetilde\Sigma_0)t]$. The prefactor is the empirical initial loss gap; the statement transfers a rate, not equality of trajectories or initial losses.

Taking $\widetilde\Sigma_0=\mathrm{diag}(2\sigma^2 I_{d_u},0)$ for $d_u\le d_v$ gives $r_0=d_u$ and $g=2\sigma^2$. This is not a new universal minimum-width threshold: it changes the initialization, whereas full-rank white sampling has $r_0=d_u+d_v$.

**Revision.** We will foreground this quantitative contribution, credit standard concentration, and compare initialization distributions as well as widths.

## 5. Residual-path assumptions and the significance of the ResNet results

> [...] cannot be checked at initialization [...]

**We agree that the ResNet theorems are conditional: initialization alone does not establish their propagation bound.** Positive layer gaps do not control the forward/backward products in $\widetilde E_s=P_{>s}^\top\nabla\ell(P_S)P_{<s}^\top$. Under the stated residual bound $\sup_{s,t}\|\Sigma_s^{vu}(t)\|_{\mathrm{op}}\le B<S$, Theorem 4.1 gives, in depth-normalized training time,

$$
\Delta_t\le\Delta_0\exp\!\left[-2\kappa\overline g_S(1-B/S)^{2(S-1)}t\right].
$$

Here $\overline g_S$ is the average initial layer gap. Theorem 4.2 gives the continuous-depth exponent $2\kappa\overline g e^{-2B}$, with $\overline g$ its depth integral, together with almost-everywhere and $L^1$ convergence of covariance profiles. It assumes a globally defined regular training trajectory; it is not an existence or discrete-to-continuum convergence theorem.

This is important because the two-layer conclusion fails at arbitrary depth: layerwise invariants must be combined with propagation and profile control. Our scalar target $-1$, with squared loss and identity layer covariances, has positive gaps but residual escape and limiting loss $1/2$. A neural ODE preserves orientation. Thus an additional hypothesis or restricted conclusion is necessary, though our example does not show that this particular path bound is the only possible hypothesis.

**Revision.** We will label the results as conditional, foreground the continuous-depth existence assumption, and bring the counterexample and depth-uniform rate comparison into the main discussion.

## 6. Does the momentum certificate only cover large initialization?

> [...] only at large initialization scales [...]

**Large position scale is one sufficient regime, not a hypothesis of the general momentum theorem.** Theorem 5.2 allows any positive position covariance and compatible phase covariance satisfying its energy/damping certificate, including zero velocity or nonzero position-velocity correlations. Write $\mathcal E_0=\Delta_0+\mathcal K_0$ for initial mechanical excess energy. At fixed $a_0,b_0,s,\gamma>0$, $\mathfrak S_s(\gamma)\to0$ as $\mathcal E_0\to0$: small energy at fixed scale also works. Conversely, large position variance accommodates fixed, non-small initial loss and kinetic energy.

The substantive difficulty is to obtain a condition on initialization rather than assume position conditioning throughout training. With $A=\Sigma^{(u,v)}$, $D=\Xi^\top A^{-1}$, and kinetic energy $\mathcal K$, we prove

$$
\dot A=DA+AD^\top,\qquad
\|D\|_{\mathrm{op}}\le\sqrt{2\mathcal K/a_s},\qquad
\int_0^t\|D_r\|_{\mathrm{op}}\,\mathrm dr\le\mathfrak S_s(\gamma)<s.
$$

A corrected mechanical energy first gives exponential decay inside $[a_s,b_s]$, where $a_s=\lambda_{\min}(A_0)e^{-2s}$ and $b_s=\lambda_{\max}(A_0)e^{2s}$; the last inequality prevents exit and makes the estimate global. The predictor's PL inequality alone cannot be applied as a global PL inequality for the factor objective. Nor do the conformal spectral identities alone supply this position bound.

**Revision.** We will distinguish the general phase-energy theorem from its large-scale corollary and foreground this bootstrap, including the fixed-scale/small-energy regime and the fact that the certificate is checkable from initialization.

## 7. Momentum rate and the overall contribution

> [...] certified rate [...]

**We do not establish acceleration over gradient flow, and we agree that the certified large-scale rate must not be presented as such.** Our claim is a global convergence guarantee for the inertial dynamics under an explicit initial-energy/damping condition, not an improvement over the first-order exponent. Its significance is that the first-order conserved conditioning argument no longer applies, yet the phase covariance can be controlled globally without assuming its conditioning along the unknown trajectory.

**Revision.** We will compare momentum and gradient-flow rates and distinguish sufficient damping windows from stability thresholds. We will center the paper on sampling, conditional depth-uniform ResNet convergence, and the initialization-based momentum theorem. We respectfully ask that these contributions be assessed on their own merits, with the first-order results properly attributed.

---

# Response to Reviewer BVHf

Thank you for identifying the overlap and recognizing the momentum result. We were not aware of the relevant results in Min et al. (2023) and Xu et al. (2025) when formulating our novelty claims, and apologize. We acknowledge that the basic gradient-flow and small-step GD convergence guarantees are not new. We will distinguish this baseline from the covariance interpretation, quantitative sampling theorem, and momentum and ResNet results, which we believe are the main new contributions.

## 1. Theorem 3.1 versus Min et al.: rate and limit statements

> [...] already established [...] straightforward consequence [...]

**We agree: Theorem 3.1's loss-decay statement is the earlier PL guarantee in covariance notation, and its additional limit statements are not our main novelty.** To make the correspondence explicit, let $\bar U=U/\sqrt n$, $\bar V=V/\sqrt n$, and $D_0=\bar U_0^\top\bar U_0-\bar V_0^\top\bar V_0$. Then

$$
\underline\Delta=[\lambda_{d_u}(D_0)]_+ +[\lambda_{d_v}(-D_0)]_+=g(\Sigma_0).
$$

Assumption A1 of [Min et al. (2023)](https://proceedings.mlr.press/v202/min23d/min23d.pdf) uses $\|\nabla f\|^2\ge\gamma(f-f_*)$, so its $\gamma$ equals $2\kappa$ in our convention. Their Theorem 1(1) therefore has exactly our loss exponent, without requiring strong convexity. Normalizing the factors gives the same flow, not a new training regime for this theorem. We also accept that the state-convergence refinements follow from finite-length/integrable-generator estimates once the rate is available.

**Revision.** We will cite this result immediately alongside Theorem 3.1, provide the explicit dictionary above, and describe our proof as a covariance-based recovery. We will also credit Min et al. (2021, 2022), distinguish their margin-enhanced bounds, and revise all first-order novelty claims in the abstract and contribution statement.

## 2. Theorem 3.6 versus Xu et al.: what actually changes?

> [...] valid extension [...] technical breakthrough [...]

**We agree that the basic GD theorem and the imbalance-control strategy are not new; we do not claim a separate technical breakthrough for Theorem 3.6.** [Xu et al. (2025), Theorems 3.1--3.2](https://arxiv.org/html/2505.11664v1) already establish local PL/descent estimates and preserve useful spectral bounds by controlling discretization-induced imbalance drift. We had cited Xu et al. (2023), but missed the 2025 paper.

The precise difference is that our predictor loss need only be smooth and PL, rather than strongly convex. Our proof controls the total gradient length and the covariance propagator instead of using strong convexity to bound the predictor's distance to a unique minimizer. Under a provisional support bound, descent and PL give a summable gradient-length estimate; this bounds covariance growth and accumulated conservation error, closing the bootstrap. This yields an explicit stepsize depending on $L,\kappa,\Delta_0,g(\Sigma_0)$, and $\|\Sigma_0\|_{\mathrm{op}}$.

This is a refinement, not our principal contribution. We claim neither sharper stepsizes nor coverage of their additional margin-based initializations.

**Revision.** We will compare assumptions and conclusions directly, credit the earlier proof mechanism, and give the smooth-PL refinement proportionate emphasis rather than claim a first GD convergence theorem.

## 3. Phase conservation versus the momentum convergence theorem

> [...] underlying mechanism [...]

**The conformal symplectic mechanism behind Proposition 5.1 is classical; the substantive additional result is the initialization-based global convergence certificate of Theorem 5.2.** The phase generator

$$
M_t=\begin{pmatrix}0&I\\-G_t&-\gamma I\end{pmatrix},\qquad
J_c=\begin{pmatrix}0&I\\-I&0\end{pmatrix}
$$

satisfies $M_t^\top J_c+J_cM_t=-\gamma J_c$ because $G_t$ is symmetric. Differentiation gives the conformal identity; this is not a new general symplectic theorem. We will explicitly relate it to [Franca et al. (2020)](https://proceedings.neurips.cc/paper/2020/file/c4b108f53550f1d5967305a9a8140ddd-Paper.pdf). Our specialization is the exact, self-consistent phase-covariance closure and its transported spectral quantities.

The convergence problem is different. Neither the phase identity nor dissipation of mechanical energy prevents degeneration of the position covariance. Write $A_t=\Sigma_t^{(u,v)}$, $\mathcal E_t=\Delta_t+\mathcal K_t$, and $D_t=\Xi_t^\top A_t^{-1}$. For a proposed spectral interval $a_s I\preceq A_t\preceq b_s I$, a corrected energy yields $\mathcal E_t\le3\mathcal E_0e^{-\lambda_s t}$. Positivity of the phase covariance then gives

$$
\dot A_t=D_tA_t+A_tD_t^\top,\qquad
\int_0^t\|D_r\|_{\mathrm{op}}\,\mathrm dr
\le\frac4{\lambda_s}\sqrt{\frac{3\mathcal E_0}{2a_s}}<s.
$$

This prevents a first exit from the proposed interval. The result is global and uses only initialization and loss parameters to verify the condition; it does not assume unknown trajectory conditioning. Here $a_0=\lambda_{\min}(A_0)$, $b_0=\lambda_{\max}(A_0)$, $a_s=a_0e^{-2s}$, $b_s=b_0e^{2s}$, and

$$
\lambda_s=\frac{a_s\kappa\gamma}{4b_sL+2\sqrt{2L\mathcal E_0}+2\gamma^2}.
$$

The theorem allows general compatible position-velocity correlations and zero velocity. It proves convergence, not acceleration. Its proof uses energy/deformation control, not a claim that conformal invariance alone supplies ellipticity.

**Revision.** We will distinguish the standard geometric identity from the new convergence certificate, add the missing geometric reference, and move this proof roadmap into the main text. We will make the admissible damping condition and its limitations more prominent, rather than lead the momentum section with a novelty claim for Proposition 5.1.

## 4. Is a genuinely mean-field argument required?

> [...] not essential [...] genuine mean-field specific argument [...]

**No infinite-width limit or measure-PDE estimate is required for the main convergence proofs; the mean-field contribution is an exact formulation and quantitative sampling interface.** We agree that the results apply to finite-dimensional matrix factorization and should be presented that way as well.

The reason to retain the measure viewpoint is that a neuron law $\mu$ determines $\Sigma(\mu)=\int ww^\top\,\mathrm d\mu(w)$ and evolves through exactly the same closed covariance equation as every empirical realization. This reduces the state relevant to the predictor from a width-dependent factorization to a matrix of size $d_u+d_v$. A mean-field law is a functional of $\mu$ inducing compatible conservation laws at every width. Appendix F characterizes the smooth covariance-level laws locally on the positive-definite simple-spectrum stratum; it does not classify all measure-level laws.

This viewpoint also gives a concrete population-to-sampling question. Corollary 3.4 transfers the rate of a prescribed population covariance to independent subgaussian samples at width

$$
n\ge CK^4\varepsilon^{-2}\bigl(\mathrm{rank}(\widetilde\Sigma_0)+\log(2/\delta)\bigr),
$$

with at most a factor $1-\varepsilon$ loss in its certified exponent. We believe this rank-adaptive rate-transfer statement is new relative to the cited results, while acknowledging that covariance concentration itself is standard. Width proportional to $\min(d_u,d_v)$ uses a one-sided initialization, not the same full-rank white law at a smaller width.

**Revision.** We will explicitly say that mean-field is an interpretation valid at every width, not an assumption needed for convergence or an asymptotic regime separating us from Min et al. We will introduce the interpretation briefly in the main text, connect it directly to sampling, and retain the technical measure formalism in the appendix.

## 5. What remains as the central contribution?

> [...] novelty is not very clear [...]

**We will reorganize the paper around the sampling guarantee and the momentum and residual-network results, rather than rely on the first-order convergence theorem as its principal contribution.** The momentum result addresses the loss of first-order conserved conditioning through the initial-energy bootstrap above. We believe it is a substantive theorem even though the Lyapunov and bootstrap tools are established.

The ResNet result addresses a separate obstruction: layerwise conditioning need not control the endpoint Jacobian. Under the stated residual-path bound, Theorem 4.1 proves the depth-normalized rate $2\kappa\overline g_S(1-B/S)^{2(S-1)}$, and Theorem 4.2 proves its continuous-depth counterpart $2\kappa\overline g e^{-2B}$ and convergence of the covariance profiles. Here $\overline g_S$ and $\overline g$ average the initial layer gaps. Establishing these statements requires controlling both forward and backward propagation and the layer profiles, rather than applying the two-layer theorem directly to the endpoint. The continuous-depth theorem assumes a globally defined regular trajectory; neither theorem removes the residual-path assumption.

These distinctions matter: unlike two layers, residual architectures can exhibit escape toward singular endpoints despite positive conserved layer gaps. Our orientation-reversing example establishes that a corresponding unconditional claim would be false. The contribution is therefore a quantitative conditional theorem with an explicit obstruction, not a claim to settle global training of arbitrary ResNets.

**Revision.** We will compare theorems individually and separate established facts, reformulations, and new statements. We respectfully ask that sampling and, especially, the momentum certificate and depth-uniform residual analysis be assessed independently of the first-order overlap, which we fully acknowledge.

---

# Response to Reviewer pzdj

Thank you for the careful reading, for identifying the missing reference, and for recognizing the geometric structure in Section 2. We accept the central criticism: the submission does not distinguish established first-order convergence results from its other contributions clearly enough. We were not aware of the relevant results in the missing Min et al. and Xu et al. references when formulating those claims, and we apologize. We acknowledge that the basic gradient-flow and small-step GD convergence guarantees are not new. We will revise the paper accordingly rather than defend the current novelty framing.

## 1. Novelty and technical difficulty of Theorem 3.1

> Core contribution lacks novelty [...]

**We agree that Theorem 3.1 should not be presented as a new first-order convergence guarantee.** [Min et al. (2021), Theorem 1](https://proceedings.mlr.press/v139/min21c/min21c.pdf) gives the corresponding quadratic-loss guarantee. The overlap extends further: [Min et al. (2023), Theorem 1(1)](https://proceedings.mlr.press/v202/min23d/min23d.pdf) already treats PL predictor losses. Thus the quadratic-to-PL distinction does not justify our claim.

Indeed, for $D_0=(U_0^\top U_0-V_0^\top V_0)/n$, the common nonzero spectra of $D_0$ and $J\Sigma_0$ give

$$
g(\Sigma_0)=[\lambda_{d_u}(D_0)]_++[\lambda_{d_v}(-D_0)]_+.
$$

This is their effective imbalance, and factor normalization yields the same flow and exponent. The covariance proof is a useful geometric presentation, but we do not claim that this reformulation makes the guarantee or its main proof mechanism new. We also withdraw any claim to a first basic GD convergence theorem; the smooth-PL distinction from Xu et al. (2025) will be presented as a refinement.

**Revision.** We will attribute these results next to their statements, explicitly identify the equivalence of assumptions and constants, and rewrite the abstract and contributions so that novelty rests elsewhere.

## 2. Precise comparisons and the remaining contribution

> [...] vague descriptions [...]

**We agree that references without theorem-level comparisons are insufficient, and will replace the current positioning with explicit comparisons of assumptions, conclusions, and proof mechanisms.** In particular, the revision will distinguish the following contributions from the established first-order baseline:

- **Sampling, Corollary 3.4:** a rank-adaptive, subgaussian population-to-finite-width guarantee preserving the certified exponent up to $1-\varepsilon$, with width $CK^4\varepsilon^{-2}(r_0+\log(2/\delta))$. We believe this quantitative statement is new relative to the cited analyses, not the concentration theorem used to prove it. The $\min(d_u,d_v)$ specialization changes the initialization to a one-sided law.
- **Momentum, Theorem 5.2:** an explicit initial-energy/damping condition that makes an initially conditional spectral bound self-consistent for all training times. The first-order covariance support bounds are not conserved by inertia. The new work is to combine a corrected mechanical energy with position-covariance deformation control and prevent exit from the conditioning region. The condition allows general compatible phase covariances, including zero initial velocity; it is not a claim of acceleration.
- **Residual networks, Theorems 4.1--4.2:** quantitative rates uniform in depth under the stated residual-path bounds, and convergence of layer covariances, including the continuous-depth profile. The proof must control forward/backward propagation as well as layerwise conditioning. The infinite-depth result assumes a globally defined regular trajectory. The orientation-reversal counterexample shows why positive layer gaps alone cannot yield the corresponding unconditional result.

We believe the momentum and residual-network statements are important because they expose and address obstructions absent from the two-layer first-order problem. The former replaces a trajectory-conditioning assumption with an explicit initialization certificate; the latter identifies sufficient depth-uniform propagation control and a concrete failure mode without it. We do not regard them as merely changing notation in Theorem 3.1, but will be more precise about what each does and does not prove.

**Revision.** We will add these comparisons beside the relevant results, cite the missing works, and make the momentum theorem and conditional ResNet results central to the paper's organization and stated contribution. The existing Tarmoun et al. (2021) and Xu et al. (2023) citations will receive substantive comparisons rather than brief mentions.

## 3. Density, perspective, and repetition

> [...] very dense [...] many variations [...] surprising repetition [...]

**We agree that the current exposition obscures the distinction between the baseline theory and the new problems.** A simpler narrative is: the predictor evolves under a time-dependent preconditioner; known invariants control it for two-layer gradient flow; sampling perturbs its initialization; residual products introduce propagation conditioning; inertia destroys the first-order argument and requires an energy bootstrap.

The overlap between Remarks 3.2 and 3.5 is avoidable. The former illustrates a singular covariance with positive support gap; the latter uses the same geometry for a sampling construction. They should share a single worked example, with the sampling result referring back to it.

**Revision.** We will rewrite the introduction around this narrative, define specialized terminology only when needed, merge the repeated example, and shorten the first-order presentation. Each main theorem will have a brief explanation of its hypothesis, conclusion, and proof mechanism, with secondary bounds left in the appendix.

## 4. Responsibility for AI-assisted writing

> [...] excessive use of a LLM [...]

**We take responsibility for the writing, attribution, and mathematical claims, including all AI-assisted material.** Our disclosure does not excuse the missing comparisons or the repetition you identified. The appropriate response is to correct the substantive positioning and improve the exposition, not merely to change its wording.

**Revision.** We will retain a transparent disclosure, revise the contribution claims against the cited theorem statements, and rewrite the affected passages with particular attention to precise attribution, definitions, and the distinction between proved conclusions and limitations.

## 5. Symmetric-matrix gradient at line 148

> [...] should be [...]

**The correct symmetric-matrix gradient is indeed $\nabla_\Sigma F=G(\Sigma^{vu})/2$.** For a symmetric variation $H$ and $E=\nabla\ell(\Sigma^{vu})$,

$$
\mathrm dF(\Sigma)[H]=\langle E,H^{vu}\rangle_F
=\tfrac12\langle G,H\rangle_F.
$$

The factor $1/2$ accounts for the two off-diagonal blocks in the symmetric Frobenius inner product. With our Wasserstein/Bures normalization, this yields the displayed covariance equation $\dot\Sigma=-(G\Sigma+\Sigma G)$.

**Revision.** We will make the symmetric-gradient convention explicit and ensure that line 148 and every corresponding formula consistently display the factor $1/2$.

## 6. Meaning of Figure 2(b)

> Figure 2(b) [...]

**The panel varies two initial neuron variances, one in each block, to illustrate deterioration of the conserved support gap.** Specifically, with $m=d_u+d_v$, the code initializes

$$
\Sigma_0(r)=I_m+(r-1)(e_1e_1^\top+e_{d_u+1}e_{d_u+1}^\top),
\qquad 0\le r\le1.
$$

All other diagonal entries remain one and the initial predictor is zero. Thus $g(\Sigma_0(r))=2r$, giving certified loss exponent $4\kappa r$ for $r>0$. At $r=0$ this guarantee vanishes; the theorem does not assert convergence in that case. The dashed curve in the panel is the $r=1$ reference rate, not the bound for every curve.

**Revision.** We will replace the ambiguous caption with this explicit initialization, explain the role of $r$ and the dashed reference, and distinguish degeneration of a guarantee from a general assertion of nonconvergence.

## 7. Meaning of a residual layer at line 330

> [...] 'layer' [...]

**A layer in Section 4 means one residual block containing both factors $U_s,V_s$ and its skip connection.** Its action is

$$
x^{s+1}=\left(I_d+\frac{V_sU_s^\top}{Sn}\right)x^s.
$$

Thus $S$ counts residual blocks, not individual factor matrices; each block contains two trainable linear factors.

**Revision.** We will use the term residual block at its first definition and wherever the factor/block distinction could be ambiguous. Together with the changes above, this should make the architecture and the scope of its conditional convergence theorem much easier to follow.

We appreciate the opportunity to correct the attribution and positioning. We respectfully ask that the paper be assessed with its first-order results properly credited and its claims focused on the quantitative sampling theorem, the explicit momentum convergence certificate, and the depth-uniform conditional ResNet analysis.
