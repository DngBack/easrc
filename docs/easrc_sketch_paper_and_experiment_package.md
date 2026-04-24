# Explanation-Audited Selective Risk Control for Biomedical Prediction

## Working Title

**Explanation-Audited Selective Risk Control for Biomedical Prediction**

Alternative titles:

1. **When Should a Biomedical Model Abstain? Explanation-Audited Selective Risk Control**
2. **Learning to Reject with Reliable Explanations for Biomedical Classification**
3. **Beyond Confidence: Explanation-Audited Selective Prediction in Biomedical Machine Learning**
4. **EASRC: Explanation-Audited Selective Risk Control for High-Stakes Biomedical Prediction**

---

# 1. Core Paper Thesis

Modern biomedical classifiers can be highly accurate but unsafe to deploy when their predictions are made for unstable, spurious, or biologically unsupported reasons. Selective prediction addresses this issue by allowing a model to abstain, but most existing rejectors are based primarily on predictive confidence, entropy, or uncertainty. In biomedical domains, confidence alone is insufficient: a model may be confident while its explanation is unstable or biologically implausible.

This paper introduces **Explanation-Audited Selective Risk Control**, a framework for learning and calibrating a rejector that accepts a prediction only when both the predicted label and the supporting explanation satisfy reliability constraints.

The central claim is:

> **A biomedical model should abstain not only when it is uncertain, but also when its explanation is unreliable.**

The main technical object is:

> **Explanation-audited selective risk**: the expected classification and explanation unreliability risk among accepted predictions.

The final goal is to maximize coverage while controlling accepted risk:

\[
\max_{\tau} \; \mathbb{E}[A_\tau(X)]
\]

subject to:

\[
\mathbb{E}[\ell_{\text{cls}}(X,Y) \mid A_\tau(X)=1] \leq \alpha
\]

and optionally:

\[
\mathbb{E}[u_{\text{xai}}(X,Y) \mid A_\tau(X)=1] \leq \beta.
\]

---

# 2. One-Sentence Contribution

We propose **EASRC**, an explanation-audited selective prediction framework that learns a rejector from confidence, explanation reliability, and biomedical evidence features, then calibrates its acceptance threshold to provide finite-sample control of classification and explanation risk among accepted predictions.

---

# 3. Target NeurIPS-Level Contribution

The paper should not be framed as a combination of existing modules:

- XAI
- learning to reject
- biomedical features

That framing is weak.

The stronger framing is:

> We introduce a new selective prediction problem where the deployed model must control not only prediction error but also explanation unreliability among accepted samples.

This makes the paper a theory-backed ML contribution rather than a biomedical feature-engineering paper.

---

# 4. Abstract Draft

Selective prediction allows machine learning models to abstain on uncertain inputs, but existing rejectors typically rely on predictive confidence alone. In biomedical domains, confidence is an incomplete notion of reliability: a classifier may be confident while relying on unstable, diffuse, or biologically unsupported evidence. We introduce **Explanation-Audited Selective Risk Control**, a framework for learning when a biomedical model should abstain based on both predictive uncertainty and explanation reliability. Our method trains a rejector using confidence features, attribution-derived explanation features, and biomedical evidence-alignment features. A held-out calibration procedure then selects an acceptance threshold that maximizes coverage subject to finite-sample control of accepted classification risk and explanation risk. We formalize explanation-audited selective risk, derive a finite-sample calibration guarantee for thresholded selectors, and show that confidence-only rejection is a restricted special case of our framework. Across RNA-seq cancer classification and medical image distribution-shift benchmarks, EASRC improves coverage over confidence-only, uncertainty-based, and learned-rejection baselines at the same risk target, while accepting samples with more stable and biologically aligned explanations. These results suggest that reliable biomedical abstention should account not only for uncertainty, but also for the quality of the evidence supporting a prediction.

---

# 5. Introduction: Main Story

## 5.1 Motivation

Biomedical machine learning systems are increasingly used for high-dimensional prediction tasks such as cancer-type classification from gene expression, tumor detection from histopathology, and clinical decision support. In these settings, wrong predictions are costly. However, a prediction may be problematic even when it is numerically confident. A cancer classifier may assign a high softmax probability to a tumor type while relying on noisy, unstable, or biologically implausible features.

Selective prediction provides one approach to safer deployment: the model may either predict or abstain. Classical selective prediction studies the risk–coverage trade-off, where the system aims to maximize the fraction of accepted samples while controlling the error rate among accepted samples. Existing rejectors often rely on max probability, entropy, margin, ensemble uncertainty, or learned selection heads. These scores are useful but incomplete for biomedical deployment because they do not audit the quality of the evidence supporting a prediction.

## 5.2 Core Gap

Existing approaches mostly ask:

> Is the model confident?

This paper asks:

> Is the model confident, and is its explanation reliable enough to justify accepting the prediction?

This distinction matters because confidence and explanation reliability can be empirically decoupled. A model may be high-confidence but explanation-unstable. Such samples can be especially dangerous under distribution shift.

## 5.3 Proposed Solution

We propose **EASRC: Explanation-Audited Selective Risk Control**. The method has three components:

1. A base biomedical predictor.
2. An explanation-aware rejector trained from confidence, attribution, and biomedical evidence features.
3. A held-out calibration procedure that selects a deployment threshold to control accepted risk.

## 5.4 Contributions

The paper should claim the following contributions:

1. **Problem formulation.** We introduce explanation-audited selective prediction, where the goal is to maximize coverage while controlling both classification error and explanation unreliability among accepted predictions.
2. **Method.** We propose EASRC, a rejector that uses confidence, attribution stability, attribution concentration, and biomedical evidence alignment to estimate whether a prediction should be accepted.
3. **Theory.** We provide finite-sample calibration guarantees for thresholded selectors under held-out exchangeable calibration data, and show that confidence-only rejection is a restricted special case of explanation-aware rejection.
4. **Benchmark package.** We evaluate on UCI RNA-seq as a smoke test, TCGA/GDC RNA-seq as the main biomedical benchmark, and Camelyon17-WILDS as an OOD medical validation benchmark.
5. **Empirical finding.** At fixed calibrated risk targets, EASRC improves accepted coverage over confidence-only and learned-rejection baselines while producing accepted predictions with more stable and biologically aligned explanations.

---

# 6. Related Work Structure

## 6.1 Selective Prediction and Learning to Reject

Discuss:

- reject option classification,
- selective prediction,
- risk–coverage trade-off,
- SelectiveNet,
- Deep Gamblers,
- confidence thresholding,
- uncertainty-based rejection.

Positioning:

> Existing methods focus primarily on predictive risk. We extend selective prediction to explanation-audited risk.

## 6.2 Risk Control and Calibration

Discuss:

- split calibration,
- Learn-then-Test,
- Conformal Risk Control,
- finite-sample risk control,
- threshold selection on held-out calibration data.

Positioning:

> We use held-out calibration to provide a deployment guarantee for accepted classification and explanation risk.

## 6.3 Explainable AI

Discuss:

- SHAP,
- Integrated Gradients,
- Gradient × Input,
- Grad-CAM for images,
- explanation stability,
- attribution entropy.

Positioning:

> We do not propose a new explainer. Instead, we audit explanation reliability and use it as part of selective deployment.

## 6.4 Biomedical Interpretability

Discuss:

- gene-expression classification,
- pathway-level interpretation,
- MSigDB / Hallmark pathways,
- biomedical evidence alignment,
- histopathology saliency localization.

Positioning:

> Biological pathway alignment is treated as weak external evidence, not as causal ground truth.

---

# 7. Formal Problem Setup

Let \((X,Y) \sim P\), where \(X\) is a biomedical input and \(Y\) is the label.

Let:

\[
f_\theta: \mathcal{X} \rightarrow \mathcal{Y}
\]

be a trained classifier.

Let:

\[
E_\theta(x)
\]

be an explanation or attribution map for input \(x\).

Let:

\[
s_\phi(x) \in \mathbb{R}
\]

be an accept score produced by the rejector.

For a threshold \(\tau\), define:

\[
A_\tau(x)=\mathbf{1}[s_\phi(x)\geq \tau].
\]

The deployed selective classifier is:

\[
h_\tau(x)=
\begin{cases}
f_\theta(x), & A_\tau(x)=1,\\
\bot, & A_\tau(x)=0.
\end{cases}
\]

Coverage is:

\[
C(\tau)=\mathbb{E}[A_\tau(X)].
\]

Selective classification risk is:

\[
R_{\text{cls}}(\tau)
=
\mathbb{E}[\ell_{\text{cls}}(f_\theta(X),Y) \mid A_\tau(X)=1].
\]

Equivalent ratio form:

\[
R_{\text{cls}}(\tau)=
\frac{\mathbb{E}[A_\tau(X)\ell_{\text{cls}}(f_\theta(X),Y)]}
{\mathbb{E}[A_\tau(X)]}.
\]

---

# 8. Explanation-Audited Risk

Define an explanation unreliability score:

\[
u_{\text{xai}}(x,y) \in [0,1].
\]

This can measure:

- explanation instability,
- attribution entropy,
- diffuse attribution,
- poor biological pathway alignment,
- saliency outside relevant image regions.

Define audited loss:

\[
\ell_\lambda(x,y)=
(1-\lambda)\ell_{\text{cls}}(f_\theta(x),y)+
\lambda u_{\text{xai}}(x,y).
\]

Then the explanation-audited selective risk is:

\[
R_\lambda(\tau)=
\mathbb{E}[\ell_\lambda(X,Y) \mid A_\tau(X)=1].
\]

Main optimization problem:

\[
\max_\tau \; C(\tau)
\]

subject to:

\[
R_\lambda(\tau)\leq \alpha.
\]

Stronger two-constraint version:

\[
R_{\text{cls}}(\tau)\leq \alpha,
\]

\[
R_{\text{xai}}(\tau)\leq \beta.
\]

Recommended main paper version:

> Use the two-constraint version for interpretability, and report weighted audited risk as an additional metric.

---

# 9. Explanation Unreliability Metrics

## 9.1 Attribution Entropy

Let \(a_j(x)\) be the attribution magnitude for feature or gene \(j\). Define:

\[
p_j(x)=\frac{|a_j(x)|}{\sum_k |a_k(x)|+\epsilon}.
\]

Normalized attribution entropy:

\[
H_{\text{attr}}(x)=
-\frac{1}{\log d}\sum_{j=1}^{d}p_j(x)\log p_j(x).
\]

High entropy means the explanation is diffuse.

## 9.2 Attribution Stability

Perturb the input:

\[
x'=x+\eta.
\]

Compute attribution again:

\[
E_\theta(x').
\]

Define stability:

\[
S_{\text{attr}}(x)=
\operatorname{corr}(|E_\theta(x)|, |E_\theta(x')|).
\]

Instability:

\[
u_{\text{stab}}(x)=1-S_{\text{attr}}(x).
\]

## 9.3 Top-k Attribution Mass

Let \(T_k(x)\) be the top-k features by attribution magnitude.

\[
M_k(x)=
\frac{\sum_{j\in T_k(x)}|a_j(x)|}{\sum_{j=1}^{d}|a_j(x)|+\epsilon}.
\]

Low top-k mass can indicate diffuse evidence.

## 9.4 Pathway Alignment for TCGA

Let \(G_y\) be a disease- or class-relevant gene set. Define:

\[
B(x,y)=
\frac{\sum_{j\in G_y}|a_j(x)|}
{\sum_{j=1}^{d}|a_j(x)|+\epsilon}.
\]

Bio-unreliability:

\[
u_{\text{bio}}(x,y)=1-B(x,y).
\]

Important deployment rule:

- rejector input may use predicted-class pathway alignment \(G_{\hat{y}}\),
- calibration and test auditing may use true-class pathway alignment \(G_y\),
- rejector input must not use the true label.

## 9.5 UCI Proxy-Bio Metric

For UCI RNA-seq, real biological pathway alignment may not be valid if the features are anonymized. Therefore:

- use UCI only as smoke test,
- implement a **proxy-bio module** for code testing,
- do not claim real pathway biology on UCI.

Proxy-bio construction:

1. Use rejector-train split only.
2. Compute class-wise top-k discriminative features.
3. Treat these top-k features as pseudo pathway groups.
4. Compute attribution mass inside the predicted-class pseudo group.
5. Compare with random size-matched feature groups.

Name in tables:

- `Bio-only-proxy`,
- `EASRC-Full-proxy`,
- `Proxy group vs random group control`.

---

# 10. EASRC Method

## 10.1 Pipeline

Input:

\[
x
\]

Base predictor:

\[
f_\theta(x)
\]

Prediction features:

- max probability,
- entropy,
- margin,
- energy,
- MC Dropout uncertainty if available.

Explanation module:

\[
E_\theta(x)
\]

Explanation features:

- attribution entropy,
- attribution stability,
- top-k attribution mass,
- explanation unreliability.

Bio features:

- predicted-class pathway alignment,
- random pathway alignment control,
- pathway misalignment.

Rejector:

\[
s_\phi(x)
\]

Calibration:

\[
\hat{\tau}
\]

Deployment:

\[
A_{\hat{\tau}}(x)=\mathbf{1}[s_\phi(x)\geq \hat{\tau}].
\]

## 10.2 Rejector Inputs

For the full model:

```text
max_prob
entropy
margin
energy
attr_entropy
attr_stability
topk_mass
xai_unreliability
pathway_alignment
bio_unreliability
```

For UCI MVP:

```text
max_prob
entropy
margin
energy
attr_entropy
attr_stability
topk_mass
xai_unreliability
proxy_alignment
proxy_bio_unreliability
```

For TCGA full version:

```text
max_prob
entropy
margin
energy
attr_entropy
attr_stability
topk_mass
xai_unreliability
predicted_class_pathway_alignment
predicted_class_pathway_misalignment
```

## 10.3 Rejector Training Objective

Basic version:

\[
z_i=\mathbf{1}[\ell_\lambda(x_i,y_i)\leq \gamma].
\]

Train with BCE:

\[
\mathcal{L}_{\text{BCE}}=
- z_i\log s_\phi(x_i)-(1-z_i)\log(1-s_\phi(x_i)).
\]

Stronger version:

Use pairwise ranking loss. If:

\[
\ell_\lambda(x_i,y_i)<\ell_\lambda(x_j,y_j),
\]

then encourage:

\[
s_\phi(x_i)>s_\phi(x_j).
\]

Ranking loss:

\[
\mathcal{L}_{\text{rank}}=
\max(0,m-(s_\phi(x_i)-s_\phi(x_j))).
\]

Final rejector loss:

\[
\mathcal{L}_{\text{rej}}=
\mathcal{L}_{\text{BCE}}+\kappa\mathcal{L}_{\text{rank}}.
\]

Recommended implementation:

- start with BCE for MVP,
- add ranking loss for paper version.

---

# 11. Calibration Algorithm

Use a held-out calibration set:

\[
\mathcal{D}_{\text{cal}}=\{(x_i,y_i)\}_{i=1}^{n}.
\]

For each threshold \(\tau\), define:

\[
\widehat{\mu}_\tau=
\frac{1}{n}\sum_i A_\tau(x_i).
\]

For classification risk:

\[
\widehat{\nu}_{\tau,\text{cls}}=
\frac{1}{n}\sum_i A_\tau(x_i)\ell_{\text{cls}}(x_i,y_i).
\]

For explanation risk:

\[
\widehat{\nu}_{\tau,\text{xai}}=
\frac{1}{n}\sum_i A_\tau(x_i)u_{\text{xai}}(x_i,y_i).
\]

Empirical risk estimate:

\[
\widehat{R}_{\text{cls}}(\tau)=
\frac{\widehat{\nu}_{\tau,\text{cls}}}{\widehat{\mu}_\tau}.
\]

\[
\widehat{R}_{\text{xai}}(\tau)=
\frac{\widehat{\nu}_{\tau,\text{xai}}}{\widehat{\mu}_\tau}.
\]

UCB version:

\[
\epsilon_n=
\sqrt{\frac{\log(2|\mathcal{T}|/\delta)}{2n}}.
\]

\[
\operatorname{UCB}_{\text{cls}}(\tau)=
\frac{\widehat{\nu}_{\tau,\text{cls}}+\epsilon_n}
{\widehat{\mu}_\tau-\epsilon_n}.
\]

\[
\operatorname{UCB}_{\text{xai}}(\tau)=
\frac{\widehat{\nu}_{\tau,\text{xai}}+\epsilon_n}
{\widehat{\mu}_\tau-\epsilon_n}.
\]

Select:

\[
\hat{\tau}=\arg\max_{\tau\in\mathcal{T}}\widehat{\mu}_\tau
\]

subject to:

\[
\operatorname{UCB}_{\text{cls}}(\tau)\leq \alpha
\]

and:

\[
\operatorname{UCB}_{\text{xai}}(\tau)\leq \beta.
\]

---

# 12. Theory Package

## Theorem 1: Finite-Sample Selective Audited-Risk Control

Assume:

1. calibration samples are exchangeable with test samples,
2. the base predictor and rejector are fixed before calibration,
3. losses are bounded in \([0,1]\),
4. thresholds are selected from a finite grid \(\mathcal{T}\).

Then with probability at least \(1-\delta\), the selected threshold \(\hat{\tau}\) satisfies:

\[
R_{\text{cls}}(\hat{\tau})\leq \alpha
\]

and:

\[
R_{\text{xai}}(\hat{\tau})\leq \beta.
\]

provided the UCB constraints are feasible.

## Proof Sketch

For every threshold, both accepted loss and coverage are bounded random variables. Apply Hoeffding's inequality uniformly over the threshold grid. With high probability, true accepted loss is upper bounded by empirical accepted loss plus \(\epsilon_n\), while true coverage is lower bounded by empirical coverage minus \(\epsilon_n\). Taking the ratio yields the UCB bound. Since the selected threshold satisfies the UCB constraints, the corresponding population risks are controlled.

## Theorem 2: Oracle Selector Thresholds Conditional Audited Risk

Define conditional audited risk:

\[
r_\lambda(x)=\mathbb{E}[\ell_\lambda(X,Y)\mid X=x].
\]

For a fixed coverage level, the optimal selector accepts examples with smallest conditional audited risk:

\[
A_c^\star(x)=\mathbf{1}[r_\lambda(x)\leq q_c].
\]

Interpretation:

> The rejector should learn to rank samples by conditional audited risk. Explanation features are useful if they improve this ranking.

## Proposition 1: Confidence-Only Rejection Is a Special Case

Let \(\mathcal{H}_{\text{conf}}\) be the class of selectors using only confidence features, and \(\mathcal{H}_{\text{xai}}\) be the class using confidence plus explanation features. If:

\[
\mathcal{H}_{\text{conf}}\subseteq \mathcal{H}_{\text{xai}},
\]

then:

\[
C^\star(\mathcal{H}_{\text{xai}},\alpha)
\geq
C^\star(\mathcal{H}_{\text{conf}},\alpha).
\]

Interpretation:

> At the population optimum, explanation-aware selection can match or improve confidence-only selection. The empirical question is whether the learned rejector realizes this advantage.

---

# 13. Experimental Package Overview

## Datasets

1. **UCI RNA-seq**: smoke test and fast high-dimensional biomedical classification.
2. **TCGA/GDC RNA-seq**: main benchmark for real gene identifiers and pathway alignment.
3. **Camelyon17-WILDS**: OOD medical validation with hospital shift.

## Baselines

1. No reject
2. MaxProb
3. Entropy
4. Margin
5. Energy
6. MC Dropout
7. SelectiveNet
8. Deep Gamblers
9. LTT/CRC confidence calibration
10. XAI-only
11. Bio-only
12. EASRC-Full

## Metrics

1. Coverage @ \(\alpha\)
2. Selective classification risk
3. Selective explanation risk
4. Audited risk
5. AURC
6. Violation rate
7. Pathway alignment
8. Explanation stability

## Tables

1. Dataset summary
2. Main coverage @ fixed risk
3. Joint risk control
4. Component ablation
5. Shift evaluation
6. Calibration size
7. Explanation robustness

## Figures

1. Method overview
2. Risk–coverage curve
3. Audited risk–coverage curve
4. Confidence vs explanation unreliability
5. Accepted vs rejected explanation quality
6. Real vs random pathway control
7. Calibration threshold plot
8. OOD shift plot

---

# 14. Dataset Details

## 14.1 UCI RNA-seq: Smoke Test

### Role

UCI RNA-seq is used for fast debugging and high-dimensional selective prediction experiments.

### Task

Multi-class cancer-type classification:

```text
BRCA, KIRC, COAD, LUAD, PRAD
```

### Recommended Split

```text
base_train: 40%
rejector_train: 20%
calibration: 20%
test: 20%
```

Use stratified splits.

Recommended number of seeds:

```text
20 seeds for final UCI reporting
5 seeds during development
```

### UCI-Specific Limitation

If features are anonymized, UCI cannot support real biological pathway alignment. Use UCI for:

- pipeline debugging,
- selective classification risk,
- XAI stability,
- attribution entropy,
- proxy-bio sanity checks.

Do not claim real pathway biology on UCI.

---

## 14.2 TCGA/GDC RNA-seq: Main Benchmark

### Role

TCGA/GDC is the main paper benchmark for real biomedical explanation auditing.

### Task A: Pan-Cancer Classification

Input:

```text
gene expression vector
```

Output:

```text
cancer type
```

Suggested cancer types:

```text
BRCA, LUAD, LUSC, KIRC, KIRP, COAD, READ, PRAD, THCA, LIHC
```

Start with 10 classes.

### Task B: Tumor vs Normal

Binary classification:

```text
tumor / normal
```

Useful as a secondary task.

### Preprocessing

```text
1. Download harmonized gene expression.
2. Keep protein-coding genes.
3. Map Ensembl IDs to gene symbols.
4. Use log2(TPM + 1) or log2(FPKM-UQ + 1).
5. Remove low-variance genes.
6. Keep top 5k or 10k genes.
7. Standardize using train statistics only.
```

### Pathway Data

Use:

- MSigDB Hallmark gene sets,
- canonical pathways,
- Reactome / KEGG / WikiPathways if available.

### Split

In-distribution split:

```text
base_train: 50%
rejector_train: 15%
calibration: 15%
test: 20%
```

Shift split:

```text
train on selected sites / cohorts
test on held-out sites / cohorts
```

---

## 14.3 Camelyon17-WILDS: OOD Medical Validation

### Role

Camelyon17-WILDS tests whether explanation-aware rejection helps under medical distribution shift.

### Task

Binary histopathology patch classification:

```text
tumor / non-tumor
```

### Domain Shift

Hospital split.

### Explanation Metrics for Images

Use:

- Grad-CAM entropy,
- saliency stability under augmentation,
- central-region focus,
- saliency outside relevant region.

### Purpose

This dataset supports the claim:

> Explanation-aware rejection is useful when distribution shift degrades explanation reliability before confidence fully reacts.

---

# 15. Baseline Definitions

## 15.1 No Reject

Always accepts.

Score:

```text
score(x)=1
```

Coverage:

```text
100%
```

Risk:

```text
ordinary test error
```

## 15.2 MaxProb

Score:

```text
max softmax probability
```

## 15.3 Entropy

Score:

```text
negative predictive entropy
```

## 15.4 Margin

Score:

```text
p_top1 - p_top2
```

## 15.5 Energy

For logits \(z\):

```text
energy = -logsumexp(z)
score = -energy
```

## 15.6 MC Dropout

Use dropout at inference:

```text
T stochastic forward passes
mean probability
predictive entropy
```

Score:

```text
negative predictive entropy
```

## 15.7 SelectiveNet

Model has:

```text
classification head
selection head
optional auxiliary head
```

Score:

```text
selection head output
```

## 15.8 Deep Gamblers

Model outputs:

```text
num_classes + 1
```

The extra class is the abstention/reservation class.

Score:

```text
1 - reservation probability
```

## 15.9 LTT/CRC Confidence Calibration

Use confidence score, but choose threshold using finite-sample calibration.

Purpose:

```text
separate score quality from calibration quality
```

## 15.10 XAI-only

Rejector uses only:

```text
attr_entropy
attr_stability
topk_mass
xai_unreliability
```

No confidence features.

## 15.11 Bio-only

For TCGA:

```text
pathway_alignment
bio_unreliability
random_pathway_alignment_gap
```

For UCI:

```text
proxy_alignment
proxy_bio_unreliability
```

## 15.12 EASRC-Full

Uses:

```text
confidence features
+ explanation features
+ bio/pathway features
+ calibration
```

This is the main method.

---

# 16. Metrics

## 16.1 Coverage @ Alpha

For target \(\alpha\), select threshold on calibration set, then report test coverage:

```text
coverage = fraction accepted on test
```

This is the primary metric.

## 16.2 Selective Classification Risk

```text
error rate among accepted samples
```

Formula:

\[
R_{\text{cls}}=\frac{\sum_i A_i \mathbf{1}[\hat{y}_i\neq y_i]}{\sum_i A_i}.
\]

## 16.3 Selective Explanation Risk

```text
mean explanation unreliability among accepted samples
```

Formula:

\[
R_{\text{xai}}=\frac{\sum_i A_i u_{\text{xai}}(x_i,y_i)}{\sum_i A_i}.
\]

## 16.4 Audited Risk

\[
R_{\lambda}=\frac{\sum_i A_i \ell_\lambda(x_i,y_i)}{\sum_i A_i}.
\]

## 16.5 AURC

Area under the selective risk–coverage curve.

Procedure:

```text
1. Sort samples by accept score descending.
2. Accept top-k samples.
3. Compute risk at each k.
4. Integrate risk over coverage.
```

## 16.6 Violation Rate

Across seeds:

```text
violation = test risk > target
violation_rate = mean violation over seeds
```

## 16.7 Pathway Alignment

For TCGA:

```text
fraction of attribution mass inside class-relevant pathway set
```

For UCI:

```text
proxy group alignment only
```

## 16.8 Explanation Stability

Correlation between attribution before and after input perturbation.

---

# 17. Main Tables

## Table 1: Dataset Summary

| Dataset | Modality | Task | Classes | Samples | Features | Role |
|---|---|---:|---:|---:|---:|---|
| UCI RNA-seq | Bulk RNA-seq | Cancer type | 5 | 801 | ~20k | Smoke test |
| TCGA/GDC RNA-seq | Bulk RNA-seq | Cancer type / tumor-normal | 2–10+ | TBD | 5k–20k genes | Main benchmark |
| Camelyon17-WILDS | Histopathology | Tumor patch | 2 | TBD | image | OOD validation |

## Table 2: Main Coverage @ Fixed Risk

| Dataset | Method | Alpha | Test Risk ↓ | Coverage ↑ | AURC ↓ | Violation ↓ |
|---|---|---:|---:|---:|---:|---:|
| UCI | No Reject | 0.05 |  | 1.000 |  |  |
| UCI | MaxProb | 0.05 |  |  |  |  |
| UCI | Entropy | 0.05 |  |  |  |  |
| UCI | Margin | 0.05 |  |  |  |  |
| UCI | Energy | 0.05 |  |  |  |  |
| UCI | MC Dropout | 0.05 |  |  |  |  |
| UCI | SelectiveNet | 0.05 |  |  |  |  |
| UCI | Deep Gamblers | 0.05 |  |  |  |  |
| UCI | LTT/CRC-Conf | 0.05 |  |  |  |  |
| UCI | XAI-only | 0.05 |  |  |  |  |
| UCI | Bio-only-proxy | 0.05 |  |  |  |  |
| UCI | EASRC-Full-proxy | 0.05 |  |  |  |  |
| TCGA | MaxProb | 0.05 |  |  |  |  |
| TCGA | SelectiveNet | 0.05 |  |  |  |  |
| TCGA | EASRC-Full | 0.05 |  |  |  |  |
| Camelyon17 | MaxProb | 0.05 |  |  |  |  |
| Camelyon17 | EASRC-Full | 0.05 |  |  |  |  |

## Table 3: Joint Risk Control

| Dataset | Method | Alpha | Beta | Cls Risk ↓ | XAI Risk ↓ | Audited Risk ↓ | Coverage ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| UCI | MaxProb | 0.05 | 0.30 |  |  |  |  |
| UCI | XAI-only | 0.05 | 0.30 |  |  |  |  |
| UCI | EASRC-Full-proxy | 0.05 | 0.30 |  |  |  |  |
| TCGA | MaxProb | 0.05 | 0.30 |  |  |  |  |
| TCGA | Bio-only | 0.05 | 0.30 |  |  |  |  |
| TCGA | EASRC-Full | 0.05 | 0.30 |  |  |  |  |

## Table 4: Component Ablation

| Dataset | Variant | Cls Risk ↓ | XAI Risk ↓ | Pathway Alignment ↑ | Coverage ↑ | AURC ↓ |
|---|---|---:|---:|---:|---:|---:|
| TCGA | EASRC-Full |  |  |  |  |  |
| TCGA | w/o confidence |  |  |  |  |  |
| TCGA | w/o XAI |  |  |  |  |  |
| TCGA | w/o bio |  |  |  |  |  |
| TCGA | w/o stability |  |  |  |  |  |
| TCGA | w/o attribution entropy |  |  |  |  |  |
| TCGA | random pathways |  |  |  |  |  |
| TCGA | shuffled attributions |  |  |  |  |  |

## Table 5: Shift Evaluation

| Dataset | Shift | Method | Test Risk ↓ | XAI Risk ↓ | Coverage ↑ | Violation ↓ |
|---|---|---|---:|---:|---:|---:|
| TCGA | held-out site | MaxProb |  |  |  |  |
| TCGA | held-out site | SelectiveNet |  |  |  |  |
| TCGA | held-out site | EASRC-Full |  |  |  |  |
| Camelyon17 | hospital shift | MaxProb |  |  |  |  |
| Camelyon17 | hospital shift | SelectiveNet |  |  |  |  |
| Camelyon17 | hospital shift | EASRC-Full |  |  |  |  |

## Table 6: Calibration Size

| Dataset | Cal Size | Method | Test Risk ↓ | Coverage ↑ | Violation ↓ | Slack |
|---|---:|---|---:|---:|---:|---:|
| UCI | 50 | MaxProb |  |  |  |  |
| UCI | 50 | EASRC |  |  |  |  |
| UCI | 100 | MaxProb |  |  |  |  |
| UCI | 100 | EASRC |  |  |  |  |
| TCGA | 250 | MaxProb |  |  |  |  |
| TCGA | 250 | EASRC |  |  |  |  |
| TCGA | 500 | MaxProb |  |  |  |  |
| TCGA | 500 | EASRC |  |  |  |  |

## Table 7: Explanation Robustness

| Dataset | Explainer | Method | XAI Risk ↓ | Pathway Alignment ↑ | Coverage ↑ | AURC ↓ |
|---|---|---|---:|---:|---:|---:|
| UCI | Grad × Input | XAI-only |  |  |  |  |
| UCI | Grad × Input | EASRC |  |  |  |  |
| UCI | Integrated Gradients | XAI-only |  |  |  |  |
| UCI | Integrated Gradients | EASRC |  |  |  |  |
| TCGA | SHAP | EASRC |  |  |  |  |
| TCGA | Integrated Gradients | EASRC |  |  |  |  |
| TCGA | Random attribution | EASRC |  |  |  |  |

---

# 18. Figures

## Figure 1: Method Overview

Show pipeline:

```text
Biomedical input
  ↓
Base predictor
  ↓
Prediction confidence features
  ↓
Explanation module
  ↓
Explanation reliability features
  ↓
Bio/pathway evidence module
  ↓
Rejector score
  ↓
Calibration threshold
  ↓
Accept / Reject
```

Caption should emphasize:

> EASRC accepts predictions only when both the prediction and explanation pass calibrated reliability constraints.

## Figure 2: Risk–Coverage Curve

x-axis:

```text
coverage
```

y-axis:

```text
selective classification risk
```

Methods:

- MaxProb,
- Entropy,
- Margin,
- Energy,
- MC Dropout,
- SelectiveNet,
- Deep Gamblers,
- EASRC.

Desired result:

> EASRC lies below or to the right of confidence-only baselines.

## Figure 3: Audited Risk–Coverage Curve

x-axis:

```text
coverage
```

y-axis:

```text
selective audited risk
```

Purpose:

> Show that EASRC dominates when explanation unreliability is part of the risk.

## Figure 4: Confidence vs Explanation Unreliability

x-axis:

```text
max probability
```

y-axis:

```text
explanation unreliability
```

color:

```text
correct / incorrect
```

Purpose:

> Demonstrate that confidence and explanation reliability are not the same signal.

Key phenomenon:

```text
high-confidence + high-explanation-unreliability samples exist
```

## Figure 5: Accepted vs Rejected Explanation Quality

Boxplots:

- accepted explanation unreliability,
- rejected explanation unreliability,
- accepted pathway alignment,
- rejected pathway alignment.

Expected:

> accepted samples have lower explanation risk and higher pathway alignment.

## Figure 6: Real vs Random Pathway Control

For TCGA:

- real pathway alignment,
- random size-matched pathway alignment,
- shuffled attribution,
- no-pathway variant.

Expected:

> real pathway alignment improves coverage or explanation risk more than random controls.

For UCI:

rename as:

```text
proxy group vs random group control
```

## Figure 7: Calibration Threshold Plot

x-axis:

```text
threshold tau
```

left y-axis:

```text
coverage
```

right y-axis:

```text
risk or UCB risk
```

Add:

- horizontal target line \(\alpha\),
- vertical selected threshold \(\hat{\tau}\).

Purpose:

> Make calibration and risk-control visually understandable.

## Figure 8: OOD Shift Plot

For TCGA and Camelyon17:

x-axis:

```text
method
```

y-axis:

```text
risk / coverage / violation rate under shift
```

Expected:

> EASRC reduces accepted risk or violation under shift compared with confidence-only methods.

---

# 19. Experimental Protocol

## 19.1 Split Discipline

Use four disjoint splits:

```text
D_base: train base predictor
D_rej: train rejector
D_cal: calibrate threshold
D_test: final evaluation
```

Do not use calibration data for training or hyperparameter selection.

Do not use test data for threshold selection.

## 19.2 Base Training

Train base predictor on \(D_{\text{base}}\).

For RNA-seq:

- Logistic regression,
- MLP,
- optional XGBoost / LightGBM.

For Camelyon17:

- ResNet-18,
- ResNet-50,
- DenseNet-121 optional.

## 19.3 Feature Computation

For every sample in \(D_{\text{rej}}\), \(D_{\text{cal}}\), and \(D_{\text{test}}\), compute:

Prediction features:

```text
max_prob
entropy
margin
energy
```

Explanation features:

```text
attr_entropy
attr_stability
topk_mass
xai_unreliability
```

Bio/pathway features:

```text
pathway_alignment
bio_unreliability
random_pathway_alignment
```

## 19.4 Rejector Training

Train rejector only on \(D_{\text{rej}}\).

Rejector variants:

```text
Conf-Rejector
XAI-only
Bio-only
EASRC-Full
```

## 19.5 Calibration

On \(D_{\text{cal}}\):

1. sweep thresholds,
2. compute empirical or UCB risks,
3. select threshold with highest coverage satisfying risk constraints.

Risk targets:

```text
alpha ∈ {0.01, 0.05, 0.10}
beta ∈ {0.20, 0.30, 0.40}
```

Main setting:

```text
alpha = 0.05
beta = 0.30
```

## 19.6 Test Evaluation

On \(D_{\text{test}}\), report:

```text
coverage
selective classification risk
selective explanation risk
audited risk
AURC
violation indicators
pathway alignment
explanation stability
```

Repeat over seeds.

Recommended:

```text
UCI: 20 seeds
TCGA: 5–10 seeds
Camelyon17: 3–5 seeds if expensive
```

---

# 20. Component Ablation Package

## 20.1 Feature Ablations

| Variant | Purpose |
|---|---|
| EASRC-Full | main method |
| w/o confidence | test confidence contribution |
| w/o XAI | test explanation contribution |
| w/o bio | test pathway contribution |
| w/o stability | test stability contribution |
| w/o attr entropy | test entropy contribution |
| XAI-only | test explanation-only signal |
| Bio-only | test pathway-only signal |
| confidence-only | test learned confidence rejector |

## 20.2 Explanation Validity Controls

| Control | Purpose |
|---|---|
| random attribution | negative control |
| shuffled attribution across samples | tests sample-specific explanation signal |
| random pathway sets | tests biological specificity |
| size-matched random gene sets | fair pathway control |
| predicted-class pathway only | deployable version |
| true-class pathway only in audit | evaluation only |

## 20.3 Calibration Ablations

| Calibration Variant | Purpose |
|---|---|
| no calibration | shows necessity |
| empirical threshold | simple baseline |
| UCB threshold | paper method |
| LTT/CRC confidence | calibration-only baseline |
| varying calibration size | sample efficiency |

## 20.4 Shift Ablations

| Shift | Dataset | Purpose |
|---|---|---|
| synthetic noise | UCI | sanity check |
| held-out site | TCGA | real domain shift |
| held-out cancer subtype | TCGA | semantic shift |
| hospital split | Camelyon17 | medical OOD |

---

# 21. Expected Result Pattern

The paper is strong only if these patterns appear.

## Pattern 1: Confidence Is Not Enough

Evidence:

```text
high-confidence samples can have high explanation unreliability
```

Shown by:

```text
confidence vs explanation unreliability scatter
```

## Pattern 2: EASRC Improves Coverage at Fixed Risk

Evidence:

```text
At alpha = 0.05, EASRC has higher coverage than MaxProb, Entropy, Margin, SelectiveNet, and Deep Gamblers.
```

Strong result:

```text
+5 to +15 coverage points over strongest baseline
```

## Pattern 3: Accepted Explanations Are Better

Evidence:

```text
accepted samples have lower xai risk and higher pathway alignment
```

## Pattern 4: Bio Signal Is Real

Evidence:

```text
real pathway alignment > random pathway alignment
shuffled attribution destroys the gain
```

This must be shown on TCGA, not only UCI.

## Pattern 5: Calibration Works

Evidence:

```text
low violation rate across seeds
risk targets mostly respected
```

## Pattern 6: OOD Shift Benefit

Evidence:

```text
under held-out site/hospital shift, EASRC catches explanation-unstable samples better than confidence-only methods
```

---

# 22. Implementation Package

## 22.1 Recommended Repository Structure

```text
easrc/
  configs/
    uci_rnaseq.yaml
    tcga_rnaseq.yaml
    camelyon17.yaml

  src/
    data/
      load_uci.py
      load_tcga.py
      load_camelyon17.py
      split.py
      preprocess.py

    models/
      mlp.py
      mlp_dropout.py
      selectivenet.py
      deep_gambler.py
      resnet.py

    explain/
      grad_input.py
      integrated_gradients.py
      shap_explainer.py
      gradcam.py
      explanation_features.py
      pathway_alignment.py
      proxy_bio.py

    rejectors/
      confidence_scores.py
      mlp_rejector.py
      xai_rejector.py
      bio_rejector.py
      easrc_rejector.py

    selective/
      calibrate.py
      risk_coverage.py
      thresholds.py

    metrics/
      classification.py
      selective.py
      explanation.py
      calibration.py

    tables/
      make_dataset_table.py
      make_main_table.py
      make_joint_table.py
      make_ablation_table.py
      make_shift_table.py
      make_calibration_table.py
      make_explanation_table.py

    figures/
      method_overview.py
      plot_risk_coverage.py
      plot_audited_risk_coverage.py
      plot_confidence_xai.py
      plot_accept_reject_xai.py
      plot_pathway_control.py
      plot_calibration.py
      plot_ood.py

  scripts/
    00_prepare_data.py
    01_train_base.py
    02_compute_features.py
    03_train_rejectors.py
    04_calibrate_eval.py
    05_make_tables.py
    06_make_figures.py
    run_all_uci.sh
    run_all_tcga.sh
    run_all_camelyon17.sh

  results/
    uci_rnaseq/
    tcga_rnaseq/
    camelyon17/

  README.md
```

## 22.2 UCI First Run

```bash
python scripts/00_prepare_data.py --config configs/uci_rnaseq.yaml
python scripts/01_train_base.py --config configs/uci_rnaseq.yaml --seed 0
python scripts/02_compute_features.py --config configs/uci_rnaseq.yaml --seed 0
python scripts/03_train_rejectors.py --config configs/uci_rnaseq.yaml --seed 0
python scripts/04_calibrate_eval.py --config configs/uci_rnaseq.yaml --seed 0
python scripts/05_make_tables.py --config configs/uci_rnaseq.yaml
python scripts/06_make_figures.py --config configs/uci_rnaseq.yaml
```

## 22.3 Output Files Per Seed

```text
results/{dataset}/seed_{seed}/
  split_indices.npz
  base_predictions.csv
  mc_dropout_predictions.csv
  explanation_features.csv
  pathway_features.csv
  baseline_scores.csv
  rejector_scores.csv
  calibration_results.csv
  test_metrics.csv
```

## 22.4 Aggregated Output Files

```text
results/{dataset}/tables/
  dataset_summary.csv
  main_coverage_fixed_risk.csv
  joint_risk_control.csv
  component_ablation.csv
  shift_evaluation.csv
  calibration_size.csv
  explanation_robustness.csv

results/{dataset}/figures/
  risk_coverage.png
  audited_risk_coverage.png
  confidence_xai_scatter.png
  accepted_rejected_xai.png
  pathway_control.png
  calibration_threshold.png
  ood_shift.png
```

---

# 23. Paper Section Outline

## 1. Introduction

- Biomedical models require reliable abstention.
- Confidence alone is insufficient.
- Explanation reliability is a missing signal.
- Introduce explanation-audited selective prediction.
- Summarize EASRC and results.

## 2. Related Work

- Selective prediction and learning to reject.
- Risk control and calibration.
- Explainable AI.
- Biomedical interpretability.

## 3. Problem Formulation

- Selective prediction.
- Explanation-audited risk.
- Coverage maximization under risk constraints.

## 4. Method

- Base model.
- Explanation features.
- Bio/pathway features.
- Rejector training.
- Calibration algorithm.
- Deployment rule.

## 5. Theory

- Oracle selector theorem.
- Finite-sample calibration theorem.
- Confidence-only special-case proposition.

## 6. Experiments

- Dataset summary.
- Baselines.
- Metrics.
- Main results.
- Joint risk control.
- Ablations.
- Shift evaluation.
- Calibration size.
- Explanation robustness.

## 7. Limitations

- No causal explanation guarantee.
- Pathway databases are incomplete.
- Calibration assumes exchangeability.
- OOD performance is empirical, not guaranteed.
- Explanations can be noisy.

## 8. Conclusion

- Selective prediction should audit both prediction and explanation.
- EASRC provides a principled framework.
- Biomedical deployment benefits from explanation-aware abstention.

---

# 24. Limitations and Safe Claims

## Safe Claims

Allowed:

```text
EASRC controls accepted classification and explanation risk under held-out exchangeable calibration.
EASRC improves coverage at fixed calibrated risk in our experiments.
Accepted samples have more stable and pathway-aligned explanations.
Pathway alignment is used as weak biological evidence.
```

## Unsafe Claims

Avoid:

```text
The explanations are causal.
The model discovers true disease mechanisms.
The guarantee holds under arbitrary distribution shift.
Pathway alignment proves biological correctness.
UCI proves real biological pathway reliability.
```

## Correct Language

Use:

```text
pathway-aligned explanation
biomedical evidence alignment
weak external biological prior
explanation unreliability
calibrated accepted risk
```

Avoid:

```text
causal gene discovery
true biological explanation
guaranteed trustworthy AI
```

---

# 25. A* Target Checklist

For a truly strong NeurIPS submission, the final paper should satisfy:

## Theory

- [ ] Definition of explanation-audited selective risk.
- [ ] Calibration theorem for accepted classification and explanation risk.
- [ ] Oracle selector theorem.
- [ ] Confidence-only special-case proposition.
- [ ] Clear assumptions and limitations.

## Experiments

- [ ] UCI RNA-seq smoke test.
- [ ] TCGA/GDC main benchmark.
- [ ] Camelyon17-WILDS OOD validation.
- [ ] All required baselines.
- [ ] Coverage @ alpha table.
- [ ] Joint risk control table.
- [ ] Component ablation.
- [ ] Real vs random pathway control.
- [ ] Calibration size study.
- [ ] Explanation robustness study.
- [ ] OOD shift result.

## Figures

- [ ] Method overview.
- [ ] Risk–coverage curve.
- [ ] Audited risk–coverage curve.
- [ ] Confidence vs explanation unreliability.
- [ ] Accepted vs rejected explanation quality.
- [ ] Real vs random pathway control.
- [ ] Calibration threshold plot.
- [ ] OOD shift plot.

## Empirical Phenomena

- [ ] Confidence and explanation reliability are decoupled.
- [ ] EASRC improves coverage at fixed risk.
- [ ] Accepted samples have better explanations.
- [ ] Real pathway signal beats random pathway controls.
- [ ] Calibration violation rate is low.
- [ ] EASRC helps under shift.

---

# 26. Final Paper Identity

The final paper should be understood as:

```text
Not: XAI + learning-to-reject + bio features.

But: A calibrated selective prediction framework that controls explanation-audited risk among accepted biomedical predictions.
```

The one-line summary for reviewers:

> **EASRC learns when not to predict by auditing not only uncertainty, but also the reliability of the evidence supporting each biomedical prediction.**

This is the core identity of the paper.

