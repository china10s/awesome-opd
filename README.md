# Awesome On-Policy Distillation [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of resources dedicated to **On-Policy Distillation (OPD)** —
> a family of post-training paradigms that fuse the *sample efficiency of
> distillation* with the *distribution matching of on-policy reinforcement
> learning*.
>
> Inspired by the layout of [aikorea/awesome-rl](https://github.com/aikorea/awesome-rl).

On-Policy Distillation trains a student model on its **own rollouts**, using a
teacher (or the model itself, conditioned on privileged context) to provide
*dense, token-level* supervision on those rollouts — instead of (a) relying
on off-policy teacher trajectories like classical SFT-distillation, or
(b) using only sparse scalar rewards like vanilla RLHF/RLVR.

A typical OPD step is:

```
1.  student π_θ samples a trajectory  τ ~ π_θ(·|x)
2.  teacher π_T (or self-teacher π_θ conditioned on demos / feedback / privileged info)
    scores τ token-by-token, producing a target distribution p_T(·|τ_<t, x)
3.  optimize θ to minimize a divergence  D(p_T || π_θ)  on the on-policy τ
    (reverse-KL / forward-KL / JSD / f-divergence)
```

The framework unifies many recent ideas — GKD, MiniLLM, SDFT, SDPO, DASD,
MOPD, OPSD, OPCD, OPSDC, TAID, SPIN, Veto, EOPD, REOPOLD, Video-OPD,
VLA-OPD, **Lightning OPD**, **DP-OPD**, GOLD, Cascade Distillation,
SKD / SpecKD, **On-Policy Cross-Stage Distillation (OPCSD)**,
**Black-box / outcome-based OPD** (GAD, OVD, ORPO-Distill),
**Inside-RL OPD hybrids** (AlignDistil, LUFFY, NPO, KEPO, BOND,
RLAD, KDRL, RLSD, HDPO), **Iterative self-bootstrapping**
(SPIN, rStar, rStar-Math, rStar2-Agent), **Speculative-decoding
drafter distillation** (EAGLE-3, OSD, SpecForge, DistillSpec, SpecKD,
ReSpec, DVI, CORAL), and many domain-specific OPD recipes — and has
been adopted in the post-training recipes of **GLM-5 / GLM-4.5 / 4.6**,
**Qwen3 / Qwen3-Coder / Qwen3.5-Omni**, **MiMo-V2-Flash**, **Ministral 3**,
**DASD-4B-Thinking**, **Gemma 2 / 3**, **Baichuan-M3**, **HY-MT**,
**HY-Embodied**, **Nemotron Cascade 2**, **KAT-Coder-V2**,
**DeepSeek-V4** and many others.

> 🔍 **Strict OPD definition (after [thinkwee/AwesomeOPD](https://github.com/thinkwee/AwesomeOPD))**:
> a method qualifies as OPD if it satisfies **C1** *(student samples its
> own trajectories during training)* and **C2** *(teacher provides
> per-token / per-trajectory supervision on those samples)*. Methods
> that only partially satisfy (e.g., sequence-level RL with KD anchor,
> or offline cached teacher logprobs) are flagged below.

---

## Contributing

Pull requests welcome! Please keep entries short, link to the paper / code /
blog directly, and place each work in the most specific subsection.

---

## Table of Contents

- [Theory](#theory)
  - [Surveys & Long-form Articles](#surveys--long-form-articles)
  - [Mechanism & Phenomenology Studies](#mechanism--phenomenology-studies)
  - [Foundational Papers](#foundational-papers)
- [Methods](#methods)
  - [Self-Distillation as On-Policy Learning](#self-distillation-as-on-policy-learning)
  - [Iterative Self-Bootstrapping](#iterative-self-bootstrapping)
  - [On-Policy KD from a Larger Teacher](#on-policy-kd-from-a-larger-teacher)
  - [Black-Box / Outcome-Based OPD](#black-box--outcome-based-opd)
  - [Inside-RL OPD Hybrids](#inside-rl-opd-hybrids)
  - [Stability & Loss Engineering](#stability--loss-engineering)
  - [Speculative / Hybrid Student-Teacher Sampling](#speculative--hybrid-student-teacher-sampling)
  - [Speculative-Decoding Drafter Distillation](#speculative-decoding-drafter-distillation)
  - [Sequence-Level & Distribution-Aligned Distillation](#sequence-level--distribution-aligned-distillation)
  - [Context Distillation on Student Rollouts](#context-distillation-on-student-rollouts)
  - [Multi-Teacher On-Policy Distillation](#multi-teacher-on-policy-distillation)
  - [Strong-to-Weak Distillation Pipelines](#strong-to-weak-distillation-pipelines)
  - [Cross-Stage / Anti-Forgetting Distillation](#cross-stage--anti-forgetting-distillation)
  - [Interpolated / Curriculum Distillation](#interpolated--curriculum-distillation)
  - [Self-Play as Implicit On-Policy Distillation](#self-play-as-implicit-on-policy-distillation)
  - [Cross-Tokenizer / Cross-Family Distillation](#cross-tokenizer--cross-family-distillation)
  - [Multimodal & Embodied On-Policy Distillation](#multimodal--embodied-on-policy-distillation)
  - [Offline / Resource-Efficient OPD](#offline--resource-efficient-opd)
  - [Privacy-Preserving / DP On-Policy Distillation](#privacy-preserving--dp-on-policy-distillation)
  - [Cascade / Pruning + Distillation](#cascade--pruning--distillation)
  - [RL × Distillation Connections (Inverse RL, Imitation)](#rl--distillation-connections-inverse-rl-imitation)
- [Applications](#applications)
  - [Long-CoT Reasoning](#long-cot-reasoning)
  - [Reasoning Compression / Token Efficiency](#reasoning-compression--token-efficiency)
  - [Agentic & Tool Use](#agentic--tool-use)
  - [Continual Learning & Catastrophic Forgetting](#continual-learning--catastrophic-forgetting)
  - [Model Compression / Small Language Models](#model-compression--small-language-models)
  - [Multimodal & Video](#multimodal--video)
  - [Robotics & Embodied AI](#robotics--embodied-ai)
  - [Privacy-Preserving Compression](#privacy-preserving-compression)
  - [Code & Math](#code--math)
- [Best Practices & Recipes](#best-practices--recipes)
- [Models & Technical Reports](#models--technical-reports)
- [Codes & Frameworks](#codes--frameworks)
- [Tutorials & Blogs](#tutorials--blogs)
- [Benchmarks & Datasets](#benchmarks--datasets)
- [Limitations & Open Problems](#limitations--open-problems)

---

## Theory

### Surveys & Long-form Articles

- **Mingyang Song, Zitai Wang, Mao Yang et al. A Survey of On-Policy
  Distillation for Large Language Models.** arXiv, Apr 2026.
  [[arXiv:2604.00626]](https://arxiv.org/abs/2604.00626)
  *The first systematic survey of OPD for LLMs. Introduces a unified
  **f-divergence framework over on-policy samples**, and organizes the
  landscape along three orthogonal axes: **feedback signal** (logit-based /
  outcome-based / hybrid), **sampling strategy** (token / sequence / mixed),
  and **teacher source** (external / self / multi-teacher).*

- **万字长文总结 RL / On-Policy Distillation 的一些进展** (知乎, 2026)
  [[Article]](https://zhuanlan.zhihu.com/p/2004506304065065334)
  *Comprehensive Chinese-language survey covering the recent surge of OPD
  methods, their relationship to RLVR, credit assignment, dense vs. sparse
  reward signals, and a taxonomy of representative works
  (SDFT / SDPO / DASD / MOPD / GKD / MiniLLM …).*

- **长文总结：近半年 On-Policy Distillation 的三大主流方向** (知乎, May 2026)
  [[Article]](https://zhuanlan.zhihu.com/p/2020191969205306820)
  *9-paper deep-dive that organizes the recent OPD wave into three lines:
  **(1) Stability & diversity** — Veto, EOPD, REOPOLD;
  **(2) Self-distillation with privileged context** — OPSD, SDFT, SDPO,
  OPSDC; **(3) Scenario expansion** — OPCD and Video-OPD. Includes
  side-by-side comparison tables and the most up-to-date taxonomy of OPD
  failure modes.*

- **近一月 On-Policy-Distillation 进展总结：密集奖励的隐患与对策**
  (知乎, May 2026) [[Article]](https://zhuanlan.zhihu.com/p/2028518805890974874)
  *5-paper "failure-modes month" review covering Rethinking OPD,
  StableOPD, Revisiting OPD, SCOPE and VLA-OPD. Argues that as Qwen3 /
  GLM-5 / MiMo-V2 push OPD into industrial production, **dense token-level
  reward turns out to be a deceptively unsafe free lunch** — repetition
  collapse, reverse-distillation paradoxes, single-token sampling
  artifacts, Pass@k destruction, and toxic prefix traps all surface
  simultaneously. The companion piece to the May survey above; together
  they are the best entry point to the **OPD-failure-mode literature**.*

- **On-Policy Distillation 是什么？如何做？** (知乎 / kxzxvbk, BUAA, Feb 2026)
  [[Article]](https://zhuanlan.zhihu.com/p/2000612721868177979) ·
  [[Mirror]](https://qingkeai.online/archives/How%20do%20On-Policy%20Distillation)
  *Tutorial-style introduction that derives both the **simple sampling
  estimator** of reverse-KL and the **vocabulary-summed gradient
  estimator** used in MiniLLM / GKD. Then walks through the self-distillation
  recipe (`π_teacher(y|x) = π_θ(y|x, c)`) that underpins OPSD / SDFT.*

- **Kevin Lu & Thinking Machines Lab. On-Policy Distillation.**
  Thinking Machines Lab: Connectionism, Oct 27, 2025.
  [[Blog]](https://thinkingmachines.ai/blog/on-policy-distillation)
  [[Code (Tinker Cookbook)]](https://github.com/thinking-machines-lab/tinker-cookbook)
  *The de-facto reference blog post. Demonstrates that sampling from the
  student and scoring with reverse-KL against a teacher matches full-RL gains
  on AIME'24 (~65 %) at roughly **1/100 of the compute**, and can also be used
  for personalization / continual learning without forgetting.*

- **On-Policy Distillation: Cheap Accuracy, Real Gains** — Medium writeup
  unpacking the Thinking Machines blog. [[Medium]](https://medium.com/@maheshlambe/mira-murati-thiking-machines-on-policy-distillation-cheap-accuracy-real-gains-85ac523b20f7)

### Mechanism & Phenomenology Studies

- **Yaxuan Li, Yuxin Zuo, Bingxiang He et al. Rethinking On-Policy
  Distillation of Large Language Models: Phenomenology, Mechanism, and
  Recipe.** arXiv, Apr 2026 (THUNLP).
  [[arXiv:2604.13016]](https://arxiv.org/abs/2604.13016) ·
  [[Code]](https://github.com/thunlp/OPD)
  *The first systematic investigation of **OPD training dynamics**.
  Identifies two conditions that govern OPD success: (i) student and teacher
  must share **compatible thinking patterns**, and (ii) even with consistent
  patterns and higher teacher scores, the teacher must offer **genuinely new
  capabilities** beyond what the student has seen during training. Validated
  via **weak-to-strong reverse distillation**: same-family 1.5B and 7B
  teachers are *distributionally indistinguishable* from the student's
  perspective, so naively distilling from a "stronger" same-family model can
  fail. At the token level, successful OPD shows progressive alignment on
  high-probability tokens at student-visited states, with a **small shared
  token set (97–99 % of the probability mass)** doing all the work. Proposes
  two rescue strategies for failing runs — **off-policy cold start** and
  **teacher-aligned prompt selection** — and warns that OPD's "free lunch"
  of dense token-level reward **does not obviously scale to long-horizon
  distillation**. The paper that crystallizes the OPD design space.*

- **Siyan Zhao et al. "Style tokens dominate the OPD training signal."**
  Finding from OPSD v3 ([arXiv:2601.18734v3](https://arxiv.org/abs/2601.18734v3), Mar 2026).
  *Stylistic reasoning tokens such as `wait`, `think` and other "meta-cognitive
  scaffolding" exhibit **6–15× higher KL divergence** than math-content tokens
  in on-policy self-distillation, and left un-checked they dominate the
  gradient. A simple **per-token JSD clip** (0.05) stabilizes training and
  improves downstream accuracy — a general lesson also applicable to SDPO /
  SDFT / GKD.*

- **Microsoft Research × KAIST × SNU. The Hidden Cost of Self-Distillation.**
  arXiv / TechTalks, Apr 2026.
  [[Article]](https://bdtechtalks.substack.com/p/the-hidden-trap-of-llms-self-distillation)
  *Warns that self-distillation (SDPO, SDFT) can **suppress exploration and
  self-correction**, producing up to **40 % accuracy drops on OOD problems**.
  Argues that models must be exposed to varied uncertainty levels during
  training to preserve robust reasoning.*

### Foundational Papers

- Geoffrey Hinton, Oriol Vinyals, Jeff Dean.
  **Distilling the Knowledge in a Neural Network.**
  NeurIPS Deep Learning Workshop, 2015.
  [[arXiv:1503.02531]](https://arxiv.org/abs/1503.02531) *— the original KD formulation.*

- Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell.
  **A Reduction of Imitation Learning and Structured Prediction to No-Regret
  Online Learning (DAgger).** AISTATS, 2011.
  [[arXiv:1011.0686]](https://arxiv.org/abs/1011.0686)
  *— canonical reference for why **on-policy** demonstrations beat off-policy
  ones; OPD is the modern, distribution-level analog.*

- Andrei A. Rusu, Sergio Gomez Colmenarejo, Caglar Gulcehre et al.
  **Policy Distillation.** ICLR, 2016.
  [[arXiv:1511.06295]](https://arxiv.org/abs/1511.06295)
  *— the original **Policy Distillation** paper in DeepRL: distill DQN
  policies via KL-on-softmax with temperature, up to **15×** compression.
  Every modern LLM OPD work ultimately inherits this formulation.*

- Yoon Kim, Alexander M. Rush. **Sequence-Level Knowledge Distillation.**
  EMNLP, 2016. [[arXiv:1606.07947]](https://arxiv.org/abs/1606.07947)
  *— introduces sequence-level KD; serves as the off-policy baseline that
  DASD critiques.*

- Tommaso Furlanello et al. **Born-Again Neural Networks.** ICML, 2018.
  [[arXiv:1805.04770]](https://arxiv.org/abs/1805.04770)
  *— the earliest "self-distillation" paper to show students can outperform
  their teachers.*

- Hossein Mobahi, Mehrdad Farajtabar, Peter L. Bartlett.
  **Self-Distillation Amplifies Regularization in Hilbert Space.**
  NeurIPS, 2020. [[arXiv:2002.05715]](https://arxiv.org/abs/2002.05715)
  *— theoretical justification for repeated self-distillation as implicit
  regularization.*

- Yuntian Deng et al. **Distilling Policy Distillation.**
  AISTATS, 2019. *— unifies policy distillation, DAgger and KD into a common
  expected-divergence framework.*

- Yee Whye Teh et al. **Distral: Robust Multitask Reinforcement Learning.**
  NeurIPS, 2017. [[arXiv:1707.04175]](https://arxiv.org/abs/1707.04175)
  *— distills a shared "centroid" policy from multiple task-specific policies
  via KL regularization; the spiritual ancestor of **Multi-Teacher On-Policy
  Distillation** (MOPD) for LLMs.*

- Emilio Parisotto, Jimmy Lei Ba, Ruslan Salakhutdinov. **Actor-Mimic:
  Deep Multitask and Transfer Reinforcement Learning.** ICLR, 2016.
  [[arXiv:1511.06342]](https://arxiv.org/abs/1511.06342)
  *— pre-dates Policy Distillation by a few months and introduces the idea of
  mimicking multiple expert Q-networks on-policy, another root of MOPD.*

---

## Methods

### Self-Distillation as On-Policy Learning

> Use the model itself (often conditioned on extra context, demonstrations,
> feedback, or privileged info) as the teacher, then distill back into the
> unconditioned policy.

- **SDFT — Self-Distillation Fine-Tuning** (MIT, 2026)
  Idan Shenfeld, Mehul Damani, Jonas Hübotter, Pulkit Agrawal.
  **Self-Distillation Enables Continual Learning.**
  [[arXiv:2601.19897]](https://arxiv.org/abs/2601.19897) ·
  [[Blog (CN)]](https://mp.weixin.qq.com/s/7BKxOoS5iqah27OrxY9BzA)
  *Frames SFT as **off-policy imitation** and shows that conditioning the
  model on demonstrations turns it into its own on-policy teacher. Position
  the method as **inverse-RL via in-context demonstrations**. Outperforms SFT
  on every continual-learning benchmark with drastically reduced forgetting.*

- **SDPO — Self-Distillation Policy Optimization** (ETH / MIT, 2026)
  **Reinforcement Learning via Self-Distillation.**
  [[arXiv:2601.20802]](https://arxiv.org/abs/2601.20802) ·
  [[Blog (CN)]](https://mp.weixin.qq.com/s/eRqxe7qcRxWJYE86UXv8Nw)
  *Tackles the credit-assignment bottleneck of RLVR by converting **rich
  textual feedback** (runtime errors, judge critiques) into a dense
  token-level signal: the feedback-conditioned model becomes the teacher,
  distilled back into the un-conditioned policy. Beats RLVR on LiveCodeBench
  v6, scientific reasoning and tool use; at test-time reaches the same
  discovery rate as best-of-k with **3× fewer attempts**.*

- **OPSD — On-Policy Self-Distillation** (UCLA / Meta, 2026; v3 Mar 2026)
  Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang,
  Feiyu Chen, Aditya Grover.
  **Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language
  Models.** [[arXiv:2601.18734v3]](https://arxiv.org/abs/2601.18734v3) ·
  [[Code]](https://github.com/siyan-zhao/OPSD)
  *A single LLM acts as **both teacher and student** by conditioning on
  different contexts: the teacher sees a **privileged verified reasoning
  trace**, the student sees only the question. Training minimizes per-token
  JSD between the two distributions **over the student's own rollouts**.
  Built on top of TRL's experimental GOLD trainer; ships with SFT / GRPO
  baselines. **Qwen3-1.7B training finishes in ~15 min on 4×H100**, peaking
  within 100 steps (AIME'24 51.5 % → 57.2 %, AIME'25 36.7 % → 43.9 %).
  Key v3 contribution: **per-token JSD clipping** (`jsd_token_clip=0.05`) —
  the authors discover that stylistic tokens (e.g., `wait`, `think`) exhibit
  **6–15× higher KL** than math-content tokens and dominate the training
  signal; clipping eliminates the instability.*

- **OPSDC — On-Policy Self-Distillation for Reasoning Compression** (2026)
  [[arXiv:2603.05433]](https://arxiv.org/abs/2603.05433) ·
  [[Code]](https://github.com/HJSang/OPSD_Reasoning_Compression)
  *Inverts the usual OPD direction: the privileged context is a **"be
  concise" instruction `c`** and the goal is to **shorten** the student's
  reasoning rather than improve its content. Uses reverse-KL on student
  rollouts with periodic teacher-weight refresh (every M=50 steps).
  Striking finding — **less reasoning, more accuracy**: on Qwen3-14B, MATH-500
  jumps from 70.0 % → 86.1 % accuracy while using **56.5 % fewer tokens**;
  AIME'24 +10 pp at 41 % compression. Theoretically the accuracy gain scales
  as \((1-p_{\text{err}})^{-(1-\alpha)L}\). Crucially shows **forward KL
  collapses on every teacher refresh** while reverse KL is stable — a clean
  empirical case study for the loss-engineering work below.*

- **OEL — Online Experiential Learning** (Microsoft Research, Mar 2026)
  [[arXiv:2603.16856]](https://arxiv.org/abs/2603.16856) ·
  [[Code (LMOps `/oel`)]](https://github.com/microsoft/LMOps)
  *Self-distillation on **interactive game / planning trajectories**:
  the same model interacts with an environment, the privileged-context
  teacher is the same model conditioned on richer environment feedback
  (executed action results, score history); per-token RKL on student
  rollouts. Demonstrates OPSD on long-horizon decision making rather
  than single-turn QA.*

- **OPCD — On-Policy Context Distillation** (Microsoft Research, Feb 2026)
  [[arXiv:2602.12275]](https://arxiv.org/abs/2602.12275) ·
  [[Code (LMOps `/opcd`)]](https://github.com/microsoft/LMOps)
  *Privileged-context teacher is the same model with **in-context
  knowledge augmentation** (retrieved docs, system prompt, scratchpad);
  student is unconditioned. Internalises the context so the student
  remains faithful even after the context is removed. Canonical
  reference for "context distillation on student rollouts".*

- **Apple SSD — Embarrassingly Simple Self-Distillation** (Apple MLR, Apr 2026)
  [[arXiv:2604.01193]](https://arxiv.org/abs/2604.01193) ·
  [[Code]](https://github.com/apple/ml-ssd)
  *Pure-sample-then-SFT recipe: the same model **with a different
  decoding config** (temperature / top-p / truncation) acts as the
  "teacher" by producing the targets, then the model SFTs on its own
  samples. Degenerate OPSD (no KL signal — supervision is hard CE),
  but ships strong code-generation results despite the simplicity. A
  useful empirical lower-bound on what OPSD costs vs gains.*

- **MTP-SD — Multi-Token Prediction via Self-Distillation** (UMD / LLNL, Feb 2026)
  [[arXiv:2602.06019]](https://arxiv.org/abs/2602.06019) ·
  [[Code]](https://github.com/jwkirchenbauer/mtp-lm)
  *Privileged-context teacher = same model with **multi-token
  prediction heads** active; student = single-token prediction. RKL on
  student rollouts. Improves both the next-token model **and** the MTP
  drafter, naturally fitting into speculative-decoding pipelines.*

- **GATES — Self-Distillation under Privileged Context** (UMD, Feb 2026)
  [[arXiv:2602.20574]](https://arxiv.org/abs/2602.20574)
  *Both tutor (= same model conditioned on full document) and student
  (no document) sample rollouts; tutor-consensus-gated RKL only updates
  the student where tutors agree. Document-QA application. ⚠️ Authors'
  own ablation: the on-policy student-rollout leg contributes only
  "modest additional improvement" on top of off-policy distillation —
  mixed OPSD.*

- **OPSDL — On-Policy Self-Distillation for Long-Context LMs** (Baidu, Apr 2026)
  [[arXiv:2604.17535]](https://arxiv.org/abs/2604.17535)
  *Privileged context = the **same model with the long context
  available**; student = same model with truncated context. Per-token
  RKL on student rollouts forces the short-context model to behave as
  if it had seen the full document. Specifically targets long-context
  reading-comprehension benchmarks where straight context truncation
  destroys accuracy.*

- **SD-Zero — Self-Revision turns binary rewards into dense supervision**
  (Princeton / Toronto / CMU, Apr 2026)
  [[arXiv:2604.12002]](https://arxiv.org/abs/2604.12002)
  *Single model plays both **Generator** (samples response on its own)
  and **Reviser** (re-samples response conditioned on the original
  response **plus its scalar binary reward**). The reviser's
  reward-conditioned token distribution becomes dense per-token
  supervision over the generator's response. C1 ✓ + C2 ✓; compared
  head-to-head with **GRPO** on Qwen3-4B-Instruct / Olmo-3-7B-Instruct,
  gains ≥10 % over base. Notably **not RL** — the reward is a
  conditioning signal, not a return; no policy gradient. The cleanest
  recipe for **converting RLVR-style binary rewards into OPSD without
  paying the RL stability tax**.*

- **π-Play — Privileged Self-Distillation for Search Agents**
  (CASIA / UCAS / Meituan, Apr 2026)
  [[arXiv:2604.14054]](https://arxiv.org/abs/2604.14054)
  *Self-play loop *examiner ↔ student/teacher* with **no external
  data**. The teacher is conditioned on the **Question Construction
  Path (QCP)** — the reverse-direction artifact emitted by the examiner
  when generating the task; the student is not. Teacher = EMA copy of
  student (τ=0.05); per-token RKL on student rollouts. **Data-free
  π-Play surpasses fully-supervised search agents** (NQ, TriviaQA,
  HotpotQA, 2WikiMQA, MuSiQue) and is 2–3× more sample-efficient than
  vanilla self-play. Converts sparse-reward self-play into dense
  per-token OPSD supervision.*

- **Skill-SD — Skill-conditioned OPSD for Multi-Turn Agents**
  (UCAS / CUHK / vivo AI Lab, Apr 2026)
  [[arXiv:2604.10674]](https://arxiv.org/abs/2604.10674) ·
  [[Project]](https://skill-sd.github.io/)
  *Extends OPSD to **multi-turn agentic interaction**: skills are
  dynamically distilled from completed trajectories; the **teacher
  conditions on these skills**, the student does not. Loss = GRPO +
  importance-weighted RKL (Schulman K3). Evaluated on AppWorld and
  Sokoban. Extends the OPSD privileged-context pattern to settings
  where the skill set evolves training-time only.*

- **Why Does Self-Distillation (Sometimes) Degrade Reasoning?**
  (MSR / KAIST / SNU, Mar 2026) — *diagnostic, not a method*
  [[arXiv:2603.24472]](https://arxiv.org/abs/2603.24472) ·
  [[Code]](https://github.com/beanie00/self-distillation-analysis)
  *Controlled study showing that **richer privileged context for the
  teacher suppresses *epistemic verbalization*** (uncertainty
  expression) in the student → fast in-domain gains but **up to 40 %
  OOD drops** on Qwen3-8B / DeepSeek-Distill-Qwen-7B / Olmo3-7B-Instruct.
  Implication: privileged-context richness is a double-edged knob —
  the OPSD analog of "stronger teacher hurts SLM" finding from
  Hsieh et al.*

- Tianduo Wang, Wei Lu et al. **Self-Distillation Bridges Distribution Gap in
  Language Model Fine-Tuning.** ACL, 2024.
  [[arXiv:2402.13669]](https://arxiv.org/abs/2402.13669)

- Yiming Zhang et al. **Self-Knowledge Distillation: A Simple Way for Better
  Generalization.** [[arXiv:2006.12000]](https://arxiv.org/abs/2006.12000)

- **SPIN — Self-Play Fine-Tuning** (UCLA, ICML 2024) — *also covered
  in [Iterative Self-Bootstrapping](#iterative-self-bootstrapping)
  below since the teacher is the previous-iteration self.*
  Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu.
  **Self-Play Fine-Tuning Converts Weak Language Models to Strong Language
  Models.** [[arXiv:2401.01335]](https://arxiv.org/abs/2401.01335) ·
  [[Code]](https://github.com/uclaml/SPIN)
  *Iterative self-play: the model at iteration t-1 generates its **own
  responses**, and iteration t is trained with a DPO-like objective that
  pushes the model to prefer human responses over its own previous
  generations. Mathematically equivalent to an **adversarial self-distillation**
  loop — a strict precursor to OPSD/SDFT, and empirically out-performs
  DPO on MT-Bench / Open LLM Leaderboard without any extra preference data.*

### Iterative Self-Bootstrapping

> Same model is the teacher, but as a **frozen earlier checkpoint**,
> not a privileged-context view. The teacher snapshot is frozen for
> one round, the student trains, then the snapshot rolls forward.
> Listed separately because supervision is typically sequence-level /
> preference / verifier-filtered, not per-token logit-distillation
> (so **C2 partially fails the strict OPD form**).

- **SPIN — Self-Play Fine-Tuning** (UCLA, ICML 2024)
  [[arXiv:2401.01335]](https://arxiv.org/abs/2401.01335) ·
  [[Code]](https://github.com/uclaml/SPIN)
  ⚠️ C1 ✓ (student samples), C2 partial (sequence-level DPO preference
  vs the frozen previous checkpoint, not per-token logit KL). Closer
  to "iterative on-policy DPO" than per-token OPD; kept here because
  the *teacher = previous self* pattern is canonical.

- **rStar / rStar-Math / rStar2-Agent** (Microsoft Research, 2024–2025)
  [[rStar-Math arXiv:2501.04519]](https://arxiv.org/abs/2501.04519) ·
  [[rStar2-Agent arXiv:2508.20722]](https://arxiv.org/abs/2508.20722) ·
  [[Code]](https://github.com/microsoft/rStar)
  *MCTS-filtered student samples + iterative SFT against a step-level
  process preference model (PPM) / discriminator. The "teacher signal"
  is a verifier score on full trajectories, not per-token logit KL ⚠️
  (C2 partial). Iterative bootstrapping rather than classical OPD,
  but operationally lives in the same design space — student rolls
  out → quality filter → re-train.*

- *NPO / AutoNPO* (IIE CAS / UCAS / JD.COM, Apr 2026) —
  see [Inside-RL OPD Hybrids](#inside-rl-opd-hybrids).
  Mixed-policy GRPO that uses verifier-filtered trajectories from a
  **near-future checkpoint of the same training run** as the teacher.
  A formalisation of "learn from your near-future self".

- *BOND / Faster WIND* — see [Inside-RL OPD Hybrids](#inside-rl-opd-hybrids).
  Treat **Best-of-N from the same model** as the iterative target.

### On-Policy KD from a Larger Teacher

> The student rolls out, a **separate (usually larger)** teacher provides
> dense targets on those rollouts.

- **GKD — Generalized Knowledge Distillation** (Google DeepMind, 2023)
  Rishabh Agarwal et al. **On-Policy Distillation of Language Models:
  Learning from Self-Generated Mistakes.** ICLR, 2024.
  [[arXiv:2306.13649]](https://arxiv.org/abs/2306.13649)
  *Defines the modern OPD recipe: student samples → teacher logprobs →
  reverse / forward / Jensen-Shannon divergence loss. Shows substantial wins
  over standard SeqKD and SFT on summarization, translation and arithmetic.*

- **MiniLLM** (THU / MSRA, 2023)
  Yuxian Gu et al. **MiniLLM: Knowledge Distillation of Large Language
  Models.** ICLR, 2024. [[arXiv:2306.08543]](https://arxiv.org/abs/2306.08543) ·
  [[Code]](https://github.com/microsoft/LMOps/tree/main/minillm)
  *Optimizes reverse-KL with a policy-gradient estimator on student samples;
  one of the first works to explicitly cast LLM distillation as an RL problem.*

- **DistiLLM / DistiLLM-2** (KAIST / Microsoft, 2024–2025)
  Jongwoo Ko et al. **DistiLLM: Towards Streamlined Distillation for Large
  Language Models.** [[arXiv:2402.03898]](https://arxiv.org/abs/2402.03898) ·
  [[Code]](https://github.com/jongwooko/distillm) ·
  **DistiLLM-2** (ICML 2025 Oral)
  [[arXiv:2503.07067]](https://arxiv.org/abs/2503.07067) ·
  [[Code]](https://github.com/jongwooko/distillm-2)
  *DistiLLM introduces **Skewed-KL** (a smooth FKL↔RKL interpolation,
  α-parameterised) with importance-reweighted student samples for
  adaptive off→on switching. DistiLLM-2 makes the loss **asymmetric
  per data source**: Skew-FKL on teacher data + Skew-RKL on student
  rollouts; one of the cleanest ablations isolating "which divergence
  for which trajectory source".*

- **G-OPD — Generalized On-Policy Distillation** (RUC / Tencent, Feb 2026)
  [[arXiv:2602.12125]](https://arxiv.org/abs/2602.12125) ·
  [[Code]](https://github.com/RUCBM/G-OPD)
  *Crystallises **OPD = KL-constrained RL with reward extrapolation**.
  Reverse-KL on student rollouts + a scaled reward extrapolation term
  with reward scale > 1 lets the student "exceed" the teacher in
  benchmarks where the teacher is the ceiling. The cleanest unifying
  formalism connecting OPD and constrained RL.*

- **DSKDv2 — Dual-Space Knowledge Distillation v2** (BJTU, Apr 2025)
  [[arXiv:2504.11426]](https://arxiv.org/abs/2504.11426) ·
  [[Code]](https://github.com/songmzhang/DSKDv2)
  *Cross-tokenizer / cross-vocabulary distillation in a dual aligned
  space, with **explicit on-policy mode**. Drops in as a teacher when
  the student family has a different tokenizer (LLaMA student / Qwen
  teacher etc.). Co-released with the **KDFlow** framework
  (see [Frameworks](#general-frameworks-that-support-opd)).*

- **AdaSwitch — Adaptive On-/Off-Policy Switching** (RUC / Baidu, Oct 2025)
  [[arXiv:2510.07842]](https://arxiv.org/abs/2510.07842)
  *Threshold-gated switching between teacher-data and student-rollout
  branches based on the running KL divergence. Avoids both the
  off-policy "frozen-target" bias and the on-policy "noise-amplified"
  variance — a practical middle-ground recipe that several frameworks
  now ship as a config flag.*

- **Constrained OPD — KL-Constrained CMDP** (Huawei Noah's Ark, Sep 2025)
  [[arXiv:2509.22921]](https://arxiv.org/abs/2509.22921)
  *Replaces the soft KL penalty with a **hard KL constraint** (CMDP
  formulation). Theoretical contribution; borderline OPD / OPD-RL
  hybrid because the trust-region structure already shows up in PPO.*

- **TIP — Token Importance Probing** (Meta / LinkedIn, Apr 2026)
  [[arXiv:2604.14084]](https://arxiv.org/abs/2604.14084) ·
  [[Code (LinkedIn OPSD repo)]](https://github.com/HJSang/OPSD_OnPolicyDistillation)
  *Only the **top-50 % high-entropy student tokens** carry the OPD
  signal; the rest are masked out. ~47 % memory savings without
  accuracy regression on reasoning benchmarks. Companion to PACED
  (frontier-curriculum self-distill) from the same LinkedIn group.*

- **PACED — Frontier-Curriculum Self-Distillation** (LinkedIn, Mar 2026)
  [[arXiv:2603.11178]](https://arxiv.org/abs/2603.11178) ·
  [[Code]](https://github.com/HJSang/OPSD_OnPolicyDistillation)
  *Difficulty weighting `w(p) = p(1-p)` concentrates training on
  prompts at the student's competence boundary. Shares the LinkedIn
  OPSD repo with TIP.*

- **TSD-KD — Token-Selective Dual KD** (Korea Univ., ICLR 2026)
  [[arXiv:2603.13260]](https://arxiv.org/abs/2603.13260) ·
  [[Code]](https://github.com/kmswin1/TSD-KD)
  *Hybrid: indirect (student-propose / teacher re-rank) + direct
  selective logit KD. Mixed-policy data, token-level, partial OPD +
  partial preference. Strong reasoning-benchmark numbers.*

- **HPD — Hybrid Policy Distillation** (Apr 2026)
  [[arXiv:2604.20244]](https://arxiv.org/abs/2604.20244) ·
  [[Code]](https://github.com/zwhong714/Hybrid-Policy-Distillation)
  *Reweighted log-likelihood that **unifies FKL + RKL** as a single
  token-level reweighted likelihood. Lightweight on-policy sampling
  preserves training efficiency. Ships with both **LlamaFactory** and
  **verl** backends.*

- **Fast OPD — Prefix-Truncated Distillation** (Feb 2026)
  [[arXiv:2602.15260]](https://arxiv.org/abs/2602.15260)
  *2× to 47× speedup via reasoning-prefix truncation: long CoT prefixes
  are truncated and the teacher only scores the suffix. Shows that
  **most of the OPD gradient comes from the answer region**, not the
  reasoning prefix — important corroboration for the offline / Lightning
  OPD line of work.*

- Cheng-Yu Hsieh et al. **Distilling Step-by-Step!** ACL, 2023.
  [[arXiv:2305.02301]](https://arxiv.org/abs/2305.02301)
  *— rationale-augmented distillation; inspires the "teacher-thought" stream
  later combined with on-policy rollouts.*

- Xinghao Chen et al. **Unveiling the Key Factors for Distilling
  Chain-of-Thought Reasoning.** ACL Findings, 2025.
  [[arXiv:2502.18001]](https://arxiv.org/abs/2502.18001) ·
  [[Code]](https://github.com/EIT-NLP/Distilling-CoT-Reasoning)
  *— empirically shows **stronger teachers are not always better** for SLMs;
  motivates teacher-aligned prompt selection in later OPD works.*

- **TAID — Temporally Adaptive Interpolated Distillation** (Sakana AI,
  ICLR 2025) — see [Interpolated / Curriculum Distillation](#interpolated--curriculum-distillation).

### Black-Box / Outcome-Based OPD

> When the teacher is **API-only** (no logits) — e.g., GPT-5,
> Claude Opus, Gemini-Ultra — OPD must replace token-level logit
> matching with scalar rewards, verbal scores, preferences, or
> adversarial discriminators evaluated on student rollouts. C1 is
> still satisfied (student samples its own trajectories); C2 is
> satisfied at the **sequence / verbal / discriminator level**
> rather than per-token logit.

- **GAD — Generative Adversarial Distillation** (Microsoft Research, Nov 2025)
  [[arXiv:2511.10643]](https://arxiv.org/abs/2511.10643) ·
  [[Project]](https://ytianzhu.github.io/Generative-Adversarial-Distillation/) ·
  [[Code (LMOps `/gad`)]](https://github.com/microsoft/LMOps)
  *The seed paper for **black-box OPD**. A discriminator distinguishes
  student rollouts from teacher (e.g. GPT-5) responses; the minimax
  game makes the discriminator co-evolve into an **on-policy reward
  model**. Result: **Qwen2.5-14B student becomes comparable to
  GPT-5-Chat on LMSYS** without any access to teacher logits — the
  cleanest evidence so far that pure outcome supervision is enough for
  black-box OPD to close most of the capability gap.*

- **OVD — On-Policy Verbal Distillation** (HKU / Huawei, Jan 2026)
  [[arXiv:2601.21968]](https://arxiv.org/abs/2601.21968)
  *Replaces logit matching with **verbal scoring (0-9)**: the teacher
  reads a student trajectory and emits a free-text rationale + scalar
  score; the score is back-propagated as a sequence-level reward.
  +25.7 % over baselines on alignment benchmarks. Useful when only a
  chat endpoint is available and even probabilities are hidden.*

- **ORPO-Distill — Student-Generated Outputs + ORPO Contrastive**
  (NeurIPS 2025 WS, Sep 2025)
  [[arXiv:2509.25100]](https://arxiv.org/abs/2509.25100)
  *Mixed-policy: student-generated negatives + teacher positives,
  trained with the ORPO contrastive objective. The first explicit
  framing of **on-policy distillation as preference optimisation**
  for cross-architecture transfer. ⚠️ Sequence-level only (C2 partial).*

### Inside-RL OPD Hybrids

> Methods that fuse OPD with **RLVR / GRPO / PPO / DPO**. Teacher
> logits become a dense reward shaping or trust-region anchor inside
> an RL objective; or BoN / preference signals are used as the
> imitation target. Strict-OPD-form C2 is sometimes only partially
> satisfied (sequence-level reward shaping rather than per-token
> logit KL); exceptions are flagged.

- **SDPO — RL via Self-Distillation** (ETH / MIT) — see
  [Self-Distillation as On-Policy Learning](#self-distillation-as-on-policy-learning).

- **AlignDistil — RLHF-Equivalent Distillation** (BJTU / Tencent, ACL 2025)
  [[arXiv:2503.02832]](https://arxiv.org/abs/2503.02832) ·
  [[Code]](https://github.com/songmzhang/AlignDistil)
  *Re-frames **DPO as policy distillation**: the target distribution
  is a DPO-derived combination of (DPO model logits + reference-model
  logits); per-token KL on student rollouts. The cleanest bridge
  between RLHF preference optimisation and OPD.*

- **LUFFY — Mixed-Policy GRPO with Off-Policy Imports** (Westlake U., Apr 2025)
  [[arXiv:2504.14945]](https://arxiv.org/abs/2504.14945) ·
  [[Code]](https://github.com/ElliottYan/LUFFY)
  *Half on-policy student rollouts + half **off-policy R1 traces**
  inserted into the GRPO buffer. Policy shaping ensures the off-policy
  half doesn't explode the importance ratio. The most-cited "learn to
  reason under off-policy guidance" recipe. ⚠️ Mixed C1.*

- **NPO / AutoNPO — Near-future Policy Optimisation** (IIE CAS / UCAS / JD.COM, Apr 2026)
  [[arXiv:2604.20733]](https://arxiv.org/abs/2604.20733)
  *Mixed-policy GRPO where the off-policy traces come from a
  **near-future checkpoint of the same training run** instead of an
  external R1. Teacher selection criterion: *strong enough* (higher Q
  than current policy) yet *close enough* (low V vs external teachers)
  → maximises effective Q/V signal. AutoNPO adaptively schedules the
  interventions; preserves higher entropy than vanilla GRPO. ⚠️
  Sequence-level (C2 partial); the paper itself invites follow-up
  work to inject the near-future-self signal via per-token OPD.*

- **KEPO — Knowledge-Enhanced PO** (Jan 2026)
  [[arXiv:2602.00400]](https://arxiv.org/abs/2602.00400) ·
  [[Code]](https://github.com/Corleno/KEPO)
  *Adds knowledge-base teacher grounding to preference RL. Mixed-policy
  rollouts; sequence-level supervision.*

- **BOND — Best-of-N Distillation** (Google DeepMind, Jul 2024)
  [[arXiv:2407.14622]](https://arxiv.org/abs/2407.14622)
  *Treats **Best-of-N from the same model** as the target distribution;
  iterative anchor; **Jeffreys divergence** loss. Sequence-level
  supervision — strictly *iterative on-policy alignment* rather than
  per-token OPD, but the algorithmic shape is identical.*

- **Faster WIND — Win-Rate Dominance** (CMU / Google, AISTATS 2025)
  [[arXiv:2410.20727]](https://arxiv.org/abs/2410.20727)
  *Game-theoretic acceleration of BOND: replaces Jeffreys with a
  win-rate-dominance objective. ~3× faster than BOND at matched
  alignment quality.*

- **KETCHUP — k-Step RL-Based KD** (U. Alberta, Apr 2025)
  [[arXiv:2504.19024]](https://arxiv.org/abs/2504.19024)
  *Sequence-level RL-based KD with k-step Bellman returns. Self-
  describes as "RL-based KD"; closer to RL-with-KD-anchor-reward
  than per-token OPD ⚠️.*

- **𝒳-KD — IRL-Style Joint Reward + Policy Distillation** (BUPT, Feb 2026)
  [[arXiv:2602.12674]](https://arxiv.org/abs/2602.12674)
  *Built on the AVRIL inverse-RL framework: jointly distills the
  teacher's reward function and policy. The IRL-flavoured experiential
  KD recipe.*

- **DDT — Distribution Discriminant Theory for On-Policy SFT**
  (MSRA / Shopee, Feb 2026)
  [[arXiv:2602.12222]](https://arxiv.org/abs/2602.12222) ·
  [[Code]](https://github.com/zhangmiaosen2000/Towards-On-Policy-SFT)
  *Theoretical foundations paper that justifies why on-policy SFT
  works at all (discriminant separation between on- and off-policy
  distributions). No deployable algorithm; the formal scaffolding for
  the entire OPD design space.*

- **RLAD — Reinforcement-Aware KD** (AWS, Feb 2026)
  [[arXiv:2602.22495]](https://arxiv.org/abs/2602.22495)
  *PPO/GRPO importance ratio anchored to a **teacher–old-policy
  mixture** (Qwen3-32B teacher). Token-level trust-region likelihood
  ratio. Reasoning benchmarks.*

- **KDRL — Joint KD + GRPO** (HIT / Huawei, Jun 2025)
  [[arXiv:2506.02208]](https://arxiv.org/abs/2506.02208)
  *Unified objective combining reverse-KL distillation with rule-based
  GRPO reward (Skywork-OR1 teacher). Token-level KD + outcome-level
  RL — the canonical "OPD-inside-RL" recipe.*

- **RLSD — Self-Distilled RLVR** (Apr 2026)
  [[arXiv:2604.03128]](https://arxiv.org/abs/2604.03128)
  *RLVR provides the *direction* of policy update; teacher evidence
  ratio modulates the *magnitude*. Same model + privileged answer
  conditioning → token-level + outcome-level signal. A symmetric
  counterpart to SD-Zero (which goes the other way: turns RLVR
  rewards into OPSD signal).*

- **HDPO — Hybrid Distillation Policy Optimisation** (NVIDIA, Mar 2026)
  [[arXiv:2603.23871]](https://arxiv.org/abs/2603.23871)
  *RL on most prompts; on **"cliff" prompts** (where RL stalls) the
  framework generates *privileged-context rollouts* and switches to
  self-distillation. **Privileged self-distillation as RL fallback**
  — a practical recipe for production scaling.*

- **OpenClaw-RL — GRPO + OPD for Coding/Tool-use Agents**
  (Gen-Verse, Mar 2026)
  [[arXiv:2603.10165]](https://arxiv.org/abs/2603.10165) ·
  [[Code]](https://github.com/Gen-Verse/OpenClaw-RL)
  *Unifies binary RL and per-token OPD in one trainer. A **judge
  model** extracts hindsight hints; the teacher–student log-prob gap
  acts as a directional advantage. Domains: terminal, GUI, SWE,
  tool-call.*

- **Open-AgentRL — GRPO-TCR Multi-Domain** (Gen-Verse, Feb 2026)
  [[Code]](https://github.com/Gen-Verse/Open-AgentRL)
  *Multi-domain teachers (reasoning / GUI / coding) with process-reward
  modelling via SandboxFusion. The agent-side counterpart to OpenClaw-RL.*

- **Probing-to-Refine / EI / EXGRPO** (UNC / ASU, Mar 2026)
  [[arXiv:2603.19266]](https://arxiv.org/abs/2603.19266) ·
  [[Code]](https://github.com/Zhen-Tan-dmml/ExGRPO)
  *"Explanatory probes" force logical articulation; GRPO + dialogue-
  structure reward. Reinforcement Distillation via Explanatory
  Inversion. ⚠️ Borderline pure RL — listed because the self-probe
  plays a teacher-like role.*

### Stability & Loss Engineering

> A separate axis of OPD research focuses **not on what the teacher is** but
> on **how the teacher's signal is shaped into a usable gradient**. Three
> canonical failure modes are addressed: (i) gradient explosion under
> forward KL on "ignorant" tokens, (ii) mode collapse under reverse KL,
> and (iii) heavy-tailed reward distributions that mimic RL pathologies.

- **Veto — Stable On-Policy Distillation through Adaptive Target
  Reformulation** (2026)
  [[arXiv:2601.07155]](https://arxiv.org/abs/2601.07155) ·
  [[HF Paper]](https://huggingface.co/papers/2601.07155)
  *Pinpoints OPD instability as a **geometry-of-divergence** problem rather
  than a data problem. Builds a *logit-space geometric bridge*
  \(P_{\text{target}} = (1-\alpha) P_T + \alpha P_S\) that simultaneously
  serves as: (a) an **adaptive gradient veto** that suppresses runaway
  forward-KL gradients on tokens where \(P_T \gg P_S\) (where naïve gradients
  reach \(10^7\) magnitudes); and (b) a **decisiveness knob** trading off
  reward-driven precision against output diversity in the reverse-KL
  regime. A single scalar \(\alpha\) defangs both classical OPD failure modes.*

- **EOPD — Entropy-Aware On-Policy Distillation of Language Models** (2026)
  [[arXiv:2603.07079]](https://arxiv.org/abs/2603.07079) ·
  [[OpenReview]](https://openreview.net/forum?id=WSRQ37tzk1)
  *Finds that pure reverse-KL **kills high-entropy tokens** — exactly the
  reasoning forks where diversity matters most. Empirically, a Qwen3-1.7B
  student trained with reverse-KL retains only **6.8 %** high-entropy tokens
  on AIME'24/25 vs. **18.5 %** for its Qwen3-8B teacher. Solution: switch
  losses **per-token** based on teacher entropy — reverse-KL on low-entropy
  tokens (fast & stable), **plus forward-KL on high-entropy tokens** (mode
  covering). Pass@8 gains +1.37 / +2.39 / +5.05 on Qwen3-{0.6B, 1.7B, 4B}
  across 6 math benchmarks; the gain **grows with model size**, suggesting
  diversity preservation matters more at scale.*

- **REOPOLD — Scaling Reasoning Efficiently via Relaxed On-Policy
  Distillation** (2026)
  [[arXiv:2603.11137]](https://arxiv.org/abs/2603.11137)
  *Formal contribution: proves that **stop-gradient OPD ≡ on-policy policy
  gradient** with a token-level reward
  \(r_{i,t}(\theta) = \log P_T(y_{i,t}\mid\cdot) / \log P_S(y_{i,t}\mid\cdot)\).
  This unlocks the entire RL toolbox for OPD: (1) **mixture reward
  clipping** (clipping the reward, not the importance ratio, to tame
  heavy-tailed negative rewards); (2) **entropy-guided token-level dynamic
  sampling** (gradients only on the top-\((1-\rho)\) most uncertain tokens);
  (3) **explore-then-refine schedule** that masks strong negatives early
  and switches to entropy masking later. Result: **6.7–12× sample efficiency
  on AIME-25** vs. ProRL / Still-3-1.5B, and a 3B vision student matches
  a 32B teacher with **3.3× inference speedup** on Geometry3K.*

- **StableOPD — Demystifying OPD: Length Inflation and Stabilization
  Strategies** (Rice University, Apr 2026)
  [[arXiv:2604.08527]](https://arxiv.org/abs/2604.08527)
  *Identifies **repetition collapse** as a built-in OPD reward-hacking
  failure mode: ~30 training steps after a phase transition, truncation
  rate spikes to 1.0, repetition rate to 0.3–0.6, and validation accuracy
  craters. Mechanism: when the student loops, the (stronger) teacher
  becomes **more confident** on the repeating context than the student,
  so \(\log P_T - \log P_S\) becomes a strongly positive reward —
  **repetition-token advantage is 4–9× normal-token advantage**, creating
  a self-reinforcing repetition cycle. Detected via zlib compression
  ratio > 10×. Solution: (a) **Reference-based KL regularization** to a
  pre-training student snapshot to slow policy drift; (b) **Rollout
  Mixture Distillation** that injects high-quality SFT examples
  (OpenR1-Math-220k, length & correctness filtered) every step. Numbers:
  Qwen2.5-1.5B 28.9 % → **35.7 %**, Qwen2.5-7B 43.8 % → **47.6 %**.*

- **Revisiting OPD — Empirical Failure Modes and Simple Fixes**
  (CASIA, Mar 2026)
  [[arXiv:2603.25562]](https://arxiv.org/abs/2603.25562) ·
  [[HF Paper]](https://huggingface.co/papers/2603.25562)
  *A patch for the **single-token-sampled OPD** that Qwen3 / MiMo-V2
  ship in production. Theoretical: token-level reverse-KL has variance
  bound \(O(T^2)\) vs sequence-level \(O(T^4)\) — token-level is a
  deliberate variance reduction, not a wrong approximation. Empirically
  diagnoses three structural bugs: (1) **signal imbalance** — most
  student samples have negative log-ratio, so the positive learning
  signal collapses onto a few tokens; (2) **out-of-support teacher
  unreliability** — when the student drifts, the teacher emits
  "plausible-looking but harmful" high-probability predictions
  (repetition, self-resets, format errors); (3) **tokenizer mismatch
  artifacts** — `<think>` split as `<,think,>` vs `<th,ink,>` makes
  single-token comparison meaningless. Fix: **Local Support Set
  Matching** — at each prefix, take teacher top-K, optionally filtered
  by top-p, **renormalize teacher and student onto this support**, then
  compute reverse-KL. +19.8 % over standard sampled-token OPD; near-zero
  compute overhead — **the cleanest drop-in upgrade for production OPD
  trainers right now**.*

- **SCOPE — Signal-Calibrated OPD Enhancement with Dual-Path Adaptive
  Weighting** (Meituan + USTC + NJU + Fudan + HUST, Apr 2026)
  [[arXiv:2604.10688]](https://arxiv.org/abs/2604.10688) ·
  [[Code]](https://github.com/machine981/SCOPE)
  *First OPD work to argue that **correct and incorrect rollouts deserve
  different objectives**. Two motivating findings: (i) **Pass@k paradox**
  — uniform rollout reinforcement on Qwen2.5-7B improves Pass@1 but
  drops Pass@32 from 93.7 % to 84.9 % by killing minority-correct paths;
  (ii) **toxic-prefix trap** — teacher recovery from bad student
  prefixes is reliable for low-PPL prefixes (64.9 %) but unreliable for
  high-PPL ones (45.4 %), and recovery degrades sharply with truncation
  depth. Solution: split rollouts by correctness, then weight per-group
  with softmax (\(\tau=1.0\)):
  **incorrect** → teacher-KL × **(1/teacher-PPL)** (down-weight
  unreliable corrections);
  **correct** → MLE × **student-PPL** (boost low-confidence "boundary"
  successes). +5.54 % Avg@32 across 6 benchmarks (R1-Distill-Qwen-1.5B
  ← Skywork-OR1-Math-7B), with +10.69 % on OlympiadBench. Reversing the
  weighting direction crashes performance, confirming the signal-quality
  hypothesis.*

### Speculative / Hybrid Student-Teacher Sampling

> Neither pure on-policy nor pure off-policy — the student proposes tokens,
> the teacher verifies / replaces bad ones.

- **SKD — Speculative Knowledge Distillation** (Google Research, 2024)
  Wenda Xu et al. **Speculative Knowledge Distillation: Bridging the
  Teacher-Student Gap Through Interleaved Sampling.** ICLR, 2025.
  [[arXiv:2410.11325]](https://arxiv.org/abs/2410.11325) ·
  [[Code]](https://github.com/google-research/google-research/tree/master/speculative_kd)
  *Student proposes tokens autoregressively; the teacher **re-samples**
  low-ranked ones based on its own distribution. Training data stays close to
  the student's inference distribution **while filtering out noise from weak
  student rollouts**.*

- **SpecKD / SelecTKD — Speculative Decoding for Effective KD** (XJTU, 2025)
  Haiduo Huang et al. [[arXiv:2510.24021]](https://arxiv.org/abs/2510.24021)
  *Instead of filtering **data**, SpecKD filters the **loss itself**: each
  student token is verified against the teacher, and the KL penalty is
  applied only on accepted tokens — drastically stabilizing training when the
  teacher–student gap is large. (v1 = SpecKD; v2 retitled SelecTKD.)*

### Speculative-Decoding Drafter Distillation

> A distinct application of OPD: the **student is a draft model** for
> speculative decoding, distilled to better mimic the verifier/target.
> The on-policy element here is over the *drafter*'s own continuations
> as judged by the *target*. Listed separately because the goal is
> **inference speedup**, not student capability — but the algorithmic
> form (student rollout + teacher per-token signal) is canonical OPD.

- **DistillSpec — OPD for Speculative Decoding** (Google DeepMind, ICLR 2024)
  [[arXiv:2310.08461]](https://arxiv.org/abs/2310.08461)
  *The seminal "OPD for SD" paper. Drafter trained on its own samples
  with a configurable choice of FKL / RKL / JSD / TVD divergence.
  Establishes that **on-policy drafter training increases acceptance
  rate** vs off-policy teacher-forced training.*

- **OSD — Online Speculative Decoding** (UCB / NVIDIA, Oct 2023)
  [[arXiv:2310.07177]](https://arxiv.org/abs/2310.07177) ·
  [[Code]](https://github.com/LiuXiaoxuanPKU/OSD)
  *The canonical **online / on-policy** SD paper: continually retrains
  the drafter during serving on rejected tokens. Uses production
  serving traces as the OPD signal.*

- **EAGLE-3 — Training-Time Test (TTT)** (PKU / Microsoft, Mar 2025)
  [[arXiv:2503.01840]](https://arxiv.org/abs/2503.01840) ·
  [[Code]](https://github.com/SafeAILab/EAGLE)
  *Self-speculative drafter that uses target features. The "TTT"
  (Training-Time Test) recipe simulates draft rollouts during
  training — i.e., **on-policy multi-step drafter distillation**.
  Smooth-L1 (feature) + CE (token) loss. Currently the SOTA
  open-source SD drafter recipe.*

- **HASS — Harmonized Self-Speculative** (Aug 2024)
  [[arXiv:2408.15766]](https://arxiv.org/abs/2408.15766) ·
  [[Code]](https://github.com/HArmonizedSS/HASS)
  *Multi-step KD CE + feature alignment with harmonized objective and
  context alignment. ⚠️ Partial on-policy: multi-step draft trajectory
  uses drafter samples for a subset of the training signal.*

- **Falcon — Coupled Sequential Glancing Distillation** (Bestpay, Dec 2024)
  [[arXiv:2412.12639]](https://arxiv.org/abs/2412.12639) ·
  [[Code]](https://github.com/Bestpay-inc/Falcon)
  *Semi-autoregressive draft model with glancing distillation: the
  glancing path uses drafter samples (partial OPD).*

- **SpecForge — Open EAGLE-3 Training Framework** (SGLang, Mar 2026)
  [[Blog]](https://www.lmsys.org/blog/2025-07-25-spec-forge/) ·
  [[Code]](https://github.com/sgl-project/SpecForge)
  *Production-grade open-source EAGLE-3 training framework with
  on-policy TTT supported. Companion to SGLang inference.*

- **ReSpec — Drafter Evolved During RL Training** (Oct 2025)
  [[arXiv:2510.26475]](https://arxiv.org/abs/2510.26475)
  *Continually retrains the drafter on RL rollouts, KD-weighted by
  rollout reward. The RL-side counterpart to OSD.*

- **DVI — Draft-Verify-Improve** (Oct 2025)
  [[arXiv:2510.05421]](https://arxiv.org/abs/2510.05421)
  *Self-speculative drafter trained online with KL → reward-masked CE
  + policy gradient on the verifier signal. Continual online training.*

- **CORAL — Cross-Step Representation Alignment** (ACL 2025)
  [[arXiv:2502.16880]](https://arxiv.org/abs/2502.16880)
  *On-policy multi-step drafter training that fixes the train–
  inference mismatch via cross-step alignment.*

- **MASSV — Multimodal Speculative Decoding** (Cerebras, May 2025)
  [[arXiv:2505.10526]](https://arxiv.org/abs/2505.10526)
  *Multimodal SD drafter trained on its own samples; extends DistillSpec
  to vision-language settings.*

### Sequence-Level & Distribution-Aligned Distillation

- **DASD — Distribution-Aligned Sequence Distillation** (Aliyun, 2026)
  **Distribution-Aligned Sequence Distillation for Superior Long-CoT
  Reasoning.** [[arXiv:2601.09088]](https://arxiv.org/abs/2601.09088) ·
  [[Blog (CN)]](https://mp.weixin.qq.com/s/TOeY5rkKUFl_cYaOKQaMaA)
  *Critically re-examines the dominant "SFT-on-teacher-responses" paradigm
  and identifies three failure modes: **inadequate sequence-level distribution
  coverage**, **teacher–student capacity mismatch**, and **exposure bias**
  from teacher-forcing vs. autoregressive inference. Achieves SOTA reasoning
  at 4B with only **448K training samples** — an order of magnitude fewer
  than peers. Releases **DASD-4B-Thinking** weights and dataset.*

- Yijia Luo et al. **Deconstructing Long Chain-of-Thought: A Structured
  Reasoning Optimization Framework for Long CoT Distillation (DLCoT).**
  [[arXiv:2503.16385]](https://arxiv.org/abs/2503.16385)

- Sunwoo Lee et al. **f-Distill: A Family of f-Divergence Distillation for
  Sequence Generation.** [[arXiv:2307.15190]](https://arxiv.org/abs/2307.15190)

### Context Distillation on Student Rollouts

- **OPCD — On-Policy Context Distillation** (Microsoft / PKU, 2026)
  Li Dong et al. **On-Policy Context Distillation for Language Models.**
  [[arXiv:2602.12275]](https://arxiv.org/abs/2602.12275)
  *A framework that **bridges on-policy distillation with context
  distillation**: train the student on its own trajectories while minimizing
  reverse-KL against a **context-conditioned teacher**. Two killer apps:
  (i) **experiential knowledge distillation**, where a model consolidates
  transferable knowledge from its own historical solution traces;
  (ii) **system-prompt distillation**, where a model internalizes beneficial
  behaviors encoded in an optimized prompt. Supports cross-size teachers.*

- Yulia Tsvetkov et al. **Context Distillation** (original concept).
  [[arXiv:2209.15189]](https://arxiv.org/abs/2209.15189) —
  off-policy precursor that OPCD upgrades.

### Multi-Teacher On-Policy Distillation

- **MOPD — Multi-Teacher On-Policy Distillation** (Xiaomi MiMo team, 2026)
  **MiMo-V2-Flash Technical Report.**
  [[arXiv:2601.02780]](https://arxiv.org/abs/2601.02780) ·
  [[Blog (CN)]](https://mp.weixin.qq.com/s/3a2xz8LYhyV6udSgxuQFoA)
  *The post-training scaling recipe of MiMo-V2-Flash: domain-specialized
  teachers (each trained via large-scale RL on its own domain — math, code,
  agent, etc.) provide **dense token-level rewards** on the student's
  rollouts. A 309B-total / 15B-active MoE student "perfectly inherits" each
  teacher's expertise, rivaling DeepSeek-V3.2 / Kimi-K2 with 1/2 – 1/3 of
  the parameters.*

- **Tinker Cookbook Multi-Teacher Recipe** (Thinking Machines, 2025)
  *Reference code for combining multiple teacher domains (DeepMath + Tulu3)
  into a single OPD run.*
  [[Code]](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation)

### Strong-to-Weak Distillation Pipelines

- **Qwen3 Technical Report** (Alibaba, 2025)
  [[arXiv:2505.09388]](https://arxiv.org/abs/2505.09388)
  *Qwen3's lightweight series (0.6B / 1.7B / 4B / 8B / 14B + 30B-A3B MoE)
  is built with a two-phase **Strong-to-Weak** pipeline:
  **(1)** off-policy SFT on teacher responses, **(2)** on-policy distillation
  where the student rolls out and aligns its logits with Qwen3-32B /
  235B-A22B via reverse-KL. Table 21 shows on-policy distillation beats
  RL on Qwen3-8B at **1/10 the GPU hours** (AIME'24 74.4 % vs 67.6 %).*

- Xiaohan Yuan et al. **Weak-to-Strong Reasoning Distillation.** 2026
  (community follow-ups to Qwen3's recipe).

- **Gemma 2 / Gemma 3** (Google DeepMind, 2024–2025)
  [[Gemma 2 TR]](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf) ·
  [[Gemma 3 TR (arXiv:2503.19786)]](https://arxiv.org/abs/2503.19786)
  *Two successive flagship open-weight families trained with distillation at
  unprecedented scale. **Pre-training**: 2B / 9B / 27B students learn from a
  large Gemini teacher via token-level cross-entropy over **256 sampled
  logits** per token, weighted by teacher probabilities — allowing students
  to see **>50× the compute-optimal token budget**. **Post-training (Gemma 3)**:
  combines improved KD from a large IT teacher with **BOND / WARM / WARP**
  RL fine-tuning, closely mirroring the Agarwal-et-al. GKD recipe. The
  resulting Gemma-3-4B-IT matches Gemma-2-27B-IT; Gemma-3-27B-IT is
  competitive with Gemini-1.5-Pro.*

- **Kimi K2 / K1.5** (Moonshot AI, 2025–2026) — 1T-total / 32B-active MoE
  agentic model. [[arXiv:2507.20534]](https://arxiv.org/abs/2507.20534) ·
  [[Code]](https://github.com/MoonshotAI/Kimi-K2)
  *Post-training uses a **joint RL + rubric-distillation** loop: verifiable
  rewards (RLVR) iteratively update an on-policy critic whose judgments are
  then **distilled into the policy** on non-verifiable tasks (creative
  writing, complex judgment). A de-facto large-scale realization of
  SDPO-style dense feedback distillation at trillion-parameter scale.*

### Cross-Stage / Anti-Forgetting Distillation

> Use **earlier checkpoints of the same model family** as teachers for the
> current training stage. Crucial when running sequential RL pipelines
> (Reasoning → Agentic → General) that would otherwise forget earlier skills.

- **GLM-5 — On-Policy Cross-Stage Distillation (OPCSD)** (Z.ai / Tsinghua, 2026)
  GLM-5 Team. **GLM-5: from Vibe Coding to Agentic Engineering.**
  [[arXiv:2602.15763]](https://arxiv.org/abs/2602.15763) ·
  [[Code]](https://github.com/zai-org/GLM-5)
  *The flagship demonstration of OPD as a **first-class building block of
  the post-training pipeline of a frontier model**. GLM-5 (744B-A40B MoE)
  runs a sequential RL cascade — multi-task SFT → **Reasoning RL** →
  **Agentic RL** → **General RL** — and, between every two stages,
  performs **On-Policy Cross-Stage Distillation**: the student rolls out
  under the new stage's policy while being scored by a **teacher checkpoint
  from an earlier stage** (typically SFT or Reasoning-RL). Advantage signals
  are derived from **teacher–student logit gaps**, not just scalar rewards.
  The effect: catastrophic forgetting is suppressed and GLM-5 retains its
  sharp reasoning edge while building agentic robustness, narrowing the gap
  with Claude Opus 4.5 on long-horizon tasks. **The first open tech report
  to name and operationalize OPD as an anti-forgetting mechanism across RL
  stages — a paradigm we refer to as OPCSD.***

- **Thinking Machines — Personalization without Forgetting.**
  [[Blog]](https://thinkingmachines.ai/blog/on-policy-distillation)
  *Complementary empirical result at smaller scale: applying OPD between a
  base-model teacher and a personalization-fine-tuned student recovers the
  lost general capabilities without re-running RL. A 1-instance-of-GLM-5
  lesson.*

- See also **SDFT** ([arXiv:2601.19897](https://arxiv.org/abs/2601.19897))
  in [Self-Distillation](#self-distillation-as-on-policy-learning) — the
  algorithmic core of cross-stage self-teaching.

### Interpolated / Curriculum Distillation

> Instead of a fixed teacher, dynamically **interpolate** between the
> student's own distribution and the teacher's, forming a curriculum of
> intermediate targets. Bridges the capacity gap without mode collapse.

- **TAID — Temporally Adaptive Interpolated Distillation** (Sakana AI,
  ICLR 2025)
  Makoto Shing, Kou Misaki, Han Bao, Sho Yokoi, Takuya Akiba.
  [[arXiv:2501.16937]](https://arxiv.org/abs/2501.16937) ·
  [[Code]](https://github.com/SakanaAI/TAID)
  *Defines a time-dependent intermediate distribution
  \(p_t = (1-\alpha_t) q_\theta + \alpha_t p_T\) that **starts at the student
  and gradually shifts to the teacher**. Proved (under a regression proxy)
  to prevent mode collapse, and empirically beats reverse-KL, forward-KL,
  f-divergence, MiniLLM and DistiLLM across sizes / architectures.
  Produces **TAID-LLM-1.5B** (SOTA <2B English LM) and **TAID-VLM-2B**
  (SOTA VLM ≤4B). A crucial missing piece in the OPD × curriculum-learning
  intersection.*

- **Born-Again Networks (self-interpolation interpretation)** — see
  [Foundational Papers](#foundational-papers).

### Self-Play as Implicit On-Policy Distillation

> When a model plays against itself or its own past, the resulting training
> signal is mathematically equivalent to distilling a *conditional* version
> of the model back into the *unconditional* one — i.e., on-policy
> self-distillation with an implicit teacher.

- **SPIN** (UCLA, ICML 2024) — *primary work*, listed in detail under
  [Self-Distillation as On-Policy Learning](#self-distillation-as-on-policy-learning).

- **Self-Rewarding Language Models** (Meta, 2024) —
  [[arXiv:2401.10020]](https://arxiv.org/abs/2401.10020)
  *Uses the model as its own judge and distills the preference signal into
  the next iteration; closely related to SPIN + SDPO.*

- **Meta-Rewarding LLMs** (Meta, 2024) —
  [[arXiv:2407.19594]](https://arxiv.org/abs/2407.19594)
  *Extends self-rewarding by additionally judging the judge, producing a
  two-level on-policy distillation loop.*

### Cross-Tokenizer / Cross-Family Distillation

> A practical bottleneck of OPD is that the teacher and student normally
> have to share a tokenizer. These works lift that constraint and turn OPD
> into a model-family-agnostic post-training tool.

- **GOLD — General Online Logit Distillation** (Hugging Face H4, 2025)
  Lewis Tunstall, Ed Beeching, Quentin Gallouédec, Patiño et al.
  **Unlocking On-Policy Distillation for Any Model Family.**
  [[Blog]](https://huggingfaceh4-on-policy-distillation.hf.space/) ·
  [[Space]](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation) ·
  [[Trainer Doc]](https://huggingface.co/docs/trl/v0.25.0/gold_trainer)
  *Extends Universal Logit Distillation (ULD) to the **on-policy** setting.
  Incrementally decodes both the student's and the teacher's tokens, groups
  passages with matching visible text, and **merges associated logits** so
  that no completion token is dropped even when token boundaries differ.
  Hybrid loss: exact match → standard logit distillation; otherwise → ULD
  fallback on sorted probabilities. Beats both ULD and GRPO on multi-step
  math, and is shipped as `GOLDTrainer` in TRL — making OPD work between
  any LLaMA / Qwen / Mistral / Gemma combination.*

### Multimodal & Embodied On-Policy Distillation

> OPD has expanded beyond text into vision-language, video, and robot
> control. The on-policy property is even more valuable here because
> rolling out a long visual / sensorimotor context is the dominant cost,
> and per-step dense supervision is much more informative than a single
> task-success bit.

- **Video-OPD — Efficient Post-Training of MLLMs for Temporal Video
  Grounding via On-Policy Distillation** (2026)
  [[arXiv:2602.02994]](https://arxiv.org/abs/2602.02994)
  *Brings OPD to **Temporal Video Grounding (TVG)**, where GRPO suffers
  from sparse sequence-level rewards and very expensive multi-rollout
  visual processing. Video-OPD has the student (Qwen3-VL-8B) roll out once,
  then a frontier teacher (Qwen3-VL-32B-GRPO) **scores** every action
  token: \(r_t = -\log \theta(a_t \mid s_t) + \log \theta_{\text{tea}}(a_t \mid s_t)\).
  Adds **TVDF curriculum** = TRPV (Teacher-Reliability Pre-Validation,
  filter unreliable teacher predictions by ground-truth IoU) + DBTP
  (Disagreement-Based Trajectory Prioritization, train hardest on
  highest-disagreement trajectories). Average **+17 %** over GRPO across
  Charades / ActivityNet / QVHighlights TimeLens (R@0.7), surpasses
  GPT-4o / GPT-5 / Gemini-2.0-Flash and approaches Gemini-2.5-Flash.*

- **VLA-OPD — Bridging Offline SFT and Online RL for Vision-Language-
  Action Models via On-Policy Distillation** (HKUST, Mar 2026)
  [[arXiv:2603.26666]](https://arxiv.org/abs/2603.26666)
  *First port of OPD to **robotic manipulation**. Replaces the sparse
  0/1 environment reward of online RL with the teacher's per-action
  log-probability on student-visited states; reward is
  \(-\log(\pi_\theta / \pi_{\text{tea}})\). The paper's central
  contribution is a **clean three-way KL ablation in OOD states**:
  *Forward-KL* makes the student copy the teacher's hesitation → entropy
  explodes, success rate drops 50 %+ early on; *Hard-CE* destroys soft
  probability information → entropy collapse; *Reverse-KL* is bounded
  mode-seeking that filters teacher uncertainty while preserving
  exploration. Results on **LIBERO**: 1-demo SFT 48.9 % → **87.4 %**
  (close to the 50-demo teacher's 93.9 %); 10 steps to 90 % vs GRPO's
  150 steps. **RoboTwin2.0 dual-arm** 45.2 % → 71.1 %. Robotics turns
  out to be a clean OPD domain — short trajectories + reliable teacher
  signal end-to-end.*

- **TAID-VLM-2B** (Sakana AI, ICLR 2025) — vision-language sister of
  TAID-LLM-1.5B; also fits here.
  [[arXiv:2501.16937]](https://arxiv.org/abs/2501.16937)

- **π-Flow — On-Policy Distillation for Image / Flow Models**
  (Multi-org, ICLR 2026)
  [[arXiv:2510.14974]](https://arxiv.org/abs/2510.14974) ·
  [[Code]](https://github.com/Lakonik/piFlow)
  *Strict OPD applied to **diffusion / flow models** for image
  generation: the student predicts the teacher velocity field at each
  timestep along its **own trajectory** (L2 imitation distillation).
  The first principled "OPD for diffusion" recipe.*

- **VOLD — LLM → VLM On-Policy Distillation** (INRIA / Goethe Univ., ICLR 2026)
  [[arXiv:2510.23497]](https://arxiv.org/abs/2510.23497) ·
  [[Project]](https://walidbousselham.com/VOLD/)
  *The flagship VLM OPD recipe: cold-start SFT alignment between the
  text LLM teacher and the VLM student, then **GRPO + on-policy KL
  distillation**. Transfers text-only LLM reasoning capability into
  the VLM without losing visual grounding.*

- **Step-Audio-R1 — Iterative Self-Distillation for Audio Reasoning**
  (StepFun, Nov 2025)
  [[arXiv:2511.15848]](https://arxiv.org/abs/2511.15848) ·
  [[Code]](https://github.com/stepfun-ai/Step-Audio-R1)
  *Iterative on-policy cycles of self-distillation + SFT + PPO/RLVR
  on audio reasoning; only audio-relevant questions are used in the
  self-distill phase. The audio counterpart to OPSD.*

- **CORD — Reasoning Text → Audio** (Baidu Ernie, Jan 2026)
  [[arXiv:2601.16547]](https://arxiv.org/abs/2601.16547)
  *Token-level RKL + sequence-level KL + GRPO transfer text reasoning
  capability into audio LLMs via OPD on student rollouts.*

- **X-OPD — Speech LLM Cross-Modal OPD** (Tencent Hunyuan / ZJU, Mar 2026)
  [[arXiv:2603.24596]](https://arxiv.org/abs/2603.24596)
  *Cross-modal token-level KL: text LLM teacher → speech LLM student.
  Capability alignment in speech LLMs via OPD.*

- **Uni-OPD — Unified OPD across LLMs & MLLMs** (Multi-org, May 2026)
  [[arXiv:2605.03677]](https://arxiv.org/abs/2605.03677) ·
  [[Code]](https://github.com/WenjinHou/Uni-OPD)
  *Dual-perspective recipe across **5 domains and 16 benchmarks**:
  (i) data balancing for insufficient exploration of informative
  student states, and (ii) **outcome-guided margin calibration** that
  restores order-consistency between correct/incorrect trajectories
  to address unreliable teacher supervision. Supports single- or
  multi-teacher; strong-to-weak and cross-modal.*

### Offline / Resource-Efficient OPD

> Standard OPD requires a **live multi-GPU teacher server co-hosted with
> the student** for the entire training run, fragmenting compute and
> putting trillion-parameter teachers out of reach for academic labs.
> This line of work asks: *can the on-policy benefit be preserved without
> the live teacher?*

- **Lightning OPD — Efficient Post-Training for Large Reasoning Models
  with Offline On-Policy Distillation** (NVIDIA, Apr 2026)
  Yecheng Wu, Song Han, Han Cai.
  [[arXiv:2604.13010]](https://arxiv.org/abs/2604.13010) ·
  [[Code]](https://github.com/jet-ai-projects/Lightning-OPD)
  *Replaces the live teacher server with a **one-time pre-computation**
  of teacher log-probabilities over rollouts sampled from the SFT
  reference policy, reused throughout training. The catch — a naïve
  offline OPD silently underperforms — is traced to an overlooked
  **design principle**: the SFT-stage and OPD-stage teachers must be the
  **same model** (a property they call **Teacher Consistency**).
  Violation introduces a gradient bias that hurts both online and
  offline OPD, with the offline variant suffering more. A canonical
  example of violation: Tinker / Thinking Machines uses **QwQ-32B SFT
  data** but **Qwen3-32B OPD teacher**.
  Theory (3 named theorems): under teacher consistency,
  **(3.5) gradient discrepancy** between online and offline OPD is
  bounded by \(G \cdot \sigma_A \cdot \sqrt{\chi^2(\pi_\theta \,\|\, \pi_{\text{ref}})}\),
  zero at initialization;
  **(3.6) shared fixed point** — when the teacher is representable, both
  objectives have the same global optimum;
  **(3.7) gradient decomposition** — \(\nabla J_{\text{off}} = \nabla J_{\text{on}} - \mathrm{Cov}_{\pi_{\text{ref}}}[w(x;\theta), f(x;\theta)]\),
  with the covariance term acting as an **implicit trust region** that
  prevents policy drift without an explicit KL penalty.
  Numbers: Qwen3-8B-Base SFT → **AIME 2024 69.9 % in 30 GPU-hours
  (4.0× speedup over standard OPD)**. First demonstration of OPD on
  trillion-parameter-class **MoE without distributed serving infra**:
  Qwen3-30B-A3B trained on a **single 8×H100 node** to **AIME 2024
  71.0 % / LiveCodeBench v5 60.8 %**. Lowers the academic OPD barrier
  by a roughly two-orders-of-magnitude factor.*

- *Open question*: extending Lightning OPD to multi-teacher (MOPD) and
  cross-stage (OPCSD) setups — Teacher Consistency would have to be
  redefined per-stage / per-domain.

### Privacy-Preserving / DP On-Policy Distillation

> When the training corpus is sensitive (medical / legal / financial /
> proprietary), the released model must satisfy a formal privacy
> guarantee. Naïvely combining DP-SGD with KD either (a) blows up the
> privacy–utility tradeoff (DP on both teacher and student) or (b)
> requires an offline DP-synthetic-text pipeline (DistilDP). On-policy
> distillation offers a third path.

- **DP-OPD — Differentially Private On-Policy Distillation for Language
  Models** (Khadem, Mousavi, Fang, Liu, Apr 2026)
  [[arXiv:2604.04461]](https://arxiv.org/abs/2604.04461) ·
  [[Code]](https://github.com/khademfatemeh/dp_opd)
  *First DP-aware OPD framework. The student is fine-tuned with **DP-SGD
  on the private corpus**, while a **frozen, public teacher** scores
  every continuation token of the student's own rollouts. Privacy is
  consumed entirely by the student updates (per-example gradient
  clipping + Gaussian noise + RDP/subsampling accountant); teacher
  inference is internal computation and consumes **zero privacy
  budget** because the teacher is never released. Loss = GKD with
  interpolated divergence \(\beta\in[0,1]\) (β=0 forward-KL, β=0.5 JSD,
  β=1 reverse-KL); on-policy mixing rate \(\lambda\in[0,1]\) trades
  rollout cost for distribution alignment.
  Why on-policy matters more under DP: **DP-SGD noise amplifies
  exposure bias** — small local mistakes compound rapidly along
  autoregressive rollouts, so matching the teacher on **states the
  student actually visits** is critical. At ε=2.0 with
  GPT-2-Large → DistilGPT-2 (9.5× compression), DP-OPD reaches
  Yelp PPL **41.68 (vs DP-SGD 48.12, DistilDP 44.15)** and BigPatent
  **30.63 (vs 41.80 / 32.43)** — **simultaneously outperforming the
  synthesis-based DistilDP while collapsing its 3-stage pipeline (DP
  teacher training + DP synthetic generation + non-DP student training)
  into a single DP loop on a single A6000**.
  Notable counter-recipe: **β=0 (forward-KL) wins on perplexity**,
  contrary to the "reverse-KL is better" wisdom from VLA-OPD / SDPO —
  because perplexity rewards mode-covering rather than mode-seeking.
  The objective shape should match the evaluation metric.*

- *Open questions*: (i) DP-OPD with **tokenizer-mismatched** teacher
  / student (would compose with HF GOLD); (ii) **rate-limited**
  teacher-query budget; (iii) sensitive **control-code metadata**
  requiring its own privatization.

### Cascade / Pruning + Distillation

- **Cascade Distillation** (Mistral AI, 2026)
  **Ministral 3.** [[arXiv:2601.08584]](https://arxiv.org/abs/2601.08584) ·
  [[Blog (CN)]](https://mp.weixin.qq.com/s/_gN-easX9VpsW1g9aixDuw)
  *The Ministral-3 (3B / 8B / 14B) family is built by **iteratively pruning
  + continued training with distillation** from a stronger ancestor.
  Each cascade step uses the previous larger model as the on-policy
  teacher, producing parameter-efficient dense models that outperform
  same-size baselines under Apache 2.0.*

- Saurav Muralidharan et al. **Compact Language Models via Pruning and
  Knowledge Distillation (Minitron).** NVIDIA, 2024.
  [[arXiv:2407.14679]](https://arxiv.org/abs/2407.14679)
  *— the precursor recipe to cascade distillation.*

- Sahaj Dixit et al. **Llama-3.1-Minitron 4B / 8B Technical Report.**
  NVIDIA, 2024.

### RL × Distillation Connections (Inverse RL, Imitation)

- Ahmed Hussein et al. **Imitation Learning: A Survey of Learning Methods.**
  ACM Computing Surveys, 2017.
- Jonathan Ho, Stefano Ermon. **Generative Adversarial Imitation Learning
  (GAIL).** NeurIPS, 2016.
  [[arXiv:1606.03476]](https://arxiv.org/abs/1606.03476)
- Brian D. Ziebart et al. **Maximum Entropy Inverse Reinforcement Learning.**
  AAAI, 2008.
- Wenhao Yu et al. **Distillation as a Form of Implicit Reward Modeling.**
  (alignment community position note)
- Charles Sun et al. **Why Distillation Can Outperform Zero-RL.**
  [[arXiv:2505.07118]](https://arxiv.org/abs/2505.07118) (representative)
- *"Learning beyond Teacher: Generalized On-Policy Distillation with Reward
  Extrapolation"* — 2026 (extends OPD with reward extrapolation to go
  **beyond** the teacher's ceiling).

---

## Applications

### Long-CoT Reasoning

- **DASD-4B-Thinking** — small-but-strong reasoning model trained purely via
  distribution-aligned sequence distillation.
  [[arXiv:2601.09088]](https://arxiv.org/abs/2601.09088)
- **OPSD** — 8–12× sample efficiency on AIME / MATH / GSM8K using a single
  LLM as both teacher and student.
  [[arXiv:2601.18734]](https://arxiv.org/abs/2601.18734)
- **Qwen3-8B / 14B** on-policy distilled from Qwen3-32B / 235B-A22B.
  [[arXiv:2505.09388]](https://arxiv.org/abs/2505.09388)
- **DeepSeek-R1-Distill-Qwen / Llama** — off-policy reasoning distillation
  baselines. [[arXiv:2501.12948]](https://arxiv.org/abs/2501.12948)
- **MiMo-V2-Flash** — long-CoT reasoning + agentic capability via MOPD.
  [[arXiv:2601.02780]](https://arxiv.org/abs/2601.02780)
- **THUNLP "Rethinking OPD"** recipe — *off-policy cold-start +
  teacher-aligned prompt selection*; the canonical recipe for
  reasoning-OPD when teacher and student share thinking patterns.
  [[arXiv:2604.13016]](https://arxiv.org/abs/2604.13016)
- **OPD-AVMP — OPD for Autonomous Vehicle Motion Planning** (Apr 2026)
  [[arXiv:2604.07944]](https://arxiv.org/abs/2604.07944)
  *GPT-Driver framework + GKD on student-generated trajectories;
  achieves a **5× model-size reduction** in the AV planner without
  performance loss. The first OPD application in autonomous driving.*

### Reasoning Compression / Token Efficiency

> A surprisingly under-discussed application: OPD as a **reasoning-length
> compressor**. Modern reasoning models often spend thousands of tokens
> on simple problems and these tokens are not just wasteful — they are
> potential error sources.

- **OPSDC** ([arXiv:2603.05433](https://arxiv.org/abs/2603.05433)) —
  conditioning the same model on a "be concise" instruction yields a
  teacher whose distillation **compresses 40–58 % of tokens while
  improving accuracy by 10–16 pp** on MATH-500 / AIME. Difficulty-adaptive
  by construction (compresses easy problems aggressively, hard problems
  gently).
- **REOPOLD** — entropy-guided masking implicitly compresses reasoning by
  zeroing gradients on low-information tokens.
  [[arXiv:2603.11137]](https://arxiv.org/abs/2603.11137)
- **OPSD** — 1,024-token rollouts vs. GRPO's 16,384, matching accuracy
  with **8–12× token efficiency**.
  [[arXiv:2601.18734]](https://arxiv.org/abs/2601.18734)

### Agentic & Tool Use

- **SDPO** — first OPD method to show explicit gains on tool use and
  competitive programming via rich textual feedback. SDPO matches GRPO's
  final accuracy with **4× fewer rollouts**, and on chemistry tasks
  Olmo3-7B reaches in 30 min what GRPO needs 5 h for (~10× speedup).
  [[arXiv:2601.20802]](https://arxiv.org/abs/2601.20802)
- **MiMo-V2-Flash** — agentic capability via teacher specialization in MOPD.
- **Tinker Cookbook Multi-Turn (Harbor)** — multi-turn agent OPD recipe.
- **OpenClaw-RL** — terminal / GUI / SWE / tool-call agent training with
  GRPO + per-token OPD; judge model extracts hindsight hints.
  [[arXiv:2603.10165]](https://arxiv.org/abs/2603.10165)
- **SCoRe — Self-Correcting Reasoning by Larger-Teacher Hindsight**
  (Alibaba ModelScope, Sep 2025)
  [[arXiv:2509.14257]](https://arxiv.org/abs/2509.14257) ·
  [[Code (easydistill `/projects/SCoRe`)]](https://github.com/modelscope/easydistill)
  *12 agent benchmarks; 72B teacher corrects the **earliest error** in
  the student's rollout, then SFT-on-corrections + short-horizon RL.
  Result: 7B student matches the 72B teacher.*
- **Skill-SD** — multi-turn agentic OPSD with dynamic skill conditioning
  on AppWorld / Sokoban.
  [[arXiv:2604.10674]](https://arxiv.org/abs/2604.10674)
- **TCOD — Temporal Curriculum OPD** (Tongyi Lab / CUHK, Apr 2026)
  [[arXiv:2604.24005]](https://arxiv.org/abs/2604.24005)
  *Multi-turn agent training with **front-to-back / back-to-front**
  temporal curriculum scheduling on trajectory-level KL. Solves
  trajectory-level KL instability that plagues vanilla multi-turn OPD.*
- **Healthcare AI GYM** (Upstage AI / Korea Univ., May 2026)
  [[arXiv:2605.02943]](https://arxiv.org/abs/2605.02943)
  *Clinical-agent RL environment + EMA teacher with outcome-privileged
  info providing **dense turn-level KL regularisation** (TT-OPD =
  turn-level truncated OPD) on top of GRPO.*
- **HyperEyes — Parallel Multimodal Search Agent**
  (Xiaohongshu / Cambridge, May 2026)
  [[arXiv:2605.07177]](https://arxiv.org/abs/2605.07177) ·
  [[Code]](https://github.com/DeepExperience/HyperEyes)
  *TRACE (trajectory-level adaptive cost efficiency) + token-level OPD
  + GRPO; macro × micro dual-grained efficiency-aware RL for parallel
  multimodal search agents.*
- **π-Play** — data-free self-play OPSD for search agents
  (NQ / TriviaQA / HotpotQA / MuSiQue …).
  [[arXiv:2604.14054]](https://arxiv.org/abs/2604.14054)
- **LLM4Teach — LLM-Guided Small RL Agent** (ZJ Lab AMMI, 2023, updated 2025)
  [[arXiv:2311.13373]](https://arxiv.org/abs/2311.13373) ·
  [[Code]](https://github.com/ZJLAB-AMMI/LLM4Teach)
  *Strict OPD for embodied agents — pre-dates the recent wave. LLM
  teacher provides action-level distillation targets while the small
  RL student trains; loss = annealed distillation + RL.*
- **RPD — Refined Policy Distillation for VLA** (TUM / Freiburg, IROS 2026)
  [[arXiv:2503.05833]](https://arxiv.org/abs/2503.05833) ·
  [[Project]](https://refined-policy-distillation.github.io/)
  *PPO + behavioural cloning on student rollouts; VLA / robot
  manipulation. The cleanest VLA-OPD recipe before VLA-OPD itself.*
- *Open question*: long-horizon agent trajectories with partial-credit
  feedback — flagged as open problem in the 2604.00626 survey.

### Continual Learning & Catastrophic Forgetting

- **GLM-5 OPCSD** — frontier-scale evidence that **On-Policy Cross-Stage
  Distillation** can eliminate inter-stage capability regression across a
  3-phase RL pipeline (Reasoning → Agentic → General).
  [[arXiv:2602.15763]](https://arxiv.org/abs/2602.15763)
- **SDFT** — establishes OPD as a *practical path* to continual learning from
  demonstrations. [[arXiv:2601.19897]](https://arxiv.org/abs/2601.19897)
- **Thinking Machines Assistant Personalization** — the Thinking Machines
  blog shows OPD recovers lost capabilities after personalization fine-tuning,
  without re-running full RL.
- James Kirkpatrick et al. **Overcoming Catastrophic Forgetting in Neural
  Networks (EWC).** PNAS, 2017.
- Zhizhong Li, Derek Hoiem. **Learning without Forgetting (LwF).**
  TPAMI, 2017 *— the spiritual ancestor: distillation as a regularizer
  against forgetting.*

### Model Compression / Small Language Models

- **TAID-LLM-1.5B / TAID-VLM-2B** (Sakana AI, ICLR 2025) — SOTA <2B models
  via **Temporally Adaptive Interpolated Distillation**.
  [[arXiv:2501.16937]](https://arxiv.org/abs/2501.16937)
- **Gemma 3** (1B / 4B / 12B / 27B) — KD during pre-training + OPD-style
  post-training. [[arXiv:2503.19786]](https://arxiv.org/abs/2503.19786)
- **Gemma 2** (2B / 9B / 27B) — the first widely-adopted frontier open-weight
  family to scale knowledge distillation to trillions of tokens.
- **Ministral 3** (Cascade Distillation, 3B / 8B / 14B).
  [[arXiv:2601.08584]](https://arxiv.org/abs/2601.08584)
- **Qwen3-0.6B / 1.7B / 4B / 8B / 14B** (Strong-to-Weak OPD).
- **Minitron / Llama-3.1-Minitron** (Pruning + KD).
- **Phi-4 / Phi-4-Reasoning** (Microsoft, 2024–2026) — synthetic-data-heavy
  14 B model. Phi-1/2/3 explicitly distilled GPT-4; Phi-4 transitions from
  *distillation* to *synthetic data generated by a teacher*, surpassing the
  teacher on STEM QA. [[arXiv:2412.08905]](https://arxiv.org/abs/2412.08905)
- **MiniCPM-4, Phi-3, Gemma-2** — community recipes that increasingly resemble
  OPD.
- **MobileLLM**, **OLMoE-1B-7B** — small-model technical reports with KD
  ablations relevant to OPD.

### Multimodal & Video

- **Video-OPD** — first OPD method on **video temporal grounding**;
  +17 % R@0.7 over GRPO across Charades / ActivityNet / QVHighlights.
  [[arXiv:2602.02994]](https://arxiv.org/abs/2602.02994)
- **REOPOLD-3B** — vision-language student matches a 32B teacher with
  3.3× speedup (Geometry3K) and 2.2× speedup (MathVerse).
  [[arXiv:2603.11137]](https://arxiv.org/abs/2603.11137)
- **TAID-VLM-2B** — best-in-class VLM ≤4B via interpolated distillation.
  [[arXiv:2501.16937]](https://arxiv.org/abs/2501.16937)

### Robotics & Embodied AI

- **VLA-OPD** — first OPD method on **robot manipulation**. Replaces
  the sparse 0/1 environment reward with the teacher's per-action
  log-prob on student-visited states; LIBERO 1-demo SFT 48.9 % →
  87.4 %, RoboTwin2.0 dual-arm 45.2 % → 71.1 %, with 15× faster
  convergence than GRPO. Empirically establishes
  **Reverse-KL > Forward-KL > Hard-CE** for OOD action distributions.
  [[arXiv:2603.26666]](https://arxiv.org/abs/2603.26666)
- **RPD — Refined Policy Distillation** (TUM / Freiburg, IROS 2026) —
  PPO + behavioural cloning on student rollouts; the cleanest VLA-OPD
  recipe before VLA-OPD itself.
  [[arXiv:2503.05833]](https://arxiv.org/abs/2503.05833)
- **HY-Embodied 0.5** (Tencent Hunyuan, Apr 2026) — FKL OPD from a 32B
  embodied teacher into a 2B MoT edge variant; downstream VLA on a
  dual-arm Xtrainer robot. [[arXiv:2604.07430](https://arxiv.org/abs/2604.07430)]
- **LLM4Teach** (ZJ Lab AMMI, 2023) — strict OPD for embodied agents,
  pre-dating the wave. LLM teacher provides action-level distillation
  targets. [[arXiv:2311.13373]](https://arxiv.org/abs/2311.13373)
- *Open question*: trillion-parameter VLA + multi-step planning + OPD —
  no work yet, but a natural next step combining MOPD-style
  multi-teacher with VLA-OPD's reverse-KL recipe.

### Privacy-Preserving Compression

> Use case: shrink a model trained on **sensitive data** (medical /
> legal / financial / proprietary corpus) under a formal differential
> privacy budget, suitable for on-device or regulated deployment.

- **DP-OPD** ([arXiv:2604.04461](https://arxiv.org/abs/2604.04461)) —
  At ε=2.0, **GPT-2-Large → DistilGPT-2 (9.5× compression)** reaches
  Yelp PPL 41.68 (vs DP-SGD 48.12, DistilDP 44.15) and BigPatent 30.63
  (vs 41.80 / 32.43), running on a **single A6000** instead of the
  multi-GPU multi-day DistilDP pipeline.
- *Open recipe*: pair DP-OPD with **HF GOLD** to enable
  cross-tokenizer DP distillation (e.g., LLaMA student trained on
  private medical text with a public Qwen teacher).

### Code & Math

- **SDPO** on LiveCodeBench v6.
- **OPSD** on AIME / MATH / GSM8K / Olympiad-Bench.
- **OPSDC** — Qwen3-14B reaches **86.1 %** on MATH-500 (up from 70.0 %)
  with 56.5 % fewer tokens. [[arXiv:2603.05433]](https://arxiv.org/abs/2603.05433)
- **EOPD** on Qwen3-4B — Pass@8 +5.05 over baseline OPD across 6 math
  benchmarks. [[arXiv:2603.07079]](https://arxiv.org/abs/2603.07079)
- **REOPOLD** — Pass@1 32–34 % on AIME-25, 6.7–12× more sample-efficient
  than ProRL. [[arXiv:2603.11137]](https://arxiv.org/abs/2603.11137)
- **DASD-4B-Thinking** on AIME / GPQA / LiveCodeBench.
- **MiMo-V2-Flash** on full math + code suite.
- **Qwen3-8B + OPD** achieves AIME'24 74.4 %, MATH500 97.0 %,
  LiveCodeBench v5 60.3 %. ([Table 21, Qwen3 TR](https://arxiv.org/abs/2505.09388))
- **Lightning-OPD-Qwen3-8B** — AIME'24 69.9 % in **30 GPU-hours**
  (vs. 120 for standard OPD), and Qwen3-30B-A3B MoE on **a single
  8×H100 node** at 71.0 % AIME / 60.8 % LCBv5 — the strongest
  small-budget reasoning result to date.
  [[arXiv:2604.13010]](https://arxiv.org/abs/2604.13010)

---

## Models & Technical Reports

| Model | Year | Distillation Recipe | Link |
|------|------|--------------------|------|
| **GLM-5 / GLM-5.1** (744B-A40B MoE) | 2026 | **On-Policy Cross-Stage Distillation (OPCSD)** across Reasoning → Agentic → General RL | [arXiv:2602.15763](https://arxiv.org/abs/2602.15763) |
| **DASD-4B-Thinking** | 2026 | Distribution-Aligned Sequence Distillation | [arXiv:2601.09088](https://arxiv.org/abs/2601.09088) |
| **Ministral 3** (3B/8B/14B) | 2026 | Cascade Distillation (iterative prune + KD) | [arXiv:2601.08584](https://arxiv.org/abs/2601.08584) |
| **MiMo-V2-Flash** (309B-A15B MoE) | 2026 | Multi-Teacher On-Policy Distillation (MOPD) | [arXiv:2601.02780](https://arxiv.org/abs/2601.02780) |
| **OPSD-Distilled Reasoner** | 2026 | On-Policy Self-Distillation (single-model) | [arXiv:2601.18734](https://arxiv.org/abs/2601.18734) |
| **Kimi K2** (1T-A32B MoE) | 2025–26 | RLVR + rubric-distillation into policy | [arXiv:2507.20534](https://arxiv.org/abs/2507.20534) |
| **Qwen3-\*-OPD** (0.6B–14B + 30B-A3B) | 2025 | Strong-to-Weak (off-policy SFT → on-policy RKL) | [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) |
| **Gemma 3** (1B/4B/12B/27B) | 2025 | KD in pre-training + GKD-style IT post-training | [arXiv:2503.19786](https://arxiv.org/abs/2503.19786) |
| **Gemma 2** (2B/9B/27B) | 2024 | Token-level KD from a Gemini teacher | [Gemma 2 TR](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf) |
| **TAID-LLM-1.5B / VLM-2B** | 2025 | Temporally Adaptive Interpolated Distillation | [arXiv:2501.16937](https://arxiv.org/abs/2501.16937) |
| **DeepSeek-R1-Distill-\*** | 2025 | Off-policy SFT distillation (baseline) | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| **Phi-4** (14B) | 2024 | Synthetic-data post-distillation ("beyond KD") | [arXiv:2412.08905](https://arxiv.org/abs/2412.08905) |
| **Llama-3.1-Minitron** (4B/8B) | 2024 | Pruning + KD | NVIDIA TR |
| **MiniLLM-OPT/Llama** | 2023 | Reverse-KL on-policy KD | [arXiv:2306.08543](https://arxiv.org/abs/2306.08543) |
| **Zephyr-7B (SPIN)** | 2024 | Self-play fine-tuning (implicit self-distillation) | [arXiv:2401.01335](https://arxiv.org/abs/2401.01335) |
| **Video-OPD-8B** (Qwen3-VL-8B) | 2026 | On-policy distillation for video temporal grounding (TVDF curriculum) | [arXiv:2602.02994](https://arxiv.org/abs/2602.02994) |
| **OPSDC-Qwen3-14B-Compact** | 2026 | "Be concise" self-distillation; –56.5 % tokens, +16 pp MATH-500 | [arXiv:2603.05433](https://arxiv.org/abs/2603.05433) |
| **REOPOLD-3B / 7B** | 2026 | Reward-clipped, entropy-masked OPD (RL-style) | [arXiv:2603.11137](https://arxiv.org/abs/2603.11137) |
| **EOPD-Qwen3-4B** | 2026 | Entropy-aware reverse-KL + forward-KL switch | [arXiv:2603.07079](https://arxiv.org/abs/2603.07079) |
| **StableOPD-Qwen2.5-1.5B / 7B** | 2026 | Reference-KL + rollout-mixture distillation; cures repetition collapse | [arXiv:2604.08527](https://arxiv.org/abs/2604.08527) |
| **Revisiting-OPD-Qwen** | 2026 | Top-K + top-p local-support reverse-KL (cures tokenizer-mismatch & signal-imbalance) | [arXiv:2603.25562](https://arxiv.org/abs/2603.25562) |
| **SCOPE-R1-Distill-Qwen-1.5B** | 2026 | Dual-path PPL-weighted (correct vs incorrect) OPD | [arXiv:2604.10688](https://arxiv.org/abs/2604.10688) |
| **VLA-OPD-LIBERO / RoboTwin** | 2026 | Reverse-KL OPD for VLA robot models (1-demo recipe) | [arXiv:2603.26666](https://arxiv.org/abs/2603.26666) |
| **Lightning-OPD-Qwen3-8B** | 2026 (NVIDIA) | **Offline OPD** with Teacher Consistency; AIME 2024 69.9 % in 30 GPU-hours, 4× speedup | [arXiv:2604.13010](https://arxiv.org/abs/2604.13010) |
| **Lightning-OPD-Qwen3-30B-A3B** (MoE) | 2026 (NVIDIA) | First single-node (8×H100) OPD on 30B-MoE; AIME 2024 71.0 % / LCBv5 60.8 % | [arXiv:2604.13010](https://arxiv.org/abs/2604.13010) |
| **DP-OPD-DistilGPT-2** | 2026 | First differentially private OPD; ε=2.0, Yelp PPL 41.68 / BigPatent PPL 30.63 (beats DistilDP) | [arXiv:2604.04461](https://arxiv.org/abs/2604.04461) |
| **GLM-4.5 / 4.6** (355B-A32B MoE) | 2025 | Expert iteration + SFT distillation; predecessor of GLM-5 | [arXiv:2508.06471](https://arxiv.org/abs/2508.06471) |
| **Baichuan-M3-235B** (MoE) | 2026 | 3-stage MOPD pipeline (TaskRL → Clip-FKL → Reverse-KL); medical-domain SOTA (HealthBench-Hard 44.4) | [arXiv:2602.06570](https://arxiv.org/abs/2602.06570) |
| **HY-MT 1.5** (1.8B / 7B) | 2025 (Tencent Hunyuan) | RKL strong-to-weak distillation; ~90 % of Gemini-3.0-Pro MT performance with 1.8B params; WMT25 champion lineage | [arXiv:2512.24092](https://arxiv.org/abs/2512.24092) |
| **HY-Embodied 0.5** (MoT-2B edge) | 2026 (Tencent Hunyuan) | FKL OPD from 32B → MoT-2B for embodied reasoning; downstream VLA for dual-arm Xtrainer robot | [arXiv:2604.07430](https://arxiv.org/abs/2604.07430) |
| **Nemotron Cascade 2** | 2026 (NVIDIA) | Multi-Domain On-Policy Distillation between Cascade RL stages; matches RL in 30–160 steps vs 1000+ | [arXiv:2603.19220](https://arxiv.org/abs/2603.19220) |
| **Qwen3-Coder-Next** (80A3 MoE) | 2026 | Combined SFT + on-policy logit alignment; multi-experts → 80A3 student | [Qwen3-Coder TR](https://github.com/QwenLM/Qwen3-Coder) |
| **KAT-Coder-V2** | 2026 (Kuaishou KwaiKAT) | RL + **step-level** OPD (reasoning-tree alignment); 79.6 % SWE-bench Verified vs Claude Opus 4.6's 80.8 %; Tree-Training 6.2× speedup | [arXiv:2603.27703](https://arxiv.org/abs/2603.27703) |
| **DeepSeek-V4** (V4-Pro 1.6T MoE / V4-Flash 284B) | 2026 (DeepSeek) | **Multi-teacher OPD replaces unified mixed-RL stage**: per-domain SFT + GRPO specialists → unified student with **full-vocabulary RKL** on its own rollouts | [Tech Report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) |
| **Qwen3.5-Omni** | 2026 (Alibaba Qwen) | Cross-modal OPD: text reasoning → audio-input reasoning; Thinker-Talker architecture with Hybrid Attention MoE; 215 audio/AV subtasks SOTA; ARIA streaming alignment | [arXiv:2604.15804](https://arxiv.org/abs/2604.15804) |
| **SD-Zero (Qwen3-4B-Instruct / Olmo-3-7B-Instruct)** | 2026 | Self-Revision converts binary RLVR rewards into dense per-token OPSD signal; ≥10 % gain over base; outperforms RFT/GRPO/SDFT at matched sample budget | [arXiv:2604.12002](https://arxiv.org/abs/2604.12002) |
| **GAD-Qwen2.5-14B** | 2025 (Microsoft) | First **black-box OPD** at scale: comparable to GPT-5-Chat on LMSYS without teacher logits | [arXiv:2511.10643](https://arxiv.org/abs/2511.10643) |

---

## Codes & Frameworks

### OPD-Specific Libraries

- **Tinker Cookbook** (Thinking Machines) — the reference implementation
  that accompanies the "On-Policy Distillation" blog; supports reasoning,
  personalization, multi-turn agent, and multi-teacher OPD recipes out of
  the box.
  <https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation>

- **FlashOPD** (china10s) — "6 files · 650 LOC" minimal OPD library, with
  `forward / reverse / JSD` KL, API-based (vLLM OpenAI) teacher, dynamic
  CE+KL loss balancing, KV-cache accelerated student rollout, DeepSpeed /
  FSDP support. CleanRL-style readability.
  <https://github.com/china10s/flash-opd>

- **OPSD official code** — On-Policy Self-Distillation training scripts.
  <https://github.com/siyan-zhao/OPSD>

- **OPSDC official code** — On-Policy Self-Distillation for Reasoning
  Compression (the "be concise" recipe).
  <https://github.com/HJSang/OPSD_Reasoning_Compression>

- **SCOPE official code** — Signal-Calibrated OPD with dual-path
  PPL-weighted training. <https://github.com/machine981/SCOPE>

- **Lightning OPD** (NVIDIA, jet-ai-projects) — offline OPD with
  pre-computed teacher log-probs; the most resource-efficient OPD
  recipe to date (4× faster than standard OPD; 30B-MoE on a single
  8×H100 node). <https://github.com/jet-ai-projects/Lightning-OPD>

- **DP-OPD** (Khadem et al.) — differentially private OPD reference
  implementation with DP-SGD on the student and a frozen public
  teacher. Single-GPU, RDP-accountant compatible.
  <https://github.com/khademfatemeh/dp_opd>

- **MiniLLM (Microsoft LMOps)** — official PyTorch implementation of
  reverse-KL on-policy distillation for LLMs.
  <https://github.com/microsoft/LMOps/tree/main/minillm>

- **DistiLLM / DistiLLM-2** — official code.
  <https://github.com/jongwooko/distillm>

- **Speculative KD** (Google Research).
  <https://github.com/google-research/google-research/tree/master/speculative_kd>

### General Frameworks That Support OPD

- **TRL** (Hugging Face) — by far the **broadest open-source OPD trainer
  collection**. `trl/experimental/` ships **gkd, gold, minillm, sdft,
  self_distillation, sdpo, nash_md, xpo, online_dpo, papo, prm**.
  Native FKL / RKL / GJSD-β. <https://github.com/huggingface/trl>
- **verl** (ByteDance Seed) — production-ready OPD recipe at
  `recipe/on_policy_distill/`; integrates with **vLLM**; supports
  FSDP / Megatron / Ray. Documents an [Async OPD](https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html)
  variant. <https://github.com/volcengine/verl>
- **NeMo-RL** (NVIDIA, Jan 2026) — native OPD with student rollouts at
  `nemo_rl/algorithms/distillation.py`; FKL / RKL / mixed configurable
  via `kl_type`. Replaces the archived NeMo-Aligner. Backbone =
  Ray + Megatron + vLLM. <https://github.com/NVIDIA-NeMo/RL>
- **SkyRL** (UC Berkeley NovaSky, Apr 2025) — OPD recipe added Nov 2025
  (PR #585) at `skyrl-train/examples/on_policy_distillation/`;
  reverse-KL with importance sampling; Ray + vLLM/SGLang.
  [[Blog]](https://novasky-ai.notion.site/on-policy-distillation) ·
  <https://github.com/NovaSky-AI/SkyRL>
- **rllm** (UC Berkeley Sky, Jan 2025) — math-distill examples include
  an `opsd/` self-distillation subdir; reverse-KL with advantage =
  log P_teacher − log P_student. Single-GPU (Tinker) + multi-GPU (verl).
  <https://github.com/rllm-org/rllm>
- **ROLL** (Alibaba, Jun 2025) — first-class `DistillPipeline`;
  `roll/pipeline/distill/` with `various_divergence.py` (multiple loss
  options); native VLM support; Megatron backbone.
  <https://github.com/alibaba/ROLL>
- **AReaL** (Ant Group / Tsinghua, Jun 2025) — async-distributed RL
  framework with `examples/distillation/gsm8k_grpo_distill.yaml`;
  `distill_loss_weight` integrates KD into GRPO.
  <https://github.com/inclusionAI/AReaL>
- **slime** (Z.ai / Tsinghua) — asynchronous RL framework behind
  **GLM-4.5 / 4.6 / 5**; OPD as additive penalty on any advantage
  estimator. `examples/on_policy_distillation/`. SGLang teacher mode.
  <https://github.com/THUDM/slime>
- **KDFlow** (BJTU, Mar 2026) — **KD-first framework** with decoupled
  SGLang teacher + FSDP2 student; transmits teacher hidden states
  (zero-copy) and recomputes logits on the student to cut comm cost,
  achieving **1.44–6.36× speedup** over homogeneous-backend baselines.
  Native cross-tokenizer + Qwen3-VL multimodal. Examples include
  `examples/on_policy_kd/` for both LLM and VLM.
  [[arXiv:2603.01875]](https://arxiv.org/abs/2603.01875) ·
  <https://github.com/songmzhang/KDFlow>
- **ms-swift** (Alibaba ModelScope) — wraps TRL `GKDTrainer`; ships
  `examples/train/rlhf/gkd/` and multimodal/Megatron variants.
  <https://github.com/modelscope/ms-swift>
- **LLaMA-Factory** — most-starred SFT/DPO framework; supports OPD
  **only via TRL integration** (no native OPD trainer).
  <https://github.com/hiyouga/LLaMA-Factory>
- **OpenRLHF** — RLHF framework with hooks suitable for OPD experiments;
  no native OPD recipe shipped. <https://github.com/OpenRLHF/OpenRLHF>
- **EasyDistill** (Alibaba) — end-to-end distillation framework covering
  both off- and on-policy modes (the latter via custom recipes).
  <https://github.com/modelscope/easydistill>
- **SakanaAI/TAID** — reference implementation of Temporally Adaptive
  Interpolated Distillation. <https://github.com/SakanaAI/TAID>
- **UCLA-AGI SPIN** — official self-play fine-tuning code.
  <https://github.com/uclaml/SPIN>

#### Speculative-Decoding Drafter Frameworks

- **SpecForge** (SGLang, Mar 2026) — open-source EAGLE-3 drafter
  training framework; on-policy TTT supported; companion to SGLang
  inference. <https://github.com/sgl-project/SpecForge>
- **EAGLE-3 official** — reference EAGLE-3 / TTT implementation.
  <https://github.com/SafeAILab/EAGLE>
- **OSD** — official Online Speculative Decoding code.
  <https://github.com/LiuXiaoxuanPKU/OSD>

---

## Tutorials & Blogs

### English

- **On-Policy Distillation** — Thinking Machines Lab, Kevin Lu, Oct 27, 2025.
  [[Blog]](https://thinkingmachines.ai/blog/on-policy-distillation)
- **Tinker Model Distillation Documentation.**
  <https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/>
- **On-Policy Distillation: Cheap Accuracy, Real Gains** — Mahesh Lambe,
  Medium, Oct 2025.
- **ML Point: On-Policy Distillation by Thinking Machines Lab** — deep-dive
  Medium article.
- **The hidden trap of LLMs self-distillation** — Ben Dickson, TechTalks,
  Apr 2026. [[Blog]](https://bdtechtalks.substack.com/p/the-hidden-trap-of-llms-self-distillation)
- **Unlocking On-Policy Distillation for Any Model Family** — Patiño,
  Tunstall, Beeching, Gallouédec et al., Hugging Face H4, Oct 2025.
  [[Blog]](https://huggingfaceh4-on-policy-distillation.hf.space/) ·
  [[Space]](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation)
  *Introduces **GOLD** (General Online Logit Distillation) — the first
  open recipe to make OPD work between **mismatched tokenizers**, e.g.
  LLaMA student with a Qwen teacher.*

### Chinese (中文资料)

- 《[万字长文总结 RL / on policy distillation 的一些进展](https://zhuanlan.zhihu.com/p/2004506304065065334)》— 知乎综述, 2026.
- 《[MIT 提出 SDFT：作为逆强化学习的在线自蒸馏](https://mp.weixin.qq.com/s/7BKxOoS5iqah27OrxY9BzA)》— SDFT 论文解读.
- 《[自蒸馏优化 SDPO：如何利用富文本反馈打破 RLVR 的信用分配瓶颈？](https://mp.weixin.qq.com/s/eRqxe7qcRxWJYE86UXv8Nw)》— SDPO 论文解读.
- 《[阿里云提出 DASD：分布对齐的序列蒸馏，实现更优的长链思维推理](https://mp.weixin.qq.com/s/TOeY5rkKUFl_cYaOKQaMaA)》— DASD 论文解读.
- 《[深度解析 Ministral 3：基于级联蒸馏的参数高效密集模型训练方法论](https://mp.weixin.qq.com/s/_gN-easX9VpsW1g9aixDuw)》— Ministral 3 论文解读.
- 《[小米 MiMo-V2-Flash 技术报告：MoE 架构、混合注意力机制与多教师在线蒸馏](https://mp.weixin.qq.com/s/3a2xz8LYhyV6udSgxuQFoA)》— MiMo-V2-Flash 论文解读.
- 《[长文总结：近半年 On-Policy Distillation 的三大主流方向](https://zhuanlan.zhihu.com/p/2020191969205306820)》— 9 篇核心 OPD 论文的深度纵览（稳定性 / 自蒸馏 / 场景扩展三大方向）.
- 《[On-Policy Distillation 是什么？如何做？](https://zhuanlan.zhihu.com/p/2000612721868177979)》— kxzxvbk (BUAA), 教程式入门与公式推导.

### Related Awesome Lists

- **[thinkwee/AwesomeOPD](https://github.com/thinkwee/AwesomeOPD)** — the
  most comprehensive sister list. Strict-OPD-formal taxonomy along
  four design axes (teacher source / supervision signal / rollout
  consumption / pipeline slot), ~80 entries with **technical-detail
  tables** and **strictness notes** distinguishing C1 / C2 satisfaction.
  Released 2026-04-28. **The reference catalogue** if you want a flat
  table view. We follow many of their classifications below; see in
  particular their separate sections for *Black-Box*, *Speculative-
  Decoding* and *Iterative Self-Bootstrapping* — all of which we
  mirror here.
- **OPD survey index** — the [Tencent OPD Survey](https://arxiv.org/abs/2604.00626)
  serves as a complementary academic index; ~50 methods catalogued.
- **THUNLP/OPD** — codebase for the *Rethinking OPD: Phenomenology,
  Mechanism, Recipe* paper; useful as both a method and a reference
  recipe. <https://github.com/thunlp/OPD>

---

## Benchmarks & Datasets

- **LiveCodeBench v5 / v6** — used by SDPO and Qwen3-OPD to demonstrate
  dense-feedback gains in competitive programming.
- **AIME'24 / AIME'25 / MATH500 / GPQA-Diamond / Olympiad-Bench** —
  standard long-CoT reasoning benchmarks for OPSD, DASD-4B-Thinking,
  MiMo-V2-Flash, Qwen3, etc.
- **MMLU-Pro / IFEval / Arena-Hard** — general capability tracking for
  Ministral 3, Qwen3 and other distilled small models.
- **Continual-LM** (introduced in SDFT) — sequential skill / knowledge
  acquisition benchmark for on-policy continual learning.
- **DeepMath** — reasoning distillation dataset used by Tinker Cookbook.
- **OpenThoughts3 / Tulu3** — personalization / instruction-following
  distillation datasets.
- **DASD-448K** — open-source distillation dataset accompanying
  DASD-4B-Thinking.

---

## Limitations & Open Problems

Recent empirical and theoretical work has flagged several *non-trivial*
limitations of OPD that are worth tracking:

1. **Thinking-pattern incompatibility & "fake" stronger teachers.**
   *Rethinking OPD* ([arXiv:2604.13016](https://arxiv.org/abs/2604.13016),
   [code](https://github.com/thunlp/OPD)) shows that teacher–student pattern
   mismatch can cause silent training failures even when the teacher is
   objectively stronger; **same-family** teachers (e.g., 7B → 1.5B of the
   same series) are often distributionally indistinguishable from the
   student, providing essentially no signal — a finding it calls
   "weak-to-strong reverse distillation."
2. **Exploration collapse in self-distillation.** The Microsoft / KAIST / SNU
   study reports up to **40 % OOD accuracy drops** when self-distillation is
   applied aggressively; epistemic-verbalization diversity in training data
   is identified as a crucial mitigation.
3. **Training instability without importance sampling.** Community
   reproduction of Qwen3's OPD (see
   [QwenLM/Qwen3#1799](https://github.com/QwenLM/Qwen3/issues/1799)) finds
   that without sentence-level importance weighting or ratio clipping, errors
   compound and training collapses.
4. **Style-token gradient dominance.** OPSD v3
   ([arXiv:2601.18734v3](https://arxiv.org/abs/2601.18734v3)) shows that a
   small number of stylistic tokens (`wait`, `think`, etc.) can absorb
   **6–15×** more KL mass than content tokens, silently hijacking the
   optimization — a finding orthogonal to the classic clipping advice above.
5. **Scaling to long-horizon agent trajectories.** The 2604.00626 survey
   lists agent-level OPD as the most important open problem — dense
   token-level feedback becomes less meaningful when the useful reward is
   many turns away. *Rethinking OPD*
   ([arXiv:2604.13016](https://arxiv.org/abs/2604.13016)) reaches the same
   conclusion empirically: dense token-level reward is "not free" and its
   benefit shrinks as the horizon grows.
6. **Distillation scaling laws.** There is currently no analog of Chinchilla
   for OPD: how does optimal compute split between teacher rollouts, student
   rollouts, and KL regularization as you scale student / teacher / data?
7. **Repetition collapse as a built-in reward-hacking failure mode.**
   *StableOPD* ([arXiv:2604.08527](https://arxiv.org/abs/2604.08527)) shows
   a phase transition ~30 steps in: when the student starts looping, the
   stronger teacher becomes *more* confident on the repeating context, so
   the OPD reward \(\log P_T - \log P_S\) becomes positive and **the
   advantage of repetition tokens spikes to 4–9× normal** — a
   self-reinforcing loop that crashes accuracy. Reverse-KL has a
   *systematic* preference for local repetition; on-policy sampling
   amplifies it.
8. **Dense reward quality decays with sequence depth.** *Rethinking OPD*
   ([arXiv:2604.13016](https://arxiv.org/abs/2604.13016)) measures
   teacher-vs-student continuation accuracy at increasing prefix lengths:
   advantage shrinks from **+0.37 at 1K tokens to +0.02 at 16K tokens**.
   The "dense reward" is densest at the start of a sequence and turns
   into noise by the end — particularly damaging for long-CoT.
9. **Single-token sampling has three structural bugs.** *Revisiting OPD*
   ([arXiv:2603.25562](https://arxiv.org/abs/2603.25562)) catalogues the
   defaults that Qwen3 / MiMo-V2 ship: signal imbalance (most samples
   negative), out-of-support teacher unreliability, and tokenizer-split
   mismatch (`<think>` → `<,think,>` vs `<th,ink,>`). Local-support-set
   matching fixes all three at near-zero compute cost.
10. **Pass@k paradox.** *SCOPE* ([arXiv:2604.10688](https://arxiv.org/abs/2604.10688))
    shows that uniform reinforcement of correct rollouts kills minority-correct
    paths: Qwen2.5-7B Pass@32 drops **93.7 % → 84.9 %** while Pass@1
    improves. Plain OPD without correctness-aware weighting silently
    sacrifices solution diversity.
11. **Toxic-prefix trap.** *SCOPE* also shows that teacher recovery from
    bad student prefixes is reliable for low-PPL prefixes (64.9 %) but
    drops to 45.4 % for high-PPL prefixes. Naïvely teaching from "fix
    this broken prefix" trajectories can inject more noise than signal.
12. **Teacher inconsistency between SFT and OPD stages.** *Lightning
    OPD* ([arXiv:2604.13010](https://arxiv.org/abs/2604.13010)) shows
    that using *different* teachers in the SFT and OPD stages — a
    convention silently inherited from RLVR pipelines — introduces a
    persistent gradient bias that degrades **both online and offline
    OPD**. Pinpointed in widely-used recipes (e.g., QwQ-32B SFT data
    with Qwen3-32B OPD teacher in Tinker). A first-principles failure
    mode that's invisible in the loss curve but compounds over training.
13. **Live-teacher serving as the dominant cost.** Standard OPD requires
    co-hosting student and teacher on the same GPU pool throughout
    training, fragmenting compute and putting trillion-parameter
    teachers out of academic reach. Lightning OPD removes this for the
    *single*-teacher case via offline pre-computation; but multi-teacher
    (MOPD), cross-stage (OPCSD), and self-evolving teacher schemes
    cannot easily reuse the same trick — an open infra problem.
14. **Privacy threat model is incomplete for OPD.** *DP-OPD*
    ([arXiv:2604.04461](https://arxiv.org/abs/2604.04461)) shows that
    OPD can be made formally DP at the student updates, but: (i) the
    **public teacher** must genuinely be independent of the private
    corpus — usually true for off-the-shelf models, hard to verify
    in industry settings; (ii) **control codes / metadata** (e.g.,
    business categories, CPC codes) used to condition the prompt may
    leak distributional information that is itself sensitive; (iii)
    **rate-limited or pay-per-token** teacher endpoints shift the
    bottleneck from compute to query budget. End-to-end privacy
    guarantees beyond the student gradient remain an open problem.

---

## Best Practices & Recipes

> A consolidated recipe shelf distilled from the failure-mode literature
> above. None of these are mandatory, but skipping any of them invites
> one of the failure modes in the previous section.

0. **Teacher Consistency** (Lightning OPD,
   [arXiv:2604.13010](https://arxiv.org/abs/2604.13010))
   *Use the same teacher model in the **SFT stage** that produces your
   reference policy and in the **OPD stage** that scores rollouts.*
   This sounds trivial but is broken in practice — e.g., Thinking
   Machines pairs **QwQ-32B SFT data** with a **Qwen3-32B OPD teacher**.
   Lightning OPD proves the mismatch yields a gradient bias bounded by
   \(G \cdot \sigma_\Delta \cdot \sqrt{\chi^2(\pi_\theta\|\pi_{\text{ref}})}\)
   that strictly degrades both online *and* offline OPD. The cleanest
   way to enforce it is to **regenerate your SFT data with the OPD
   teacher** before starting the pipeline.
1. **Pre-flight diagnosis (Rethinking OPD, [arXiv:2604.13016](https://arxiv.org/abs/2604.13016))**
   - Measure **overlap ratio** = (student top-k ∩ teacher top-k) / k at
     student-visited states. Successful OPD trends from ~72 % to ≥91 %;
     a flat curve means the teacher offers no new signal — abort.
   - Run **reverse-distillation sanity check**: if "stronger" teacher
     pulls a strong RL'd student *down* to its un-RL'd sibling, the
     teacher is the same distribution as your student — find a teacher
     from a *different* training pipeline.
2. **SFT cold start + prompt-template alignment** (Rethinking OPD)
   *Generate ~200K demonstrations from the teacher and run a brief SFT
   pass before OPD; use the teacher's training prompt format verbatim
   for the student rollouts.* Single biggest stability win in the paper.
   **Bonus**: this is one of two ways (the other being SFT-data
   regeneration) to satisfy the Teacher Consistency principle above.
3. **KL anchor + golden mixture** (StableOPD,
   [arXiv:2604.08527](https://arxiv.org/abs/2604.08527))
   - Add a reference-model KL term against the **initial student
     checkpoint** to bound policy drift speed.
   - Mix in **filtered SFT** examples (e.g., OpenR1-Math-220k filtered
     by length & correctness) every step. Detect repetition via zlib
     compression ratio > 10× and trigger early stopping if it appears.
4. **Local-support reverse-KL** (Revisiting OPD,
   [arXiv:2603.25562](https://arxiv.org/abs/2603.25562))
   At each prefix, compute reverse-KL **only on teacher's top-K (with
   optional top-p filter), with both distributions renormalized onto
   that support**. Fixes signal imbalance, OOS unreliability, and
   tokenizer artifacts in one stroke. Drop-in upgrade for any GKD-style
   trainer.
5. **Dual-path PPL weighting** (SCOPE,
   [arXiv:2604.10688](https://arxiv.org/abs/2604.10688))
   Split rollouts by correctness:
   - **Wrong** → teacher-KL weighted by **1/teacher-PPL** (group-softmax,
     τ = 1.0).
   - **Right** → MLE weighted by **student-PPL** (boost
     low-confidence successes).
6. **Choose Reverse-KL for OOD-heavy problems** (VLA-OPD,
   [arXiv:2603.26666](https://arxiv.org/abs/2603.26666))
   - Forward-KL → entropy explosion when teacher hesitates.
   - Hard-CE → entropy collapse when teacher is on the decision boundary.
   - Reverse-KL → bounded mode-seeking that filters teacher noise while
     preserving student exploration. Use Reverse-KL by default in
     robotics / OOD-heavy settings.
7. **Token-level entropy guard** (EOPD,
   [arXiv:2603.07079]; OPSD v3,
   [arXiv:2601.18734v3](https://arxiv.org/abs/2601.18734v3))
   - Switch to **forward-KL** on high-teacher-entropy tokens to preserve
     reasoning diversity.
   - Apply a **per-token JSD clip** (~0.05) to prevent style tokens
     (`wait`, `think`) from monopolizing the gradient.
8. **Reward / log-ratio clipping** (REOPOLD,
   [arXiv:2603.11137](https://arxiv.org/abs/2603.11137))
   Clip **the reward**, not the importance ratio:
   \(\tilde{R} = \max(\text{sg}(R),\ \log\frac{\alpha}{1-\alpha})\).
   Prevents heavy-tailed negative rewards from dominating.
9. **Match the divergence to the evaluation metric.** *DP-OPD*
   ([arXiv:2604.04461](https://arxiv.org/abs/2604.04461)) shows that
   when the goal is **perplexity / coverage** (e.g., language modeling
   on private text), **forward-KL (β=0)** beats reverse-KL because
   perplexity rewards mode-covering over mode-seeking. The reverse-KL
   default favored by VLA-OPD / SDPO / OPSDC is correct for
   **decision / reasoning** tasks where mode-seeking precision matters.
   Pick β by the downstream metric, not by tradition. A practical
   compromise: **JSD (β≈0.5)** with on-policy mixing rate **λ≈0.5**
   gives the best empirical PPL / throughput tradeoff in DP-OPD.
10. **Use OPD instead of RL when running under DP-SGD.** Same DP-OPD
    paper observes that **DP noise amplifies exposure bias**: on-policy
    targets on student-visited states, rather than teacher-forced
    targets, are dramatically more sample-efficient under privacy
    noise. Translate: **OPD is strictly preferable to off-policy KD or
    DP-RL whenever the privacy budget is the binding constraint.**

---

## License

This list is released under
[CC0 1.0 Universal (Public Domain)](https://creativecommons.org/publicdomain/zero/1.0/).
Contributions are welcome via pull request.
