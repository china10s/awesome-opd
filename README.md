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
GOLD, Cascade Distillation, SKD / SpecKD, **On-Policy Cross-Stage
Distillation (OPCSD)** — and has been adopted in the post-training recipes
of **GLM-5**, **Qwen3**, **MiMo-V2-Flash**, **Ministral 3**,
**DASD-4B-Thinking**, **Gemma 3**, and many others.

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
  - [On-Policy KD from a Larger Teacher](#on-policy-kd-from-a-larger-teacher)
  - [Stability & Loss Engineering](#stability--loss-engineering)
  - [Speculative / Hybrid Student-Teacher Sampling](#speculative--hybrid-student-teacher-sampling)
  - [Sequence-Level & Distribution-Aligned Distillation](#sequence-level--distribution-aligned-distillation)
  - [Context Distillation on Student Rollouts](#context-distillation-on-student-rollouts)
  - [Multi-Teacher On-Policy Distillation](#multi-teacher-on-policy-distillation)
  - [Strong-to-Weak Distillation Pipelines](#strong-to-weak-distillation-pipelines)
  - [Cross-Stage / Anti-Forgetting Distillation](#cross-stage--anti-forgetting-distillation)
  - [Interpolated / Curriculum Distillation](#interpolated--curriculum-distillation)
  - [Self-Play as Implicit On-Policy Distillation](#self-play-as-implicit-on-policy-distillation)
  - [Cross-Tokenizer / Cross-Family Distillation](#cross-tokenizer--cross-family-distillation)
  - [Multimodal & Embodied On-Policy Distillation](#multimodal--embodied-on-policy-distillation)
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

- Tianduo Wang, Wei Lu et al. **Self-Distillation Bridges Distribution Gap in
  Language Model Fine-Tuning.** ACL, 2024.
  [[arXiv:2402.13669]](https://arxiv.org/abs/2402.13669)

- Yiming Zhang et al. **Self-Knowledge Distillation: A Simple Way for Better
  Generalization.** [[arXiv:2006.12000]](https://arxiv.org/abs/2006.12000)

- **SPIN — Self-Play Fine-Tuning** (UCLA, ICML 2024)
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

- **DistiLLM / DistiLLM-2** (KAIST, 2024–2025)
  Jongwoo Ko et al. **DistiLLM: Towards Streamlined Distillation for Large
  Language Models.** [[arXiv:2402.03898]](https://arxiv.org/abs/2402.03898) ·
  [[Code]](https://github.com/jongwooko/distillm)

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

- **SpecKD — Speculative Decoding for Effective KD** (XJTU, 2025)
  Haiduo Huang et al. [[arXiv:2510.24021]](https://arxiv.org/abs/2510.24021)
  *Instead of filtering **data**, SpecKD filters the **loss itself**: each
  student token is verified against the teacher, and the KL penalty is
  applied only on accepted tokens — drastically stabilizing training when the
  teacher–student gap is large.*

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
- *Open question*: trillion-parameter VLA + multi-step planning + OPD —
  no work yet, but a natural next step combining MOPD-style
  multi-teacher with VLA-OPD's reverse-KL recipe.

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

- **MiniLLM (Microsoft LMOps)** — official PyTorch implementation of
  reverse-KL on-policy distillation for LLMs.
  <https://github.com/microsoft/LMOps/tree/main/minillm>

- **DistiLLM / DistiLLM-2** — official code.
  <https://github.com/jongwooko/distillm>

- **Speculative KD** (Google Research).
  <https://github.com/google-research/google-research/tree/master/speculative_kd>

### General Frameworks That Support OPD

- **TRL** (Hugging Face) — `GKDTrainer` implements the Agarwal et al.
  recipe; experimental `GOLDTrainer` adds **cross-tokenizer** OPD; legacy
  `GOLDTrainer` (older spelling) underpins OPSD.
  <https://github.com/huggingface/trl>
- **OpenRLHF** — RLHF-style framework with hooks suitable for OPD
  experiments.
  <https://github.com/OpenRLHF/OpenRLHF>
- **veRL** (ByteDance) — large-scale RL framework that supports rollout-based
  distillation losses.
  <https://github.com/volcengine/verl>
- **NVIDIA NeMo-Aligner** — has a knowledge distillation trainer that can be
  configured to be on-policy.
  <https://github.com/NVIDIA/NeMo-Aligner>
- **EasyDistill** (Alibaba) — end-to-end distillation framework covering
  both off- and on-policy modes.
- **ms-swift** (ModelScope / Alibaba) — has a first-class `swift distill`
  entry point with GKD / reverse-KL / forward-KL / JSD loss options.
  <https://github.com/modelscope/ms-swift>
- **LLaMA-Factory** — popular SFT / DPO framework; `--stage kd` mode ships
  an on-policy GKD-style trainer.
  <https://github.com/hiyouga/LLaMA-Factory>
- **SakanaAI/TAID** — reference implementation of Temporally Adaptive
  Interpolated Distillation. <https://github.com/SakanaAI/TAID>
- **UCLA-AGI SPIN** — official self-play fine-tuning code.
  <https://github.com/uclaml/SPIN>
- **slime** (Z.ai / Tsinghua) — the asynchronous RL framework behind GLM-5,
  used to implement OPCSD at trillion-parameter scale.
  <https://github.com/THUDM/slime>

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

- **thinkwee/AwesomeOPD** — sister awesome list maintained by the THUNLP
  community, accompanies the [arXiv:2604.13016](https://arxiv.org/abs/2604.13016)
  paper. <https://github.com/thinkwee/AwesomeOPD>

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

---

## Best Practices & Recipes

> A consolidated recipe shelf distilled from the failure-mode literature
> above. None of these are mandatory, but skipping any of them invites
> one of the failure modes in the previous section.

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

---

## License

This list is released under
[CC0 1.0 Universal (Public Domain)](https://creativecommons.org/publicdomain/zero/1.0/).
Contributions are welcome via pull request.
