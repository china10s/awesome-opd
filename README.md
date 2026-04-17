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
MOPD, OPSD, OPCD, Cascade Distillation, SKD / SpecKD — and has been adopted
in the post-training recipes of **Qwen3**, **MiMo-V2-Flash**, **Ministral 3**,
**DASD-4B-Thinking**, and many others.

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
  - [Speculative / Hybrid Student-Teacher Sampling](#speculative--hybrid-student-teacher-sampling)
  - [Sequence-Level & Distribution-Aligned Distillation](#sequence-level--distribution-aligned-distillation)
  - [Context Distillation on Student Rollouts](#context-distillation-on-student-rollouts)
  - [Multi-Teacher On-Policy Distillation](#multi-teacher-on-policy-distillation)
  - [Strong-to-Weak Distillation Pipelines](#strong-to-weak-distillation-pipelines)
  - [Cascade / Pruning + Distillation](#cascade--pruning--distillation)
  - [RL × Distillation Connections (Inverse RL, Imitation)](#rl--distillation-connections-inverse-rl-imitation)
- [Applications](#applications)
  - [Long-CoT Reasoning](#long-cot-reasoning)
  - [Agentic & Tool Use](#agentic--tool-use)
  - [Continual Learning & Catastrophic Forgetting](#continual-learning--catastrophic-forgetting)
  - [Model Compression / Small Language Models](#model-compression--small-language-models)
  - [Code & Math](#code--math)
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

- **Bingxiang He et al. Rethinking On-Policy Distillation of Large Language
  Models: Phenomenology, Mechanism, and Recipe.** arXiv, Apr 2026.
  [[arXiv:2604.13016]](https://arxiv.org/abs/2604.13016)
  *Identifies two conditions that govern OPD success: (i) the student and
  teacher must share **compatible thinking patterns**, and (ii) even with
  consistent patterns, teacher–student **score gap** matters. Shows that
  successful OPD is characterized by progressive alignment on
  high-probability tokens at student-visited states, with a **small shared
  token set concentrating 97–99 % of the probability mass**. Proposes two
  rescue strategies for failing runs: **off-policy cold start** and
  **teacher-aligned prompt selection**.*

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

- Tianduo Wang, Wei Lu et al. **Self-Distillation Bridges Distribution Gap in
  Language Model Fine-Tuning.** ACL, 2024.
  [[arXiv:2402.13669]](https://arxiv.org/abs/2402.13669)

- Yiming Zhang et al. **Self-Knowledge Distillation: A Simple Way for Better
  Generalization.** [[arXiv:2006.12000]](https://arxiv.org/abs/2006.12000)

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

### Agentic & Tool Use

- **SDPO** — first OPD method to show explicit gains on tool use and
  competitive programming via rich textual feedback.
  [[arXiv:2601.20802]](https://arxiv.org/abs/2601.20802)
- **MiMo-V2-Flash** — agentic capability via teacher specialization in MOPD.
- **Tinker Cookbook Multi-Turn (Harbor)** — multi-turn agent OPD recipe.
- *Open question*: long-horizon agent trajectories with partial-credit
  feedback — flagged as open problem in the 2604.00626 survey.

### Continual Learning & Catastrophic Forgetting

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

- **Ministral 3** (Cascade Distillation, 3B / 8B / 14B).
  [[arXiv:2601.08584]](https://arxiv.org/abs/2601.08584)
- **Qwen3-0.6B / 1.7B / 4B / 8B / 14B** (Strong-to-Weak OPD).
- **Minitron / Llama-3.1-Minitron** (Pruning + KD).
- **MiniCPM, Phi-3, Gemma-2** — community recipes that increasingly resemble
  OPD.
- **MobileLLM**, **OLMoE-1B-7B** — small-model technical reports with KD
  ablations relevant to OPD.

### Code & Math

- **SDPO** on LiveCodeBench v6.
- **OPSD** on AIME / MATH / GSM8K / Olympiad-Bench.
- **DASD-4B-Thinking** on AIME / GPQA / LiveCodeBench.
- **MiMo-V2-Flash** on full math + code suite.
- **Qwen3-8B + OPD** achieves AIME'24 74.4 %, MATH500 97.0 %,
  LiveCodeBench v5 60.3 %. ([Table 21, Qwen3 TR](https://arxiv.org/abs/2505.09388))

---

## Models & Technical Reports

| Model | Year | Distillation Recipe | Link |
|------|------|--------------------|------|
| **DASD-4B-Thinking** | 2026 | Distribution-Aligned Sequence Distillation | [arXiv:2601.09088](https://arxiv.org/abs/2601.09088) |
| **Ministral 3** (3B/8B/14B) | 2026 | Cascade Distillation (iterative prune + KD) | [arXiv:2601.08584](https://arxiv.org/abs/2601.08584) |
| **MiMo-V2-Flash** (309B-A15B MoE) | 2026 | Multi-Teacher On-Policy Distillation (MOPD) | [arXiv:2601.02780](https://arxiv.org/abs/2601.02780) |
| **OPSD-Distilled Reasoner** | 2026 | On-Policy Self-Distillation (single-model) | [arXiv:2601.18734](https://arxiv.org/abs/2601.18734) |
| **Qwen3-\*-OPD** | 2025 | Strong-to-Weak (off-policy SFT → on-policy RKL) | [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) |
| **DeepSeek-R1-Distill-\*** | 2025 | Off-policy SFT distillation (baseline) | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| **Llama-3.1-Minitron** (4B/8B) | 2024 | Pruning + KD | NVIDIA TR |
| **MiniLLM-OPT/Llama** | 2023 | Reverse-KL on-policy KD | [arXiv:2306.08543](https://arxiv.org/abs/2306.08543) |

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

- **MiniLLM (Microsoft LMOps)** — official PyTorch implementation of
  reverse-KL on-policy distillation for LLMs.
  <https://github.com/microsoft/LMOps/tree/main/minillm>

- **DistiLLM / DistiLLM-2** — official code.
  <https://github.com/jongwooko/distillm>

- **Speculative KD** (Google Research).
  <https://github.com/google-research/google-research/tree/master/speculative_kd>

### General Frameworks That Support OPD

- **TRL** (Hugging Face) — `GKDTrainer` implements the Agarwal et al. recipe.
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

### Chinese (中文资料)

- 《[万字长文总结 RL / on policy distillation 的一些进展](https://zhuanlan.zhihu.com/p/2004506304065065334)》— 知乎综述, 2026.
- 《[MIT 提出 SDFT：作为逆强化学习的在线自蒸馏](https://mp.weixin.qq.com/s/7BKxOoS5iqah27OrxY9BzA)》— SDFT 论文解读.
- 《[自蒸馏优化 SDPO：如何利用富文本反馈打破 RLVR 的信用分配瓶颈？](https://mp.weixin.qq.com/s/eRqxe7qcRxWJYE86UXv8Nw)》— SDPO 论文解读.
- 《[阿里云提出 DASD：分布对齐的序列蒸馏，实现更优的长链思维推理](https://mp.weixin.qq.com/s/TOeY5rkKUFl_cYaOKQaMaA)》— DASD 论文解读.
- 《[深度解析 Ministral 3：基于级联蒸馏的参数高效密集模型训练方法论](https://mp.weixin.qq.com/s/_gN-easX9VpsW1g9aixDuw)》— Ministral 3 论文解读.
- 《[小米 MiMo-V2-Flash 技术报告：MoE 架构、混合注意力机制与多教师在线蒸馏](https://mp.weixin.qq.com/s/3a2xz8LYhyV6udSgxuQFoA)》— MiMo-V2-Flash 论文解读.

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

1. **Thinking-pattern incompatibility.** *Rethinking OPD*
   ([arXiv:2604.13016](https://arxiv.org/abs/2604.13016)) shows that
   teacher–student pattern mismatch can cause silent training failures even
   when the teacher is objectively stronger.
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
   many turns away.
6. **Distillation scaling laws.** There is currently no analog of Chinchilla
   for OPD: how does optimal compute split between teacher rollouts, student
   rollouts, and KL regularization as you scale student / teacher / data?

---

## License

This list is released under
[CC0 1.0 Universal (Public Domain)](https://creativecommons.org/publicdomain/zero/1.0/).
Contributions are welcome via pull request.
