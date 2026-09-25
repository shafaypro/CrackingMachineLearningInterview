# Reinforcement Learning

Supervised learning learns from labelled answers; reinforcement learning learns from the consequences of its own actions, with feedback that is delayed, sparse and dependent on what the agent chose to try. It comes up in interviews in three forms: as fundamentals (Bellman equations, Q-learning vs SARSA, on- vs off-policy), as applied industry questions dressed up as something else (bandits for recommendations, ads, pricing, and evaluating a new policy from logged data), and, increasingly, as "explain RLHF and why DPO exists" for anyone working near LLMs. The strongest answers also know when RL is the wrong tool.

---

## Table of Contents
1. [When RL Is the Right Frame](#when-rl-is-the-right-frame)
2. [The MDP Formalism](#the-mdp-formalism)
3. [Returns, Value Functions and Bellman Equations](#returns-value-functions-and-bellman-equations)
4. [The Taxonomy](#the-taxonomy)
5. [Dynamic Programming](#dynamic-programming)
6. [Monte Carlo vs Temporal Difference](#monte-carlo-vs-temporal-difference)
7. [Q-Learning vs SARSA](#q-learning-vs-sarsa)
8. [Exploration](#exploration)
9. [Multi-Armed and Contextual Bandits](#multi-armed-and-contextual-bandits)
10. [DQN](#dqn)
11. [Policy Gradients and REINFORCE](#policy-gradients-and-reinforce)
12. [Actor-Critic and PPO](#actor-critic-and-ppo)
13. [Offline RL and Off-Policy Evaluation](#offline-rl-and-off-policy-evaluation)
14. [Reward Design, Shaping and Hacking](#reward-design-shaping-and-hacking)
15. [RL for LLMs](#rl-for-llms)
16. [When Not to Use RL](#when-not-to-use-rl)
17. [Algorithm Comparison](#algorithm-comparison)
18. [Interview Q&A](#interview-qa)
19. [Common Pitfalls](#common-pitfalls)
20. [Related Topics](#related-topics)

---

## When RL Is the Right Frame

RL fits when **actions change what data you see next** and the goal is a long-run outcome rather than a one-step prediction.

| Problem | Why it's RL-shaped |
|---|---|
| Recommendations / ads ranking | Choosing what to show determines what feedback you get (partial feedback) |
| Dynamic pricing | Price is an action; revenue feedback is only observed for the price you set |
| Robotics, game playing | Sequential control with delayed reward |
| Resource allocation (bidding, scheduling) | Actions consume budget that constrains future actions |
| LLM alignment | Optimize a sequence-level preference signal that isn't differentiable w.r.t. tokens |

The key distinction from supervised learning: **you only observe the outcome of the action you took**. There is no label telling you what would have happened under the other action. That single fact is the source of the exploration problem, the off-policy evaluation problem, and most of the difficulty.

---

## The MDP Formalism

A Markov Decision Process is a tuple `(S, A, P, R, γ)`:

| Symbol | Meaning | Example (recsys session) |
|---|---|---|
| `S` | States | User context + session history |
| `A` | Actions | Which item to show |
| `P(s' \| s, a)` | Transition dynamics | How the user's state changes after seeing the item |
| `R(s, a)` | Reward | Click, watch time, purchase |
| `γ ∈ [0, 1)` | Discount factor | How much future reward counts relative to now |

**The Markov property**: the next state depends only on the current state and action, not on the full history. If it doesn't hold for your raw observations, you fold history into the state (stacked frames, an RNN hidden state) or treat it as a POMDP.

A **policy** `π(a | s)` maps states to a distribution over actions. The goal is to find a policy that maximizes expected discounted return.

**Why discount?** Three reasons worth stating: it makes infinite-horizon returns finite (for bounded rewards and `γ < 1`), it encodes a preference for sooner reward, and it reduces variance by down-weighting distant, noisy rewards. The effective horizon is roughly `1 / (1 − γ)`, so `γ = 0.99` looks about 100 steps ahead.

---

## Returns, Value Functions and Bellman Equations

The **return** from time `t`:

```
G_t = r_{t+1} + γ r_{t+2} + γ² r_{t+3} + ... = Σ_k γ^k r_{t+k+1}
```

Two value functions:

```
V^π(s)    = E_π[ G_t | s_t = s ]                  # how good is this state under π
Q^π(s, a) = E_π[ G_t | s_t = s, a_t = a ]         # how good is this action here, then follow π
A^π(s, a) = Q^π(s, a) − V^π(s)                    # advantage: better or worse than average
```

The **Bellman expectation equation** expresses value recursively, one step of reward plus the discounted value of where you land:

```
V^π(s)    = Σ_a π(a|s) Σ_s' P(s'|s,a) [ R(s,a) + γ V^π(s') ]
Q^π(s, a) = Σ_s' P(s'|s,a) [ R(s,a) + γ Σ_a' π(a'|s') Q^π(s',a') ]
```

The **Bellman optimality equation** replaces the policy average with a max:

```
V*(s)    = max_a Σ_s' P(s'|s,a) [ R(s,a) + γ V*(s') ]
Q*(s, a) = Σ_s' P(s'|s,a) [ R(s,a) + γ max_a' Q*(s',a') ]
```

Once you have `Q*`, the optimal policy is just `argmax_a Q*(s, a)` — no model needed. That is why so much of RL is about estimating `Q`.

---

## The Taxonomy

Three axes that interviewers use to check you have a map of the field:

**Value-based vs policy-based**
- *Value-based* (Q-learning, DQN): learn `Q`, act greedily. Works well for discrete actions; awkward for continuous ones because of the `max_a`.
- *Policy-based* (REINFORCE): parameterize `π_θ` directly and do gradient ascent on return. Handles continuous and stochastic policies naturally; high variance.
- *Actor-critic* (A2C, PPO, SAC): both — a policy (actor) updated with help from a learned value function (critic).

**Model-based vs model-free**
- *Model-based*: learn or are given `P` and `R`, then plan (dynamic programming, MCTS, model-predictive control). Much more sample-efficient; errors in the model compound over long rollouts.
- *Model-free*: learn values or policies directly from experience. Simpler and robust to model error, but data-hungry.

**On-policy vs off-policy**
- *On-policy* (SARSA, REINFORCE, PPO): learn about the policy that is currently collecting the data. Data goes stale after each update.
- *Off-policy* (Q-learning, DQN, SAC): learn about a *target* policy from data collected by a different *behaviour* policy. Can reuse old data from a replay buffer or logs — which is what makes offline RL and logged-data evaluation possible.

---

## Dynamic Programming

When the MDP is fully known and small, you can solve it exactly.

- **Policy evaluation**: iterate the Bellman expectation equation until `V^π` converges.
- **Policy iteration**: alternate evaluation and greedy improvement `π(s) ← argmax_a Q^π(s, a)`. Converges in a finite number of iterations for finite MDPs.
- **Value iteration**: apply the Bellman optimality update directly, `V(s) ← max_a Σ P [R + γ V(s')]`. Each sweep is a contraction with factor `γ`, so it converges to `V*`.

```python
import numpy as np

def value_iteration(P, R, gamma=0.99, tol=1e-8):
    """P: (S, A, S) transition probs; R: (S, A) expected reward."""
    V = np.zeros(P.shape[0])
    while True:
        Q = R + gamma * P @ V              # (S, A): one-step lookahead
        V_new = Q.max(axis=1)
        if np.abs(V_new - V).max() < tol:
            return V_new, Q.argmax(axis=1)
        V = V_new
```

DP is rarely the production answer — you almost never know `P` — but it's the conceptual base: every model-free method is an approximation of one of these updates using samples instead of the true expectation.

---

## Monte Carlo vs Temporal Difference

Both estimate value from experience without a model. They differ in what target they update towards.

| | Monte Carlo | TD(0) |
|---|---|---|
| Target | Full observed return `G_t` | `r + γ V(s')` (bootstrapped) |
| Needs episode to finish? | Yes | No — updates every step |
| Bias | Unbiased | Biased (depends on current estimate of `V(s')`) |
| Variance | High (sum of many random rewards) | Low (one reward + an estimate) |
| Uses Markov structure? | No | Yes |
| Continuing (non-episodic) tasks | Awkward | Natural |

```
MC:     V(s) ← V(s) + α [ G_t − V(s) ]
TD(0):  V(s) ← V(s) + α [ r + γ V(s') − V(s) ]      # bracket = TD error δ
```

**The middle ground** is n-step returns and TD(λ): use `n` real rewards and then bootstrap. Generalized Advantage Estimation (GAE), used in PPO, is the same idea applied to advantage estimation, with `λ` trading bias for variance.

---

## Q-Learning vs SARSA

Both are tabular TD control methods. The difference is one term in the target:

```
SARSA:      Q(s,a) ← Q(s,a) + α [ r + γ Q(s', a')        − Q(s,a) ]   # a' actually taken
Q-learning: Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s', a') − Q(s,a) ]   # greedy a'
```

- **SARSA is on-policy**: it learns the value of the policy it is actually following, exploration included.
- **Q-learning is off-policy**: it learns the value of the greedy policy regardless of how actions were chosen.

The textbook illustration is the **cliff-walking** gridworld. With ε-greedy exploration, Q-learning learns the optimal path along the cliff edge but falls off during training because of random exploratory steps. SARSA learns a safer path further from the edge because its values account for the fact that it will sometimes explore. If exploration happens at deployment too, SARSA's answer is the better one.

A minimal tabular Q-learning agent on a 1-D corridor (reach the right end for +1):

```python
import numpy as np

n_states, n_actions = 6, 2          # actions: 0 = left, 1 = right
goal = n_states - 1
Q = np.zeros((n_states, n_actions))
alpha, gamma, eps = 0.1, 0.95, 0.1
rng = np.random.default_rng(0)

def step(s, a):
    s2 = min(s + 1, goal) if a == 1 else max(s - 1, 0)
    r = 1.0 if s2 == goal else 0.0
    return s2, r, s2 == goal

for episode in range(500):
    s, done = 0, False
    while not done:
        # epsilon-greedy with random tie-breaking
        if rng.random() < eps:
            a = int(rng.integers(n_actions))
        else:
            a = int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))
        s2, r, done = step(s, a)
        target = r if done else r + gamma * Q[s2].max()   # no bootstrap past terminal
        Q[s, a] += alpha * (target - Q[s, a])
        s = s2

print(Q.argmax(axis=1)[:-1])   # [1 1 1 1 1] -> always move right
```

Note the `if done` in the target: bootstrapping from a terminal state is a common bug. Tabular Q-learning converges to `Q*` given every state-action pair is visited infinitely often and the learning rate decays appropriately.

---

## Exploration

The agent only learns about actions it tries. Too little exploration locks in an early mistake; too much wastes reward.

| Strategy | How | Notes |
|---|---|---|
| **ε-greedy** | Random action with probability ε, else greedy | Simple, default; explores uniformly, including obviously bad actions; decay ε over time |
| **Softmax / Boltzmann** | Sample `a ∝ exp(Q(s,a)/τ)` | Explores near-best actions more than terrible ones |
| **UCB** | Pick `argmax_a Q(a) + c √(ln t / N(a))` | Optimism in the face of uncertainty; bonus shrinks as an arm is tried; deterministic |
| **Thompson sampling** | Sample a plausible model from the posterior, act greedily under it | Explores in proportion to probability of being optimal; strong empirically; handles delayed batch updates well |
| **Entropy bonus** | Add `β · H(π(·\|s))` to the objective | Standard in policy-gradient methods to prevent premature collapse |
| **Intrinsic motivation** | Reward novelty / prediction error | For sparse-reward environments |

UCB and Thompson sampling come with logarithmic regret guarantees in the stochastic multi-armed bandit setting; plain ε-greedy with fixed ε has linear regret because it never stops exploring.

---

## Multi-Armed and Contextual Bandits

A bandit is RL with **one step**: no state transitions, so no credit assignment over time — only the exploration-exploitation trade-off. This is where most industrial RL actually lives.

| Setting | State | Example |
|---|---|---|
| **Multi-armed bandit** | None | Which of 5 headlines gets the most clicks |
| **Contextual bandit** | Context `x` observed before acting | Which article to show *this* user; which ad; what discount to offer |
| **Full RL** | State evolves with actions | Session-level recommendations optimizing long-term retention |

**Bandits vs A/B tests.** An A/B test explores uniformly for a fixed period and then exploits; it is built to estimate an effect with a confidence interval. A bandit shifts traffic toward winners as evidence accumulates, which reduces regret (lost reward during the experiment) but makes clean inference about effect sizes harder. Use A/B tests when you need a defensible measurement; use bandits when you care about reward during learning, have many arms, or the best arm drifts (headlines, promos, creatives).

Beta-Bernoulli Thompson sampling for click-through rates:

```python
import numpy as np

rng = np.random.default_rng(0)
true_ctr = np.array([0.04, 0.05, 0.07])          # unknown to the agent
alpha = np.ones(3); beta = np.ones(3)            # Beta(1,1) priors

for t in range(20_000):
    theta = rng.beta(alpha, beta)                # sample a plausible CTR per arm
    arm = int(theta.argmax())
    click = rng.random() < true_ctr[arm]
    alpha[arm] += click
    beta[arm] += 1 - click

print((alpha + beta - 2).astype(int))            # pulls concentrate on arm 2
```

**Contextual bandits** in practice: LinUCB (a linear reward model per arm with a UCB bonus from its covariance), linear or neural Thompson sampling, or a greedy model with ε-exploration. The operational essentials are the same across methods: **log the propensity** (probability the chosen action had) with every decision, so the data can later be used for off-policy evaluation, and keep some exploration forever so the logs cover alternatives.

**Pricing** is a contextual bandit with a twist: reward is `price × P(purchase | price)`, the arms are ordered, and demand is usually monotone in price, so parametric demand models share information across price points far more efficiently than independent arms.

---

## DQN

Deep Q-Networks replace the Q-table with a neural network `Q_θ(s, a)`. Naively combining Q-learning with function approximation is unstable; DQN's two main stabilizers are the interview answer:

1. **Experience replay**: store transitions `(s, a, r, s', done)` in a buffer and train on random minibatches. Breaks the temporal correlation between consecutive samples and reuses each transition many times. Only valid because Q-learning is off-policy.
2. **Target network**: compute targets with a lagged copy `θ⁻`, updated every `C` steps (or by Polyak averaging). Without it, every update moves the target it is regressing toward.

```python
import torch
import torch.nn.functional as F

def dqn_loss(q_net, target_net, batch, gamma=0.99):
    s, a, r, s2, done = batch                        # tensors; done is 0/1 float
    q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        # Double DQN: online net selects the action, target net evaluates it
        a2 = q_net(s2).argmax(dim=1, keepdim=True)
        q2 = target_net(s2).gather(1, a2).squeeze(1)
        target = r + gamma * (1 - done) * q2
    return F.smooth_l1_loss(q, target)               # Huber loss
```

**The deadly triad** (Sutton & Barto): function approximation + bootstrapping + off-policy learning together can diverge. DQN has all three; replay and target networks mitigate rather than eliminate it.

**Common improvements**: Double DQN (shown above; reduces the overestimation bias from `max` over noisy estimates), dueling architecture (separate `V` and `A` streams), prioritized replay (sample high-TD-error transitions more often), n-step returns, distributional RL. Rainbow combines several of these.

---

## Policy Gradients and REINFORCE

Parameterize the policy `π_θ(a|s)` and ascend `J(θ) = E_π[G]`. The **policy gradient theorem** gives a gradient that does not require differentiating the environment:

```
∇_θ J(θ) = E_π[ Σ_t ∇_θ log π_θ(a_t | s_t) · G_t ]
```

Intuition: increase the log-probability of actions in proportion to how good the return after them was. This is the **log-derivative (score function) trick**, the same estimator used whenever you need gradients through a sampling step you can't differentiate.

**REINFORCE** is the Monte Carlo version: roll out a full episode, compute `G_t`, step along the gradient. It is unbiased but very high variance.

**Baselines** are the standard variance reduction: subtract `b(s_t)` from `G_t`. Any baseline that doesn't depend on the action leaves the gradient unbiased (because `E[∇ log π] = 0`). Using `b = V(s)` turns `G_t − V(s_t)` into an advantage estimate — which is the step to actor-critic.

```python
import torch

def reinforce_loss(log_probs, rewards, gamma=0.99):
    """log_probs: list of log π(a_t|s_t) tensors for one episode."""
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)   # simple baseline
    return -(torch.stack(log_probs) * returns).sum()
```

Policy gradients handle continuous actions (output a Gaussian mean and std) and stochastic optimal policies, which value-based methods cannot represent with a greedy argmax.

---

## Actor-Critic and PPO

**Actor-critic** learns a critic `V_φ(s)` alongside the actor `π_θ`, and uses the critic to form a lower-variance advantage estimate, e.g. the TD error `δ = r + γ V(s') − V(s)` or GAE. A2C/A3C are synchronous/asynchronous versions of this. SAC and TD3 are off-policy actor-critic methods popular for continuous control.

The problem with vanilla policy gradients: step size. One large update can collapse the policy, and because the policy generates its own data, a bad policy then collects bad data and may not recover. TRPO constrains each update with a KL-divergence trust region; **PPO** gets most of the benefit with a simple clipped objective:

```
r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)

L_CLIP(θ) = E_t[ min( r_t(θ) Â_t,  clip(r_t(θ), 1−ε, 1+ε) Â_t ) ]
```

- If the advantage is positive, the objective stops rewarding increases in `r_t` beyond `1 + ε`.
- If negative, it stops rewarding decreases beyond `1 − ε`.
- The `min` makes it a pessimistic bound: the clip removes the incentive to move far, but never hides a change that makes things worse.

```python
import torch

def ppo_clip_loss(logp_new, logp_old, adv, eps=0.2):
    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    return -torch.min(unclipped, clipped).mean()
```

The full PPO loss adds a value-function loss and an entropy bonus. PPO is on-policy but reuses each batch for a few epochs of minibatch updates, which the clipping makes safe. It is the default general-purpose deep RL algorithm and was the standard optimizer in RLHF pipelines.

---

## Offline RL and Off-Policy Evaluation

In industry you usually cannot let an untested policy explore on live users. You have **logs** from the current production policy (the behaviour policy `π_0`) and want to either evaluate or learn a new policy `π` from them.

**Off-policy evaluation (OPE)** for contextual bandits, with logged tuples `(x_i, a_i, r_i, p_i)` where `p_i = π_0(a_i | x_i)`:

| Estimator | Formula (per-sample average) | Trade-off |
|---|---|---|
| **Direct method (DM)** | `Σ_a π(a\|x) r̂(x, a)` | Low variance; biased if the reward model is wrong |
| **IPS** | `π(a_i\|x_i) / p_i · r_i` | Unbiased if propensities are correct and overlap holds; high variance when weights are large |
| **SNIPS** | IPS divided by the mean weight | Small bias, much lower variance |
| **Doubly robust (DR)** | `DM + π(a_i\|x_i)/p_i · (r_i − r̂(x_i, a_i))` | Unbiased if *either* the propensities or the reward model is correct; usually lower variance than IPS |

```python
import numpy as np

def ope_estimates(pi_new, p_logged, r, r_hat_logged, r_hat_policy):
    """pi_new: π(a_i|x_i) for logged actions; p_logged: π_0(a_i|x_i);
    r_hat_logged: r̂(x_i, a_i); r_hat_policy: Σ_a π(a|x_i) r̂(x_i, a)."""
    w = pi_new / p_logged
    ips = np.mean(w * r)
    snips = np.sum(w * r) / np.sum(w)
    dr = np.mean(r_hat_policy + w * (r - r_hat_logged))
    return {"IPS": ips, "SNIPS": snips, "DR": dr}
```

Requirements that interviews probe: **logged propensities** (reconstructing them after the fact is error-prone), **overlap/support** (if `π_0` never takes an action that `π` takes, no estimator can evaluate it — this is why production systems keep some randomization), and **weight clipping** to control variance at the cost of bias. This is the same machinery as inverse propensity weighting in causal inference.

For sequential problems, importance weights multiply across time steps and variance explodes with horizon — a big reason long-horizon OPE is hard.

**Offline RL** (learning a policy from a fixed dataset) faces **distributional shift**: the learned `Q` overestimates the value of actions the data never tried, and the policy exploits those errors. Methods constrain the policy toward the data: BCQ, CQL (penalizes Q-values for out-of-distribution actions), IQL (avoids querying unseen actions entirely). Behaviour cloning on the best logged trajectories is a strong baseline that is often hard to beat.

---

## Reward Design, Shaping and Hacking

The agent optimizes exactly the reward you wrote, not the one you meant.

**Reward hacking / specification gaming**: the policy finds a way to score highly that violates the intent. Classic forms: exploiting a simulator bug, looping to collect a repeatable reward, optimizing a proxy (clicks) at the expense of the goal (satisfaction) — clickbait is reward hacking. In RLHF, the policy learns to exploit weaknesses of the learned reward model: longer answers, confident tone, sycophancy.

**Reward shaping** adds intermediate rewards to speed up learning in sparse-reward problems, and can change the optimal policy if done carelessly. **Potential-based shaping** `F(s, s') = γ Φ(s') − Φ(s)` provably preserves the optimal policy (Ng, Harada & Russell, 1999), because the shaping terms telescope along any trajectory.

Mitigations: reward the outcome, not the method; combine multiple signals with guardrail metrics; add a KL penalty to a trusted reference policy; audit high-reward episodes by hand; and refresh learned reward models on the new policy's outputs.

---

## RL for LLMs

**Why RL at all?** Next-token cross-entropy can only imitate. Preferences ("answer A is better than B") and verifiable outcomes ("the code passes tests") are sequence-level signals that aren't a per-token target, so you optimize them with RL or an RL-derived objective. See [Fine-Tuning](../deep_learning/intro_fine_tuning.md) for the full SFT → preference-tuning pipeline.

**RLHF pipeline**
1. **SFT**: supervised fine-tune on demonstrations.
2. **Reward model**: train `r_φ(x, y)` on human preference pairs with a Bradley-Terry loss, `−log σ(r_φ(x, y_w) − r_φ(x, y_l))`.
3. **RL**: optimize the policy with PPO to maximize `r_φ(x, y) − β · KL(π_θ(·|x) ‖ π_ref(·|x))`.

In the MDP framing, the state is the prompt plus tokens so far, the action is the next token, and reward typically arrives only at the end of the sequence. The **KL penalty** keeps the policy near the SFT model: it limits reward hacking of an imperfect reward model and preserves fluency.

| Method | What it needs | How it works | Trade-offs |
|---|---|---|---|
| **PPO (RLHF)** | Reward model, value model, reference model, policy | Online RL with clipped updates and KL penalty | Most flexible; four models in memory; many hyperparameters; unstable |
| **DPO** | Preference pairs, reference model | Closed-form reparameterization of the KL-regularized objective turns it into a classification loss on pairs | No reward model, no sampling, simple and stable; offline — limited to the preference data's distribution |
| **GRPO** | Reward function or model, reference model | Sample a group of responses per prompt; advantage = reward normalized by the group mean and std; PPO-style clipped update | No value network; works well with verifiable rewards (math, code); popularized by DeepSeek's reasoning models |

**RL with verifiable rewards (RLVR)**: for math and code, the reward comes from checking the answer or running tests rather than a learned reward model, which removes one source of reward hacking (though models can still exploit weak test suites or answer-format checks).

---

## When Not to Use RL

This is where interviewers separate practitioners from enthusiasts.

- **Actions don't affect future states** → it's a bandit, not full RL. Use a contextual bandit.
- **You can get labels for the right action** → it's supervised learning. Imitation/behaviour cloning beats RL when good demonstrations exist.
- **You only need a ranking score and exploration is cheap to fake** → a supervised model plus a small ε of randomization and logged propensities covers most recsys needs.
- **You need a defensible effect estimate** → run an A/B test.
- **No simulator and online exploration is unsafe or expensive** → RL's sample appetite is a real barrier; offline RL is possible but hard to validate.
- **The reward is a proxy you don't trust** → RL will optimize the gap between the proxy and the goal.

Deep RL is also notoriously sensitive to seeds, hyperparameters and implementation details; results should be reported over multiple seeds. The practical ladder is: supervised model → bandit → offline evaluation of a candidate policy → cautious online RL.

---

## Algorithm Comparison

| | Q-learning / DQN | SARSA | REINFORCE | PPO | SAC | Contextual bandit |
|---|---|---|---|---|---|---|
| Family | Value | Value | Policy | Actor-critic | Actor-critic | One-step |
| On/off-policy | Off | On | On | On | Off | Either (off with logged propensities) |
| Action space | Discrete | Discrete | Any | Any | Continuous | Discrete, usually |
| Sample efficiency | Good (replay) | Moderate | Poor | Moderate | Good | Good |
| Stability | Moderate | Good (tabular) | Poor (variance) | **Good** | Good | **Good** |
| Typical use | Atari, discrete control | Safe tabular control | Teaching, simple tasks | General default; RLHF | Robotics | Recsys, ads, pricing |

---

## Interview Q&A

#### What is the difference between Q-learning and SARSA?

Both are TD control methods that update `Q(s, a)` toward a one-step target. SARSA uses the action the agent actually takes next, `r + γ Q(s', a')`, so it learns the value of its current (exploring) policy — it is on-policy. Q-learning uses `r + γ max_a' Q(s', a')`, the greedy action, so it learns the value of the optimal policy regardless of how data was collected — it is off-policy.

The practical consequence shows up in cliff walking: under ε-greedy exploration, Q-learning learns the risky optimal path along the edge and falls off during training; SARSA learns a safer path because its values include the cost of occasional random moves. If you'll keep exploring when deployed, SARSA's policy performs better. Q-learning's off-policy nature is also what allows DQN's experience replay.

#### Explain the Bellman equation and why it matters.

It writes value recursively: the value of a state equals the expected immediate reward plus the discounted value of the next state, `V(s) = E[r + γ V(s')]`. The optimality version takes a max over actions.

It matters because it turns a sum over infinite futures into a local consistency condition. Dynamic programming iterates it to the fixed point when the model is known. TD learning and Q-learning replace the expectation with a single sampled transition and nudge the estimate toward it. DQN is Q-learning with a neural network regressing onto Bellman targets. Almost every value-based method is some approximation of this equation.

#### Monte Carlo vs TD — which would you use?

MC waits for the episode's full return: unbiased but high variance, and it needs episodes to terminate. TD bootstraps from the current estimate of the next state: biased, but much lower variance, updates online, and works on continuing tasks. TD usually learns faster in practice because it exploits the Markov structure.

The real answer is often in between: n-step returns or GAE, which use a few real rewards before bootstrapping and expose the bias-variance trade-off as a tunable knob (`n` or `λ`). If the Markov property is badly violated, MC-style targets are more robust because they don't rely on the state being a sufficient summary.

#### Why does DQN need a replay buffer and a target network?

Q-learning with a neural network is unstable for two reasons. Consecutive transitions are highly correlated, which violates the i.i.d. assumption SGD relies on and makes the network overfit to the recent trajectory. And the regression target `r + γ max Q(s', a')` depends on the same parameters being updated, so the target moves with every step.

The replay buffer samples random past transitions, decorrelating minibatches and reusing data — valid because Q-learning is off-policy. The target network freezes the parameters used for targets for many steps, so the network regresses toward a fixed objective for a while. Double DQN additionally decouples action selection from evaluation to reduce the overestimation bias caused by taking a max over noisy estimates.

#### Explain the policy gradient theorem and why baselines help.

The gradient of expected return is `E[∇ log π(a|s) · G]`: push up the log-probability of actions in proportion to the return that followed. It doesn't require a differentiable environment — only the ability to differentiate the policy's log-probability.

The estimator is unbiased but very noisy, because `G` varies a lot for reasons that have nothing to do with the action. Subtracting a baseline `b(s)` keeps it unbiased, since the expected score function is zero, but reduces variance substantially. The natural baseline is `V(s)`, which turns the multiplier into an advantage — "was this action better than usual here?" — and that is exactly the actor-critic setup.

#### What problem does PPO solve, and how does the clipped objective work?

Policy gradient step size is dangerous: a single oversized update can wreck the policy, and since the policy collects its own data, a bad policy may never recover. TRPO fixes this with a hard KL constraint, but it's complex to implement.

PPO uses the probability ratio `r = π_new/π_old` and optimizes `min(r·A, clip(r, 1−ε, 1+ε)·A)`. For a good action it stops rewarding the policy for raising its probability past `1+ε`; for a bad action, past `1−ε` downward. The min makes it pessimistic, so the clip never masks an update that makes things worse. This lets PPO run several epochs of minibatch SGD on each batch of rollouts without the policy drifting too far, with a first-order optimizer and few moving parts.

#### When would you use a bandit instead of an A/B test, and instead of full RL?

Instead of an A/B test when regret during the experiment matters and I don't need a precise effect estimate: many creatives, short-lived content like headlines or promos, or a best option that drifts. A/B tests are the right tool when I need a clean, defensible measurement for a launch decision.

Instead of full RL when my action doesn't meaningfully change the user's future state — or when I'm willing to ignore that effect. Most recommendation, ad and pricing decisions are modelled as contextual bandits because they're far easier to train, evaluate offline and debug. I'd move to full RL only if there's clear evidence that myopic optimization hurts long-term outcomes, for example that optimizing immediate clicks is reducing retention.

#### How do you evaluate a new recommendation policy without deploying it?

Off-policy evaluation on logged data. For each logged decision I need the context, action, reward and the propensity with which the production policy chose that action. IPS reweights each logged reward by `π_new(a|x) / π_old(a|x)`; it's unbiased if propensities are correct and the old policy gave positive probability to every action the new one takes, but variance blows up when weights are large. SNIPS normalizes by the sum of weights to cut variance. Doubly robust combines a reward model with an IPS correction on its residual, and stays unbiased if either component is correct.

In practice I'd report DR and SNIPS with confidence intervals, check the effective sample size of the weights, clip extreme weights, and treat OPE as a filter for which candidates go to an online A/B test rather than a replacement for it. If the logging policy was deterministic, there's no overlap and OPE is impossible — which is why you keep some randomization in production.

#### What is the exploration-exploitation trade-off and how do UCB and Thompson sampling handle it?

Exploiting means choosing what currently looks best; exploring means trying things to find out whether something else is better. Without exploration you can lock in an early wrong estimate forever.

UCB adds an optimism bonus that shrinks with the number of times an action has been tried, so under-sampled actions get a chance until their uncertainty is resolved. Thompson sampling keeps a posterior over each action's reward, samples once from it, and acts greedily on the sample; an action is chosen roughly in proportion to the probability that it's the best. Both achieve logarithmic regret in stochastic bandits. Thompson sampling tends to do well empirically and copes well with delayed, batched feedback, since the randomization avoids every server picking the same arm between updates.

#### Explain RLHF, and why DPO and GRPO exist.

RLHF trains a reward model on human preference pairs, then uses PPO to maximize that reward minus a KL penalty to the SFT reference model. The KL term prevents the policy from drifting into regions where the reward model is wrong and being exploited. It works, but it holds a policy, reference model, reward model and value model in memory, is sensitive to hyperparameters, and the reward model can be hacked — longer, more confident or more flattering outputs.

DPO shows that the optimal policy of the KL-regularized objective has a closed form in terms of the reward, so you can substitute it back and train the policy directly on preference pairs with a logistic loss — no reward model, no sampling, no RL loop. The cost is that it's offline: it only learns from the pairs you have.

GRPO keeps online sampling but drops the value network: for each prompt it samples a group of responses, uses their mean and standard deviation as the baseline to normalize rewards into advantages, and applies a PPO-style clipped update with a KL penalty. It's cheaper than PPO and particularly effective with verifiable rewards like passing unit tests or matching a math answer.

#### What is reward hacking, and how do you guard against it?

The policy finds behaviour that scores well on the reward you specified but violates what you intended — the gap between a proxy and the true goal gets optimized. Examples: a recommender maximizing clicks learns clickbait; an RLHF model learns that longer answers score higher with the reward model; a code model special-cases the test inputs.

Guards: use rewards close to the true outcome, pair the primary metric with guardrail metrics, regularize toward a trusted reference policy with a KL penalty, and inspect the highest-reward samples manually, because that's where hacks show up first. For learned reward models, periodically collect fresh human labels on the current policy's outputs and retrain. If you need to shape rewards, potential-based shaping is the form that provably doesn't change the optimal policy.

#### When should a company not use RL?

When the problem doesn't actually need it. If actions don't affect future states, it's a bandit. If you can label the correct action, it's supervised learning, and imitation learning on good demonstrations is far more sample-efficient. If the need is a launch decision, it's an A/B test.

RL also needs either a trustworthy simulator or safe online exploration, a reward you actually trust, and the engineering capacity to handle instability, seed sensitivity and hard offline evaluation. Lacking those, a supervised ranker with logged exploration and OPE, or a contextual bandit, captures most of the value with a fraction of the risk. I'd want evidence that myopic decisions are measurably hurting long-term metrics before committing to full sequential RL.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Bootstrapping from terminal states | Adds phantom future value; wrong targets | Multiply the next-state term by `(1 − done)` |
| Treating time-limit truncation as termination | Agent learns the episode "ends" when it doesn't | Bootstrap on truncation; only zero out on true terminal |
| No logged propensities | Off-policy evaluation becomes impossible or biased | Log `π(a\|x)` with every decision |
| Deterministic production policy | No overlap; can't evaluate alternatives from logs | Keep a small amount of randomization |
| Plain IPS with tiny propensities | Variance explodes; estimates are noise | SNIPS, doubly robust, weight clipping, check effective sample size |
| Fixed ε forever | Linear regret; wastes reward | Decay ε, or use UCB / Thompson sampling |
| Optimizing a proxy reward | Reward hacking — clickbait, verbosity | Guardrail metrics, KL to reference, audit top-reward samples |
| Careless reward shaping | Changes the optimal policy | Potential-based shaping |
| DQN without a target network or replay | Moving targets and correlated samples cause divergence | Replay buffer + lagged target network |
| Single-seed RL results | High variance across seeds; false conclusions | Report mean and spread over several seeds |
| Offline RL with unconstrained Q-learning | Overestimates unseen actions and exploits them | CQL / IQL / BCQ; compare against behaviour cloning |
| Using full RL when a bandit suffices | Complexity and instability with no gain | Start with a contextual bandit |
| Dropping the KL penalty in RLHF | Policy drifts, exploits reward model, loses fluency | Tune `β`; monitor KL during training |

---

## Related Topics

- [Fine-Tuning](../deep_learning/intro_fine_tuning.md)
- [Neural Network Training](./intro_neural_network_training.md)
- [Transformers](./intro_transformers.md)
- [Sequence Models](./intro_sequence_models.md)
- [Causal Inference and Uplift Modeling](../classical_ml/intro_causal_inference.md)
- [Recommender Systems](../classical_ml/intro_recommender_systems.md)
- [A/B Testing & Experimentation](../mlops/intro_ab_testing.md)
- [LLM Fundamentals](../ai_genai/intro_llm_fundamentals.md)
- [LLM Evaluation](../mlops/intro_llm_evaluation.md)
- [Recommendation System Design](../system_design/recommendation_system.md)
- [PyTorch](../frameworks/intro_pytorch.md)
- [Deep Learning Overview](./README.md)
