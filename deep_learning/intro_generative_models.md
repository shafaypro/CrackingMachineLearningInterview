# Generative Models: VAEs, GANs, and Diffusion

Every image, audio, and video generation product rests on this lineage, and the trajectory (autoencoder → VAE → GAN → diffusion → flow matching) is one of the cleanest "explain the evolution and why each step happened" narratives in ML interviews. Diffusion in particular is now standard interview material for anyone touching generative AI.

---

## Table of Contents
1. [What Generative Means](#what-generative-means)
2. [The Family Tree](#the-family-tree)
3. [Autoencoders Are Not Generative](#autoencoders-are-not-generative)
4. [Variational Autoencoders](#variational-autoencoders)
5. [GANs](#gans)
6. [Diffusion Models](#diffusion-models)
7. [Latent Diffusion and Stable Diffusion](#latent-diffusion-and-stable-diffusion)
8. [Conditioning and Guidance](#conditioning-and-guidance)
9. [Samplers and Step Count](#samplers-and-step-count)
10. [Flow Matching and Consistency Models](#flow-matching-and-consistency-models)
11. [Evaluating Generative Models](#evaluating-generative-models)
12. [Model Comparison](#model-comparison)
13. [Interview Q&A](#interview-qa)
14. [Common Pitfalls](#common-pitfalls)
15. [Related Topics](#related-topics)

---

## What Generative Means

A **discriminative** model learns `p(y|x)`: a decision boundary. A **generative** model learns `p(x)` (or `p(x|c)`), the data distribution itself, well enough to draw new samples from it.

Three properties are in tension, and every architecture trades among them. This "generative trilemma" is the organizing idea of the whole field:

| Property | Meaning |
|---|---|
| **Sample quality** | Do outputs look real? |
| **Mode coverage / diversity** | Does it capture the *whole* distribution, not a corner of it? |
| **Sampling speed** | How many forward passes per sample? |

GANs picked quality and speed, sacrificing coverage. VAEs picked coverage and speed, sacrificing sharpness. Diffusion picked quality and coverage, sacrificing speed, and the last five years of research has been about buying that speed back.

---

## The Family Tree

```
Autoencoder ──► VAE ──────────┐
   (compression)  (probabilistic latent)
                               ├──► Latent Diffusion (Stable Diffusion)
GAN ──► StyleGAN ─────────────┤     (VAE compresses, diffusion generates)
   (adversarial)               │
                               │
Diffusion ──► DDPM ──► DDIM ──┘──► Flow Matching / Consistency
   (iterative denoising)              (fewer steps)

Autoregressive (PixelCNN, GPT-style image tokens): a parallel branch
```

---

## Autoencoders Are Not Generative

An autoencoder learns `encoder: x → z` and `decoder: z → x̂`, minimizing reconstruction error. It compresses well, but you **cannot sample from it**: the latent space has no known distribution, so a random `z` decodes to noise. There are "holes": regions of latent space no training example maps to, which decode to nothing meaningful.

That gap is exactly what the VAE fixes, and stating it is the natural way into a VAE explanation.

---

## Variational Autoencoders

A VAE makes the latent space **probabilistic and structured**. The encoder outputs a distribution (a mean and variance) rather than a point, and a KL term forces that distribution toward a standard Gaussian prior. Now sampling `z ~ N(0, I)` and decoding produces valid samples.

The loss has two terms:

```
L = E[log p(x|z)]  -  β · KL(q(z|x) ‖ p(z))
    └ reconstruction ┘   └ regularizer: keep latents near N(0,I) ┘
```

```python
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, d_in, d_hidden=512, d_latent=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU())
        self.mu = nn.Linear(d_hidden, d_latent)
        self.logvar = nn.Linear(d_hidden, d_latent)
        self.dec = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_in),
        )

    def reparameterize(self, mu, logvar):
        # THE trick: sample via z = mu + sigma*eps so gradients flow through mu/sigma.
        # Sampling z directly from N(mu, sigma) is not differentiable.
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
    # Closed form KL between N(mu, sigma) and N(0, I)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl
```

**The reparameterization trick** is the most-asked VAE question. You need to backpropagate through a sampling operation, but sampling is not differentiable. Rewriting `z ~ N(μ, σ²)` as `z = μ + σ·ε` with `ε ~ N(0, I)` moves the randomness into `ε`, which has no parameters, so gradients flow cleanly through `μ` and `σ`.

**Why VAE samples are blurry**: the reconstruction term is typically a pixel-wise Gaussian likelihood (i.e. MSE), which is minimized by the *conditional mean*. When several plausible images fit a latent code, the optimum is their average, and the average of sharp images is blurry. It's not a bug in the optimizer; it's what the objective asks for.

**β-VAE** raises the KL weight to encourage disentangled latent dimensions, trading reconstruction fidelity for interpretability. **Posterior collapse** is the classic failure: the KL term drives `q(z|x)` to the prior, the latent carries no information, and the decoder ignores it, mitigated by KL annealing (warm up β from 0) or free bits.

---

## GANs

Two networks in a minimax game: a **generator** maps noise to samples, a **discriminator** classifies real versus generated. The generator learns from the discriminator's gradients.

```
min_G max_D  E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

```python
# One training step: the alternating structure is the whole idea
# 1) Discriminator: push real toward 1, fake toward 0
d_loss = bce(D(real), ones) + bce(D(G(z).detach()), zeros)   # detach: no G gradients here
d_loss.backward(); opt_D.step()

# 2) Generator: use the NON-saturating loss, maximize log D(G(z))
#    rather than minimize log(1 - D(G(z))), which has vanishing gradients early
g_loss = bce(D(G(z)), ones)
g_loss.backward(); opt_G.step()
```

GANs produce sharp results because there is no pixel-wise averaging anywhere: the discriminator only asks "does this look real?", and a blurry image is easy to reject.

**Why GANs are hard to train**, the classic list:
- **Mode collapse**: the generator finds one output that fools the discriminator and produces only that, losing diversity. Mitigations: minibatch discrimination, unrolled GANs, WGAN-GP.
- **Non-convergence**: it's a minimax game, not a minimization, so the pair can oscillate indefinitely rather than settle.
- **Vanishing gradients**: a discriminator that wins too decisively gives the generator nothing to learn from; hence the non-saturating loss.
- **No meaningful loss curve**: the losses are adversarial and don't indicate quality, so you cannot early-stop on them. You evaluate with FID instead.

**WGAN** replaces Jensen-Shannon divergence with the Wasserstein (earth-mover) distance, which stays informative even when the distributions don't overlap, that's why it gives usable gradients and a loss that actually correlates with quality. Gradient penalty (WGAN-GP) enforces the required Lipschitz constraint better than the original weight clipping.

GANs remain competitive where **single-step generation** matters: real-time super-resolution, face generation, and as decoders inside other systems.

---

## Diffusion Models

The idea that displaced GANs. Split generation into many small, easy steps instead of one hard one.

**Forward process** (fixed, no learning): gradually add Gaussian noise over `T` steps until the image is pure noise.

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t)·x_{t-1}, β_t·I)
```

A key property is that you can jump to any timestep in closed form, which is what makes training tractable: no need to simulate the chain:

```
x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε        where  ᾱ_t = Π(1-β_s)
```

**Reverse process** (learned): a network predicts the noise that was added, and you subtract it step by step.

The training objective reduces to something remarkably simple: predict the noise:

```python
def diffusion_training_step(model, x0, scheduler):
    B = x0.size(0)
    t = torch.randint(0, scheduler.T, (B,), device=x0.device)   # random timestep per sample
    noise = torch.randn_like(x0)

    # Jump straight to x_t in closed form
    a_bar = scheduler.alpha_bar[t].view(-1, 1, 1, 1)
    x_t = a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * noise

    noise_pred = model(x_t, t)              # the network predicts the noise
    return nn.functional.mse_loss(noise_pred, noise)   # that's the entire loss
```

**Why this is so much more stable than a GAN**: it's a plain regression problem with a fixed target and a single network: no adversary, no minimax, no mode collapse. The loss is meaningful and decreases monotonically. Coverage is good because the model must explain the *whole* data distribution rather than find one region that fools a critic.

The cost is sampling: generation requires many sequential forward passes (originally 1,000), versus one for a GAN. Everything since has been about reducing that.

The backbone is typically a **U-Net** with skip connections, self-attention at lower resolutions, and timestep embeddings (sinusoidal, like positional encodings) so one network handles all noise levels. Recent systems increasingly use a **Diffusion Transformer (DiT)** instead, which scales better.

---

## Latent Diffusion and Stable Diffusion

Running diffusion directly on 512×512 pixels is enormously expensive. **Latent diffusion** fixes this with a two-stage design:

1. A **VAE** compresses the image to a small latent (512×512×3 → 64×64×4, roughly 48× fewer values).
2. **Diffusion runs entirely in that latent space.**
3. The VAE decoder maps the final latent back to pixels.

This is the architectural insight behind Stable Diffusion, and it's a satisfying answer because it shows how two families combine: the VAE handles *perceptual compression* (throwing away imperceptible detail), while diffusion handles *semantic generation* in a space small enough to be affordable. The speedup is roughly an order of magnitude with no meaningful quality loss.

Text conditioning enters through **cross-attention** layers in the U-Net, where the keys and values come from a text encoder (CLIP or T5) and queries come from the image features.

---

## Conditioning and Guidance

**Classifier-free guidance (CFG)** is how text-to-image models follow prompts strongly, and it comes up often.

During training, the conditioning is randomly dropped (say 10% of the time), so one network learns both conditional and unconditional prediction. At sampling time, extrapolate away from the unconditional prediction:

```python
def cfg_predict(model, x_t, t, cond, uncond, guidance_scale=7.5):
    eps_cond = model(x_t, t, cond)
    eps_uncond = model(x_t, t, uncond)
    # Push away from unconditional, toward conditional
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)
```

The scale trades prompt adherence against diversity and realism: around 1 ignores the prompt, 7-8 is the usual sweet spot, and very high values produce over-saturated, artifact-heavy images that rigidly obey the prompt. It costs **two forward passes per step**, which is why it roughly doubles inference cost.

Other conditioning mechanisms worth naming: **ControlNet** (a trainable copy of the encoder that injects spatial conditions like depth maps or poses), **IP-Adapter** (image prompting), and **LoRA** (the same parameter-efficient fine-tuning idea as in LLMs, used for styles and characters).

---

## Samplers and Step Count

| Sampler | Steps | Notes |
|---|---|---|
| **DDPM** | ~1000 | Original, stochastic, slow |
| **DDIM** | 20-50 | **Deterministic**, skips steps, enables interpolation and inversion |
| **DPM-Solver++** | 15-25 | ODE solver; strong quality per step |
| **Euler / Heun** | 20-40 | Simple ODE integrators, widely used |
| **LCM / Turbo** | **1-4** | Distilled models trading some quality for speed |

**DDIM** is the important conceptual jump: it reformulates the reverse process as a deterministic ODE rather than a stochastic chain, which means you can take far larger steps and get reproducible outputs from a fixed seed. Determinism also enables *inversion* mapping a real image back to a latent for editing.

**Distillation** (LCM, Turbo, consistency models) trains a student to reproduce many teacher steps in one, reaching 1-4 step generation. That is what makes real-time image generation feasible, at some cost in fine detail and diversity.

---

## Flow Matching and Consistency Models

The current frontier, worth a sentence to show you're up to date.

**Flow matching** trains a model to predict a velocity field that transports noise to data along a straight path, rather than learning to reverse a stochastic noising chain. Straighter trajectories mean fewer integration steps for the same quality, and the training objective is simpler. Rectified flow is the variant behind several recent leading image and video models.

**Consistency models** are trained so that points along the same trajectory map to the same endpoint, enabling genuine single-step generation while retaining diffusion-like coverage.

The through-line: every step in this lineage is an attempt to keep diffusion's quality and coverage while approaching GAN sampling speed.

---

## Evaluating Generative Models

There is no likelihood to report for GANs and no single satisfying metric anywhere, so evaluation is a basket.

| Metric | Measures | Weakness |
|---|---|---|
| **FID** | Distance between real/generated feature distributions (Inception) | Sensitive to sample count; biased; Inception features are ImageNet-specific |
| **Inception Score** | Sharpness + class diversity | Ignores real data entirely; largely superseded |
| **CLIP Score** | Text-image alignment | Says nothing about image quality |
| **Precision / Recall** | Separates fidelity from coverage | More informative than FID alone |
| **Human preference** | What actually matters | Expensive, slow, subjective |

**FID's key property**: and the standard interview point: is that it compares *distributions*, not individual images, so it penalizes mode collapse in a way per-sample metrics cannot. Its weaknesses are worth naming too: it's biased by sample size (always compare at the same N, typically 50k), and it inherits ImageNet's notion of similarity, which is a poor fit for medical or artistic domains.

**Precision and recall for generative models** decompose quality: precision measures whether samples fall in the real distribution (fidelity), recall whether the real distribution is covered (diversity). A mode-collapsed GAN scores high precision and low recall: exactly the diagnosis FID alone obscures.

---

## Model Comparison

| | VAE | GAN | Diffusion |
|---|---|---|---|
| Training stability | **Stable** | Unstable | **Stable** |
| Sample quality | Blurry | **Sharp** | **Sharp** |
| Mode coverage | **Good** | Poor (collapse) | **Excellent** |
| Sampling speed | **1 pass** | **1 pass** | 20-1000 passes |
| Likelihood | Lower bound (ELBO) | None | Approximate |
| Latent space | **Structured, interpolable** | Structured (StyleGAN) | Not natively |
| Controllability | Moderate | Moderate | **Excellent** (guidance, ControlNet) |
| Best for | Representation learning, anomaly detection, compression | Real-time single-step generation | Text-to-image/video, editing, most 2026 products |

---

## Interview Q&A

#### Why can't you sample from a plain autoencoder, and how does a VAE fix it?

An autoencoder only learns to reconstruct: nothing constrains the *shape* of the latent space, so it has arbitrary geometry with gaps that no training example maps to. Pick a random `z` and the decoder produces noise, because that region was never trained.

A VAE makes the encoder output a distribution (mean and log-variance) instead of a point, and adds a KL term pulling those posteriors toward a standard Gaussian prior. Latents are now spread over a known distribution with no holes, so sampling `z ~ N(0, I)` and decoding gives valid samples. The reparameterization trick (`z = μ + σ·ε`) makes the sampling step differentiable so the whole thing trains end to end.

#### Why are VAE samples blurry while GAN samples are sharp?

It follows from the objective. The VAE's reconstruction term is usually a pixel-wise Gaussian likelihood, which is just MSE, and MSE is minimized by the **conditional mean**. When one latent code is consistent with several plausible images, the loss-optimal output is their average, and averaging sharp images produces blur. The model is doing exactly what you asked.

A GAN has no pixel-wise term at all. The discriminator only judges realism, and a blurry image is trivially identified as fake, so blur is heavily penalized. That's why GANs are sharp, and also why they can collapse onto a few realistic outputs, since nothing forces them to cover the distribution.

#### Explain how diffusion models work and why they beat GANs.

Diffusion defines a fixed forward process that gradually adds Gaussian noise until the image is pure noise, then trains a network to reverse it. The training objective collapses to something simple: sample a random timestep, add the corresponding noise in closed form, and have the network predict the noise that was added, plain MSE regression.

They beat GANs on stability and coverage. There's no adversary, so no minimax dynamics, no mode collapse, and a loss that actually decreases and means something. Because the model must explain the whole distribution at every noise level, coverage is excellent. And splitting generation into many small denoising steps makes each step an easy problem, whereas a GAN must map noise to a photorealistic image in one shot.

The cost is sampling speed: many sequential forward passes versus one. DDIM, ODE solvers, and distillation to 1-4 steps are the response to that.

#### What is classifier-free guidance and what does it trade off?

During training, the conditioning signal is randomly dropped some fraction of the time, so a single network learns both conditional and unconditional noise prediction. At sampling, you extrapolate: `ε = ε_uncond + s·(ε_cond - ε_uncond)`, pushing the prediction away from the unconditional direction and further toward the conditional one.

The guidance scale trades prompt adherence against diversity and naturalness. Near 1, the prompt is barely followed; around 7-8 is typically the sweet spot; very high values give rigid prompt-following with over-saturated colours and artifacts, and noticeably less variation across seeds. It also costs two forward passes per step, roughly doubling inference cost, which is why some deployments use distilled guidance to fold it into one pass.

#### What problem does latent diffusion solve, and how?

Pixel-space diffusion at 512×512 is prohibitively expensive: every one of dozens of denoising steps runs a U-Net over ~786k values.

Latent diffusion inserts a VAE that compresses the image into a much smaller latent (64×64×4 instead of 512×512×3, roughly 48× fewer values), and runs the entire diffusion process there, decoding only once at the end. The division of labour is the elegant part: the VAE handles perceptual compression, discarding detail humans don't notice, while diffusion does semantic generation in a space small enough to be affordable. Roughly an order of magnitude cheaper with negligible quality cost, and it's the reason Stable Diffusion could run on consumer GPUs.

#### Why is FID the standard metric, and what are its limitations?

FID embeds real and generated images with an Inception network and compares the two feature distributions as Gaussians via Fréchet distance. The key property is that it compares **distributions rather than individual samples**, so it penalizes mode collapse: a GAN producing ten perfect images scores terribly, which per-sample quality measures would miss.

Limitations worth naming: it's biased by sample count, so comparisons are only valid at the same N (50k is conventional); Inception features encode ImageNet's notion of similarity, which transfers poorly to medical imaging or line art; it's insensitive to some artifacts humans find glaring; and it says nothing about prompt alignment. I'd report FID alongside precision/recall (which separate fidelity from coverage), plus CLIP score for text alignment and a human preference study for anything shipping.

#### When would you still use a GAN in 2026?

When single-step generation is a hard requirement. A GAN generates in one forward pass; even a heavily distilled diffusion model needs 1-4, and undistilled needs 20+. For real-time video super-resolution, live face manipulation, or on-device generation with a tight latency budget, that gap decides it.

GANs also remain common as *components* the decoder in a VQGAN, or an adversarial loss added to a reconstruction objective to remove blur. What they've largely lost is text-to-image generation, where diffusion's coverage, controllability, and training stability win decisively.

#### What is posterior collapse in a VAE?

The KL term drives the approximate posterior `q(z|x)` to match the prior for every input, so the latent carries no information about `x`. The decoder learns to ignore `z` and models the data unconditionally, which makes the model useless as a representation learner even though the loss may look fine.

It's most common with powerful autoregressive decoders that can model the data well on their own, so the cheapest way to reduce loss is to zero out the KL. Mitigations: **KL annealing** (warm β up from 0 so reconstruction establishes itself first), **free bits** (don't penalize KL below a floor per dimension), weakening the decoder, or switching to a discrete latent (VQ-VAE) where collapse can't happen the same way.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Expecting to sample from a plain autoencoder | Latent space has no known distribution | Use a VAE, or fit a density over the latents |
| Sampling `z` directly in a VAE | Not differentiable; no gradients | Reparameterization: `z = μ + σ·ε` |
| Judging GAN quality by the loss curve | Adversarial losses don't track quality | FID over training; visual checks |
| Forgetting `.detach()` on fakes in the D step | Generator gets wrong gradients | Detach `G(z)` when training D |
| Using the saturating generator loss | Vanishing gradients early in training | Non-saturating: maximize `log D(G(z))` |
| Comparing FID at different sample counts | FID is biased by N | Fix N (typically 50k) across comparisons |
| Reporting FID alone | Hides the fidelity/coverage tradeoff | Add precision/recall and CLIP score |
| Cranking guidance scale for "better" prompts | Over-saturation, artifacts, low diversity | Stay near 7-8; tune per model |
| Pixel-space diffusion at high resolution | Enormously expensive for no gain | Latent diffusion |
| Ignoring posterior collapse | Latent is uninformative; model is useless as an encoder | KL annealing, free bits, weaker decoder |
| Assuming more diffusion steps always help | Quality plateaus; cost grows linearly | Tune steps per sampler; 20-30 is often enough |

---

## Related Topics

- [Neural Network Training](./intro_neural_network_training.md)
- [Transformers](./intro_transformers.md)
- [Computer Vision](./intro_computer_vision.md)
- [Sequence Models](./intro_sequence_models.md)
- [Fine-Tuning](./intro_fine_tuning.md)
- [Multimodal AI](../ai_genai/intro_multimodal_ai.md)
- [LLM Inference Optimization](../ai_genai/intro_llm_inference_optimization.md)
- [Anomaly Detection](../classical_ml/intro_anomaly_detection.md)
- [Deep Learning Overview](./README.md)
