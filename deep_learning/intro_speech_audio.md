# Speech and Audio ML

Speech is the interface behind voice assistants, call-center analytics, meeting transcription, dictation, accessibility tools, and now real-time voice agents built on LLMs. It is also a field with its own vocabulary — spectrograms, CTC, WER, EER, vocoders — that interviewers use to check whether you have actually shipped audio models or only read about them. This guide covers the signal-processing basics, the main model families for recognition and synthesis, the metrics, and the production issues that decide whether a speech system works outside the lab.

---

## Table of Contents
1. [Digital Audio Basics](#digital-audio-basics)
2. [Features: From Waveform to Log-Mel](#features-from-waveform-to-log-mel)
3. [Data Augmentation](#data-augmentation)
4. [Automatic Speech Recognition](#automatic-speech-recognition)
5. [Decoding](#decoding)
6. [ASR Metrics](#asr-metrics)
7. [Streaming vs Offline](#streaming-vs-offline)
8. [Speaker Tasks](#speaker-tasks)
9. [Keyword Spotting and Wake Words](#keyword-spotting-and-wake-words)
10. [Text-to-Speech](#text-to-speech)
11. [Audio Classification and Sound Event Detection](#audio-classification-and-sound-event-detection)
12. [Speech LLMs and Voice Agents](#speech-llms-and-voice-agents)
13. [Production Concerns](#production-concerns)
14. [Interview Q&A](#interview-qa)
15. [Common Pitfalls](#common-pitfalls)
16. [Related Topics](#related-topics)

---

## Digital Audio Basics

Audio is a pressure wave sampled at regular intervals and quantized to integers.

| Term | Meaning | Typical values |
|---|---|---|
| **Sample rate** `fs` | Samples per second | 8 kHz (telephony), 16 kHz (ASR standard), 44.1/48 kHz (music, TTS) |
| **Nyquist frequency** | Highest representable frequency, `fs / 2` | 8 kHz for 16 kHz audio |
| **Bit depth** | Bits per sample | 16-bit PCM (about 96 dB dynamic range, ~6 dB per bit); 32-bit float in pipelines |
| **Channels** | Mono / stereo / multi-mic arrays | ASR models almost always take mono |

**Nyquist–Shannon**: a signal can be reconstructed exactly only if it contains no energy above `fs / 2`. Anything above that folds back into the representable band as a false lower frequency. That is **aliasing**, and it cannot be removed after the fact.

**Resampling** is where aliasing bites in practice. Downsampling 48 kHz to 16 kHz by keeping every third sample lets content between 8 and 24 kHz fold into the speech band. A proper resampler low-pass filters at the new Nyquist first, then decimates.

```python
import numpy as np
from scipy.signal import resample_poly

sr = 48_000
t = np.arange(sr) / sr
x = np.sin(2 * np.pi * 10_000 * t)          # 10 kHz tone, above the 8 kHz target Nyquist

naive = x[::3]                               # 16 kHz, no filter: tone aliases to 16k - 10k = 6 kHz
proper = resample_poly(x, up=1, down=3)      # anti-alias low-pass, then decimate

def peak_hz(sig, fs):
    spec = np.abs(np.fft.rfft(sig))
    return np.fft.rfftfreq(len(sig), 1 / fs)[spec.argmax()], spec.max()

print(peak_hz(naive, 16_000))    # (6000.0, ~8000) -> a phantom tone in the speech band
print(peak_hz(proper, 16_000))   # (6000.0, ~11)   -> residual ~57 dB down: filtered out
```

**Sample-rate mismatch** is the most common silent bug in speech work: a model trained on 16 kHz fed 8 kHz telephone audio upsampled naively, or 44.1 kHz audio passed in unresampled, produces garbage without raising an error. Always assert the sample rate at the model boundary.

---

## Features: From Waveform to Log-Mel

Raw waveforms are long (16,000 numbers per second) and the relevant information is in how energy is distributed across frequencies over time. The standard pipeline turns the waveform into a time-frequency image.

**Short-Time Fourier Transform (STFT).** Slice the signal into overlapping frames, multiply each by a window (Hann or Hamming) to reduce spectral leakage from the frame edges, and take an FFT of each frame. The magnitude squared is the **power spectrogram**, shape `(frames, n_fft/2 + 1)`.

**Window and hop trade off time against frequency resolution.** Frequency resolution is `fs / n_fft`; time resolution is the window length. A long window resolves pitch harmonics but smears fast events like plosives; a short window does the opposite. Speech convention is a **25 ms window with a 10 ms hop** (400 and 160 samples at 16 kHz), which gives 100 frames per second.

| Choice | Effect |
|---|---|
| Longer window | Finer frequency detail, blurrier timing |
| Shorter hop | More frames, more compute, smoother in time |
| Larger `n_fft` (zero-padding) | Interpolated frequency bins, not real extra resolution |

**Mel scale.** Human pitch perception is roughly linear below ~1 kHz and logarithmic above. The mel scale `m = 2595 · log10(1 + f / 700)` spaces triangular filters to match, concentrating resolution where speech information lives. Typical models use 80 mel bins (Whisper large-v3 uses 128).

**Log compression.** Loudness perception is roughly logarithmic, and the log turns multiplicative channel effects into additive ones. **Log-mel spectrograms are the default input for nearly every modern speech model.**

**MFCCs** apply a DCT to the log-mel energies and keep the first ~13 coefficients. The DCT decorrelates the features, which mattered for GMMs with diagonal covariances. Neural networks handle correlated inputs fine and do better with the richer log-mel, so MFCCs are now mostly a legacy or tiny-device feature.

```python
import numpy as np

def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
def mel_to_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    fmax = fmax or sr / 2
    hz = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)                 # centre freq of each FFT bin
    fb = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        lo, mid, hi = hz[i], hz[i + 1], hz[i + 2]
        rising = (bins - lo) / (mid - lo)
        falling = (hi - bins) / (hi - mid)
        fb[i] = np.maximum(0.0, np.minimum(rising, falling))  # triangle
    return fb

def log_mel(x, sr=16_000, win_ms=25, hop_ms=10, n_fft=512, n_mels=80):
    win, hop = sr * win_ms // 1000, sr * hop_ms // 1000       # 400, 160
    n_frames = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(win)                         # (T, win)
    power = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2         # (T, n_fft//2 + 1)
    mel = power @ mel_filterbank(sr, n_fft, n_mels).T         # (T, n_mels)
    return np.log(mel + 1e-10)

x = np.random.randn(16_000).astype(np.float32)                # 1 s of audio
print(log_mel(x).shape)                                       # (98, 80)
```

Per-utterance or global **mean/variance normalization** of these features (CMVN) is standard and removes some channel effects. Some models (wav2vec 2.0, HuBERT) skip hand-crafted features and learn a convolutional front end directly on the waveform.

---

## Data Augmentation

Labeled speech is expensive, and real audio varies in speaker, microphone, room, and noise. Augmentation is not optional.

| Technique | What it does | Why it helps |
|---|---|---|
| **SpecAugment** | Masks random frequency bands and time spans of the log-mel (plus optional time warping) | Cheap, on-the-fly; forces the model to use context instead of any single band or frame |
| **Additive noise** | Mix in noise (e.g. MUSAN, recorded background) at a random SNR, say 0–20 dB | Robustness to cafes, cars, TVs |
| **Speed perturbation** | Resample to 0.9x / 1.0x / 1.1x speed | Changes tempo and pitch together; triples effective data; strong, well-established gain |
| **Room impulse responses** | Convolve clean speech with a real or simulated RIR | Simulates reverberation and far-field microphones |
| **Codec / channel simulation** | Pass through telephone band-pass, MP3/Opus, packet loss | Matches deployment channel |

```python
import numpy as np
from scipy.signal import fftconvolve

def add_noise(x, noise, snr_db):
    noise = noise[: len(x)]
    p_sig, p_noise = np.mean(x ** 2), np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))
    return x + scale * noise

def add_reverb(x, rir):
    rir = rir / (np.abs(rir).max() + 1e-12)
    return fftconvolve(x, rir)[: len(x)]

def spec_augment(S, F=27, n_freq=2, T=100, n_time=2, rng=np.random.default_rng(0)):
    S = S.copy()                                   # S: (frames, mels)
    n_t, n_f = S.shape
    for _ in range(n_freq):
        f = rng.integers(0, F + 1); f0 = rng.integers(0, max(1, n_f - f))
        S[:, f0:f0 + f] = S.mean()
    for _ in range(n_time):
        t = rng.integers(0, min(T, n_t) + 1); t0 = rng.integers(0, max(1, n_t - t))
        S[t0:t0 + t, :] = S.mean()
    return S
```

Apply waveform augmentations (noise, reverb, speed) before feature extraction and SpecAugment after. Never augment the evaluation set, and keep a clean and a noisy test slice so you can see which one a change actually moved.

---

## Automatic Speech Recognition

ASR maps a variable-length audio sequence (T frames) to a shorter, variable-length token sequence (U tokens) with **no given alignment** between them. Every architecture below is a different answer to the alignment problem.

### HMM-GMM and hybrid systems (history)

The classic pipeline factored the problem: an **acoustic model** (GMMs, later DNNs) scored frames against context-dependent phone states of an HMM, a **pronunciation lexicon** mapped words to phones, and an **n-gram language model** scored word sequences. These were composed into a weighted finite-state transducer and searched with Viterbi decoding (Kaldi). Hybrid DNN-HMM systems dominated until around 2016–2019. They still show up where a lexicon and tight control over vocabulary matter, but training requires forced alignments and many separate components.

### CTC

**Connectionist Temporal Classification** lets an encoder emit one distribution per frame over the vocabulary **plus a special blank token**, and defines the probability of a transcript as the sum over every frame-level path that collapses to it. Collapse rule: merge consecutive repeats, then delete blanks.

```
frames:  h h _ e l l _ l o _      ->  "hello"
         (the blank between the two l-runs is what keeps "ll" from merging)
```

The blank does two jobs: it lets the model output "nothing" on frames between tokens (most frames), and it separates genuinely repeated characters. The sum over alignments is computed efficiently with a forward-backward dynamic program, so training needs only (audio, transcript) pairs.

The cost is a **conditional independence assumption**: each frame's output is predicted independently given the audio, so CTC has no internal language model and benefits a lot from an external one. It also requires `T` to be at least `U` plus the number of repeated adjacent labels. CTC models tend to produce **peaky** outputs: sharp spikes on a single frame per token with blank everywhere else.

```python
def ctc_greedy_decode(log_probs, blank=0):
    """log_probs: (T, V). Best token per frame, merge repeats, drop blanks."""
    best = log_probs.argmax(axis=-1)
    out, prev = [], None
    for tok in best:
        if tok != prev and tok != blank:
            out.append(int(tok))
        prev = tok
    return out
```

### Attention encoder-decoder

Listen, Attend and Spell style models use an encoder over audio and an autoregressive decoder that attends to it, like machine translation. No independence assumption, so the decoder acts as an implicit language model and accuracy is strong. Weaknesses: the decoder needs the whole utterance (not naturally streaming), attention can lose its place on long audio, producing **repeated or dropped phrases and hallucinated text**, and it has no hard monotonic alignment. **Joint CTC-attention** training (ESPnet) adds a CTC loss on the encoder to encourage monotonic alignment and is a common fix.

### RNN-Transducer

**RNN-T** combines an audio encoder, a **prediction network** that conditions on previously emitted tokens (an internal LM), and a **joint network** that combines both to output a distribution over vocabulary plus blank. Emitting blank means "advance to the next audio frame"; emitting a token means "stay on this frame". Training sums over all paths through the `T × U` lattice.

It removes CTC's independence assumption while staying **frame-synchronous and streaming-friendly**, which is why it became the standard for on-device and real-time ASR. The practical cost is memory: the joint output is `B × T × U × V`, so implementations use pruned RNN-T losses, function merging, or small vocabularies. Despite the name, the encoder is now usually a Conformer or transformer.

### Conformer

The **Conformer** encoder block combines self-attention (global context) with depthwise convolution (local patterns such as formant transitions), sandwiched between two half-step feed-forward layers. It beat pure transformer and pure CNN encoders on LibriSpeech and is the default encoder for CTC, RNN-T, and attention models alike. Encoders usually subsample the 100 fps features by 4x or more with a convolutional front end so attention cost stays manageable.

### Whisper and weakly supervised models

**Whisper** is a transformer encoder-decoder trained on 680k hours of audio paired with transcripts scraped from the web (later versions used more). The labels are noisy, but the scale and diversity make it robust zero-shot across accents, domains, and ~100 languages. Special tokens select the task: language ID, transcribe vs translate-to-English, and timestamps. It processes **fixed 30-second windows**, so long audio needs chunking, and it is known to **hallucinate fluent text on silence or noise**, which a VAD in front of it largely mitigates.

### Self-supervised: wav2vec 2.0 and HuBERT

Unlabeled audio is plentiful. Self-supervised models learn representations from it, then fine-tune with a small labeled set (typically with a CTC head).

| Model | Input | Pretext task |
|---|---|---|
| **wav2vec 2.0** | Raw waveform through a CNN (one latent per 20 ms) | Mask spans of latents; transformer must pick the true **quantized** latent from distractors (contrastive loss) plus a codebook diversity loss |
| **HuBERT** | Same front end | Masked prediction of **offline cluster IDs** (k-means on MFCCs, then on its own features in later iterations) with cross-entropy |
| **WavLM** | Same | HuBERT-style plus simulated overlapping speech and noise, which helps speaker and diarization tasks |

wav2vec 2.0 showed usable ASR from ten minutes of labeled data. These encoders are also the standard backbones for speaker, emotion, and low-resource language tasks, and the discrete units HuBERT produces feed into speech language models.

| Architecture | Streaming | Internal LM | Main weakness |
|---|---|---|---|
| CTC | Yes | No | Independence assumption; needs external LM |
| Attention enc-dec | No (not natively) | Yes | Hallucination and looping on long audio |
| RNN-T | **Yes** | Yes (prediction net) | Training memory; harder to train |
| Whisper-style | No (30 s windows) | Yes | Hallucination on silence; latency |

---

## Decoding

**Greedy decoding** takes the best token at each step. It is fast and, for strong models, often within a small relative WER of beam search.

**Beam search** keeps the top-k partial hypotheses. For CTC this is **prefix beam search**, which merges paths that collapse to the same prefix, tracking separately whether each prefix ended in blank or not.

**Language model fusion** brings in text-only knowledge, which is cheap to collect for a new domain:

```
score(y) = log p_AM(y | x) + λ · log p_LM(y) + β · |y|
```

- **Shallow fusion**: add the external LM score during beam search. The length bonus `β` counters the LM's bias toward short outputs. Tune `λ` and `β` on a dev set.
- **Rescoring**: generate an n-best list or lattice with a cheap LM, then rescore with a large neural LM or an LLM.
- **Deep/cold fusion**: combine hidden states of AM and LM during training; less common in practice.
- **Contextual biasing**: boost specific phrases (contact names, product SKUs) at decode time, via a prefix trie over bias phrases or a learned biasing module. This is often the highest-leverage fix for domain vocabulary.

An end-to-end model already contains an implicit LM learned from its training transcripts, so fusing an external LM can double-count; **internal LM estimation** subtracts an estimate of it and helps on cross-domain audio.

---

## ASR Metrics

**Word Error Rate** is the word-level edit distance between reference and hypothesis, normalized by reference length:

```
WER = (S + D + I) / N        substitutions, deletions, insertions, reference words
```

It can exceed 100% because insertions are unbounded.

```python
import numpy as np

def edit_distance(ref, hyp):
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    d[:, 0] = np.arange(len(ref) + 1)
    d[0, :] = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            sub = d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1])
            d[i, j] = min(sub, d[i - 1, j] + 1, d[i, j - 1] + 1)   # sub, del, ins
    return d[-1, -1]

def wer(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    return edit_distance(r, h) / max(len(r), 1)

def cer(ref, hyp):
    return edit_distance(list(ref), list(hyp)) / max(len(ref), 1)

print(wer("turn on the kitchen lights", "turn the kitchen light on"))   # 0.6
```

**Text normalization before scoring matters as much as the model.** "Dr." vs "doctor", "twenty five" vs "25", casing, and punctuation can move WER by several points. Fix one normalizer and apply it to both reference and hypothesis in every comparison.

| Metric | Measures | Use when |
|---|---|---|
| **WER** | Word-level accuracy | Default for space-delimited languages |
| **CER** | Character-level accuracy | Chinese, Japanese, Thai; or spelling-sensitive tasks |
| **Entity / keyword error** | Accuracy on names, numbers, key terms | When a few words carry most of the business value |
| **RTF** (real-time factor) | Processing time / audio duration | Throughput and cost; RTF 0.1 = 10x faster than real time |
| **Latency** | Time to first partial, time to final after the user stops | Interactive and streaming systems |

Report WER **per slice** (accent, noise level, device, domain), not only as an average. Aggregate WER is dominated by long, clean utterances and hides the users who are failing.

---

## Streaming vs Offline

| | Offline | Streaming |
|---|---|---|
| Context | Full utterance, both directions | Past plus a small lookahead |
| Accuracy | Best | Worse, typically by a noticeable relative margin |
| Latency | Waits for end of audio | Partials in hundreds of ms |
| Architectures | Attention enc-dec, full-context Conformer, Whisper | RNN-T, CTC with chunked or causal encoders |
| Use cases | Transcribing recordings, captions after the fact, analytics | Dictation, live captions, voice assistants and agents |

Streaming is built with **causal convolutions** and **chunked or limited-context attention**, where each frame attends to its chunk plus some left context and a fixed number of future frames. More lookahead buys accuracy at the cost of latency, and that knob is the main design decision.

Common hybrid designs: **two-pass** systems where a streaming RNN-T produces partials and a second, full-context pass rescores or rewrites the final result; and **cascaded or dual-mode encoders** that share weights between streaming and full-context modes.

**Endpointing**, deciding the user has finished, often dominates perceived latency more than the recognizer does. A fixed silence timeout of 500–800 ms is simple but cuts off slow speakers and makes fast ones wait. Learned endpointers that combine acoustic cues with whether the partial transcript sounds complete do better.

A subtle streaming issue is **emission delay**: an RNN-T trained without constraints can learn to wait for more future context before emitting a token, adding latency. Delay-penalized or FastEmit-style regularization counters it.

---

## Speaker Tasks

| Task | Question | Output |
|---|---|---|
| **Verification** | Is this the claimed speaker? (1:1) | Accept / reject |
| **Identification** | Which enrolled speaker is this? (1:N) | Speaker ID |
| **Diarization** | Who spoke when? (unknown speakers) | Time segments labeled speaker A, B, C |

All three rest on **speaker embeddings**: fixed-size vectors where the same voice maps close together regardless of words.

- **i-vectors**: factor-analysis based, pre-deep-learning.
- **x-vectors**: a TDNN over frames, a **statistics pooling** layer (mean and std over time) to get an utterance-level vector, trained to classify thousands of training speakers. The penultimate layer is the embedding.
- **ECAPA-TDNN**: adds squeeze-excitation Res2Net blocks, multi-layer feature aggregation, and attentive statistics pooling; a strong default.
- Trained with margin losses such as **AAM-softmax** (ArcFace-style) so unseen speakers still cluster, then scored with cosine similarity (or PLDA).

**Equal Error Rate** is the threshold-free verification metric: the operating point where the false accept rate equals the false reject rate. Deployed systems pick a threshold for the real cost trade-off, reported with minDCF or FAR at a fixed FRR.

```python
import numpy as np

def eer(scores, labels):
    """labels: 1 = same speaker, 0 = different."""
    best = (1.0, None)
    for t in np.unique(scores):
        far = np.mean(scores[labels == 0] >= t)     # impostors accepted
        frr = np.mean(scores[labels == 1] < t)      # genuine rejected
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

**Diarization** pipelines run VAD, split speech into short segments, embed each, cluster (agglomerative or spectral, with the number of speakers estimated), then resegment. The weakness is **overlapping speech**, which a one-speaker-per-segment pipeline cannot represent. End-to-end neural diarization (EEND) predicts multi-label speaker activity per frame and handles overlap; hybrid systems like pyannote combine local neural segmentation with global clustering. The metric is **DER** = missed speech + false alarm + speaker confusion, as a fraction of total speech time.

Voice is a biometric, and verification is vulnerable to replay and synthesized-voice attacks, so production systems pair it with **anti-spoofing** detectors and do not rely on voice alone for high-value authentication.

---

## Keyword Spotting and Wake Words

A wake word detector ("Hey ...") runs continuously on the device, so the constraints are extreme: tens to hundreds of KB of model, milliwatts of power, often on a DSP or microcontroller, with audio never leaving the device until triggered.

- **Models**: small CNNs, depthwise-separable CNNs, or small streaming RNN/attention models over MFCC or log-mel frames, int8-quantized.
- **Cascades**: a tiny always-on first stage with high recall, a larger on-device verifier, and often a server-side check on the uploaded audio. Each stage trades power for precision.
- **Metrics**: **false accepts per hour** of background audio (TV, conversation) against **false reject rate**. A 5% false reject rate annoys users; a false accept every few hours is a privacy incident. Evaluate on many hours of realistic negative audio, not just short clips.
- **Data**: positives are scarce and must cover accents, distances, and far-field conditions; hard negatives are phonetically similar phrases. TTS-generated positives and heavy augmentation are common.

Custom keyword spotting (user-defined phrases) is typically done with query-by-example embeddings or small ASR-based detectors instead of per-keyword classifiers.

---

## Text-to-Speech

Modern TTS has three stages, which are increasingly merged:

1. **Text front end**: **text normalization** ("$3.50" to "three dollars fifty", "Dr. Smith" vs "Smith Dr.", dates, units), then grapheme-to-phoneme conversion for pronunciation. Normalization errors are the most visible TTS failures, and they are rule-heavy and locale-specific.
2. **Acoustic model**: text or phonemes to a mel spectrogram. **Tacotron 2** did this autoregressively with attention, which occasionally skipped or repeated words. **FastSpeech 2** is non-autoregressive with an explicit duration predictor (plus pitch and energy predictors), so it is fast, parallel, and robust.
3. **Vocoder**: mel spectrogram to waveform. WaveNet produced high quality slowly; **GAN vocoders such as HiFi-GAN** are fast enough for real time on CPU. **VITS** trains acoustic model and vocoder end to end.

**Neural codec / token-based TTS.** Neural audio codecs (SoundStream, EnCodec, and descendants) compress audio into a few streams of discrete tokens using **residual vector quantization**: each codebook quantizes the residual left by the previous one, so the first codebook carries coarse content and later ones add detail. Once audio is tokens, TTS becomes language modeling: **VALL-E** style models predict codec tokens conditioned on text and a few seconds of a speaker prompt, enabling **zero-shot voice cloning**. Many current systems combine an autoregressive LM for semantic or coarse tokens with a non-autoregressive or flow-matching / diffusion stage for fine acoustic detail.

| Evaluation | What it captures |
|---|---|
| **MOS / MUSHRA** (human ratings) | Naturalness; still the gold standard |
| **ASR WER on TTS output** | Intelligibility, and catches skipped or repeated words |
| **Speaker similarity** (embedding cosine) | Cloning fidelity |
| **Latency to first audio** | Whether it works in a conversation |

Voice cloning raises consent and misuse risks; production systems gate it on verified consent and often add audio watermarking.

---

## Audio Classification and Sound Event Detection

**Audio classification** assigns labels to a clip: environmental sounds, music genre, machine faults, speech emotion, language ID. The standard recipe treats the log-mel spectrogram as an image: CNNs (PANNs) or transformers over spectrogram patches (**AST**, BEATs), usually pretrained on **AudioSet** (about 2M ten-second YouTube clips, 527 classes). Labels are usually **multi-label**, so use sigmoid outputs with binary cross-entropy, and evaluate with mAP.

**Sound event detection (SED)** also asks *when*: onset and offset of each event. Training data is often **weakly labeled** (clip-level tags only), so models predict frame-level probabilities and pool them to clip level (a multiple-instance learning setup), learning localization without timestamps. Evaluation uses event-based or segment-based F1 with time tolerances, or PSDS in the DCASE challenges.

Contrastive audio-text models (**CLAP**) enable zero-shot classification by comparing audio embeddings to text prompts, the audio counterpart of CLIP.

Practical notes: class imbalance is severe (speech and music dominate AudioSet), so use balanced sampling; mixup on spectrograms works well; and clip length at inference should match training or use sliding windows with aggregation.

---

## Speech LLMs and Voice Agents

Two architectures for a voice agent:

**Cascaded: VAD → streaming ASR → LLM → streaming TTS.** Each component is swappable and debuggable, the text intermediate is easy to log, evaluate, apply guardrails to, and use for tool calls. The costs are accumulated latency and loss of paralinguistics: tone, emotion, hesitation, and emphasis vanish at the ASR step.

**End-to-end speech-to-speech.** A single model consumes audio tokens (or encoder features) and emits audio tokens directly, sometimes with an interleaved text stream as an internal scaffold. It can respond faster, keep prosody and emotion, and some designs are **full-duplex** (listening while speaking, allowing backchannels like "mm-hm"). The costs: harder to control, evaluate, and constrain; tool use and guardrails are less mature; and debugging a bad response is harder without a transcript.

A middle ground is a **speech-in LLM** (audio encoder feeding an LLM directly, like LLaVA for audio) with a separate streaming TTS on the output.

**Latency budget.** Human conversational turn gaps are around 200 ms on average; voice agents feel responsive under roughly 800 ms from the user stopping to the agent's first audio, and awkward past about 1.5 s. A cascaded budget looks like:

| Stage | Typical budget |
|---|---|
| Endpointing (deciding the user finished) | 200–500 ms |
| ASR finalization | 50–200 ms |
| LLM time to first token | 150–500 ms |
| TTS time to first audio | 100–250 ms |
| Network and audio buffering | 50–150 ms |

The levers: stream everything (partial ASR, LLM token streaming, TTS starting on the first clause), speculatively start the LLM on a stable partial transcript, use smaller or faster LLMs for the first sentence, colocate components, and use semantic turn detection rather than long silence timeouts.

**VAD and barge-in.** A voice activity detector (WebRTC VAD, Silero VAD, or a small neural model) gates everything: it saves ASR compute, prevents Whisper-style hallucination on silence, and drives turn-taking. **Barge-in** means the user can interrupt: when VAD detects user speech during agent playback, stop TTS immediately, cancel the in-flight LLM generation, and **truncate the conversation history to what was actually played**, otherwise the LLM believes it said things the user never heard. Barge-in requires **acoustic echo cancellation** so the agent does not hear its own voice through the speaker and interrupt itself, and some filtering so a cough or "uh-huh" does not count as an interruption.

---

## Production Concerns

**Robustness.** WER gaps across accents, dialects, age groups, and non-native speakers are well documented and large. Noise, far-field microphones, reverberation, overlapping speakers, code-switching between languages, and domain vocabulary each add error. Build evaluation sets that represent your actual users, report per slice, and treat the worst slice as a release criterion.

**Domain adaptation**, roughly cheapest first:

1. Contextual biasing with domain phrases and entity lists.
2. LM adaptation or fusion with in-domain text (no audio needed).
3. Better text normalization and post-processing for numbers, units, and formatting.
4. Fine-tuning on in-domain audio, often parameter-efficient (LoRA or adapters) to avoid forgetting.
5. Pseudo-labeling unlabeled in-domain audio with a large model, filtering by confidence, and training on it (noisy student / distillation).

**PII and privacy.** Transcripts contain names, addresses, card numbers, health details, and anything else people say. Redact with NER plus pattern rules on the transcript, and mute or bleep the matching audio span using word timestamps. Voice itself is biometric data under laws such as GDPR and Illinois BIPA, which affects retention and consent for recording. Minimize raw audio retention, keep training on customer audio opt-in, and check that logs, analytics pipelines, and labeling vendors do not receive unredacted transcripts.

**Hallucination.** Encoder-decoder models, Whisper in particular, can produce fluent text that was never spoken, especially on silence, music, or noise. In high-stakes settings (medical, legal) this is worse than a visible error. Mitigate with VAD, no-speech probability and log-probability thresholds, compression-ratio checks for repetition loops, and CTC or RNN-T models where hallucination risk is unacceptable.

**Serving.** Batch offline transcription for throughput (RTF matters); streaming needs stateful sessions pinned to a worker with carried encoder state, which complicates autoscaling. Quantized models on-device avoid network latency and keep audio local.

```python
# Optional: requires `pip install transformers torch` and ffmpeg for decoding
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="openai/whisper-small",
               chunk_length_s=30)                        # chunk long audio into 30 s windows
out = asr("meeting.wav", return_timestamps=True)
print(out["text"])
```

---

## Interview Q&A

#### Walk me through turning a raw waveform into model input.

Resample to the model's rate (usually 16 kHz) with a proper anti-aliasing resampler and convert to mono float. Frame it with a 25 ms window and 10 ms hop, apply a Hann window, and take the FFT of each frame to get a power spectrogram. Multiply by a bank of about 80 triangular mel filters, which compresses the frequency axis the way hearing does, then take the log so the dynamic range is manageable and channel effects become additive. Normalize per utterance or with global statistics.

The result is a `(frames, 80)` log-mel spectrogram at 100 frames per second. The encoder usually subsamples it by 4x or more before attention. MFCCs add a DCT on top, which decorrelated features for GMMs but loses information neural networks can use, so log-mel is the modern default.

#### What is CTC and why does it need a blank token?

CTC trains a frame-level classifier without frame-level alignments. The encoder outputs a distribution over tokens plus blank at every frame, and the probability of a transcript is the sum over all frame paths that collapse to it (merge repeats, delete blanks), computed with forward-backward dynamic programming.

The blank serves two purposes. Most frames fall between tokens, and blank lets the model say "nothing new here". It also separates genuine repeats: without a blank between them, "l l" would merge into one "l", so "hello" would be impossible to emit.

The limitation is that frame outputs are conditionally independent given the audio, so CTC has no notion of which token sequences are likely. That is why CTC models gain a lot from an external language model, and why RNN-T and attention models, which condition on previous outputs, usually beat plain CTC at equal size.

#### CTC vs attention encoder-decoder vs RNN-T. When would you pick each?

**CTC** is simple, fast, non-autoregressive, and streams easily. It is a good choice for fine-tuning self-supervised encoders, for alignment and timestamps, and for anywhere hallucination is unacceptable, since it cannot invent text unrelated to the frames. It needs an LM for best accuracy.

**Attention encoder-decoder** is the most accurate for offline transcription because the decoder is a full conditional LM, and it handles translation and multitask setups naturally (Whisper). It is not natively streaming and can hallucinate or loop on long or silent audio.

**RNN-T** is what I would pick for streaming and on-device: it keeps a prediction network (so no independence assumption) while emitting frame-synchronously. The cost is training memory from the `T × U × V` joint tensor and more tuning. Many production systems combine them: a Conformer encoder with RNN-T for streaming, plus a CTC or attention head for rescoring.

#### How do wav2vec 2.0 and HuBERT learn without labels?

Both mask spans of the encoder's latent sequence and train a transformer to predict something about the masked positions from context, like BERT for audio. They differ in the target.

wav2vec 2.0 quantizes the CNN latents into a learned codebook and uses a contrastive loss: for each masked position, identify the true quantized latent among distractors sampled from other positions. A diversity loss keeps the codebook from collapsing.

HuBERT makes targets offline: cluster MFCCs with k-means, train to predict the cluster IDs of masked frames with cross-entropy, then re-cluster using the model's own intermediate features and train again. Its targets are more stable than a jointly learned codebook.

Fine-tuning either with a CTC head on a small labeled set gives strong ASR, which matters most for low-resource languages and domains. The learned representations also transfer well to speaker, emotion, and diarization tasks.

#### Your model's WER is 8% on the test set but users say it's bad. What do you check?

First, whether the test set represents users. Aggregate WER hides slices, so break it down by accent, device, noise level, utterance length, and domain. Often clean read speech dominates the test set while users are in cars with accents.

Second, whether WER measures what users care about. Getting a name, address, or number wrong costs more than dropping "the", so measure entity error rate on the words that drive the task. Also check that text normalization is not hiding or inflating errors.

Third, latency and endpointing. Users often describe "cut me off" or "slow" as "bad recognition". Measure time to first partial and endpoint behavior.

Fourth, the input path in production: sample rate, codec, gain, channel selection, and VAD truncating the start of utterances. A pipeline bug upstream of the model is common and does not appear in offline evaluation.

#### How do you adapt a general ASR model to a medical or legal domain?

Start cheap. Contextual biasing with domain term lists and shallow fusion with an LM trained on in-domain text often fix most vocabulary errors without any audio. Then fix normalization for domain formats such as dosages, statute citations, and units.

If that plateaus, collect in-domain audio. Pseudo-label a large amount of unlabeled audio with the strongest model available, filter by confidence, and have humans correct a smaller subset. Fine-tune with LoRA or adapters, or a low learning rate with a mix of general data, to avoid catastrophic forgetting of general speech.

Evaluate on a held-out in-domain set sliced by speaker and recording condition, and track entity error on domain terms specifically. For medical, also measure hallucination rate, since inserted text in a clinical note is worse than a deletion.

#### Why do streaming models lose accuracy, and how do you reduce the gap?

A streaming model sees only past audio plus a small lookahead, while an offline model uses the entire utterance in both directions. Right context matters for speech: the end of a word or the next word often disambiguates the current one.

Levers: add a small amount of lookahead (a few hundred ms) and accept the latency; chunked attention with left context caching; a second pass that rescores or rewrites the final hypothesis with full context once the user stops, while partials come from the first pass; distillation from an offline teacher into the streaming student; and shared-weight dual-mode encoders trained for both. Also regularize emission delay so the model does not learn to wait internally.

#### Explain speaker verification, identification, and diarization, and how you'd evaluate verification.

Verification is 1:1: does this voice match the claimed identity. Identification is 1:N over enrolled speakers. Diarization labels who spoke when for an unknown set of speakers in a recording.

All use speaker embeddings from x-vector or ECAPA-TDNN style networks: frame-level layers, a pooling layer that summarizes over time, and a margin-based classification loss over many training speakers. At test time the classifier is dropped and embeddings are compared with cosine similarity.

For verification I report EER, the point where false accepts equal false rejects, as a threshold-free summary, but I choose the deployed threshold from the actual cost trade-off, often a fixed low false accept rate. Evaluate on trials with realistic channel mismatch between enrollment and test, include spoofing attacks (replay, TTS clones), and check EER by demographic group.

#### How does a wake-word system avoid draining the battery and triggering constantly?

It is a cascade. A tiny always-on model, tens of KB and int8, runs on a low-power DSP and is tuned for high recall. When it fires, a larger on-device model verifies the detection, and often the server re-checks the audio before acting. Only the first stage runs all the time, so power stays low.

False accepts are controlled with hard negatives in training (phonetically similar phrases, TV audio), evaluation measured as false accepts per hour over many hours of realistic background audio, and per-stage thresholds set against that metric. False rejects are controlled with augmentation for distance, noise, and accents. The trade-off is set explicitly because a false accept means unintended recording, which is a privacy failure, not just a quality one.

#### Walk through a modern TTS pipeline and where codec language models fit.

Text normalization expands numbers, dates, abbreviations, and symbols into words; grapheme-to-phoneme handles pronunciation. An acoustic model predicts a mel spectrogram, either autoregressively (Tacotron 2) or non-autoregressively with explicit durations (FastSpeech 2), and a neural vocoder like HiFi-GAN turns the mel spectrogram into a waveform.

Codec language models replace the mel-plus-vocoder interface with discrete tokens from a neural codec that uses residual vector quantization. TTS then becomes next-token prediction over codec tokens conditioned on text and a short speaker prompt, which gives zero-shot voice cloning and natural prosody from scale. The codec decoder turns tokens back into audio. The trade-offs are autoregressive latency and occasional skipped or repeated words, so systems often add non-autoregressive or flow-matching stages and measure intelligibility with ASR WER on the output.

#### Design a low-latency voice agent. Cascaded or end-to-end?

I would default to cascaded for most business use cases: streaming ASR, an LLM with tools, streaming TTS. The text intermediate makes it auditable, lets me apply guardrails and tool calls, and lets me swap or upgrade each component independently. End-to-end speech-to-speech is attractive when prosody, emotion, and full-duplex interaction matter more than control, for example companionship or language practice.

For latency, target first audio within about 800 ms of the user finishing. Everything streams: ASR partials, LLM tokens, and TTS synthesis starting on the first clause. Endpointing is usually the biggest single cost, so use a semantic turn detector rather than a long silence timeout, and speculatively start the LLM on a stable partial.

For turn-taking: VAD gates the pipeline; barge-in stops TTS and cancels generation when the user speaks; echo cancellation keeps the agent from hearing itself; and the conversation history is truncated to the audio actually played. Monitor end-to-end latency percentiles, not averages, since p95 is what users remember.

#### What are the privacy risks in a speech product and how do you handle them?

Audio and transcripts carry PII people did not intend to share: names, account numbers, health information, and third parties in the background. Voice itself is biometric data under some laws, so it needs consent and retention limits.

Controls: tell users when they are recorded; redact transcripts with NER and pattern rules and mute the corresponding audio using word timestamps; minimize raw audio retention; restrict training on customer data to opt-in; and audit every downstream consumer (logs, analytics, labeling vendors). For wake-word devices, keep audio on the device until the wake word is confirmed. On-device ASR removes the question of where audio goes entirely, at some accuracy cost.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Sample-rate mismatch between training and serving | Model sees shifted spectra; accuracy collapses silently | Assert sample rate at the model boundary; resample once, correctly |
| Downsampling by dropping samples | High frequencies alias into the speech band | Use a resampler with an anti-aliasing filter (`resample_poly`) |
| Different text normalization for reference and hypothesis | WER changes by points for reasons unrelated to the model | One shared normalizer for every comparison |
| Reporting only aggregate WER | Hides accent, noise, and device slices that fail | Slice evaluation; gate releases on the worst slice |
| Test speakers overlapping with training speakers | Inflated WER and EER; model memorized voices | Split by speaker (and by session or device) |
| Feeding silence or music to Whisper-style models | Fluent hallucinated text | VAD first; no-speech and log-prob thresholds |
| Silence-timeout endpointing only | Cuts off slow speakers, adds latency for fast ones | Learned or semantic endpointing |
| No echo cancellation in a voice agent | Agent hears itself and triggers false barge-in | AEC plus VAD tuned for playback |
| Keeping the full LLM reply after barge-in | Model thinks it said things the user never heard | Truncate history to the audio actually played |
| Evaluating wake words on short clips only | False accepts per hour are invisible | Many hours of realistic negative audio |
| External LM weight tuned on the test set | Optimistic results that do not transfer | Tune `λ` and `β` on a dev set |
| Unredacted transcripts in logs and label queues | PII leak and compliance exposure | Redact at ingestion; restrict raw audio access |

---

## Related Topics

- [Sequence Models](./intro_sequence_models.md)
- [Transformers](./intro_transformers.md)
- [Generative Models](./intro_generative_models.md)
- [Model Compression](./intro_model_compression.md)
- [Fine-Tuning](./intro_fine_tuning.md)
- [Neural Network Training](./intro_neural_network_training.md)
- [Multimodal AI](../ai_genai/intro_multimodal_ai.md)
- [LLM Inference Optimization](../ai_genai/intro_llm_inference_optimization.md)
- [NLP Fundamentals](../classical_ml/intro_nlp_fundamentals.md)
- [Time Series](../classical_ml/intro_time_series.md)
- [Model Serving](../mlops/intro_model_serving.md)
- [Responsible AI and Fairness](../mlops/intro_responsible_ai_fairness.md)
- [Hugging Face](../frameworks/intro_huggingface.md)
- [Deep Learning Overview](./README.md)
