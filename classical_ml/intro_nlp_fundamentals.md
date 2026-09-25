# NLP Fundamentals

Before transformers there was a whole discipline of turning text into features, and it hasn't gone away: TF-IDF still beats a fine-tuned BERT on plenty of small-data classification problems, tokenization decisions still determine what an LLM can represent, and "what's your baseline?" is still the first question about any text model.

This covers the classical foundation plus the tokenization and representation concepts that carry straight into modern systems.

---

## Table of Contents
1. [Why This Still Matters](#why-this-still-matters)
2. [Text Preprocessing](#text-preprocessing)
3. [Tokenization](#tokenization)
4. [Bag of Words and TF-IDF](#bag-of-words-and-tf-idf)
5. [Word Embeddings](#word-embeddings)
6. [From Static to Contextual](#from-static-to-contextual)
7. [Text Classification](#text-classification)
8. [Named Entity Recognition](#named-entity-recognition)
9. [Topic Modeling](#topic-modeling)
10. [Text Similarity](#text-similarity)
11. [Evaluation](#evaluation)
12. [Choosing an Approach](#choosing-an-approach)
13. [Interview Q&A](#interview-qa)
14. [Common Pitfalls](#common-pitfalls)
15. [Related Topics](#related-topics)

---

## Why This Still Matters

Three reasons interviewers ask about it:

1. **Baselines.** A TF-IDF + logistic regression baseline takes ten minutes and frequently gets within a few points of a fine-tuned transformer on domain classification. Skipping it makes any subsequent result uninterpretable.
2. **Tokenization is upstream of everything.** Vocabulary decisions determine what the model can represent, how much your prompt costs, and why some languages are 3× more expensive per character.
3. **Small data.** With 500 labeled examples, a transformer overfits and a linear model on n-grams doesn't.

---

## Text Preprocessing

The classical pipeline, and knowing when *not* to apply each step:

| Step | What it does | Skip it when |
|---|---|---|
| **Lowercasing** | `Apple` → `apple` | Case is signal (NER: `Apple` vs `apple`) |
| **Punctuation removal** | Strips `!?,.` | Sentiment (`!!!` matters), code |
| **Stopword removal** | Drops `the, is, at` | Using transformers; phrase matching ("to be or not to be") |
| **Stemming** | `running` → `run` (crude, rule-based) | Precision matters: it produces non-words |
| **Lemmatization** | `better` → `good` (dictionary-based) | Speed matters: it's slower than stemming |
| **Normalization** | Unicode, accents, whitespace | Rarely: nearly always worth doing |

**Stemming vs lemmatization** is a standard question: stemming chops suffixes with rules (fast, crude, `studies → studi`), lemmatization maps to a dictionary base form using part-of-speech (slower, correct, `studies → study`, `better → good`). Use lemmatization when the output is read by humans or precision matters; stemming when you're building a search index and speed dominates.

**With modern transformers you generally skip all of it.** Subword tokenizers were trained on raw text, so lowercasing or stripping punctuation moves your input off the distribution the model saw. Aggressive preprocessing actively *hurts* BERT-family models: a good thing to state, since candidates often apply 2010-era preprocessing to 2020s models out of habit.

---

## Tokenization

How text becomes model inputs, and the source of many practical surprises.

| Level | Example | Vocabulary | Problem |
|---|---|---|---|
| **Character** | `c,a,t` | ~100 | Sequences far too long |
| **Word** | `cat` | 100k+ | Out-of-vocabulary words; huge embedding table |
| **Subword** | `un,happi,ness` | 30–100k | **The standard**: best of both |

**Subword tokenization** solves the out-of-vocabulary problem: any unseen word decomposes into known pieces, so nothing is ever `<UNK>`, while common words stay single tokens.

| Algorithm | How | Used by |
|---|---|---|
| **BPE** | Iteratively merge the most frequent adjacent pair | GPT family |
| **WordPiece** | Merge the pair that most increases likelihood | BERT |
| **Unigram / SentencePiece** | Start large, prune tokens by loss impact | T5, Llama |

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("bert-base-uncased")

tok.tokenize("tokenization")     # ['token', '##ization']: ## marks continuation
tok.tokenize("antidisestablish") # splits into several known subwords
```

**Practical consequences worth naming in an interview:**

- **Token count ≠ word count.** English averages ~1.3 tokens per word; code, JSON, and rare identifiers are far denser. Estimating cost or context limits from character counts is unreliable: use the actual tokenizer.
- **Non-English text costs more.** Tokenizers trained mostly on English fragment other scripts heavily, so the same sentence in Thai or Hindi can cost several times more tokens. This is a real fairness and cost issue.
- **Numbers tokenize badly.** `1234` may split into `12` + `34`, which is part of why LLMs are unreliable at arithmetic.
- **Trailing whitespace changes tokenization**, which is why prompt formatting sometimes has surprising effects.

---

## Bag of Words and TF-IDF

Represent a document as a vector of term counts, discarding order.

**TF-IDF** weights terms by how often they appear in this document against how rare they are across the corpus:

```
tf-idf(t, d) = tf(t, d) · log(N / df(t))
```

The intuition: a term appearing often in *this* document but rarely elsewhere is distinctive. Terms appearing everywhere ("the") get near-zero weight, which is why stopword removal is largely redundant when using TF-IDF.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    TfidfVectorizer(
        ngram_range=(1, 2),     # unigrams + bigrams recovers some word order
        min_df=2,               # ignore terms in fewer than 2 docs (noise)
        max_df=0.9,             # ignore terms in >90% of docs (uninformative)
        sublinear_tf=True,      # 1+log(tf): dampens repeated-term dominance
    ),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)
model.fit(train_texts, train_labels)
```

**This is the baseline to beat**, and it's strong on domain-specific classification with limited data. It's also fully interpretable: you can read the coefficients and see which terms drive each class, which matters for stakeholder trust and debugging.

Its limits: no word order beyond n-grams, no synonymy (`car` and `automobile` are unrelated dimensions), and high-dimensional sparse vectors.

---

## Word Embeddings

**Word2Vec** learns dense vectors by predicting context. Two variants: **skip-gram** (predict context from the word (better for rare words) and **CBOW** (predict the word from context) faster). Negative sampling makes it tractable by replacing the full softmax with a handful of binary decisions.

The famous property: `king - man + woman ≈ queen`. Vector arithmetic captures relational structure because the training objective places words in similar contexts near each other.

**GloVe** factorizes a global co-occurrence matrix instead of using local windows: different route, comparable result.

**FastText** represents words as bags of character n-grams, so it handles morphology and can embed **unseen words** by composing their n-grams. That makes it strong for morphologically rich languages and noisy user-generated text with typos.

**The fatal limitation of all three**: one vector per word, regardless of context. "River **bank**" and "investment **bank**" share an embedding, so the representation is an average of all senses. Fixing that is exactly what contextual models did.

---

## From Static to Contextual

```
Word2Vec (2013)  →  one vector per word, no context
ELMo    (2018)   →  bidirectional LSTM, context-dependent vectors
BERT    (2018)   →  transformer, deeply bidirectional, masked LM pretraining
Sentence-BERT    →  fine-tuned so vector distance means semantic similarity
```

**Why raw BERT embeddings are bad for similarity**: a favourite interview question: BERT is trained on masked-token prediction, not on making sentence vectors comparable. Mean-pooled BERT vectors occupy a narrow cone where almost every pair scores above 0.8 cosine, so the numbers barely discriminate. Sentence-BERT fixes it by fine-tuning with a contrastive or triplet objective so distance actually corresponds to semantic difference. See [Embeddings](../ai_genai/intro_embeddings.md).

---

## Text Classification

| Approach | Data needed | When |
|---|---|---|
| **TF-IDF + linear model** | 100s | Baseline; small data; interpretability required |
| **FastText classifier** | 1000s | Fast, strong, handles typos |
| **Fine-tuned transformer** | 1000s–10,000s | Best accuracy when data supports it |
| **Zero/few-shot LLM** | **0** | No labels; rapid prototyping; rare classes |
| **Embedding + classifier** | 100s | Sentence embeddings + logistic regression: strong and cheap |

The last row is underrated and worth mentioning: encode with a good sentence embedding model, train logistic regression on the vectors. It needs no fine-tuning, trains in seconds, handles small data well, and often lands close to a fine-tuned transformer.

**Class imbalance** is the norm in real text data. Use `class_weight="balanced"`, evaluate with macro-F1 or PR-AUC rather than accuracy, and be careful that rare classes have enough examples to be learnable at all.

---

## Named Entity Recognition

Sequence labeling (one tag per token) using the **BIO scheme**:

```
Tim    Cook   visited  Apple  Park   in  Cupertino
B-PER  I-PER  O        B-ORG  I-ORG  O   B-LOC
```

`B-` begins an entity, `I-` continues it, `O` is outside. The scheme exists so adjacent entities of the same type stay separable.

**Evaluation must be entity-level, not token-level.** Getting 3 of 4 tokens right in "Tim Cook Jr" is not 75% correct: the entity is wrong, full stop. Token-level F1 systematically overstates performance, and using it is a tell that someone hasn't shipped an NER system.

Approaches, in historical order: CRF over hand-crafted features (still competitive on small data, and it enforces valid tag sequences), BiLSTM-CRF, and fine-tuned transformers with a token-classification head (the current default). For zero-shot or rare entity types, LLMs with structured output work well.

---

## Topic Modeling

**LDA** models each document as a mixture of topics and each topic as a distribution over words. Unsupervised, and requires choosing the number of topics, usually by coherence score rather than perplexity, since perplexity correlates poorly with human judgments of topic quality.

**Modern alternative**: embed documents, cluster (HDBSCAN), then label each cluster with its distinctive terms, which is essentially what BERTopic does. It handles short text far better than LDA, which struggles when documents are tweets rather than articles.

Topic models are exploratory tools. Topics are interpretations, not ground truth, and they shift with hyperparameters: worth saying rather than presenting them as discovered facts.

---

## Text Similarity

| Measure | Type | Use |
|---|---|---|
| **Jaccard** | Set overlap | Deduplication, shingling |
| **Edit (Levenshtein)** | Character | Typo correction, fuzzy name matching |
| **Cosine on TF-IDF** | Lexical | Document similarity with shared vocabulary |
| **Cosine on embeddings** | Semantic | Paraphrase, cross-vocabulary matching |
| **Cross-encoder** | Semantic, joint | Highest accuracy, too slow for retrieval |

**MinHash + LSH** is the scalable deduplication answer: exact pairwise Jaccard over `n` documents is `O(n²)`, which is infeasible at corpus scale. MinHash approximates Jaccard with small signatures, and LSH buckets likely-similar pairs so you only compare within buckets. This is how large corpora get deduplicated before pretraining, and naming it signals real-world scale experience.

---

## Evaluation

| Task | Metric | Note |
|---|---|---|
| Classification | Macro-F1, PR-AUC | Accuracy misleads on imbalanced text |
| NER | **Entity-level** F1 | Token-level overstates |
| Retrieval | Recall@k, NDCG, MRR | Recall is the ceiling downstream |
| Summarization | ROUGE + human | ROUGE measures overlap, not quality |
| Translation | BLEU, COMET + human | BLEU correlates weakly with fluency |
| Generation | Task success, LLM judge + human | No single automatic metric suffices |

**ROUGE and BLEU measure n-gram overlap with a reference**, which means a correct paraphrase scores poorly and a fluent-but-wrong output can score well. They're useful for tracking regressions, not for deciding whether output is good. Say that when asked how to evaluate summarization: the expected answer includes human evaluation or a validated LLM judge.

---

## Choosing an Approach

| Situation | Approach |
|---|---|
| Any text task, first hour | **TF-IDF + linear model** as a baseline |
| < 1,000 labels | Sentence embeddings + logistic regression |
| Plenty of labels, accuracy matters | Fine-tuned transformer |
| No labels at all | Zero/few-shot LLM, or embedding clustering |
| Interpretability required | TF-IDF + linear (readable coefficients) |
| Noisy user text, typos | FastText (character n-grams) |
| Millions of docs, need speed | FastText or a linear model on hashed features |
| Semantic search | Bi-encoder embeddings + reranker |
| Deduplication at scale | MinHash + LSH |

---

## Interview Q&A

#### Why start with TF-IDF when transformers exist?

Because it's ten minutes of work and it makes every later result interpretable. Without a baseline, "my model gets 0.89 F1" means nothing: 0.89 could be worse than a linear model on n-grams, and you'd never know.

It's also competitive in the situations that come up most: small labeled datasets where a transformer overfits, domain-specific vocabulary where pretrained semantics don't transfer, and tasks driven by distinctive keywords rather than nuanced meaning. Plus it trains in seconds, runs anywhere, and its coefficients are directly readable, which matters when a stakeholder asks why a document was classified a certain way.

The framing I'd use: TF-IDF isn't the answer, it's the number the answer has to beat by enough to justify the added complexity in training, serving, and monitoring.

#### Explain subword tokenization and why it replaced word-level.

Word-level tokenization has two fatal problems: any word not in the vocabulary becomes `<UNK>`, destroying information, and covering enough words requires an enormous embedding table.

Subword tokenization splits rare words into known pieces while keeping common words whole. `tokenization` becomes `token` + `##ization`. Nothing is ever out-of-vocabulary because worst case you fall back to characters, morphology is partially captured for free, and vocabulary stays around 30–100k.

BPE merges the most frequent adjacent pair iteratively; WordPiece merges the pair that most improves likelihood; Unigram starts large and prunes by loss impact. The practical consequences are worth knowing: token count doesn't track word count, non-English text costs substantially more tokens because tokenizers are English-dominant, numbers split awkwardly (part of why LLMs struggle with arithmetic), and trailing whitespace changes the tokenization.

#### Why are Word2Vec embeddings insufficient, and what replaced them?

One vector per word, fixed regardless of context. "River bank" and "investment bank" get the identical vector, so it's an average over all senses of the word: the representation is systematically wrong for every polysemous word.

Contextual models fixed this: ELMo with bidirectional LSTMs, then BERT with transformers, produce a *different* vector for each occurrence based on surrounding text. Same word, different sentence, different embedding.

One caveat worth adding: raw BERT embeddings are poor for *similarity* comparisons despite being contextual, because BERT is trained on masked-token prediction rather than on making vectors comparable: mean-pooled BERT vectors sit in a narrow cone where nearly everything scores above 0.8 cosine. Sentence-BERT fine-tunes with a contrastive objective so distance actually means something.

#### How do you evaluate a NER model?

**Entity-level F1**, not token-level. Predicting 3 of the 4 tokens in an entity span isn't partial credit: the extracted entity is wrong and downstream systems get bad data. Token-level metrics systematically overstate performance, sometimes dramatically for long entities.

Concretely, a predicted entity counts as correct only if both its span boundaries and its type match the gold annotation exactly. `seqeval` implements this correctly; hand-rolled token accuracy generally doesn't.

I'd also report per-entity-type F1, since aggregate F1 hides that `PERSON` works well and `PRODUCT` doesn't, and error-analyze the boundary cases specifically, because boundary errors and type errors have different fixes.

#### You have 300 labeled examples for a text classification task. What do you do?

Not fine-tune a transformer: it overfits at that size and the result won't be trustworthy.

I'd try three things and compare with proper cross-validation, since with 300 examples a single split is far too noisy: **TF-IDF plus regularized logistic regression** as the baseline; **sentence embeddings plus logistic regression**, which is usually the strongest option at this scale because the embedding model brings pretrained semantics and only the classifier needs fitting; and a **zero/few-shot LLM**, which needs no labels at all and gives an immediate reference point.

Alongside that I'd think about getting more data cheaply: LLM-assisted labeling with human verification, or active learning to prioritize which examples to label next. And I'd report results as mean ± std across folds, because at 300 examples the confidence interval is wide and a single number would be misleading.

#### How would you deduplicate 100 million documents?

Exact pairwise comparison is `O(n²)`: 10¹⁶ comparisons, which is impossible.

**MinHash plus LSH.** MinHash produces a compact signature such that the probability two signatures agree equals their Jaccard similarity, so similarity is estimable from small fixed-size sketches. LSH then hashes signatures into buckets so that similar documents collide, and you only compare within buckets, turning it into roughly linear work.

The parameters (number of bands and rows) tune the similarity threshold, trading false positives against false negatives, so you set them from what "duplicate" means for your use case.

For exact duplicates, a content hash is far simpler and should be tried first. And for *semantic* near-duplicates (same meaning, different words) MinHash won't catch them; that needs embeddings plus ANN search, which is more expensive but a different notion of duplicate.

#### When is preprocessing like stopword removal and stemming harmful?

Whenever you're using a pretrained transformer. Those models were trained on raw natural text with its punctuation, casing, and function words intact, so stripping them moves your input off the training distribution and degrades performance. Their tokenizers already handle morphology through subwords, so stemming is redundant and lossy.

It's also harmful in specific classical settings: stopwords carry signal in sentiment ("not good") and in phrase matching, casing carries signal in NER, and punctuation carries signal in sentiment and code.

The general rule I'd give: preprocess to match how the representation was built. TF-IDF benefits from normalization because it treats tokens as independent symbols; transformers want raw text. Applying 2010-era preprocessing to a modern model is a common and costly habit.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| No baseline before a transformer | Results become uninterpretable | TF-IDF + linear model first |
| Heavy preprocessing before BERT | Moves input off the pretraining distribution | Feed raw text |
| Fitting the vectorizer before splitting | IDF computed on test data leaks | Fit inside a `Pipeline` on train only |
| Token-level NER metrics | Overstates real performance | Entity-level F1 (`seqeval`) |
| Accuracy on imbalanced text | Majority class dominates | Macro-F1, PR-AUC |
| Estimating tokens from character count | Wrong for code, JSON, non-English | Use the actual tokenizer |
| Assuming equal token cost across languages | Non-English can cost several times more | Measure per language |
| Raw BERT vectors for similarity | Anisotropic; nearly everything scores high | Sentence-BERT or a trained embedding model |
| ROUGE/BLEU as the sole quality metric | Measures overlap, not correctness or fluency | Add human eval or a validated judge |
| Single train/test split on small data | Estimate is dominated by split noise | Cross-validate; report mean ± std |
| Treating LDA topics as ground truth | Topics shift with hyperparameters | Treat as exploratory; check coherence |
| `O(n²)` deduplication | Infeasible past ~100k documents | MinHash + LSH |

---

## Related Topics

- [Embeddings](../ai_genai/intro_embeddings.md)
- [LLM Fundamentals](../ai_genai/intro_llm_fundamentals.md)
- [Transformers](../deep_learning/intro_transformers.md)
- [Sequence Models](../deep_learning/intro_sequence_models.md)
- [Feature Engineering](./intro_feature_engineering.md)
- [Model Evaluation and Metrics](./intro_model_evaluation.md)
- [Search and Ranking System Design](../system_design/search_ranking_system.md)
- [HuggingFace](../frameworks/intro_huggingface.md)
- [Classical ML Overview](./README.md)
