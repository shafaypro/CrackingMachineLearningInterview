# Preparation Resources and References

## Preparation Resources
1. [All of Statistics: A Concise Course in Statistical Inference](https://amzn.to/3r87WGa) by Larry Wasserman
2. [Machine Learning](https://amzn.to/3RdiFK3) by Tom Mitchell
3. [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://amzn.to/3LiVgD2) by Chip Huyen
4. [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) by Aurelien Geron
5. [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html) by Kevin P. Murphy
6. [Deep Learning](https://www.deeplearningbook.org/) by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
7. [Generative Deep Learning](https://www.oreilly.com/library/view/generative-deep-learning/9781098134174/) by David Foster
8. [Reliable Machine Learning](https://www.oreilly.com/library/view/reliable-machine-learning/9781098106218/) by Cathy Chen, Niall Richard Murphy, Kranti Parisa, and D. Sculley

## Modern Interview Reference Links
These references are especially useful for 2025-2026 style interviews focused on LLM applications and production AI systems:

1. [OpenAI function calling guide](https://platform.openai.com/docs/guides/function-calling)
2. [OpenAI embeddings guide](https://platform.openai.com/docs/guides/embeddings)
3. [OpenAI Cookbook](https://cookbook.openai.com/)
4. [Anthropic: Building Effective AI Agents](https://resources.anthropic.com/ty-building-effective-ai-agents)
5. [Full Stack Deep Learning](https://fullstackdeeplearning.com/)
6. [Made With ML](https://madewithml.com/)
7. [Chip Huyen blog](https://huyenchip.com/blog/)

## Additional Topics

#### How would you define GAN (Generative Adversarial Networks)?
GANs are generative models composed of two networks trained together:
* A generator that tries to create realistic synthetic samples.
* A discriminator that tries to distinguish real samples from generated samples.

The generator improves by learning to fool the discriminator, and the discriminator improves by learning to detect generated outputs. GANs are useful for image synthesis, super-resolution, data augmentation, style transfer, and simulation.

#### What are Gaussian Processes?
Gaussian Processes are non-parametric probabilistic models often used for regression and uncertainty estimation. Instead of learning a single fixed function, they define a distribution over possible functions. They are especially useful when:
* data is relatively small,
* uncertainty estimates matter,
* smoothness assumptions can be encoded through kernels.

A key limitation is that they become computationally expensive as the dataset grows.

#### What is a Graph Neural Network?
A Graph Neural Network (GNN) is a neural network designed for graph-structured data, where objects are represented as nodes and relationships are represented as edges. GNNs learn by aggregating information from neighboring nodes. Common use cases include fraud detection, recommender systems, molecule modeling, knowledge graphs, and social-network analysis.

#### What is Language Modeling (LM)? Give examples.
Language modeling is the task of learning the probability of token sequences in natural language. In simple terms, a language model predicts the next token or estimates how likely a sequence is. Examples include:
* next-word prediction in mobile keyboards,
* text generation assistants,
* summarization systems,
* machine translation,
* code completion models,
* question-answering systems built on top of LLMs.

#### Define Named Entity Recognition. Give some use cases where it can come in handy.
Named Entity Recognition (NER) is an NLP task where a model identifies and classifies entities in text, such as person names, organizations, locations, dates, products, or medical terms. Common use cases include:
* extracting companies and skills from resumes,
* identifying diseases and drugs in clinical notes,
* tagging organizations and locations in news articles,
* compliance and document review,
* search enrichment and knowledge graph construction.
