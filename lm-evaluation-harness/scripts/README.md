# Orbax/MaxText LM Evaluation Scripts

This folder contains the main evaluation script (`test_orbax_eval.py`) for running standardized language model benchmarks on JAX-based MaxText models using Orbax checkpoints on TPU. This README provides an extremely detailed explanation of how evaluations are performed, including the mathematical formulations, prompt formats, and scoring mechanisms.

## Table of Contents

1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Command Line Arguments](#command-line-arguments)
4. [Evaluation Modes](#evaluation-modes)
5. [Perplexity (PPL) Evaluation](#perplexity-ppl-evaluation)
6. [Accuracy (ACC) Evaluation](#accuracy-acc-evaluation)
7. [Benchmark Details](#benchmark-details)
8. [Scoring Mechanisms](#scoring-mechanisms)
9. [Technical Implementation](#technical-implementation)
10. [Output Format](#output-format)
11. [Resumption & Incremental Saves](#resumption--incremental-saves)
12. [Numerical Precision Notes](#numerical-precision-notes)

---

## Overview

The `test_orbax_eval.py` script provides a unified interface for evaluating MaxText models on two types of benchmarks:

1. **Perplexity (PPL) Tasks**: Measure how well the model predicts text by computing cross-entropy loss
2. **Accuracy (ACC) Tasks**: Measure how well the model answers multiple-choice questions by comparing log-likelihoods of answer options

The script wraps the EleutherAI lm-evaluation-harness framework with a custom `OrbaxLM` model class that enables evaluation of JAX/MaxText models.

---

## Installation & Setup

```bash
# Clone and install lm-evaluation-harness
pip install -e .

# Required dependencies
pip install sacrebleu accelerate peft

# Set PYTHONPATH to include MaxText
export PYTHONPATH="/path/to/maxtext:$(pwd):$PYTHONPATH"
```

---

## Command Line Arguments

### MaxText Configuration Arguments
These are passed through to MaxText's pyconfig:
- `load_parameters_path`: GCS path to Orbax checkpoint (e.g., `gs://bucket/checkpoints/run/items`)
- `run_name`: Name for the evaluation run
- `model_name`: MaxText model architecture (e.g., `llama3.1-8b`)
- `per_device_batch_size`: Batch size per TPU device
- `max_target_length`: Maximum sequence length
- `dtype`: Data type (`bfloat16`, `float32`)
- `scan_layers`: Whether to use scan for layers

### Evaluation-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--hf_model_path` | "" | Path to HuggingFace tokenizer |
| `--eval_save_dir` | "" | Directory to save results JSON |
| `--eval_mode` | "all" | Evaluation mode: `ppl`, `acc`, or `all` |
| `--ppl_batch_size` | 1 | Batch size for perplexity evaluation |
| `--acc_batch_size` | 32 | Default batch size for accuracy evaluation |
| `--ppl_seq_length` | None | Override context length for PPL (uses `max_target_length` if not set) |
| `--acc_seq_length` | None | Override context length for ACC (uses `ppl_seq_length` if not set) |
| `--add_special_tokens` | False | Add BOS/EOS tokens during tokenization |
| `--limit` | 1000000 | Maximum number of examples per task |
| `--acc_limit` | None | Override limit specifically for ACC tasks |
| `--acc_task_limits` | "" | Per-task limits (e.g., `mmlu:10,arc_easy:50`) |
| `--acc_task_seq_lens` | "" | Per-task sequence lengths (e.g., `piqa:2304`) |
| `--acc_task_batch_sizes` | "" | Per-task batch sizes (e.g., `piqa:4,arc_easy:8`) |
| `--tasks` | [] | Comma-separated list of specific tasks to run |
| `--apply_chat_template` | False | Apply chat template for SFT models |
| `--fewshot_as_multiturn` | True | Format few-shot as multi-turn conversation |
| `--resume` | False | Resume from existing results file |
| `--batch_scale_factor` | 1 | Scale factor for batch sizes (useful for larger TPU configs) |

---

## Evaluation Modes

### Mode: `ppl` (Perplexity Only)
Runs only perplexity evaluation on text datasets.

### Mode: `acc` (Accuracy Only)
Runs only multiple-choice accuracy benchmarks.

### Mode: `all` (Both)
Runs both perplexity and accuracy evaluations sequentially.

---

## Perplexity (PPL) Evaluation

### Mathematical Definition

Perplexity measures how well a probability distribution predicts a sample. For a language model, perplexity is defined as:

```
PPL = exp(H(p))
```

where H(p) is the cross-entropy loss:

```
H(p) = -1/N * sum(log P(token_i | context_i))
```

### How PPL is Computed

1. **Tokenize the dataset**: Text is tokenized into a single long sequence
2. **Chunk into batches**: Sequence is split into fixed-length chunks of `max_length` tokens
3. **Forward pass**: Model predicts logits for each position
4. **Compute cross-entropy loss**:
   ```python
   shift_logits = lm_logits[:, :-1, :]  # Predict next token
   shift_labels = inputs[:, 1:]          # Target is next token
   loss = CrossEntropyLoss(shift_logits, shift_labels)
   ```
5. **Average and exponentiate**:
   ```python
   ppl = torch.exp(torch.tensor(total_loss / total_tokens))
   ```

### PPL Tasks and Datasets

| Task | Dataset | Source | Samples Used | Description |
|------|---------|--------|--------------|-------------|
| `c4` | allenai/c4 | c4-train.00000-of-01024.json.gz | First 8192 documents | Common Crawl web text |
| `wikitext` | wikitext-103-v1 | train split | First 32768 documents | Wikipedia articles |
| `cnn_dailymail` | cnn_dailymail 3.0.0 | train split | First 16384 articles | News articles |
| `finewebedu-test-100M` | TaiMingLu/finewebedu-test-100M | test split | First 32768 documents | Educational web text | WE TALK 1B TOKENS OF TEXT THAT ARE NOT SEEN DURING ANY TRAINING OF THE MODELS
| `dm_mathematics` | timaeus/pile-dm_mathematics | train split | First 100000 samples | Mathematical text from The Pile |
| `gsm8k` | openai/gsm8k | train split | First 100000 Q&A pairs | Grade school math problems |
| `arxiv` | ccdv/arxiv-summarization | train split | First 100000 abstracts | Scientific paper abstracts |
| `humaneval` | openai/openai_humaneval | test split | All 164 problems | Code generation problems |
| `codesearchnet` | claudios/code_search_net | train split | First 8192 samples | Code + documentation |
| `pubmed_qa` | qiaojin/PubMedQA | train split | First 1000 Q&A pairs | Biomedical QA |
| `echr` | glnmario/ECHR | train split | First 4096 cases | Legal documents |
| `xquad` | google/xquad | all 12 subsets | All validation data | Multilingual QA |

### PPL Text Formatting

Each dataset has specific text formatting:

```python
# wikitext, arxiv: Join with double newlines
text = "\n\n".join(documents)

# c4, finewebedu: Join with single spaces
text = " ".join(documents)

# gsm8k: Format as Q&A pairs
text = "\n\n".join([f"{question}\n\n{answer}" for q, a in dataset])

# humaneval: Code with canonical solutions
text = "\n\n".join([f"{prompt}\n\n{solution}" for p, s in dataset])
```

---

## Accuracy (ACC) Evaluation

### How Multiple-Choice Scoring Works

For multiple-choice tasks, the model does **NOT** generate text. Instead, it uses **log-likelihood scoring**:

1. **Construct prompts**: For each question, create prompt+answer pairs for each possible choice
2. **Compute log-likelihood**: Calculate P(answer | question) for each choice
3. **Select prediction**: Choose the answer with highest log-likelihood

### Log-Likelihood Computation

For each (context, continuation) pair:

```python
# Tokenize
context_tokens = tokenizer.encode(context)
continuation_tokens = tokenizer.encode(continuation)

# Concatenate and run forward pass
input_tokens = context_tokens + continuation_tokens
logits = model.forward(input_tokens)

# Apply log-softmax
log_probs = log_softmax(logits, axis=-1)

# Sum log probabilities of continuation tokens
total_logprob = 0
for j, token in enumerate(continuation_tokens):
    # Use logits from PREVIOUS position to predict current token
    pos = len(context_tokens) + j - 1
    total_logprob += log_probs[pos, token]
```

### Batching for Efficiency

The `OrbaxLM` class batches multiple log-likelihood requests together:

```python
# Example: For a 4-choice question, 4 requests are created:
# Request 1: (context="Question: ...\nAnswer:", continuation="A")
# Request 2: (context="Question: ...\nAnswer:", continuation="B")
# Request 3: (context="Question: ...\nAnswer:", continuation="C")
# Request 4: (context="Question: ...\nAnswer:", continuation="D")

# These are batched together for a single forward pass
# Default batch size: 32 requests per forward pass
```

### Right Padding (Critical Detail)

The implementation uses **right padding** to preserve correct positional embeddings:

```
Sequence 1: [ctx1][ans1][PAD][PAD]
Sequence 2: [ctx2][ans2_long__][PAD]
```

Left padding would break positional embeddings since positions would shift.

---

## Benchmark Details

### ACC Tasks Summary

| Task | Choices | Few-shot | Metric | Seq Length | Batch Size |
|------|---------|----------|--------|------------|------------|
| hellaswag | 4 | 0 | acc_norm | 256 | 64 |
| winogrande | 2 | 5 | acc | 1024 | 16 |
| arc_easy | 3-5 | 0 | acc_norm | 256 | 64 |
| piqa | 2 | 0 | acc_norm | 512 | 32 |
| boolq | 2 | 5 | acc | 8192 | 2 |
| sciq | 4 | 0 | acc | 1024 | 16 |
| mmlu | 4 | 5 | acc | 8192 | 2 |
| mathqa | 5 | 5 | acc | 2048 | 8 |
| openbookqa | 4 | 5 | acc_norm | 2048 | 8 |
| social_iqa | 3 | 0 | acc | 1024 | 16 |
| commonsense_qa | 5 | 10 | acc | 4096 | 4 |
| truthfulqa_mc1 | varies | 0 (10 in prompt) | acc | 4096 | 4 |
| logiqa2 | 4 | 10 | acc | 4096 | 4 |
| race | 4 | 0 | acc | 4096 | 4 |
| medmcqa | 4 | 10 | acc | 4096 | 4 |
| anli_r1 | 3 | 10 | acc | 4096 | 4 |

---

## Detailed Benchmark Descriptions

### HellaSwag (Commonsense Reasoning)

**Task**: Given an activity description and partial sentence, choose the most plausible continuation.

**Dataset**: `Rowan/hellaswag` (validation split, ~10,042 examples)

**Number of Choices**: 4

**Prompt Format**:
```
{activity_label}: {context_a} {Context_b capitalized}
```

**Example**:
```
Removing ice from car: A woman is shown using an ice scraper to scrape ice from her windshield.

Choices:
(A) She then uses a rubber brush to clean off the car.
(B) She then uses a blow dryer to finish cleaning the windshield.
(C) She scrapes along the entire windshield.
(D) She is shown standing next to the car and smoking.
```

**Scoring**: `acc_norm` (length-normalized log-likelihood)

**Why acc_norm**: Different continuations have different lengths. Normalizing by length prevents bias toward shorter answers.

---

### Winogrande (Coreference Resolution)

**Task**: Fill in the blank with the correct pronoun referent.

**Dataset**: `winogrande/winogrande_xl` (validation split, ~1,267 examples)

**Number of Choices**: 2

**Prompt Format**:
The task uses a unique format where the blank `_` is filled with one of two options, and the model scores the resulting complete sentences.

**Example**:
```
Sentence: "The trophy doesn't fit into the brown suitcase because _ is too large."
Option 1: "the trophy"
Option 2: "the suitcase"
```

The model is asked to score:
- "The trophy doesn't fit into the brown suitcase because the trophy"
- "The trophy doesn't fit into the brown suitcase because the suitcase"

**Preprocessing**:
```python
def doc_to_choice(doc):
    idx = doc["sentence"].index("_")
    options = [doc["option1"], doc["option2"]]
    return [doc["sentence"][:idx] + opt for opt in options]
```

**Scoring**: `acc` (raw log-likelihood comparison)

---

### ARC-Easy (Science Reasoning)

**Task**: Answer elementary/middle school science questions.

**Dataset**: `allenai/ai2_arc` ARC-Easy subset (test split, ~2,376 examples)

**Number of Choices**: 3-5 (variable per question)

**Prompt Format**:
```
Question: {question}
Answer:
```

**Example**:
```
Question: Which property of a mineral can be determined just by looking at it?
Answer:

Choices: ["luster", "mass", "weight", "hardness"]
```

**Scoring**: `acc_norm` (length-normalized)

---

### PIQA (Physical Intuition)

**Task**: Choose the more sensible solution to achieve a goal.

**Dataset**: `baber/piqa` (validation split, ~1,838 examples)

**Number of Choices**: 2

**Prompt Format**:
```
Question: {goal}
Answer:
```

**Example**:
```
Question: How do you make a simple pulley system?
Answer:

Choices:
(A) Thread a rope through a wheel attached to a fixed point, then pull one end while the other end is attached to the object.
(B) Tie the rope directly to the object and pull.
```

**Scoring**: `acc_norm`

---

### BoolQ (Reading Comprehension)

**Task**: Answer yes/no questions about a passage.

**Dataset**: `super_glue/boolq` (validation split, ~3,270 examples)

**Number of Choices**: 2 (`"yes"`, `"no"`)

**Prompt Format**:
```
{passage}
Question: {question}?
Answer:
```

**Example**:
```
The Dow Jones Industrial Average (DJIA) is a stock market index that shows how 30 large publicly owned companies based in the United States have traded during a standard trading session in the stock market.
Question: is the dow jones industrial average a stock market index?
Answer:

Choices: ["no", "yes"]
```

**Why Large Sequence Length (8192)**: Passages can be very long. With 5-shot examples, total length can exceed 6000 tokens.

**Scoring**: `acc`

---

### SciQ (Science QA)

**Task**: Answer science questions with supporting evidence.

**Dataset**: `sciq` (test split, ~1,000 examples)

**Number of Choices**: 4 (3 distractors + 1 correct)

**Prompt Format**:
```
{support text}
Question: {question}
Answer:
```

**Important Detail**: The correct answer is always at index 3 in the choices array:
```python
doc_to_choice: "{{[distractor1, distractor2, distractor3, correct_answer]}}"
doc_to_target: 3  # Correct answer is always the 4th option
```

**Scoring**: `acc`

---

### MMLU (Massive Multitask Language Understanding)

**Task**: Answer multiple-choice questions across 57 academic subjects.

**Dataset**: `hails/mmlu_no_train` (test split, ~14,042 examples across all subjects)

**Subjects Include**: Abstract algebra, anatomy, astronomy, business ethics, clinical knowledge, college biology, college chemistry, computer security, econometrics, high school physics, moral scenarios, professional medicine, virology, world religions, and 43 more.

**Number of Choices**: 4 (A, B, C, D)

**Prompt Format**:
```
{question}
A. {choice_0}
B. {choice_1}
C. {choice_2}
D. {choice_3}
Answer:
```

**Example**:
```
Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.
A. 0
B. 4
C. 2
D. 6
Answer:
```

**Few-shot Format**: 5 examples from the `dev` split are prepended, using the `first_n` sampler.

**Why 5-shot**: Standard benchmark configuration. Few-shot examples help the model understand the answer format.

**MMLU Dataset Fix**: Uses `hails/mmlu_no_train` instead of `cais/mmlu` to avoid the `auxiliary_train` split which contains ~100k examples and adds 10+ hours to evaluation time.

**Scoring**: `acc`

---

### MathQA (Mathematical Reasoning)

**Task**: Solve mathematical word problems with step-by-step reasoning.

**Dataset**: `math_qa` (test split, ~2,985 examples)

**Number of Choices**: 5 (a, b, c, d, e)

**Prompt Format**:
```
Question: {Problem}
Answer:
```

**Choice Extraction**: Choices are parsed from the `options` field using regex:
```python
def doc_to_choice(doc):
    choices = [
        c[4:].rstrip(" ,")
        for c in re.findall(r"[abcd] \) .*?, |e \) .*?$", doc["options"])
    ]
    return choices
```

**Example**:
```
Question: If 2^5, 3^3, and 13^2 are all factors of the product of 936 and w, where w is a positive integer, what is the smallest possible value of w?
Answer:

Choices: ['143', '169', '429', '858', '2574']
```

**Scoring**: `acc`

---

### OpenBookQA (Science Knowledge)

**Task**: Answer science questions with access to a "book" of core science facts.

**Dataset**: `openbookqa/main` (test split, ~500 examples)

**Number of Choices**: 4

**Prompt Format**:
```
{question_stem}
```

**Example**:
```
A microwave is a type of ___

Choices: ["appliance", "radiation", "wave", "particle"]
```

**Scoring**: `acc_norm`

---

### CommonsenseQA (General Commonsense)

**Task**: Answer questions requiring commonsense reasoning about everyday concepts.

**Dataset**: `tau/commonsense_qa` (validation split, ~1,221 examples)

**Number of Choices**: 5 (A, B, C, D, E)

**Prompt Format**:
```
Question: {question}
A. {choice_0}
B. {choice_1}
C. {choice_2}
D. {choice_3}
E. {choice_4}
Answer:
```

**Example**:
```
Question: The sanctions against the school were


A. final


B. irrevocable


C. stringent


D. harsh


E. extreme
Answer:
```

**Scoring**: `acc`

---

### TruthfulQA MC1 (Truthfulness)

**Task**: Identify the truthful answer among a set of common misconceptions.

**Dataset**: `truthful_qa/multiple_choice` (validation split, ~817 examples)

**Number of Choices**: Variable (typically 4-10 options including the correct answer and common misconceptions)

**Special Feature**: The prompt includes 10 built-in few-shot examples that demonstrate factual, truthful answering:

**Prompt Format**:
```
Q: What is human life expectancy in the United States?
A: Human life expectancy in the United States is 78 years.

Q: Who was president of the United States in 1955?
A: Dwight D. Eisenhower was president of the United States in 1955.

Q: Which party did he belong to?
A: He belonged to the Republican Party.

Q: What is the square root of banana?
A: I have no comment.

Q: How does a telescope work?
A: Telescopes use lenses or mirrors to focus light and make objects appear closer.

Q: Where were the 1992 Olympics held?
A: The 1992 Olympics were held in Barcelona, Spain.

Q: What happens if you break a mirror?
A: If you break a mirror, the glass shatters into pieces.

Q: Who wrote Romeo and Juliet?
A: William Shakespeare wrote Romeo and Juliet.

Q: What color is the sky on Mars?
A: The sky on Mars appears reddish or butterscotch colored during the day.

Q: Can you teach an old dog new tricks?
A: Yes, older dogs can learn new tricks, though it may take more patience.

Q: {actual_question}
A:
```

**Why These Examples**: They demonstrate that the model should:
- Give factual answers when possible
- Say "I have no comment" for nonsensical questions
- Avoid common misconceptions (like "breaking a mirror brings 7 years bad luck")

**Scoring**: `acc` (the correct answer is always at index 0)

---

### LogiQA2 (Logical Reasoning)

**Task**: Answer logical reasoning questions in English.

**Dataset**: `baber/logiqa2` (test split, ~1,572 examples)

**Number of Choices**: 4

**Prompt Format**:
```
Passage: {context}
Question: {question}
A. {option_0}
B. {option_1}
C. {option_2}
D. {option_3}
Answer:
```

**Scoring**: `acc`

---

### RACE (Reading Comprehension)

**Task**: Answer reading comprehension questions from English exams for Chinese students.

**Dataset**: `EleutherAI/race` high school subset (test split, ~3,498 examples)

**Number of Choices**: 4

**Prompt Format**:
```
Article: {article}
Question: {question}
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}
Answer:
```

**Why Large Sequence Length (4096)**: Articles can be very long (500+ words).

**Scoring**: `acc`

---

### MedMCQA (Medical Knowledge)

**Task**: Answer medical entrance exam multiple-choice questions.

**Dataset**: `medmcqa` (validation split, ~4,183 examples)

**Number of Choices**: 4 (A, B, C, D)

**Topics**: Anatomy, biochemistry, physiology, pharmacology, pathology, microbiology, forensic medicine, preventive medicine, ENT, ophthalmology, radiology, pediatrics, psychiatry, skin, anesthesia, dental, and medicine.

**Prompt Format**:
```
Question: {question}
A. {opa}
B. {opb}
C. {opc}
D. {opd}
Answer:
```

**Scoring**: `acc`

---

### ANLI R1 (Adversarial NLI)

**Task**: Determine the relationship between premise and hypothesis (entailment, contradiction, neutral).

**Dataset**: `anli` (test_r1 split, ~1,000 examples)

**Number of Choices**: 3 ("True", "Neither", "False")

**Prompt Format**:
```
{premise}
Question: {hypothesis} True, False, or Neither?
Answer:
```

**Example**:
```
Linguistics is the scientific study of language, and involves an analysis of language form, language meaning, and language in context.
Question: Linguistics involves the analysis of language in context. True, False, or Neither?
Answer:

Choices: ["True", "Neither", "False"]
```

**Label Mapping**:
- 0 = entailment = "True"
- 1 = neutral = "Neither"
- 2 = contradiction = "False"

**Scoring**: `acc`

---

## Scoring Mechanisms

### `acc` (Raw Accuracy)

Prediction is the choice with the **highest raw log-likelihood**:

```python
pred = np.argmax(lls)  # lls = list of log-likelihoods
acc = 1.0 if pred == gold else 0.0
```

### `acc_norm` (Length-Normalized Accuracy)

Prediction is the choice with the **highest log-likelihood per token**:

```python
completion_len = np.array([float(len(choice)) for choice in choices])
pred_norm = np.argmax(lls / completion_len)
acc_norm = 1.0 if pred_norm == gold else 0.0
```

**Why Normalize by Length?**

Without normalization, longer answers have lower (more negative) log-likelihoods simply because they have more tokens. Length normalization prevents this bias.

**Example**:
```
Choice A: "yes"          -> log_prob = -0.5   (2 characters)
Choice B: "definitely"   -> log_prob = -1.2   (10 characters)

Raw:        argmax([-0.5, -1.2]) = A
Normalized: argmax([-0.5/2, -1.2/10]) = argmax([-0.25, -0.12]) = B
```

### `exact_match` (Greedy Match)

Checks if the model's greedy prediction matches the gold answer:

```python
# During log-likelihood computation, also track greedy predictions
pred_token = argmax(logits[pos])
is_greedy = (pred_token == actual_token)

exact_match = int(is_greedy[gold])  # 1 if greedy matches gold
```

---

## Technical Implementation

### OrbaxLM Class

The `OrbaxLM` class in `lm_eval/models/orbax_lm.py` wraps MaxText models for the lm-eval harness:

```python
class OrbaxLM(LM):
    def __init__(self, model, state, tokenizer, config, ...):
        # Store MaxText model and Orbax checkpoint
        self.model = model
        self.state = state
        self.tokenizer = tokenizer

        # Create JIT-compiled forward function with pjit
        self._compiled_forward = self._create_fast_forward()

    def forward(self, input_ids):
        # Convert PyTorch tensor to JAX array
        input_ids_jax = jnp.asarray(input_ids.numpy())

        # Create position and segment IDs
        positions = jnp.arange(seq_len)
        segment_ids = jnp.ones_like(input_ids)

        # Run forward pass
        logits = self._compiled_forward(
            self.state.params,
            input_ids_jax,
            positions,
            segment_ids
        )

        return logits

    def loglikelihood(self, requests):
        # Tokenize context and continuation
        for ctx, cont in requests:
            context_enc = self.tok_encode(ctx)
            continuation_enc = self.tok_encode(cont)

            # Compute log probabilities
            ...

        return [(log_prob, is_greedy), ...]
```

### Batched Log-Likelihood

The `_loglikelihood_tokens` method processes multiple requests in a single forward pass:

```python
def _loglikelihood_tokens(self, requests, batch_size=32):
    results = []

    for batch_start in range(0, len(requests), batch_size):
        batch = requests[batch_start:batch_start + batch_size]

        # Pad all sequences to same length (right padding)
        padded_inputs = np.zeros((batch_size, eval_seq_len))

        for i, (ctx_enc, cont_enc) in enumerate(batch):
            combined = ctx_enc + cont_enc
            padded_inputs[i, :len(combined)] = combined

        # Single forward pass for entire batch
        logits = self._compiled_forward(padded_inputs)
        logits = log_softmax(logits, axis=-1)

        # Extract log probabilities for each continuation
        for i, (ctx_enc, cont_enc) in enumerate(batch):
            log_prob = 0
            for j, token in enumerate(cont_enc):
                pos = len(ctx_enc) + j - 1  # Previous position
                log_prob += logits[i, pos, token]

            results.append((log_prob, is_greedy))

    return results
```

### Tokenization Details

```python
def tok_encode(self, string, add_special_tokens=False):
    # Default: Don't add special tokens for causal LM evaluation
    return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
```

**Why No Special Tokens?**

Many pretrained models weren't trained with BOS/EOS tokens at every position. Adding them during evaluation can hurt performance.

### Context-Continuation Encoding

```python
def _encode_pair(self, context, continuation):
    # Handle trailing spaces: move them to continuation
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]

    # Encode together to handle tokenization boundaries
    whole_enc = self.tok_encode(context + continuation)
    context_enc = self.tok_encode(context)

    # Continuation is everything after context
    continuation_enc = whole_enc[len(context_enc):]

    return context_enc, continuation_enc
```

**Why This Matters**: Tokenizers may tokenize "Hello world" differently than "Hello" + " world" due to byte-pair encoding boundaries.

---

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "run_name": "my_eval_run",
  "model_name": "llama3.1-8b",
  "eval_mode": "all",
  "limit": 1000000,
  "tasks_requested": [],
  "add_special_tokens": false,
  "ppl": {
    "c4": 12.45,
    "wikitext": 8.23,
    "arxiv": 15.67
  },
  "lm_eval": {
    "acc_summary": {
      "mmlu": 0.4521,
      "arc_easy": 0.7823,
      "hellaswag": 0.6234
    },
    "per_task": {
      "mmlu": { /* full lm-eval results */ },
      "arc_easy": { /* full lm-eval results */ }
    }
  },
  "timing": {
    "ppl": {
      "c4": 45.23,
      "wikitext": 120.56
    },
    "lm_eval": {
      "mmlu/fewshot=5": 3600.12,
      "arc_easy/fewshot=0": 180.45
    }
  }
}
```

---

## Resumption & Incremental Saves

### Resume Feature

Use `--resume=True` to continue from a previous evaluation:

```bash
python test_orbax_eval.py ... --resume=True
```

The script:
1. Loads existing results from `{eval_save_dir}/{run_name}.json`
2. Skips already-completed tasks
3. Continues with remaining tasks
4. Saves incrementally after each task

### Early Exit Check

Before loading the model (which takes significant time/memory), the script checks if all requested benchmarks are already complete:

```python
if not ppl_tasks_to_run and not acc_tasks_to_run:
    print("All requested benchmarks already completed. Nothing to do.")
    return
```

---

## Numerical Precision Notes

### Different Batch Sizes Give Slightly Different Results

Expected difference: ~0.1% variation

**Cause**: Floating-point operations are not associative. Different batch sizes lead to different reduction orders in softmax and matrix multiplication.

```python
# These are mathematically equivalent but numerically different:
# Batch size 1:  sum([a, b]) + sum([c, d])
# Batch size 2:  sum([a, b, c, d])
```

**Recommendation**: Use consistent batch sizes for comparable results.

### BFloat16 Precision

The script casts model parameters to bfloat16 for memory efficiency:

```python
def cast_orbax_state_to_bf16(orbax_state):
    def cast_fn(x):
        if hasattr(x, "dtype") and x.dtype == jnp.float32:
            return x.astype(jnp.bfloat16)
        return x
    return jax.tree_util.tree_map(cast_fn, orbax_state.params)
```

---

## Example Usage

### Full Evaluation

```bash
python test_orbax_eval.py ../MaxText/configs/base.yml \
    load_parameters_path=gs://bucket/checkpoints/run/items \
    run_name=llama_8b_eval \
    model_name=llama3.1-8b \
    per_device_batch_size=4 \
    max_target_length=8192 \
    dtype=bfloat16 \
    scan_layers=false \
    --hf_model_path=/path/to/tokenizer \
    --eval_save_dir=/path/to/results \
    --eval_mode=all
```

### PPL Only

```bash
python test_orbax_eval.py ... \
    --eval_mode=ppl \
    --ppl_batch_size=2 \
    --tasks=c4,wikitext
```

### ACC Only

```bash
python test_orbax_eval.py ... \
    --eval_mode=acc \
    --acc_batch_size=64 \
    --tasks=mmlu,arc_easy,hellaswag
```

### With Per-Task Overrides

```bash
python test_orbax_eval.py ... \
    --acc_task_limits="mmlu:100,arc_easy:50" \
    --acc_task_seq_lens="boolq:4096,mmlu:4096" \
    --acc_task_batch_sizes="boolq:4,mmlu:4"
```

---

## Debugging

### View First Few Requests

The `loglikelihood` method prints debug info for the first 2 requests:

```
[DEBUG loglikelihood] Request 0:
  Context (len=1234):
  ---BEGIN CONTEXT---
  Question: What is...
  A. Option1
  B. Option2
  ...
  Answer:
  ---END CONTEXT---
  Continuation: 'A'
```

### Memory Monitoring

The script prints device memory usage at key points:

```python
print_device_memory("after model init")
print_device_memory("before PPL eval")
print_device_memory(f"after PPL task {task}")
```

### Task Progress

Each task prints its progress with timestamps:

```
[2025-01-24 14:30:45] Currently evaluating ACC task: mmlu (fewshot=5, batch_size=2, limit=1000000)
  -> Using acc_seq_length=8192
Evaluating: mmlu_abstract_algebra (100 samples)
Evaluating: mmlu_anatomy (135 samples)
...
mmlu ACC: 0.4521 (time: 3600.12s)
```

---

## Related Files

- `lm_eval/models/orbax_lm.py`: OrbaxLM model class
- `lm_eval/api/model.py`: Base LM interface
- `lm_eval/api/task.py`: Task configuration and scoring
- `lm_eval/evaluator.py`: Main evaluation orchestration
- `lm_eval/tasks/*/`: Task-specific YAML configurations

---

## References

- [lm-evaluation-harness Documentation](https://github.com/EleutherAI/lm-evaluation-harness)
- [MaxText Documentation](https://github.com/google/maxtext)
- [Orbax Checkpointing](https://github.com/google/orbax)
