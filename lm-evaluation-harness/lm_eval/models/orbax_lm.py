import random

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from functools import partial
from transformers import AutoTokenizer

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.layers import models
from MaxText.layers import quantizations
from MaxText.utils.ckpt_conversion.utils.hf_utils import (
    convert_jax_weight_to_torch,
)
from MaxText import maxengine

from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.experimental.pjit import pjit
from flax.linen import partitioning as nn_partitioning


# =============================================================================
# SFT Chat Template (matches training format exactly)
# Format: NO system prompt, NO BOS token
# =============================================================================
SFT_TEMPLATE = {
    "user_start": "<|start_header_id|>user<|end_header_id|>\n\n",
    "user_end": "<|eot_id|>",
    "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "assistant_end": "<|eot_id|>",
}


def loss_fn(model, config, data, dropout_rng, params, is_train=True):
  """loss_fn for both train and eval.

  Args:
    model: A nn.Module
    config: Config of parameters
    data: Batch of data to apply to the model
    dropout_rng: A key to use to generate rng for dropout
    params: Model params
    is_train: True for train_step and False for eval_step

  Returns:
    loss: average loss
    aux: a dictionary including intermediate_outputs, total_loss, and total_weights
  """
  # inputs, targets, segments, positions = apply_args
  rng1, aqt_rng = jax.random.split(dropout_rng)

  # decimate proportion of data when per_device_batch_size<1
  if is_train:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_train_on, :]
  else:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_eval_on, :]
  mutable_collections = ["intermediates"]
  if config.mtp_num_layers > 0 and is_train:
    # The single model.apply call now triggers the entire chain if MTP is enabled:
    # Decoder runs -> returns hidden_state -> MTPBlock uses it -> MTPBlock sows losses -> we reap them here.
    mutable_collections.append("mtp_losses")

  # During evaluation, if the acceptance rate test is enabled, we must
  # make its specific collection mutable so the MTPBlock can sow into it.
  if config.mtp_eval_target_module > 0 and not is_train:
    mutable_collections.append("mtp_acceptance")

  logits, intermediate_outputs = model.apply(
      params,
      data["inputs"],
      data["inputs_position"],
      decoder_segment_ids=data["inputs_segmentation"],
      encoder_images=data["images"] if config.use_multimodal else None,
      enable_dropout=config.enable_dropout if is_train else False,
      rngs={"dropout": rng1, "params": aqt_rng},
      mutable=mutable_collections,
      decoder_target_tokens=data["targets"],
      decoder_target_mask=data["targets_segmentation"],
  )
  one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
  xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets, 0.0)
  xent = nn.with_logical_constraint(xent, ("activation_embed_and_logits_batch", "activation_length"))
  # Mask out paddings at the end of each example.
  xent = xent * (data["targets_segmentation"] != 0)
  total_loss = jnp.sum(xent)
  total_weights = jnp.sum(data["targets_segmentation"] != 0)
  loss = total_loss / (total_weights + EPS)

  # Calculate and Add MTP Loss
  mtp_loss = 0.0
  if config.mtp_num_layers > 0 and is_train:
    mtp_loss = calculate_mtp_loss(intermediate_outputs, config)
    loss += mtp_loss

  # get moe load balance loss
  moe_lb_loss = 0.0
  if config.num_experts > 1:
    nested_key = ("intermediates", "decoder", "layers", "moe_lb_loss")
    total_moe_lb_loss = maxtext_utils.get_nested_value(intermediate_outputs, nested_key, 0.0)
    moe_lb_loss = jnp.mean(jnp.array(total_moe_lb_loss))
    loss += moe_lb_loss

  # Add the model's primary output to the intermediates dict so it can be used
  # by the acceptance rate calculation in eval_step.
  intermediate_outputs["logits"] = logits

  aux = {
      "intermediate_outputs": intermediate_outputs,
      "total_loss": total_loss,
      "total_weights": total_weights,
      "moe_lb_loss": moe_lb_loss,
      "mtp_loss": mtp_loss,
  }
  return loss, aux

@partial(jax.jit, static_argnames=["model"])
def forward_jax(model, params, input_ids, positions, segment_ids):
    return model.apply(
        params,
        input_ids,
        positions,
        segment_ids,
        enable_dropout=False,
        rngs={"aqt": jax.random.PRNGKey(0)},
    )

@register_model("orbax_lm")
class OrbaxLM(LM):
    def __init__(
        self,
        model,
        state,
        tokenizer,
        config,
        state_mesh_shardings,
        mesh,
        loglikelihood_batch_size=32,
        max_loglikelihood_seq_length=None,
    ) -> None:
        super().__init__()
        self.model = model
        self.state = state
        self.tokenizer = tokenizer
        self.config = config
        self.state_mesh_shardings = state_mesh_shardings
        self.mesh = mesh
        self.loglikelihood_batch_size = loglikelihood_batch_size
        self._default_loglikelihood_batch_size = loglikelihood_batch_size
        max_seq = max_loglikelihood_seq_length or getattr(self.config, "max_target_length", None)
        if max_seq is None:
            raise ValueError("A max sequence length is required for OrbaxLM loglikelihood evaluation")
        self.eval_seq_len = int(max_seq)

        # Store tokenizer name for chat template support
        self._tokenizer_name = getattr(tokenizer, "name_or_path", None) or "unknown"

        self._compiled_forward = self._create_fast_forward()

    @property
    def tokenizer_name(self) -> str:
        """Return tokenizer name for chat template support."""
        return self._tokenizer_name

    @property
    def eot_token_id(self) -> int:
        """Return end-of-text token ID (used as prefix for loglikelihood)."""
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self) -> int:
        """Return prefix token ID for loglikelihood calculations."""
        # Use BOS token if available, otherwise fall back to EOS
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.eot_token_id

    def apply_chat_template(self, chat_history: list[dict[str, str]], add_generation_prompt=True) -> str:
        """
        Format chat history using our SFT training format.

        Format (NO system prompt, NO BOS token):
          <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>

        For generation, appends: <|start_header_id|>assistant<|end_header_id|>\n\n
        """
        result = []

        for message in chat_history:
            role = message.get("role", "")
            content = message.get("content", "")

            # Skip system messages - SFT training didn't include them
            if role == "system":
                continue

            if role == "user":
                text = SFT_TEMPLATE["user_start"] + content + SFT_TEMPLATE["user_end"]
                result.append(text)
            elif role == "assistant":
                text = SFT_TEMPLATE["assistant_start"] + content + SFT_TEMPLATE["assistant_end"]
                result.append(text)

        formatted = "".join(result)

        # Add generation prompt (assistant header) for model to continue
        if add_generation_prompt:
            formatted += SFT_TEMPLATE["assistant_start"]

        return formatted

    def _create_fast_forward(self):
        # Extract shardings from actual loaded params (jax arrays have .sharding attribute)
        def extract_sharding(x):
            if hasattr(x, 'sharding'):
                return x.sharding
            return None
        params_shardings = jax.tree_util.tree_map(extract_sharding, self.state.params)

        @partial(pjit,
                 in_shardings=(params_shardings, None, None, None),
                 out_shardings=None)
        def fast_forward(params, input_ids, positions, segment_ids):
            return self.model.apply(
                params,
                input_ids,
                positions,
                segment_ids,
                enable_dropout=False,
                rngs={"aqt": jax.random.PRNGKey(0)},
            )
        return fast_forward
       
    def forward(self, input_ids, **kwargs):
        input_ids_np = input_ids.cpu().numpy()
        input_ids_jax = jnp.asarray(input_ids_np, dtype=jnp.int32)

        batch_size, seq_len = input_ids_jax.shape
        segment_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        positions = jnp.tile(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, 1))

        with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
            jax_logits = self._compiled_forward(
                self.state.params,
                input_ids_jax,
                positions,
                segment_ids,
            )

        class Output:
            def __init__(self, logits):
                self.logits = convert_jax_weight_to_torch(logits)

        return Output(jax_logits)

    def set_eval_seq_length(self, seq_len: int):
        if seq_len <= 0:
            raise ValueError("Evaluation sequence length must be positive")
        if seq_len == self.eval_seq_len:
            return
        print(f"Updating OrbaxLM eval_seq_len from {self.eval_seq_len} to {seq_len}")
        self.eval_seq_len = int(seq_len)

    def set_loglikelihood_batch_size(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("loglikelihood batch size must be positive")
        if batch_size == self.loglikelihood_batch_size:
            return
        print(
            f"Updating OrbaxLM loglikelihood_batch_size from {self.loglikelihood_batch_size} to {batch_size}"
        )
        self.loglikelihood_batch_size = int(batch_size)

    def reset_loglikelihood_batch_size(self):
        self.loglikelihood_batch_size = self._default_loglikelihood_batch_size

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        return cls

    def _tokenize(self, text):
        enc = self.tokenizer(text, return_tensors="pt", padding=True, max_length=self.config.max_target_length, truncation=True)
        return enc["input_ids"].numpy(), enc["attention_mask"].numpy()

    def tok_encode(
        self,
        string: str,
        left_truncate_len: int | None = None,
        add_special_tokens: bool | None = None,
    ) -> list[int]:
        """ """
        # Default to False for causal LM - don't add special tokens
        if add_special_tokens is None:
            add_special_tokens = False

        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def loglikelihood(
        self, requests: list["Instance"], disable_tqdm: bool = False
    ) -> list[tuple[float, bool]]:
        new_reqs = []
        # Debug: print first few requests to understand the format
        debug_count = 0
        for context, continuation in [req.args for req in requests]:
            if debug_count < 2:
                print(f"\n[DEBUG loglikelihood] Request {debug_count}:")
                print(f"  Context (len={len(context)}):")
                print(f"  ---BEGIN CONTEXT---")
                print(context)
                print(f"  ---END CONTEXT---")
                print(f"  Continuation: '{continuation}'")
                debug_count += 1
            if context == "":
                # BOS or EOS as context
                context_enc, continuation_enc = (
                    [self.prefix_token_id],
                    self.tok_encode(continuation),
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)

    def _truncate_to_eval_window(self, context_enc, continuation_enc):
        if not context_enc:
            context_enc = [self.prefix_token_id]

        max_len = self.eval_seq_len
        total = len(context_enc) + len(continuation_enc)
        if total > max_len:
            raise ValueError(
                f"Combined context ({len(context_enc)}) + continuation ({len(continuation_enc)}) length "
                f"{total} exceeds evaluation window {max_len}. Reduce --eval_seq_length or num_fewshot."
            )

        return context_enc, continuation_enc

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: int | None = None,
    ) -> list[tuple[float, bool]]:
        results = []

        # Use override_bs if provided, otherwise use instance's loglikelihood_batch_size
        batch_size = override_bs if override_bs is not None else self.loglikelihood_batch_size
        if batch_size <= 0:
            raise ValueError("loglikelihood_batch_size must be > 0")

        eval_seq_len = self.eval_seq_len
        static_positions = jnp.broadcast_to(
            jnp.arange(eval_seq_len, dtype=jnp.int32), (batch_size, eval_seq_len)
        )
        static_segment_ids = jnp.ones((batch_size, eval_seq_len), dtype=jnp.int32)

        # Process in batches
        for batch_start in tqdm(range(0, len(requests), batch_size), disable=disable_tqdm, desc="Batched loglikelihood"):
            batch_end = min(batch_start + batch_size, len(requests))
            batch_requests = requests[batch_start:batch_end]

            # Prepare batch data
            truncated_pairs = []
            for (_text, context_enc, continuation_enc) in batch_requests:
                c_enc, cont_enc = self._truncate_to_eval_window(context_enc, continuation_enc)
                truncated_pairs.append((c_enc, cont_enc))

            if truncated_pairs:
                filler = truncated_pairs[0]
            else:
                filler = ([self.prefix_token_id], [self.prefix_token_id])

            while len(truncated_pairs) < batch_size:
                truncated_pairs.append(filler)

            padded_input_ids = np.zeros((batch_size, eval_seq_len), dtype=np.int32)
            seq_lens = []
            cont_lens = []
            cont_encs = []
            for idx, (context_enc, continuation_enc) in enumerate(truncated_pairs):
                combined = context_enc + continuation_enc
                seq_len = len(combined)
                if seq_len > eval_seq_len:
                    combined = combined[-eval_seq_len:]
                    seq_len = len(combined)
                padded_input_ids[idx, :seq_len] = combined
                seq_lens.append(seq_len)
                cont_lens.append(len(continuation_enc))
                cont_encs.append(continuation_enc)

            input_ids_batch = jnp.asarray(padded_input_ids, dtype=jnp.int32)

            # Forward pass for entire batch
            with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
                logits = self._compiled_forward(
                    self.state.params,
                    input_ids_batch,
                    static_positions,
                    static_segment_ids,
                )

            # logits: [batch_size, seq_len, vocab_size]
            logits = jax.nn.log_softmax(logits, axis=-1)

            # Extract results for each item in batch
            for i in range(batch_end - batch_start):
                orig_len = seq_lens[i]
                cont_len = cont_lens[i]
                continuation_enc = cont_encs[i]
                # With right padding, continuation starts at (orig_len - cont_len)
                cont_start = orig_len - cont_len

                log_probs = []
                match = True
                for j, tok in enumerate(continuation_enc):
                    logp = logits[i, cont_start + j - 1, tok]  # use previous token's logits
                    log_probs.append(logp)
                    pred = int(jnp.argmax(logits[i, cont_start + j - 1]))
                    if pred != tok:
                        match = False

                loglikelihood = float(jnp.sum(jnp.stack(log_probs)))
                results.append((loglikelihood, match))

        return results
    
    def _encode_pair(
        self, context: str, continuation: str
    ) -> tuple[list[int], list[int]]:
        import transformers

        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        model_class = getattr(self, "AUTO_MODEL_CLASS", None)

        if model_class == transformers.AutoModelForSeq2SeqLM:
            context_enc = self.tok_encode(context)
            continuation_enc = self.tok_encode(continuation, add_special_tokens=False)
        else:
            whole_enc = self.tok_encode(context + continuation)
            context_enc = self.tok_encode(context)

            context_enc_len = len(context_enc)
            continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def generate_until(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError
        
        # res = []

        # for request in tqdm(requests, disable=disable_tqdm):
        #     res.append("lol")
        #     assert request.arguments[0].strip() != ""

        # return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError
    
        results = []
        for text in tqdm(requests, disable=disable_tqdm):
            input_ids, _ = self._tokenize(text)
            mt_ids = jnp.asarray(input_ids, dtype=jnp.int32)
            actual_seq_len = mt_ids.shape[1]

            mt_decoder_segment_ids = jnp.zeros((mt_ids.shape[0], actual_seq_len), dtype=jnp.int32) + 1
            mt_decoder_positions = jnp.stack([jnp.arange(actual_seq_len, dtype=jnp.int32)] * mt_ids.shape[0])

            logits = self.model.apply(
                self.state.params,
                mt_ids,
                mt_decoder_positions,
                mt_decoder_segment_ids,
                enable_dropout=False,
                rngs={"aqt": jax.random.PRNGKey(0)},
            )

            # rolling logp placeholder
            loglikelihood = -1.0  # TODO: compute rolling log-likelihood from logits
            results.append(loglikelihood)

        return results
