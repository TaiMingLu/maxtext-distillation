# import hydra
# requirements: 
# pip install sacrebleu accelerate peft 
import os
import json
import jax
import jax.numpy as jnp
import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse

from tqdm import tqdm
from functools import partial
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import evaluator
from lm_eval.models.orbax_lm import OrbaxLM

from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.layers import models
from MaxText.layers import quantizations

from jax.sharding import Mesh
from jax.experimental import mesh_utils

import math

def _human_readable_bytes(num_bytes):
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(num_bytes)
    unit_idx = 0
    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    return f"{size:.2f} {units[unit_idx]}"

def print_device_memory(note=""):
    try:
        devices = jax.devices()
    except Exception:
        devices = []
    try:
        backend = jax.default_backend()
    except Exception:
        backend = "unknown"
    header = f"[Device Memory]{' ' + note if note else ''} | backend={backend} | devices={len(devices)}"
    print(header)
    for dev in devices:
        # Compose a friendly name
        dev_kind = getattr(dev, "device_kind", getattr(dev, "kind", ""))
        dev_name = f"{getattr(dev, 'platform', 'unknown')}:{getattr(dev, 'id', '?')} ({dev_kind})"

        printed = False
        # Preferred: memory_stats() if available (often on TPU)
        try:
            if hasattr(dev, "memory_stats"):
                stats = dev.memory_stats()
                if isinstance(stats, dict) and stats:
                    in_use = stats.get("bytes_in_use") or stats.get("kb_in_use", 0) * 1024
                    peak = stats.get("peak_bytes_in_use") or stats.get("peak_kb_in_use", 0) * 1024
                    total = stats.get("total_memory") or stats.get("kb_total", 0) * 1024
                    parts = []
                    if in_use:
                        parts.append(f"in_use={_human_readable_bytes(in_use)}")
                    if peak:
                        parts.append(f"peak={_human_readable_bytes(peak)}")
                    if total:
                        parts.append(f"total={_human_readable_bytes(total)}")
                    if parts:
                        print(f"  {dev_name}: " + ", ".join(parts))
                        printed = True
        except Exception:
            pass

        # Fallbacks commonly available on GPU/others
        if not printed:
            alloc = None
            limit = None
            try:
                if hasattr(dev, "memory_allocated"):
                    alloc = dev.memory_allocated()
            except Exception:
                pass
            try:
                if hasattr(dev, "memory_limit"):
                    limit = dev.memory_limit()
            except Exception:
                pass
            try:
                if limit is None and hasattr(dev, "total_memory"):
                    limit = dev.total_memory()
            except Exception:
                pass

            if alloc is not None or limit is not None:
                parts = []
                if alloc is not None:
                    parts.append(f"in_use={_human_readable_bytes(int(alloc))}")
                if limit is not None:
                    parts.append(f"total={_human_readable_bytes(int(limit))}")
                print(f"  {dev_name}: " + ", ".join(parts))
                printed = True

        if not printed:
            print(f"  {dev_name}: memory stats unavailable")

def str2bool(v):
    if isinstance(v, bool):
        return v
    val = v.lower()
    if val in ("yes", "true", "t", "y", "1"):
        return True
    elif val in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (yes/no/true/false)")

PPL_TASKS = [
    "c4",
    "wikitext",
    "wikitext2",
    "cnn_dailymail",
    # "dclm"
]

ACC_TASKS = [
    {
        "name": "winogrande",
        "num_fewshot": 0,
        "acc_key": "acc,none",
    },
    # {
    #     "name": "winogrande",
    #     "num_fewshot": 5,
    #     "acc_key": "acc,none",
    # },
    {
        "name": "arc_easy",
        "num_fewshot": 0,
        "acc_key": "acc_norm,none",
    },
    {
        "name": "arc_challenge",
        "num_fewshot": 0,
        "acc_key": "acc_norm,none",
    },
    # {
    #     "name": "arc_challenge",
    #     "num_fewshot": 25,
    #     "acc_key": "acc_norm,none",
    # },
    # {
    #     "name": "hellaswag",
    #     "num_fewshot": 0,
    #     "acc_key": "acc_norm,none",
    # },
    # {
    #     "name": "hellaswag",        
    #     "num_fewshot": 10,
    #     "acc_key": "acc_norm,none",
    # },
    # {
    #     "name": "mmlu",
    #     "num_fewshot": 0,
    #     "acc_key": None,
    # },
    # {
    #     "name": "mmlu",
    #     "num_fewshot": 5,
    #     "acc_key": None,
    # },
    # {
    #     "name": "truthfulqa_mc1",
    #     "num_fewshot": 0,
    #     "acc_key": "acc,none",
    # },
    # {
    #     "name": "truthfulqa_mc2",
    #     "num_fewshot": 0,
    #     "acc_key": "acc,none",
    # },
    # {
    #     "name": "piqa",
    #     "num_fewshot": 0,
    #     "acc_key": "acc_norm,none",
    # },
    # {
    #     "name": "sciq",
    #     "num_fewshot": 0,
    #     "acc_key": "acc,none",
    # },
    # {
    #     "name": "boolq",
    #     "num_fewshot": 0,
    #     "acc_key": "acc,none",
    # },
    # {
    #     "name": "anli_r1",
    #     "num_fewshot": 0,
    #     "acc_key": None,
    # },
    # {
    #     "name": "anli_r2",
    #     "num_fewshot": 0,
    #     "acc_key": None,
    # },
    # {
    #     "name": "anli_r3",
    #     "num_fewshot": 0,
    #     "acc_key": None,
    # },
    # {
    #     "name": "openbookqa",
    #     "num_fewshot": 0,
    #     "acc_key": None,
    # },
    # {
    #     "name": "rte",
    #     "num_fewshot": 0,
    #     "acc_key": None,
    # },
    # {
    #     "name": "record",
    #     "num_fewshot": 0,
    #     "acc_key": None,
    # },
]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_ppl_enc(task, tokenizer, add_special_tokens: bool = True):
    if task == 'wikitext':
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", trust_remote_code=True)
        text_column = "text"
        testenc = tokenizer.encode("\n\n".join(dataset[:32768][text_column]), return_tensors='pt', add_special_tokens=add_special_tokens)
    elif task == 'wikitext2':
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
        text_column = "text"
        testenc = tokenizer.encode("\n\n".join(dataset[:32768][text_column]), return_tensors='pt', add_special_tokens=add_special_tokens)
    elif task == 'cnn_dailymail':
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="train", trust_remote_code=True)
        text_column = "article"
        testenc = tokenizer.encode(" ".join(dataset[:16384][text_column]), return_tensors='pt', add_special_tokens=add_special_tokens)
    elif task == 'c4':
        dataset = load_dataset(
            "allenai/c4", 
            data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, 
            split="train", 
            verification_mode="no_checks",
            trust_remote_code=True
        )
        text_column = "text"
        testenc = tokenizer.encode(" ".join(dataset[:8192][text_column]), return_tensors='pt', add_special_tokens=add_special_tokens)
    elif task == 'dclm':
        data_path = "/home/zephyr/gcs-bucket/datasets/dclm/dclm_baseline_1.0.val.jsonl"
        dataset = load_dataset(
            "json",
            data_files={"train": data_path},
            split="train",
            verification_mode="no_checks"
        )
        text_column = "text"
        testenc = tokenizer.encode(" ".join(dataset[:8192][text_column]), return_tensors='pt', add_special_tokens=add_special_tokens)
    else:
        raise NotImplementedError(f"Unsupported task: {task}")
    return testenc

def get_ppl(
    model, 
    tokenizer, 
    tasks,
    batch_size: int = 1,
    calib_size: int = 256,
    max_length: int = 8192,
    add_special_tokens: bool = True,
    task_range: list = []
):
    # devices_in_data_fsdp = model.devices_in_data_fsdp
    # if batch_size % devices_in_data_fsdp != 0:
    #     print(f"🔁 Adjusting batch_size {batch_size} → {devices_in_data_fsdp * ((batch_size + devices_in_data_fsdp - 1) // devices_in_data_fsdp)} for device mesh compatibility.")
    #     batch_size = devices_in_data_fsdp * ((batch_size + devices_in_data_fsdp - 1) // devices_in_data_fsdp)
    if task_range:
        tasks = [t for t in tasks if t in task_range]
    
    print(f"Starting PPL evaluation for tasks: {tasks}")
    ppl_res = {}
    ppl_times = {}
    for task in tasks:
        print(f"Currently evaluating PPL task: {task}")
        start_ts = time.perf_counter()
        testenc = get_ppl_enc(task, tokenizer, add_special_tokens=add_special_tokens)
        tot_loss = 0
        tot_tokens = 0
        bs = batch_size
        seq_len = max_length
        nsamples = min(testenc.numel() // seq_len, calib_size)
        with torch.no_grad():
            for i in tqdm(range(0, nsamples, bs), desc=f"Evaluating PPL for {task}"):
                j = min(i + bs, nsamples)
                inputs = testenc[:,(i * seq_len):(j * seq_len)]
                inputs = inputs.reshape(j - i, seq_len)
                # import pdb; pdb.set_trace()
                
                outputs = model.forward(inputs)
                if hasattr(outputs, "logits"):
                    lm_logits = outputs.logits
                else:
                    lm_logits = outputs
                
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = inputs[:, 1:]
                
                loss_fct = nn.CrossEntropyLoss().to(shift_logits.device)
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
                
                tot_loss += loss.item() * seq_len * (j - i)
                tot_tokens += seq_len * (j - i)
                
            ppl_res[task] = torch.exp(torch.tensor(tot_loss / tot_tokens)).item()
            duration_s = time.perf_counter() - start_ts
            ppl_times[task] = duration_s
            print(f"{task} PPL: {ppl_res[task]} (time: {duration_s:.2f}s)")
            if task == "dclm":
                print("dclm val loss", math.log(ppl_res[task]))
            print_device_memory(f"after PPL task {task}")
                
    return ppl_res, ppl_times

def get_acc(model, tokenizer, tasks, task_range=[], limit=1000000, batch_size=32):
    # lm_eval_model = models.orbax_lm.HFLM(
    #     pretrained=model,
    #     tokenizer=tokenizer,
    #     generation_kwargs={
    #         "do_sample": True,
    #         "temperature": 0.2,
    #         "top_p": 0.95,
    #     }
    # )
    if task_range:
        tasks = [cfg for cfg in tasks if cfg["name"] in task_range]

    print("tasks to evaluate:")
    print(json.dumps(tasks, indent=2))
    print(f"Starting accuracy evaluation with batch_size={batch_size}...")
    acc_res = {}
    full_res_by_task = {}
    acc_times = {}
    for cfg in tasks:
        task = cfg["name"]
        print(f"Currently evaluating ACC task: {task} (fewshot={cfg['num_fewshot']})")
        start_ts = time.perf_counter()
        res = evaluator.simple_evaluate(
            model=model,
            tasks=[task],
            num_fewshot=cfg["num_fewshot"],
            max_batch_size=batch_size,
            log_samples=True,
            # task_kwargs={"limit": 256},
            confirm_run_unsafe_code=True,
            limit=limit
        )
        
        print(res['results'][task])
        duration_s = time.perf_counter() - start_ts
        times_key = f"{task}/fewshot={cfg['num_fewshot']}"
        acc_times[times_key] = duration_s
        print(f"{times_key} ACC eval time: {duration_s:.2f}s")
        acc_key = cfg["acc_key"]
        if acc_key is not None:
            acc_res[task] = res['results'][task][acc_key]
        full_res_by_task[task] = res
        print_device_memory(f"after ACC task {times_key}")
    return acc_res, full_res_by_task, acc_times

def cast_orbax_state_to_bf16(orbax_state):
    casted_params = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if hasattr(x, "dtype") and x.dtype == jnp.float32 else x,
        orbax_state.params
    )
    orbax_state = orbax_state.replace(params=casted_params)
    return orbax_state

def main(config, test_args):
    tokenizer = AutoTokenizer.from_pretrained(test_args.hf_model_path)
    
    init_rng = jax.random.PRNGKey(config.init_weights_seed)
    init_rng, rng1 = jax.random.split(init_rng)
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
    quant = quantizations.configure_quantization(config)
    orbax_model = models.Transformer(config, mesh, quant=quant)
    orbax_state, _ = maxtext_utils.setup_decode_state(orbax_model, config, rng1, mesh, None)
    
    orbax_state = cast_orbax_state_to_bf16(orbax_state)
    print_device_memory("after model init")
    
    _, _, state_mesh_shardings = maxtext_utils.get_abstract_state(
        orbax_model, None, config, rng1, mesh, is_training=False
    )

    model = OrbaxLM(orbax_model, orbax_state, tokenizer, config, state_mesh_shardings, mesh)
    
    print_device_memory("before PPL eval")
    ppl_res, ppl_times = get_ppl(
        model,
        tokenizer,
        batch_size=test_args.ppl_batch_size,
        calib_size=min(256, test_args.limit),
        max_length=config.max_target_length,
        tasks=PPL_TASKS,
        add_special_tokens=test_args.add_special_tokens,
        task_range=test_args.tasks,
    )
    print(ppl_res)
    print({"ppl_times_s": ppl_times})

    print_device_memory("before ACC eval")
    acc_res, acc_full, acc_times = get_acc(
        model,
        tokenizer,
        tasks=ACC_TASKS,
        task_range=test_args.tasks,
        limit=test_args.limit,
        batch_size=test_args.acc_batch_size
    )
    print(acc_res)
    print({"acc_times_s": acc_times})

    # Optionally save results to disk
    if getattr(test_args, "eval_save_dir", ""):
        os.makedirs(test_args.eval_save_dir, exist_ok=True)

        def to_serializable(obj):
            try:
                import numpy as _np
                import jax.numpy as _jnp
            except Exception:  # pragma: no cover
                _np, _jnp = None, None
            if _np is not None and isinstance(obj, _np.generic):
                return obj.item()
            if _jnp is not None and hasattr(obj, "dtype") and hasattr(obj, "tolist"):
                return obj.tolist()
            if hasattr(obj, "tolist"):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        results_payload = {
            "run_name": getattr(config, "run_name", ""),
            "model_name": getattr(config, "model_name", ""),
            "limit": test_args.limit,
            "tasks_requested": test_args.tasks,
            "add_special_tokens": test_args.add_special_tokens,
            "ppl": ppl_res,
            "lm_eval": {
                "acc_summary": acc_res,
                "per_task": acc_full,
            },
            "timing": {
                "ppl": ppl_times,
                "lm_eval": acc_times,
            },
        }

        save_path = os.path.join(test_args.eval_save_dir, f"{getattr(config, 'run_name', 'results')}.json")
        with open(save_path, "w") as f:
            json.dump(results_payload, f, indent=2, default=to_serializable)
        print(f"Saved results to {save_path}")
    
if __name__ == "__main__":
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, required=False, default=0.1)
    parser.add_argument("--rtol", type=float, required=False, default=0.1)
    parser.add_argument("--token_size", type=int, required=False)
    parser.add_argument("--max_kl_div", type=float, required=False, default=None)
    parser.add_argument("--golden_logits_path", type=str, required=False, default="")
    parser.add_argument("--hf_model_path", type=str, required=False, default="")
    parser.add_argument("--run_hf_model", type=bool, required=False, default=False)
    parser.add_argument('--add_special_tokens', type=str2bool, default=True)
    parser.add_argument("--limit", type=int, default=1000000)
    parser.add_argument("--tasks", type=lambda x: [] if not x else x.split(","), default=[])
    parser.add_argument("--eval_save_dir", type=str, required=False, default="")
    parser.add_argument("--ppl_batch_size", type=int, default=1, help="Batch size for PPL evaluation (default: 1)")
    parser.add_argument("--acc_batch_size", type=int, default=32, help="Batch size for accuracy evaluation (default: 32)")
    test_args, _ = parser.parse_known_args()

    # Remove args defined in this test file to avoid error from pyconfig
    model_args = sys.argv
    to_remove_args = [
        "--atol",
        "--rtol",
        "--token_size",
        "--max_kl_div",
        "--golden_logits_path",
        "--hf_model_path",
        "--run_hf_model",
        "--add_special_tokens",
        "--limit",
        "--tasks",
        "--save_dir",
        "--eval_save_dir",
        "--ppl_batch_size",
        "--acc_batch_size"
    ]
    for arg in to_remove_args:
        model_args = [s for s in model_args if not s.startswith(arg)]

    cfg = pyconfig.initialize(model_args)
    main(cfg, test_args)
