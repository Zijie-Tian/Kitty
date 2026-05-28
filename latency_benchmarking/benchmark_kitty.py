# examples/run_qwen3_kitty.py

from typing import Optional
import time
import shutil
import gc
import statistics
import torch
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig
from transformers.tokenization_utils_base import BatchEncoding
from transformers.models.qwen3 import Qwen3Config
from transformers.models.qwen3 import Qwen3ForCausalLM
from kitty.models.qwen3 import Qwen3ForCausalLM_Kitty
from kitty.kvcache import get_kvcache_kitty
#


def get_prompt(prompt_choice: int) -> tuple[str, str]:
    if prompt_choice == 1:
        task_name = "gsm8k"
        prompt = """"Given the following problem, think step by step and give a final answer to the problem.
        Problem: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
        Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
        There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6
        Given the following problem, think step by step and give a final answer to the problem.
        Problem: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
        Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
        There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5
        Given the following problem, think step by step and give a final answer to the problem.
        Problem: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
        Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
        Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39
        Given the following problem, think step by step and give a final answer to the problem.
        Problem: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
        Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
        Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8
        Given the following problem, think step by step and give a final answer to the problem.
        Problem: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
        Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
        Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9
        Given the following problem, think step by step and give a final answer to the problem.
        Problem: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
        Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
        There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29
        Given the following problem, think step by step and give a final answer to the problem.
        Problem: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
        Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
        Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33
        Given the following problem, think step by step and give a final answer to the problem.
        Problem: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
        Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
        Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8

        Given the following problem, think step by step and give a final answer to the problem.
        Problem: Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
        Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
        """
    elif prompt_choice == 2:
        task_name = "gpqa"
        prompt = """"Here are some example questions from experts. Think step by step and then give the final answer, following the format of the previous questions exactly.
        Question: In a given population, 1 out of every 400 people has a cancer caused by a completely recessive allele, b. Assuming the population is in Hardy-Weinberg equilibrium, which of the following is the expected proportion of individuals who carry the b allele but are not expected to develop the cancer?
        Choices:
        (A) 1/400
        (B) 19/400
        (C) 20/400
        (D) 38/400
        Let's think step by step:  The expected proportion of individuals who carry the b allele but are not expected to develop the cancer equals to the frequency of heterozygous allele in the given population. According to the Hardy-Weinberg equation p^2 + 2pq + q^2 = 1, where p is the frequency of dominant allele frequency, q is the frequency of recessive allele frequency, p^2 is the frequency of the homozygous dominant allele, q^2 is the frequency of the recessive allele, and 2pq is the frequency of the heterozygous allele. Given that q^2=1/400, hence, q=0.05 and p=1-q=0.95. The frequency of the heterozygous allele is 2pq=2*0.05*0.95=38/400. The final answer is (D).

        Question: A Fe pellet of 0.056 g is first dissolved in 10 mL of hydrobromic acid HBr (0.1 M). The resulting solution is then titrated by KMnO4 (0.02 M). How many equivalence points are there?
        Choices:
        (A) Two points, 25 ml and 35 ml
        (B) One point, 25 mL
        (C) One point, 10 ml
        (D) Two points, 25 ml and 30 ml
        Let's think step by step:  HBr will react with Fe to produce Fe2+. MnO4- will first react with Fe2+ then Br-. Two equivalence points will exist 25 ml and 35 ml. In the beaker there is Fe2+ and Br-. When considering titration with two analytes one will have to consider which reaction will occur first. Since it is a redox titration consider the reduction potential of: E0 (Br2 /Br- ) = 1.09 V  \tE0 (MnO4-/ Mn2+) = 1.49 V\tE0 (Fe3+/Fe2+) =0.77 V. [Fe2+]=m/MV=0.1M. Reaction 1: MnO4- + 5Fe2+ + 8H+ \u2192 Mn2+ + 5Fe3+ + 4H2O. Reaction 2: 2MnO4- + 10Br- + 16H+ \u2192 2Mn2+ + 5Br2 + 8H2O. So MnO4- will first react with Fe2+ with a stoichiometry of 1:5 so Veq1 will be 10 ml. Then when Fe2+ is used up, MnO4- will react with Br- with a stoichiometry of 2:10 then V added will be 25 ml so Veq2=25+10=35 ml. The final answer is (A).

        Question: Consider a quantum mechanical system containing a particle of mass $m$ moving in an isotropic three-dimensional potential of the form $V(r) = 1/2 m \u03c9^2 r^2$ corresponding to the acted force obeying Hooke's law. Here, $\u03c9$ is the angular frequency of oscillation and $r$ is the radial distance of the particle from the origin in spherical polar coordinates. What is the value of energy of the third excited state, and how many linearly independent eigenfunctions are possible for the same energy eigenvalue?
        Choices:
        (A) 11 \u03c0\u00b2 \u210f\u00b2 / (2m r\u00b2), 3
        (B) (9/2) \u210f \u03c9, 10
        (C) 11 \u03c0\u00b2 \u210f\u00b2 / (2m r\u00b2), 10
        (D) (9/2) \u210f \u03c9, 3
        Let's think step by step:  This problem is nothing but the three-dimensional simple harmonic oscillator (SHO) problem. The energy spectrum of three-dimensional SHO is $E_n= (n+3/2)\u210f \u03c9$ where $n=0,1,2,3\u2026$. For third excited state n=3. 3+3/2=6/2+3/2=9/2. Thus the corresponding energy is $(9/2)\u210f \u03c9$. The degeneracy of the state is $g_n= (n+1)(n+2)/2$. For n=3, degeneracy is (3+1)*(3+2)/2=4*5/2=10. The final answer is (B).

        Question: Your overhear two chemists talking to each other as they leave a synthetic organic chemistry lab. One asks the other 'So, how did it go?' The second chemist replies, 'Not well - my compounds are on top of each other.' What is the second chemist most likely referring to?
        Choices:
        (A) The compounds they are working with have similar polarities.
        (B) The compounds they are working with have similar boiling points.
        (C) The compounds they are working with are bonding to each other through non-covalent/van der Waals interactions.
        (D) The compounds they are working with have similar optical rotations.
        Let's think step by step:  On top of each other commonly refers to two compounds that have similar Rf values on chromatography (a common operation in synthetic chemistry). Similar Rf values arise for compounds with similar polarities. The final answer is (A).

        Question: Two people are playing the following game. A fair coin is tossed into the air. Person A says that in a single toss of the coin, the tail will come. So it's like the first shot or the third shot or the fifth shot. Person B says that the coin will come with a double toss. So like the second, fourth, sixth or eighth shot. Imagine this game played forever. What is the probability that person A wins this game?
        Choices:
        (A) 1/2
        (B) 1/4
        (C) 2/3
        (D) 1/8
        Let's think step by step:  When finding the correct answer, the probability of playing forever and the coin's single-point toss will be calculated. For example, a tail may appear on the first shot. This probability is 1/2. if the first toss doesn't come up, it shouldn't come to the second roll either, because the second throw is an even number. So it can come in the third shot. This is (1/2)(1/2)(1/2). So (1/2)^3=1/8. Or it could come on the fifth shot. This is (1/2)^5=1/32. This is actually a geometric series that goes on forever. We can write this series as follows. (1/2) + (1/2)^3 + (1/2)^5 + (1/2)^7 + \u2026\u2026\u2026. The solution for this series is as follows : a1/(1-r) where a1 is the first number and r is the sequence or r= a2/a1 or a3/a2 etc. a1=1/2 r=(1/2)^2=1/4 So a1/(1-r)=(1/2)/(1-1/4)=(1/2)/(3/4)=2/3. The final answer is (C).

        Question: Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?
        Choices:
        (A) 10^-9 eV
        (B) 10^-11 eV
        (C) 10^-8 eV
        (D) 10^-4 eV
        Let's think step by step: ",
        """
    else:
        task_name = "general"
        prompt = "现在父亲的年龄是儿子的 3 倍。再过 15 年后，父亲的年龄会是儿子的 2 倍。请回答以下两个问题： 1.现在父亲几岁？ 2.现在儿子几岁？"
    return task_name, prompt


#
import argparse
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',                  type=str, default="Qwen/Qwen3-8B",  help='Qwen3 model to load')
    parser.add_argument("--cache_implementation",   type=int, default=0,                help="Choosing KV cache implementation: (0) for Kitty; (1) for FP16 static cache; (2) for FP16 dynamic cache; (3) for INT4 quantized cache with quanto backend.")
    parser.add_argument("--max_seq_len",            type=int, default=2048,             help="Maximum sequence length for KV cache.")
    parser.add_argument("--batch_size",             type=int, default=32,               help="Batch size for generation, repeat the prompt for each batch")
    parser.add_argument("--prompt_choice",          type=int, default=1,                help="Choice of prompt to use")
    parser.add_argument("--warmup_runs",            type=int, default=2,                help="Number of warmup runs")
    parser.add_argument("--repeat_runs",            type=int, default=3,                help="Number of repeat runs for benchmarking")
    parser.add_argument("--max_new_tokens",         type=int, default=None,             help="Maximum number of decode tokens to generate. If omitted, generation uses --max_seq_len as the total max length.")
    parser.add_argument("--page_size",              type=int, default=128,              help="Kitty KV-cache page size. Keep 128 for the paper default; use 16 for QUEST-aligned experiments.")
    parser.add_argument("--promote_ratio",          type=float, default=0.125,            help="Fraction of key-cache channels promoted to INT4. Use 0.125 for paper Kitty; 0.25 for Kitty-Pro.")
    parser.add_argument("--quest-enabled",          action="store_true",                 help="Enable correctness-first QUEST sparse page selection in the Kitty cache.")
    parser.add_argument("--quest-topk-pages",       type=int, default=None,               help="QUEST sparse page budget in page16 logical pages.")
    parser.add_argument("--quest-token-budget",     type=int, default=None,               help="QUEST sparse token budget; converted to pages by the cache page size.")
    parser.add_argument("--quest-skip-layers",      type=int, default=2,                  help="Disable QUEST sparse selection for the first N layers.")
    parser.add_argument("--force-sparse-for-equivalence", action="store_true",           help="Force sparse all-pages path for dense-equivalence tests.")
    parser.add_argument("--compare-quest-kitty",    action="store_true",                 help="Run a decode-only comparison: pure Kitty page16 vs QUEST+Kitty page16. Defaults QUEST token budget to 2048 when no QUEST budget is supplied.")
    parser.add_argument("--attn-implementation",    type=str, default="sdpa",             choices=["eager", "sdpa", "flash_attention_2"], help="Transformers attention backend used while loading the model. Use sdpa when flash_attn is not installed.")
    return parser


def _generation_length_kwargs(max_seq_len: int, max_new_tokens: int | None) -> dict:
    if max_new_tokens is None:
        return {"max_length": max_seq_len, "max_new_tokens": None}
    return {"max_new_tokens": max_new_tokens}


def _repeat_inputs_to_length(inputs: BatchEncoding, target_length: int) -> BatchEncoding:
    if target_length <= 0:
        raise ValueError(f"target_length must be positive; got {target_length}.")
    input_ids = inputs.input_ids
    current_length = input_ids.size(1)
    if current_length <= 0:
        raise ValueError("Cannot expand an empty prompt for a decode-speed gate.")
    repeats = (target_length + current_length - 1) // current_length
    expanded_input_ids = input_ids.repeat(1, repeats)[:, :target_length].contiguous()
    if "attention_mask" in inputs:
        attention_mask = inputs.attention_mask.repeat(1, repeats)[:, :target_length].contiguous()
    else:
        attention_mask = torch.ones_like(expanded_input_ids)
    return BatchEncoding({"input_ids": expanded_input_ids, "attention_mask": attention_mask})


def _new_kitty_cache(
    model_config: PretrainedConfig,
    max_batch_size: int,
    max_length: int,
    page_size: int,
    promote_ratio: float,
    quest_enabled: bool,
    quest_topk_pages: int | None,
    quest_token_budget: int | None,
    quest_skip_layers: int,
    force_sparse_for_equivalence: bool,
):
    return get_kvcache_kitty(
        model_config,
        max_batch_size,
        max_length,
        page_size=page_size,
        promote_ratio=promote_ratio,
        quest_enabled=quest_enabled,
        quest_topk_pages=quest_topk_pages,
        quest_token_budget=quest_token_budget,
        quest_skip_layers=quest_skip_layers,
        force_sparse_for_equivalence=force_sparse_for_equivalence,
    )


def _collect_quest_cache_stats(kitty_kv_cache) -> dict:
    stats = {
        "qk_page_loads": 0,
        "sv_page_loads": 0,
        "sparse_qk_hits": 0,
        "sparse_sv_hits": 0,
        "selected_pages": [],
        "selected_tokens": [],
        "shared_pages": [],
        "paths": {},
    }
    for layer in getattr(kitty_kv_cache, "kv_cache", []):
        path = getattr(layer, "last_quest_path", "unknown")
        stats["paths"][path] = stats["paths"].get(path, 0) + 1
        stats["sparse_qk_hits"] += int(getattr(layer, "last_sparse_qk_hits", 0) or 0)
        stats["sparse_sv_hits"] += int(getattr(layer, "last_sparse_sv_hits", 0) or 0)
        stats["shared_pages"].append(int(getattr(layer, "last_shared_page_count", 0) or 0))
        selected = getattr(layer, "last_selected_pages", None)
        if selected is not None:
            qk_page_loads = int(getattr(layer, "last_sparse_qk_pages_loaded", 0) or 0)
            sv_page_loads = int(getattr(layer, "last_sparse_sv_pages_loaded", 0) or 0)
            # Fall back to selected-page count for older caches, but prefer real
            # sparse-kernel page-load evidence when available.
            if qk_page_loads == 0 and sv_page_loads == 0:
                qk_page_loads = int(selected.numel())
                sv_page_loads = int(selected.numel())
            stats["qk_page_loads"] += qk_page_loads
            stats["sv_page_loads"] += sv_page_loads
            if selected.ndim > 0:
                stats["selected_pages"].append(int(selected.shape[-1]))
        selected_tokens = int(getattr(layer, "last_selected_tokens", 0) or 0)
        if selected_tokens:
            stats["selected_tokens"].append(selected_tokens)
    return stats


def _merge_quest_cache_stats(total: dict, step: dict) -> None:
    total["qk_page_loads"] += step["qk_page_loads"]
    total["sv_page_loads"] += step["sv_page_loads"]
    total["sparse_qk_hits"] += step["sparse_qk_hits"]
    total["sparse_sv_hits"] += step["sparse_sv_hits"]
    total["selected_pages"].extend(step["selected_pages"])
    total["selected_tokens"].extend(step["selected_tokens"])
    total["shared_pages"].extend(step["shared_pages"])
    for path, count in step["paths"].items():
        total["paths"][path] = total["paths"].get(path, 0) + count


def _summarize_ints(values: list[int]) -> str:
    if not values:
        return "n/a"
    return f"min={min(values)}, median={statistics.median(values):.1f}, max={max(values)}, mean={statistics.mean(values):.1f}"


def _run_kitty_decode_only_trial(
    model: PreTrainedModel,
    inputs: BatchEncoding,
    model_config: PretrainedConfig,
    max_new_tokens: int,
    cache_length: int,
    page_size: int,
    promote_ratio: float,
    quest_enabled: bool,
    quest_topk_pages: int | None,
    quest_token_budget: int | None,
    quest_skip_layers: int,
    force_sparse_for_equivalence: bool,
) -> dict:
    max_batch_size = inputs.input_ids.size(0)
    kitty_kv_cache = _new_kitty_cache(
        model_config,
        max_batch_size,
        cache_length,
        page_size,
        promote_ratio,
        quest_enabled,
        quest_topk_pages,
        quest_token_budget,
        quest_skip_layers,
        force_sparse_for_equivalence,
    )
    input_ids = inputs.input_ids.cuda()
    attention_mask = inputs.attention_mask.cuda() if "attention_mask" in inputs else None
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=kitty_kv_cache,
            use_cache=True,
            logits_to_keep=1,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        quest_stats = {
            "qk_page_loads": 0,
            "sv_page_loads": 0,
            "sparse_qk_hits": 0,
            "sparse_sv_hits": 0,
            "selected_pages": [],
            "selected_tokens": [],
            "shared_pages": [],
            "paths": {},
        }
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=next_token,
                attention_mask=None,
                past_key_values=kitty_kv_cache,
                use_cache=True,
                logits_to_keep=1,
            )
            _merge_quest_cache_stats(quest_stats, _collect_quest_cache_stats(kitty_kv_cache))
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        elapsed_time = time.perf_counter() - start_time
    kitty_kv_cache = None
    return {
        "elapsed_s": elapsed_time,
        "decode_tokens": max_new_tokens,
        "ms_per_token": (elapsed_time / max_new_tokens) * 1000,
        "tokens_per_s": max_new_tokens / elapsed_time,
        "quest_stats": quest_stats,
    }


def benchmark_kitty_decode_only(
    label: str,
    model: PreTrainedModel,
    inputs: BatchEncoding,
    model_config: PretrainedConfig,
    warmup_runs: int,
    repeat_runs: int,
    page_size: int,
    promote_ratio: float,
    max_new_tokens: int,
    quest_enabled: bool = False,
    quest_topk_pages: int | None = None,
    quest_token_budget: int | None = None,
    quest_skip_layers: int = 2,
    force_sparse_for_equivalence: bool = False,
) -> dict:
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive for decode-only benchmarking; got {max_new_tokens}.")
    if repeat_runs <= 0:
        raise ValueError(f"repeat_runs must be positive for decode-only benchmarking; got {repeat_runs}.")
    input_length = inputs.input_ids.size(1)
    cache_length = input_length + max_new_tokens
    print(
        f"[decode-only:{label}] prefill_tokens={input_length}, decode_tokens={max_new_tokens}, "
        f"page_size={page_size}, promote_ratio={promote_ratio}, quest_enabled={quest_enabled}, "
        f"quest_topk_pages={quest_topk_pages}, quest_token_budget={quest_token_budget}, quest_skip_layers={quest_skip_layers}"
    )
    for _ in range(warmup_runs):
        _run_kitty_decode_only_trial(
            model,
            inputs,
            model_config,
            max_new_tokens,
            cache_length,
            page_size,
            promote_ratio,
            quest_enabled,
            quest_topk_pages,
            quest_token_budget,
            quest_skip_layers,
            force_sparse_for_equivalence,
        )
    if warmup_runs:
        print(f"[decode-only:{label}] Warmup done.")

    runs = []
    for _ in range(repeat_runs):
        runs.append(
            _run_kitty_decode_only_trial(
                model,
                inputs,
                model_config,
                max_new_tokens,
                cache_length,
                page_size,
                promote_ratio,
                quest_enabled,
                quest_topk_pages,
                quest_token_budget,
                quest_skip_layers,
                force_sparse_for_equivalence,
            )
        )

    ms_per_token_values = [run["ms_per_token"] for run in runs]
    tokens_per_s_values = [run["tokens_per_s"] for run in runs]
    elapsed_values = [run["elapsed_s"] for run in runs]
    total_stats = {
        "qk_page_loads": 0,
        "sv_page_loads": 0,
        "sparse_qk_hits": 0,
        "sparse_sv_hits": 0,
        "selected_pages": [],
        "selected_tokens": [],
        "shared_pages": [],
        "paths": {},
    }
    for run in runs:
        _merge_quest_cache_stats(total_stats, run["quest_stats"])

    result = {
        "label": label,
        "avg_elapsed_s": statistics.mean(elapsed_values),
        "median_elapsed_s": statistics.median(elapsed_values),
        "avg_ms_per_token": statistics.mean(ms_per_token_values),
        "median_ms_per_token": statistics.median(ms_per_token_values),
        "avg_tokens_per_s": statistics.mean(tokens_per_s_values),
        "median_tokens_per_s": statistics.median(tokens_per_s_values),
        "quest_stats": total_stats,
    }
    print(
        f"[decode-only:{label}] avg_elapsed_s={result['avg_elapsed_s']:.6f}, "
        f"median_elapsed_s={result['median_elapsed_s']:.6f}, "
        f"avg_ms_per_token={result['avg_ms_per_token']:.4f}, "
        f"median_ms_per_token={result['median_ms_per_token']:.4f}, "
        f"avg_tokens_per_s={result['avg_tokens_per_s']:.2f}, "
        f"median_tokens_per_s={result['median_tokens_per_s']:.2f}"
    )
    print(
        f"[decode-only:{label}] qk_page_loads={total_stats['qk_page_loads']}, "
        f"sv_page_loads={total_stats['sv_page_loads']}, sparse_qk_hits={total_stats['sparse_qk_hits']}, "
        f"sparse_sv_hits={total_stats['sparse_sv_hits']}, paths={total_stats['paths']}, "
        f"selected_pages={_summarize_ints(total_stats['selected_pages'])}, "
        f"selected_tokens={_summarize_ints(total_stats['selected_tokens'])}, "
        f"shared_pages={_summarize_ints(total_stats['shared_pages'])}"
    )
    return result


def benchmark_quest_kitty_comparison(
    model: PreTrainedModel,
    inputs: BatchEncoding,
    model_config: PretrainedConfig,
    max_seq_len: int,
    warmup_runs: int,
    repeat_runs: int,
    page_size: int,
    promote_ratio: float,
    max_new_tokens: int,
    quest_topk_pages: int | None = None,
    quest_token_budget: int | None = None,
    quest_skip_layers: int = 2,
    force_sparse_for_equivalence: bool = False,
) -> None:
    if page_size != 16:
        raise ValueError("--compare-quest-kitty is the page16 speed gate; pass --page_size 16.")
    if quest_topk_pages is None and quest_token_budget is None:
        quest_token_budget = 2048
    gate_inputs = _repeat_inputs_to_length(inputs, max_seq_len)
    print(
        f"[decode-only:comparison] expanded prompt to {gate_inputs.input_ids.size(1)} tokens; "
        f"QUEST budget defaults to {quest_token_budget} tokens when no explicit QUEST budget is supplied."
    )
    pure = benchmark_kitty_decode_only(
        "pure_kitty_page16",
        model,
        gate_inputs,
        model_config,
        warmup_runs,
        repeat_runs,
        page_size,
        promote_ratio,
        max_new_tokens,
        quest_enabled=False,
        quest_topk_pages=None,
        quest_token_budget=None,
        quest_skip_layers=quest_skip_layers,
        force_sparse_for_equivalence=False,
    )
    quest = benchmark_kitty_decode_only(
        "quest_kitty_page16",
        model,
        gate_inputs,
        model_config,
        warmup_runs,
        repeat_runs,
        page_size,
        promote_ratio,
        max_new_tokens,
        quest_enabled=True,
        quest_topk_pages=quest_topk_pages,
        quest_token_budget=quest_token_budget,
        quest_skip_layers=quest_skip_layers,
        force_sparse_for_equivalence=force_sparse_for_equivalence,
    )
    decode_speedup = pure["avg_ms_per_token"] / quest["avg_ms_per_token"]
    median_decode_speedup = pure["median_ms_per_token"] / quest["median_ms_per_token"]
    print(
        f"[decode-only:comparison] decode_speedup={decode_speedup:.3f} "
        f"(kitty_avg_ms_per_token / quest_avg_ms_per_token); "
        f"median_decode_speedup={median_decode_speedup:.3f}"
    )


def benchmark_kitty(model: PreTrainedModel, tokenizer: AutoTokenizer, inputs: dict, max_seq_len, model_config: PretrainedConfig, warmup_runs: int, repeat_runs: int, page_size: int, promote_ratio: float, max_new_tokens: int | None = None, quest_enabled: bool = False, quest_topk_pages: int | None = None, quest_token_budget: int | None = None, quest_skip_layers: int = 2, force_sparse_for_equivalence: bool = False) -> None:
    max_batch_size = inputs.input_ids.size(0)
    print(f"Kitty page_size: {page_size}")
    print(f"Kitty promote_ratio: {promote_ratio}")
    print(f"Kitty QUEST: enabled={quest_enabled}, topk_pages={quest_topk_pages}, token_budget={quest_token_budget}, skip_layers={quest_skip_layers}, force_sparse_for_equivalence={force_sparse_for_equivalence}")
    length_kwargs = _generation_length_kwargs(max_seq_len, max_new_tokens)
    cache_max_length = max_seq_len
    if max_new_tokens is not None:
        cache_max_length = max(cache_max_length, inputs.input_ids.size(1) + max_new_tokens)
    # Warm up
    for _ in range(warmup_runs):
        kitty_kv_cache = _new_kitty_cache(
            model_config,
            max_batch_size,
            cache_max_length,
            page_size,
            promote_ratio,
            quest_enabled,
            quest_topk_pages,
            quest_token_budget,
            quest_skip_layers,
            force_sparse_for_equivalence,
        )
        outputs = model.generate(
            input_ids=inputs.input_ids.cuda(),
            attention_mask=inputs.attention_mask.cuda() if "attention_mask" in inputs else None,
            **length_kwargs,
            return_dict_in_generate=False,
            do_sample=False,
            temperature=None, # No temperature for deterministic generation
            top_p=None,       # No top-p sampling
            top_k=None,       # No top-k sampling
            past_key_values=kitty_kv_cache,                   # Use Kitty KV cache
            eos_token_id = None,                                   # Disable early stopping for fair comparison
            disable_compile = True,                                # Disable torch.compile
        )
        kitty_kv_cache = None
    print("Warmup done.")

    # Execution
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(repeat_runs):
        kitty_kv_cache = _new_kitty_cache(
            model_config,
            max_batch_size,
            cache_max_length,
            page_size,
            promote_ratio,
            quest_enabled,
            quest_topk_pages,
            quest_token_budget,
            quest_skip_layers,
            force_sparse_for_equivalence,
        )
        outputs = model.generate(
            input_ids=inputs.input_ids.cuda(),
            attention_mask=inputs.attention_mask.cuda() if "attention_mask" in inputs else None,
            **length_kwargs,
            return_dict_in_generate=False,
            do_sample=False,
            temperature=None, # No temperature for deterministic generation
            top_p=None,       # No top-p sampling
            top_k=None,       # No top-k sampling
            #
            past_key_values=kitty_kv_cache,                   # Use Kitty KV cache
            eos_token_id = None,                                   # Disable early stopping for fair comparison
            disable_compile = True,                                # Disable torch.compile
        )
        kitty_kv_cache = None
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Model.generate() average execution time: { (elapsed_time/repeat_runs):.4f} seconds.")

    # Decode the output sequences
    output_sequences = outputs
    output_texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=False)
    width = shutil.get_terminal_size().columns
    print('-' * width)
    for i, text in enumerate(output_texts):
        if i>=1:
            continue
        print(f"\n======== BATCH {i} OUTPUT ========\n")
        print(text)
    print('-' * width)

    print(f"Model.generate() average execution time: { (elapsed_time/repeat_runs):.4f} seconds.")


def benchmark_fp16_kv(model: PreTrainedModel, tokenizer: AutoTokenizer, inputs: dict, max_seq_len, max_new_tokens: int | None, warmup_runs: int, repeat_runs: int, cache_implementation: str, cache_config: Optional[dict]) -> None:
    length_kwargs = _generation_length_kwargs(max_seq_len, max_new_tokens)
    # Warm up
    for _ in range(warmup_runs):
        outputs = model.generate(
        input_ids=inputs.input_ids.cuda(),
        attention_mask=inputs.attention_mask.cuda() if "attention_mask" in inputs else None,
        **length_kwargs,
        return_dict_in_generate=False,
        do_sample=False,
        temperature=None, # No temperature for deterministic generation
        top_p=None,       # No top-p sampling
        top_k=None,       # No top-k sampling
        #
        cache_implementation=cache_implementation,             # Use static / dynamic / quantized KV cache.
        cache_config=cache_config,
        eos_token_id = None,                                   # Disable early stopping for fair comparison
        disable_compile = True,                                # Disable torch.compile
        )
    print("Warmup done.")
    
    # Execution
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(repeat_runs):
        outputs = model.generate(
        input_ids=inputs.input_ids.cuda(),
        attention_mask=inputs.attention_mask.cuda() if "attention_mask" in inputs else None,
        **length_kwargs,
        return_dict_in_generate=False,
        do_sample=False,
        temperature=None, # No temperature for deterministic generation
        top_p=None,       # No top-p sampling
        top_k=None,       # No top-k sampling
        #
        cache_implementation=cache_implementation,             # Use static / dynamic / quantized KV cache.
        cache_config=cache_config,
        eos_token_id = None,                                   # Disable early stopping for fair comparison
        disable_compile = True,                                # Disable torch.compile
        )

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Model.generate() average execution time: { (elapsed_time/repeat_runs):.4f} seconds.")

    # Decode the output sequences
    output_sequences = outputs
    output_texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=False)
    # Print outputs
    width = shutil.get_terminal_size().columns
    print('-' * width)
    for i, text in enumerate(output_texts):
        if i>=1:
            continue
        print(f"\n======== BATCH {i} OUTPUT ========\n")
        print(text)
    print('-' * width)

    print(f"Model.generate() average execution time: { (elapsed_time/repeat_runs):.4f} seconds.")


def main() -> None:
    args = build_parser().parse_args()
    if args.compare_quest_kitty:
        if args.cache_implementation != 0:
            raise ValueError("--compare-quest-kitty only supports --cache_implementation 0.")
        if args.max_new_tokens is None:
            raise ValueError("--compare-quest-kitty requires --max_new_tokens so decode-only timing is explicit.")
        if args.page_size != 16:
            raise ValueError("--compare-quest-kitty is the page16 speed gate; pass --page_size 16.")
    print("Model: ",args.model)
    #
    prompt_choice = args.prompt_choice
    task_name, prompt = get_prompt(prompt_choice)
    print(f"Task: {task_name}, Prompt: {prompt}")
    #
    prompt = [prompt for _ in range(args.batch_size)]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    #
    APPLY_CHAT_TEMPLATE = True
    if APPLY_CHAT_TEMPLATE:
        messages = [[{"role": "user", "content": p}] for p in prompt]
        texts = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        inputs = tokenizer(text=texts, return_tensors="pt")
    else:
        inputs = tokenizer(text=prompt, return_tensors="pt")
    #
    config = Qwen3Config.from_pretrained(args.model)
    #
    if args.cache_implementation == 0:
        print("Using the customized Qwen3 modeling of Kitty.")
        model_class = Qwen3ForCausalLM_Kitty
    else:
        assert args.cache_implementation in [1,2,3]
        print("Using the original Qwen3 modeling of Huggingface transformers.")
        model_class = Qwen3ForCausalLM

    model = model_class.from_pretrained(
        pretrained_model_name_or_path=args.model,
        config = config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
        device_map='auto'
    )
    model.eval()

    # Benchmarking
    if args.compare_quest_kitty:
        print("Using decode-only comparison: pure Kitty page16 vs QUEST+Kitty page16.")
        benchmark_quest_kitty_comparison(
            model,
            inputs,
            config,
            args.max_seq_len,
            args.warmup_runs,
            args.repeat_runs,
            args.page_size,
            args.promote_ratio,
            args.max_new_tokens,
            quest_topk_pages=args.quest_topk_pages,
            quest_token_budget=args.quest_token_budget,
            quest_skip_layers=args.quest_skip_layers,
            force_sparse_for_equivalence=args.force_sparse_for_equivalence,
        )
    elif args.cache_implementation == 1:
        print("Using FP16 static KV cache implementation of Huggingface transformers.")
        benchmark_fp16_kv(model, tokenizer, inputs, args.max_seq_len, args.max_new_tokens, args.warmup_runs, args.repeat_runs, "static", None)
    elif args.cache_implementation == 2:
        print("Using FP16 dynamic KV cache implementation of Huggingface transformers.")
        benchmark_fp16_kv(model, tokenizer, inputs, args.max_seq_len, args.max_new_tokens, args.warmup_runs, args.repeat_runs, "dynamic", None)
    elif args.cache_implementation == 3:
        print("Using INT4 quantized KV cache implementation with HQQ backend of Huggingface transformers.")
        cache_config = {"backend": "quanto"}
        benchmark_fp16_kv(model, tokenizer, inputs, args.max_seq_len, args.max_new_tokens, args.warmup_runs, args.repeat_runs, "quantized", cache_config)
    else:
        assert args.cache_implementation == 0
        print("Using Kitty KV cache implementation.")
        benchmark_kitty(
            model,
            tokenizer,
            inputs,
            args.max_seq_len,
            config,
            args.warmup_runs,
            args.repeat_runs,
            args.page_size,
            args.promote_ratio,
            max_new_tokens=args.max_new_tokens,
            quest_enabled=args.quest_enabled,
            quest_topk_pages=args.quest_topk_pages,
            quest_token_budget=args.quest_token_budget,
            quest_skip_layers=args.quest_skip_layers,
            force_sparse_for_equivalence=args.force_sparse_for_equivalence,
        )

    #
    model.to("cpu")
    del model


if __name__ == "__main__":
    main()
