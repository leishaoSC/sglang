# Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py
import argparse
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple, TypedDict
import os
from contextlib import nullcontext

import ray
import torch
import triton
from ray.experimental.tqdm_ray import tqdm
from transformers import AutoConfig

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    fused_moe,
    get_config_dtype_str,
    get_config_file_name,
    get_default_config,
    get_moe_configs,
)
from sglang.srt.utils import is_hip

_is_hip_ = is_hip()

padding_size = 128 if bool(int(os.getenv("MOE_PADDING", "0"))) else 0

class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int

def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_fp8: bool,
    block_shape: List[int] = None,
    num_iters: int = 100,
) -> float:
    if use_int4_fp8:
        init_dtype = torch.int8
    else:
        init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    w1_scale_int4 = None
    w2_scale_int4 = None
    if use_int8_w8a16 or use_int8_w8a8:
        w1 = torch.randint(
            -127,
            127,
            (
                num_experts,
                shard_intermediate_size,
                hidden_size,
            ),
            dtype=torch.int8,
        )
        w2 = torch.randint(
            -127,
            127,
            (
                num_experts,
                hidden_size,
                shard_intermediate_size // 2, 
            ),
            dtype=torch.int8,
        )
    elif use_int4_fp8: # new packing: 2 INT4s packed into 1 INT8
        #hard coded w1,w2 for INT4-FP8 new packing case ( int8, //2 in the last dimension)#############################??????
        w1 = torch.randint(
            -127,
            127,
            (
                num_experts,
                shard_intermediate_size,
                hidden_size //2,
            ),
            dtype=torch.int8,
        )
        w2 = torch.randint(
            -127,
            127,
            (
                num_experts,
                hidden_size,
                shard_intermediate_size // 2 //2, 
            ),
            dtype=torch.int8,
        )
        w1_scale_int4 = torch.randn(num_experts, shard_intermediate_size, dtype=torch.float32)
        w2_scale_int4 = torch.randn(num_experts, hidden_size, dtype=torch.float32)
        #print("inside tuning script, ", "w1.shape=", w1.shape, "dtype of w1=", w1.dtype, "w2.shape=", w2.shape)
    else:
        w1 = torch.randn(
            num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype 
        )
        w2 = torch.randn(
            num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype 
        )

    gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    if use_int8_w8a16:
        w1_scale = torch.randn(
            (num_experts, 2 * shard_intermediate_size), dtype=torch.float32
        )
        w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32)
    if use_fp8_w8a8 or use_int8_w8a8:
        if use_int8_w8a8 and block_shape is None:
            w1_scale = torch.randn(
                num_experts, shard_intermediate_size, dtype=torch.float32
            )
            w2_scale = torch.randn(num_experts, hidden_size, dtype=torch.float32)
        elif block_shape is None:
            w1_scale = torch.randn(num_experts, dtype=torch.float32)
            w2_scale = torch.randn(num_experts, dtype=torch.float32)
            a1_scale = torch.randn(1, dtype=torch.float32)
            a2_scale = torch.randn(1, dtype=torch.float32)
        else:
            block_n, block_k = block_shape[0], block_shape[1]
            n_tiles_w1 = (shard_intermediate_size + block_n - 1) // block_n
            n_tiles_w2 = (hidden_size + block_n - 1) // block_n
            k_tiles_w1 = (hidden_size + block_k - 1) // block_k
            k_tiles_w2 = (shard_intermediate_size // 2 + block_k - 1) // block_k
            w1_scale = torch.rand(
                (num_experts, n_tiles_w1, k_tiles_w1), dtype=torch.float32
            )
            w2_scale = torch.rand(
                (num_experts, n_tiles_w2, k_tiles_w2), dtype=torch.float32
            )

    if use_fp8_w8a8 and (not use_int4_fp8):
        w1 = w1.to(torch.float8_e4m3fnuz if _is_hip_ else torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fnuz if _is_hip_ else torch.float8_e4m3fn)

    input_gating = torch.empty(num_tokens, num_experts, dtype=torch.float32)

    def prepare(i: int):
        input_gating.copy_(gating_output[i])

    def run():
        from sglang.srt.layers.moe.fused_moe_triton import override_config

        with override_config(config):
            fused_moe(
                x,
                w1,
                w2,
                input_gating,
                topk,
                renormalize=True,
                inplace=True,
                use_fp8_w8a8=use_fp8_w8a8,
                #use_int8_w8a8=use_int8_w8a8, #not in int4 triton code
                use_int8_w8a16=use_int8_w8a16,
                use_int4_fp8=use_int4_fp8, ##############
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_scale_int4=w1_scale_int4,##############
                w2_scale_int4=w2_scale_int4,##############
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                block_shape=block_shape,
            )

    # JIT compilation & warmup
    run()
    torch.cuda.synchronize()

    # Capture 10 invocations with CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies: List[float] = []
    for i in range(num_iters):
        prepare(i)
        torch.cuda.synchronize()

        start_event.record()
        graph.replay()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    graph.reset()
    return avg


def get_rocm_configs_compute_bound() -> List[Dict[str, int]]:
    configs: List[BenchmarkConfig] = []
    waves_per_eu_range = 0
    for num_stages in [2]:
        for block_m in [32, 64, 128, 256]:
            for block_k in [32, 64, 128, 256]:
                for block_n in [16, 32, 64, 128, 256]:
                    for num_warps in [1, 2, 4, 8]:
                        for group_size in [1, 4, 8, 16, 32]:
                                configs.append(
                                    {
                                        "BLOCK_SIZE_M": block_m,
                                        "BLOCK_SIZE_N": block_n,
                                        "BLOCK_SIZE_K": block_k,
                                        "GROUP_SIZE_M": group_size,
                                        "num_warps": num_warps,
                                        "num_stages": num_stages,
                                        "waves_per_eu": waves_per_eu_range,
                                    }
                                )
    return configs


def get_configs_compute_bound() -> List[Dict[str, int]]:
    # Reduced search space for faster tuning.
    # TODO(woosuk): Increase the search space and use a performance model to
    # prune the search space.
    configs: List[BenchmarkConfig] = []
    if _is_hip_:
        configs = get_rocm_configs_compute_bound()
    else:
        for num_stages in [2, 3, 4, 5]:
            for block_m in [16, 32, 64, 128, 256]:
                for block_k in [64, 128, 256]:
                    for block_n in [32, 64, 128, 256]:
                        for num_warps in [4, 8]:
                            for group_size in [1, 16, 32, 64]:
                                configs.append(
                                    {
                                        "BLOCK_SIZE_M": block_m,
                                        "BLOCK_SIZE_N": block_n,
                                        "BLOCK_SIZE_K": block_k,
                                        "GROUP_SIZE_M": group_size,
                                        "num_warps": num_warps,
                                        "num_stages": num_stages,
                                    }
                                )
    return configs

def prune_rocm_search_space(num_tokens, shard_intermediate_size, hidden_size,
                            search_space, is_fp16, topk):
    N1, K1 = shard_intermediate_size, hidden_size
    N2, K2 = hidden_size, shard_intermediate_size // 2
    pruned_space_1 = prune_rocm_configs(num_tokens * topk, N1, K1,
                                        search_space, is_fp16)
    pruned_space_2 = prune_rocm_configs(num_tokens * topk, N2, K2,
                                        search_space, is_fp16)
    search_space = merge_unique_dicts(pruned_space_1, pruned_space_2)
    return search_space


# The following code is inspired by ROCm/Triton GEMM tuning script:
# https://github.com/ROCm/triton/blob/triton-mlir/scripts/amd/gemm/tune_gemm.py#L89
def prune_rocm_configs(M, N, K, configs, is_fp16=True):
    pruned_configs = []
    elemBytes_a = 2 if is_fp16 else 1
    elemBytes_b = 2 if is_fp16 else 1

    mfma = 16 if M < 32 or N < 32 else 32

    # TODO (zhanglx): figure out the boundary between large and small gemms
    large_gemm = False
    if M >= 2048 and N >= 2048:
        large_gemm = True

    for config in configs:
        BLOCK_SIZE_M = config.get("BLOCK_SIZE_M")
        BLOCK_SIZE_N = config.get("BLOCK_SIZE_N")
        BLOCK_SIZE_K = config.get("BLOCK_SIZE_K")
        num_warps = config.get("num_warps")

        if is_fp16:
            matrix_instr_nonkdim = config.get("matrix_instr_nonkdim")
            if matrix_instr_nonkdim > mfma:
                continue
        if mfma == 4 and BLOCK_SIZE_K < 64:
            continue
        # some layouts could not work properly in case
        # number elements per thread is less 1
        if BLOCK_SIZE_M * BLOCK_SIZE_N < 64:
            continue
        SPLIT_K = 1 #config.get("SPLIT_K", 1)
        GROUP_M = config.get("GROUP_SIZE_M")
        if is_fp16:
            if (matrix_instr_nonkdim > BLOCK_SIZE_M
                    or matrix_instr_nonkdim > BLOCK_SIZE_N):
                continue
            if (matrix_instr_nonkdim >= M
                    and matrix_instr_nonkdim != BLOCK_SIZE_M):
                continue
            if (matrix_instr_nonkdim >= N
                    and matrix_instr_nonkdim != BLOCK_SIZE_N):
                continue
        # Skip BLOCK_SIZE that is too large compare to M/N
        # unless BLOCK_SIZE is already small enough
        if M * 2 < BLOCK_SIZE_M and BLOCK_SIZE_M != 16:
            continue
        if N * 2 < BLOCK_SIZE_N and BLOCK_SIZE_N != 16:
            continue
        # skip large split_k when not necessary
        if SPLIT_K != 1 and not need_split_k(M, N, K):
            continue
        # skip split_k that leads to EVEN_K = false
        leap = SPLIT_K * BLOCK_SIZE_K
        modv = K % leap
        if modv != 0:
            continue
        # skip large GROUP_M
        if GROUP_M * BLOCK_SIZE_M > M and GROUP_M != 1:
            continue
        # out of shared memory resource
        # TODO (zhanglx): This does not consider the LDS usage in the epilogue
        LDS = (BLOCK_SIZE_K * BLOCK_SIZE_M * elemBytes_a +
               BLOCK_SIZE_K * BLOCK_SIZE_N * elemBytes_b)
        if LDS > 65536:
            continue
        # Skip small block sizes and num_warps for large gemm
        # For fp16 and f8, we want to only use BLOCK_SIZE >= 64
        if large_gemm:
            if BLOCK_SIZE_M < 64 or BLOCK_SIZE_N < 64:
                continue
            if BLOCK_SIZE_K < 64:
                continue
            if num_warps < 4:
                continue

        pruned_configs.append(config)

    return pruned_configs


def need_split_k(SIZE_M, SIZE_N, SIZE_K):
    return (SIZE_M < 64 or SIZE_N < 64) and SIZE_K > 1024


def merge_unique_dicts(list1, list2):
    result = []
    combined_list = list1.copy()
    combined_list.extend(list2)
    for dictionary in combined_list:
        if dictionary not in result:
            result.append(dictionary)
    return result

@ray.remote(num_gpus=1)
class BenchmarkWorker:

    def __init__(self, seed: int) -> None:
        torch.set_default_device("cuda")
        torch.cuda.manual_seed_all(0)
        self.seed = seed
        # Get the device ID to allocate tensors and kernels
        # on the respective GPU. This is required for Ray to work
        # correctly with multi-GPU tuning on the ROCm platform.
        self.device_id = int(ray.get_gpu_ids()[0])

    def benchmark(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int4_fp8: bool, ##############
        block_shape: List[int],
    ) -> Tuple[Dict[str, int], float]:
        torch.cuda.manual_seed_all(0)
        dtype_str = get_config_dtype_str(
            dtype, use_int8_w8a16=use_int8_w8a16, use_fp8_w8a8=use_fp8_w8a8
        )
        # NOTE(woosuk): The current naming convention uses w2.shape[2], which
        # is the intermediate size after silu_and_mul.
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        op_config = get_moe_configs(
            num_experts, shard_intermediate_size // 2, dtype_str, block_n, block_k
        )
        if op_config is None:
            config = get_default_config(
                num_tokens,
                num_experts,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype_str,
                False,
            )
        else:
            config = op_config[min(op_config.keys(), key=lambda x: abs(x - num_tokens))]
        kernel_time = benchmark_config(
            config,
            num_tokens,
            num_experts,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_fp8, ##############
            block_shape,
        )
        return config, kernel_time

    def tune(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int4_fp8: bool, ##############
        block_shape: List[int],
        search_space: List[Dict[str, int]],
    ) -> Dict[str, int]:
        best_config = None
        best_time = float("inf")
        if _is_hip_:
            print("######################len of search space before pruning=", len(search_space))
            is_fp16 = not (use_fp8_w8a8 or use_int8_w8a16)
            search_space = prune_rocm_search_space(num_tokens,
                                                    shard_intermediate_size,
                                                    hidden_size, search_space,
                                                    is_fp16, topk)
            print("######################len of search space after pruning=", len(search_space))

        with torch.cuda.device(self.device_id)  if _is_hip_ else nullcontext():
            for config in tqdm(search_space):
                try:
                    kernel_time = benchmark_config(
                        config,
                        num_tokens,
                        num_experts,
                        shard_intermediate_size,
                        hidden_size,
                        topk,
                        dtype,
                        use_fp8_w8a8,
                        use_int8_w8a8,
                        use_int8_w8a16,
                        use_int4_fp8, ##############
                        block_shape,
                        num_iters=10,
                    )
                except triton.runtime.autotuner.OutOfResources:
                    # Some configurations may be invalid and fail to compile.
                    continue

                if kernel_time < best_time:
                    best_time = kernel_time
                    best_config = config
        now = datetime.now()
        print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")
        assert best_config is not None
        return best_config


def sort_config(config: BenchmarkConfig) -> BenchmarkConfig:
    return {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": config["GROUP_SIZE_M"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
        **(
            {"waves_per_eu": config["waves_per_eu"]} if "waves_per_eu" in config else {}
        ),
    }


def save_configs(
    configs: Dict[int, BenchmarkConfig],
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_fp8: bool, ##############
    block_shape: List[int],
) -> None:
    dtype_str = get_config_dtype_str(
        dtype,
        use_int8_w8a16=use_int8_w8a16,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
    )

    # NOTE(woosuk): The current naming convention uses w2.shape[2], which
    # is the intermediate size after silu_and_mul.
    filename = get_config_file_name(
        num_experts,
        shard_intermediate_size // 2,
        dtype_str,
        block_shape,
    )

    print(f"Writing best config to {filename}...")
    with open(filename, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")


def main(args: argparse.Namespace):
    print(args)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    print("#######################config.architectures[0]=", config.architectures[0])
    if config.architectures[0] == "DbrxForCausalLM":
        E = config.ffn_config.moe_num_experts
        topk = config.ffn_config.moe_top_k
        intermediate_size = config.ffn_config.ffn_hidden_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] == "JambaForCausalLM":
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] == "Qwen2MoeForCausalLM":
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] in ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"]:
        E = config.n_routed_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] in [
        "Grok1ForCausalLM",
        "Grok1ImgGen",
        "Grok1ModelForCausalLM", 
    ]:
        E = config.num_experts if config.architectures[0] == "Grok1ModelForCausalLM" else config.num_local_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
        print("############grok1: E=", E, "topk=", topk, "intermediate_size=", intermediate_size, "shard_intermediate_size=", shard_intermediate_size, "hidden_size=", config.hidden_size )
    else:
        # Default: Mixtral
        E = config.num_local_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
        print("############mixtral: E=", E, "topk=", topk, "intermediate_size=", intermediate_size, "shard_intermediate_size=", shard_intermediate_size)

    hidden_size = config.hidden_size
    dtype = config.torch_dtype
    use_fp8_w8a8 = args.dtype == "fp8_w8a8"
    use_int8_w8a8 = args.dtype == "int8_w8a8"
    use_int8_w8a16 = args.dtype == "int8_w8a16"
    use_int4_fp8 = args.use_int4_fp8
    block_shape = None
    if (
        hasattr(config, "quantization_config")
        and "weight_block_size" in config.quantization_config
    ):
        block_shape = config.quantization_config["weight_block_size"]
        assert len(block_shape) == 2

    if args.batch_size is None:
        # batch_sizes = [
        #     1,
        #     2,
        #     4,
        #     8,
        #     16,
        #     24,
        #     32,
        #     48,
        #     64,
        #     96,
        #     128,
        #     256,
        #     512,
        #     1024,
        #     1536,
        #     2048,
        #     3072,
        #     4096,
        # ]
        batch_sizes = [
            #4, #other left after pruning
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
            131072,
        ]
    else:
        batch_sizes = [args.batch_size]

    ray.init()
    num_gpus = 1 #int(ray.available_resources()["GPU"]) #force to use 1gpu, as multi-gpu ray still has load imbalance issue
    workers = [BenchmarkWorker.remote(args.seed) for _ in range(num_gpus)]
    print("#############################################num_gpus=",  num_gpus)
    print("############################################workers are", workers)
    print("#################################################prune search space")

    def _distribute(method: str, inputs: List[Any]) -> List[Any]:
        outputs = []
        worker_idx = 0
        for input_args in inputs:
            worker = workers[worker_idx]
            worker_method = getattr(worker, method)
            output = worker_method.remote(*input_args)
            outputs.append(output)
            worker_idx = (worker_idx + 1) % num_gpus
        return ray.get(outputs)

    if args.tune:
        search_space = get_configs_compute_bound()
        if block_shape is not None:
            block_n, block_k = block_shape[0], block_shape[1]
            search_space = [
                config
                for config in search_space
                if block_k % config["BLOCK_SIZE_K"] == 0
            ]
        print(f"Start tuning over {len(search_space)} configurations...")

        start = time.time()
        configs = _distribute(
            "tune",
            [
                (
                    batch_size,
                    E,
                    shard_intermediate_size,
                    hidden_size,
                    topk,
                    dtype,
                    use_fp8_w8a8,
                    use_int8_w8a8,
                    use_int8_w8a16,
                    use_int4_fp8, ##############
                    block_shape,
                    search_space,
                )
                for batch_size in batch_sizes
            ],
        )
        best_configs = {
            M: sort_config(config) for M, config in zip(batch_sizes, configs)
        }
        save_configs(
            best_configs,
            E,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_fp8, ##############
            block_shape,
        )
        end = time.time()
        print(f"Tuning took {end - start:.2f} seconds")
    else:
        outputs = _distribute(
            "benchmark",
            [
                (
                    batch_size,
                    E,
                    shard_intermediate_size,
                    hidden_size,
                    topk,
                    dtype,
                    use_fp8_w8a8,
                    use_int8_w8a8,
                    use_int8_w8a16,
                    use_int4_fp8, ##############
                    block_shape,
                )
                for batch_size in batch_sizes
            ],
        )

        for batch_size, (config, kernel_time) in zip(batch_sizes, outputs):
            print(f"Batch size: {batch_size}, config: {config}")
            print(f"Kernel time: {kernel_time:.2f} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    parser.add_argument("--tp-size", "-tp", type=int, default=2)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "fp8_w8a8", "int8_w8a16", "int8_w8a8"],
        default="auto",
    )
    parser.add_argument("--use_int4_fp8", action="store_true")  #int4 weights and fp8 activation. Must have fp8_w8a8 in --dtype also
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()

    main(args)
