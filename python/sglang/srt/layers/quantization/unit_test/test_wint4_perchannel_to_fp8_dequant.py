import math
import pytest
import torch
import triton
import triton.language as tl

################################################################################
# Detect AMD ROCm vs. CPU fallback
################################################################################
if torch.version.hip is not None and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

################################################################################
# Triton Kernel: INT4 -> float16 per-channel dequantization
#                then you can cast float16 -> float8 in Python
################################################################################

@triton.jit
def wint4_perchannel_to_fp16_dequantize_kernel(
    qweight_ptr,    # [K, M//8], each int32 stores 8 nibbles
    scales_ptr,     # [K], float32
    result_ptr,     # [K, M], float16
    BLOCK_SIZE: tl.constexpr,
    K_SIZE: tl.constexpr,
):
    """
    Each Triton program processes one row (pid) of the QWeight matrix:
      1. Loads the row's int32 values (each containing 8 nibble-weights).
      2. Expands them into 4-bit values and reorders the nibbles.
      3. Applies a per-channel scale (one float per row).
      4. Stores float16 results into 'result_ptr'.
    """

    # pid = which row of [K, M//8] this program handles
    pid = tl.program_id(0)

    # Compute 1D offsets for the int32 row
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_elements = K_SIZE * BLOCK_SIZE  # K * (M//8)
    mask_offsets = offsets < total_elements

    # Compute 1D offsets for the output row
    result_offsets = pid * BLOCK_SIZE * 8 + tl.arange(0, BLOCK_SIZE * 8)
    total_out_elems = K_SIZE * BLOCK_SIZE * 8
    mask_result = result_offsets < total_out_elems

    # 1) Load the int32 data
    iweights = tl.load(qweight_ptr + offsets, mask=mask_offsets, other=0)
    # iweights shape: [BLOCK_SIZE]
    iweights = tl.reshape(iweights, (BLOCK_SIZE, 1))

    # 2) Expand int32 -> 8 nibbles via repeated `tl.interleave`
    iweights = tl.interleave(iweights, iweights)  # -> [BLOCK_SIZE, 2]
    iweights = tl.interleave(iweights, iweights)  # -> [BLOCK_SIZE, 4]
    iweights = tl.interleave(iweights, iweights)  # -> [BLOCK_SIZE, 8]

    # 3) Reorder nibble pattern from [0,2,4,6,1,3,5,7] to [0,4,1,5,2,6,3,7]
    reverse_order_tensor = (
        (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
    ).reshape(8)  # shape: [8]
    shifts = reverse_order_tensor * 4  # shape: [8]
    # Broadcast to [BLOCK_SIZE, 8]
    shifts = shifts + tl.zeros([BLOCK_SIZE, 8], dtype=tl.int32)
    # Extract the nibbles
    iweights = (iweights >> shifts) & 0xF  # shape: [BLOCK_SIZE, 8]

    # 4) Load the per-channel float32 scale
    scale_val = tl.load(scales_ptr + pid)  # shape: scalar
    # Broadcast scale to [BLOCK_SIZE, 8]
    scales = scale_val + tl.zeros([BLOCK_SIZE, 8], dtype=tl.float32)

    # 5) Multiply nibble-weights by scale, cast to float16
    iweights = (iweights * scales).to(tl.float16)

    # 6) Flatten from [BLOCK_SIZE, 8] -> [BLOCK_SIZE * 8]
    iweights = tl.reshape(iweights, [BLOCK_SIZE * 8])

    # 7) Store results
    tl.store(result_ptr + result_offsets, iweights, mask=mask_result)


def wint4_perchannel_to_fp8_dequantize_triton(qweight: torch.Tensor,
                                              weight_scale: torch.Tensor
                                              ) -> torch.Tensor:
    """
    1) Launch the Triton kernel to dequantize from INT4 -> float16.
    2) Then cast the result to float8 in Python code.
    """
    K = qweight.shape[0]
    M_8 = qweight.shape[1]            # M//8
    M = M_8 * 8

    # Temporary float16 buffer
    tmp_result = torch.zeros((K, M), device=qweight.device, dtype=torch.float16)

    # We'll pick a BLOCK_SIZE = M_8 for best coverage
    # or keep it as a small debug choice if needed.
    # For demonstration, we set BLOCK_SIZE=2 in code. 
    BLOCK_SIZE = M_8  # or keep a small debug value if you prefer

    # Launch the Triton kernel
    grid = (K,)
    wint4_perchannel_to_fp16_dequantize_kernel[grid](
        qweight,
        weight_scale,
        tmp_result,
        BLOCK_SIZE=BLOCK_SIZE,
        K_SIZE=K
    )

    # Convert float16 to float8 in Python
    out_fp8 = tmp_result.to(torch.float8_e4m3fn)
    return out_fp8

################################################################################
# Reference Implementation (PyTorch-based)
################################################################################

def reverse_4bit_order(x: torch.Tensor) -> torch.Tensor:
    """
    Reverse nibble blocks [0,4,1,5,2,6,3,7] in groups of 8 for AMD 4-bit layout.
    """
    AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
    orig_shape = x.shape
    n_nibbles = x.shape[-1]
    assert n_nibbles % 8 == 0, f"Expected multiple of 8 nibbles, got {n_nibbles}"

    x = x.view(*x.shape[:-1], n_nibbles // 8, 8)
    x = x[..., :, AWQ_REVERSE_ORDER]
    x = x.view(*orig_shape)
    return x


def wint4_perchannel_to_fp8_dequantize_torch(qweight: torch.Tensor,
                                             weight_scale: torch.Tensor
                                             ) -> torch.Tensor:
    """
    Reference CPU/GPU PyTorch path:
      1) Shift out each nibble from int32.
      2) Reverse nibble order.
      3) Multiply by per-channel scale -> float.
      4) Cast float -> float16 -> float8.
    """
    K, col_32 = qweight.shape
    M = col_32 * 8

    bits = 4
    shifts = torch.arange(0, 32, bits, device=qweight.device, dtype=torch.int32)
    shifts = shifts.view(1, -1)

    # 1) Unpack each int32 to 8 nibbles
    iweights = (qweight.unsqueeze(-1) >> shifts) & 0xF
    iweights = iweights.view(K, M)

    # 2) Reverse nibble order
    iweights = iweights.view(K, M // 8, 8)
    iweights = reverse_4bit_order(iweights)
    iweights = iweights.view(K, M)

    # 3) Multiply per-channel scale -> float
    iweights = iweights.float() * weight_scale.view(K, 1)

    # 4) Float -> float16 -> float8
    iweights = iweights.to(torch.float16).to(torch.float8_e4m3fn)
    return iweights


################################################################################
# Pytest Test
################################################################################

@pytest.mark.parametrize("K", [4, 16, 32])
@pytest.mark.parametrize("M", [8, 16, 24, 32])  # must be multiple of 8
def test_wint4_perchannel_to_fp8_dequantize(K, M):
    """
    Compares the Triton kernel vs. a reference PyTorch function for
    INT4 -> float8 (e4m3) per-channel dequantization (on AMD GPU if available).
    """
    if device == "cpu":
        pytest.skip("No AMD GPU found. Skipping test on CPU-only environment.")

    assert M % 8 == 0, "M must be multiple of 8."

    # qweight: [K, M//8]
    qweight_cols = M // 8
    qweight = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (K, qweight_cols),
        device=device,
        dtype=torch.int32
    )

    # scales: [K], using all ones here for consistent results
    weight_scale = torch.ones((K,), dtype=torch.float32, device=device)

    print("qweight shape:", qweight.shape)
    print("weight_scale shape:", weight_scale.shape)

    # Triton path
    out_triton = wint4_perchannel_to_fp8_dequantize_triton(qweight, weight_scale)

    # Reference path
    out_ref = wint4_perchannel_to_fp8_dequantize_torch(qweight, weight_scale)

    # Check for NaNs
    assert not torch.any(torch.isnan(out_triton)), "Triton result has NaNs!"
    assert not torch.any(torch.isnan(out_ref)),    "Reference result has NaNs!"

    # Compare in float32 (to avoid float8 1-bit differences)
    torch.testing.assert_close(
        out_triton.cpu().float(),
        out_ref.cpu().float(),
        atol=1e-1,
        rtol=1e-1
    )
