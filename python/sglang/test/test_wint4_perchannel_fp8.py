#    packed_4_bits (pack)   = [0, 2, 4, 6, 1, 3, 5, 7]
#                  (unpack) = [0, 4, 1, 5, 2, 6, 3, 7]

import math
import pytest
import torch
from sglang.srt.layers.quantization.w_int4_perchannel_to_fp8_dequant_triton import wint4_perchannel_to_fp8_dequantize_triton
################################################################################
# Detect AMD ROCm device vs CPU fallback
################################################################################

if torch.version.hip is not None and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

################################################################################
# Reference (also does float16 -> float8)
################################################################################

def reverse_4bit_order(x: torch.Tensor) -> torch.Tensor:
    """Reorder nibble blocks [0,4,1,5,2,6,3,7] in groups of 8."""
    AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
    orig_shape = x.shape
    n_nibbles = x.shape[-1]
    assert n_nibbles % 8 == 0, f"Expected multiple of 8 nibbles, got {n_nibbles}"

    x = x.view(*x.shape[:-1], n_nibbles // 8, 8)
    x = x[..., :, AWQ_REVERSE_ORDER]
    x = x.view(*orig_shape)
    return x



def wint4_perchannel_to_fp8_dequantize_torch(
    qweight: torch.Tensor,       # [K, M//8], int32,   M // 8 --> 2      M = 1024 // 8 = 128   (it will fail BLOCK_SIZE < M // 8)
    weight_scale1: torch.Tensor  # [K], float32
) -> torch.Tensor:
    """Reference: do everything in float16, then cast to float8."""
    K, col_32 = qweight.shape
    M = col_32 * 8

    # 1) Unpack each int32 to 8 nibbles
    bits = 4
    shifts = torch.arange(0, 32, bits, device=qweight.device, dtype=torch.int32)
    shifts = shifts.view(1, -1)

    iweights = (qweight.unsqueeze(-1) >> shifts).to(torch.int8) 
    iweights = iweights.view(K, M)
    

    # 2) Reverse nibble order
    iweights = reverse_4bit_order(iweights)
    iweights = iweights.view(K, M)
    iweights = iweights & 0xF

    # make sure negative number of int4 is handled correctly
    # Use -3 as an example, we have to restore 00001101 to 11111101, so we can check the fourth digit of the unzipped number,
    # and if the fourth digit == 1 it proves that the number is negative
    mask = (iweights & 0x08).bool()
    iweights[mask] = iweights[mask] | 0xF0
    

    # 3) Multiply by scale => float
    iweights = iweights.float() * weight_scale1.view(K, 1)

    # 4) Convert to float16 first, then cast to float8
    iweights = iweights.to(torch.float8_e4m3fnuz)

    return iweights

################################################################################
# Pytest Test
################################################################################
@pytest.mark.parametrize("K", [32768, 45056, 65536])
@pytest.mark.parametrize("M", [6144, 11264, 16384])
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

    scale: [K]
    weight_scale1 = torch.rand(
        (K,),
        dtype=torch.float32,
        device=device
    ) 

    print("qweight shape:", qweight.shape)
    print("weight_scale1 shape:", weight_scale1.shape)
    out_triton = wint4_perchannel_to_fp8_dequantize_triton(qweight, weight_scale1)
    out_ref = wint4_perchannel_to_fp8_dequantize_torch(qweight, weight_scale1)

    # Check for NaN/Inf (isinf not implemented for 'Float8_e4m3fn')
    assert not torch.any(torch.isnan(out_triton)), "Triton result has NaNs!"
    assert not torch.any(torch.isnan(out_ref)),    "Reference result has NaNs!"

    # Float8 is low precision => use a looser tolerance
    # torch.testing.assert_close(out_triton.cpu(), out_ref.cpu(), atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(
        out_triton.cpu().float(),  # Convert float8 -> float32
        out_ref.cpu().float(),
        atol=1e-1,
        rtol=1e-1)

