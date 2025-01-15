#    packed_4_bits (pack)   = [0, 2, 4, 6, 1, 3, 5, 7]
#                  (unpack) = [0, 4, 1, 5, 2, 6, 3, 7]
import torch
import triton
import triton.language as tl

@triton.jit
def wint4_perchannel_to_fp8_dequantize_kernel(
    qweight_ptr,    # [K, M//8], each int32 has 8 nibbles,  quantized matrix
    scales_ptr,     # [K], float32,  scales, per channel
    result_ptr,     # [K, M], FP8,  Output matrix
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
    P2: tl.constexpr,
):
    # Each program handles 1 row
    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, P2)
    offsets = block_start + arange
    mask_offsets = arange < BLOCK_SIZE

    # Offsets in the output
    result_block_start = pid * BLOCK_SIZE * 8
    result_arange = tl.arange(0, P2 * 8)
    result_offsets = result_block_start + result_arange
    mask_result = result_arange < BLOCK_SIZE * 8

    # Load the int32 packed 4-bit
    iweights = tl.load(qweight_ptr + offsets, mask=mask_offsets, other=0.0) # (BLOCK_SIZE)
    iweights = tl.reshape(iweights, (P2, 1)) # to handle the cases where BLOCK_SIZE is not power of 2.

    # Expand from int32 -> 8 nibbles with repeated tl.interleave calls
    iweights = tl.interleave(iweights, iweights)     # (BLOCK_SIZE, 1) -> (BLOCK_SIZE, 2)
    iweights = tl.interleave(iweights, iweights)     # (BLOCK_SIZE, 2) -> (BLOCK_SIZE, 4)
    iweights = tl.interleave(iweights, iweights)     # (BLOCK_SIZE, 4) -> (BLOCK_SIZE, 8)

    # Reverse nibble order: [0,4,1,5,2,6,3,7]
    reverse_order_tensor = (
        (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
    ).reshape(8)
    shifts = reverse_order_tensor * 4  # (8)

    # Broadcast shifts
    shifts = shifts + tl.zeros([P2, 8], dtype=shifts.dtype) # to handle the cases where BLOCK_SIZE is not power of 2.

    # Extract nibbles
    iweights = (iweights >> shifts) & 0xF    # (BLOCK_SIZE, 8), this is unsigned [0..15]

    # Sign-extend nibble => range [-8..7]
    iweights = tl.where(iweights >= 8, iweights - 16, iweights)
    """
    0x0..0xF [0..15]
    
    int32 --> uint4 --> int4* [7..0..-8]
    0x0 --> 0           (0)
    0x1 --> 1
    .
    .
    0x7 --> 7           (7)
    0x8 --> 8           (-8)
    ---------
    0x9 --> 9           (-7)
    0xA --> 10          (-6)
    0xB --> 11
    0xC --> 12
    0xD --> 13
    0xE --> 14
    0xF --> 15          (-1)
    """
    # Load the per-channel scale (float32)
    scale_offsets = pid
    scales = tl.load(scales_ptr + scale_offsets) # every row (pid) has one float32 (1 element vector)
    scales = scales + tl.zeros([P2, 8], dtype=tl.float32)    # to handle the cases where BLOCK_SIZE is not power of 2.

    # Multiply
    iweights = iweights * scales   # symmetric, so no zeros
    iweights = iweights.to(result_ptr.type.element_ty)  # Ref: https://github.com/triton-lang/triton/blob/main/python/triton/runtime/jit.py#L409-L410 (tl.float8e4b8)

    # Flatten from [BLOCK_SIZE, 8] => [BLOCK_SIZE*8]
    iweights = tl.reshape(iweights, [P2 * 8])  # to handle the cases where BLOCK_SIZE is not power of 2.

    # Store to result
    tl.store(result_ptr + result_offsets, iweights, mask=mask_result)


def wint4_perchannel_to_fp8_dequantize_triton(
    qweight: torch.Tensor,  # [K, M//8], pack 8 INT4s into one int32
    weight_scale1: torch.Tensor  # [K], float32, (scaling factor for per-channel INT4)
) -> torch.Tensor:

    K = qweight.shape[0]
    M_8 = qweight.shape[1]  # M//8

    M = M_8 * 8

    assert weight_scale1.shape[0] == K 

    # Result tensor:
    # number of rows = same as input tensor
    # number of cols = 8 x input tensor num cols
    result = torch.zeros((K, M), device=qweight.device, dtype=torch.float8_e4m3fnuz)  # Ref: https://github.com/triton-lang/triton/blob/main/python/triton/runtime/jit.py#L409-L410

    # Next power of two above M_8
    P2 = triton.next_power_of_2(M_8) # the smallest power of 2 that’s ≥ M_8

    # Launch the Triton kernel
    grid = (K,)
    wint4_perchannel_to_fp8_dequantize_kernel[grid](
        qweight,
        weight_scale1,
        result,
        BLOCK_SIZE=M_8,
        K=K,
        P2=P2
    )
    return result