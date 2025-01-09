import torch
import triton
import triton.language as tl
import math

#This code is adapted from https://github.com/ROCm/vllm/blob/main/vllm/model_executor/layers/quantization/awq_triton.py

#zeros are ignored since we use symmetric quantization

@triton.jit
def wint4_perchannel_to_fp8_dequantize_kernel(
        qweight_ptr,  # quantized matrix
        scales_ptr,  # scales, per channel
        result_ptr,  # Output matrix
        BLOCK_SIZE: tl.constexpr,
        P2: tl.constexpr,
        ):
    # Setup the pids.
    pid = tl.program_id(0)

    # Compute offsets and masks for qweight_ptr.
    offsets = pid * BLOCK_SIZE + tl.arange(0, P2)
    masks = offsets < BLOCK_SIZE

    # Compute offsets and masks for result output ptr.
    result_offsets = pid * BLOCK_SIZE * 8 + tl.arange(
        0, P2 * 8)

    result_masks = result_offsets < BLOCK_SIZE * 8

    # Load the weights.
    iweights = tl.load(qweight_ptr + offsets, masks, 0.0)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)

    # Create reverse  order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    reverse_order_tensor = ((tl.arange(0, 2) * 4)[None, :] +
                                tl.arange(0, 4)[:, None]).reshape(8)

    # Use this to compute a set of shifts that can be used to unpack and
    # reorder the values in iweights and zeros.
    shifts = reverse_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE, 8))
    shifts = tl.reshape(shifts, (1, BLOCK_SIZE * 8))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    iweights = (iweights >> shifts) & 0xF


    # Compute scale offsets and masks.
    scale_offsets = (pid * BLOCK_SIZE * 8 +
                       tl.arange(0, P2 * 8))
    scale_masks = scale_offsets < BLOCK_SIZE * 8

    # Load the scales.
    scales = tl.load(scales_ptr + scale_offsets, scale_masks, 0.0)
    scales = tl.broadcast_to(scales, (1, BLOCK_SIZE * 8))

    # Dequantize.
    iweights = iweights  * scales
    iweights = iweights.to(result_ptr.type.element_ty)

    # Finally, store.
    tl.store(result_ptr + result_offsets, iweights, result_masks)




# qweights - [K, M // 8], int32
# weight_scale1 - [K ], float32 (scaling factor for per-channel INT4)

def wint4_perchannel_to_fp8_dequantize_triton(qweight: torch.Tensor,
                                            weight_scale1: torch.Tensor,
) -> torch.Tensor:
    
    K = qweight.shape[0]

    assert K > 0 
    assert weight_scale1.shape[0] == K 

    # Result tensor:
    # number of rows = same as input tensor
    # number of cols = 8 x input tensor num cols
    result = torch.empty(qweight.shape[0],
                         qweight.shape[1] * 8,
                         device=qweight.device,
                         dtype=torch.float8_e4m3fn)
    
    P2 = int(2 ** (math.ceil(math.log2(qweight.shape[1]))))

    grid = lambda META: (qweight.shape[0],)

    wint4_perchannel_to_fp8_dequantize_kernel[grid](qweight,
                                weight_scale1,
                                result,
                                BLOCK_SIZE=qweight.shape[1],
                                P2=P2)

    return result


