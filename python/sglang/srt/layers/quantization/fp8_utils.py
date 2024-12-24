from typing import Optional, Tuple

import torch
from vllm import _custom_ops as ops

def normalize_e4m3fn_to_e4m3fnuz(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    assert weight.dtype == torch.float8_e4m3fn
    # The bits pattern 10000000(-128) represents zero in e4m3fn
    # but NaN in e4m3fnuz. So here we set it to 0.
    # https://onnx.ai/onnx/technical/float8.html
    weight_as_int8 = weight.view(torch.int8)
    ROCM_FP8_NAN_AS_INT = -128
    weight_as_int8[weight_as_int8 == ROCM_FP8_NAN_AS_INT] = 0
    weight = weight_as_int8.view(torch.float8_e4m3fnuz)

    # For the same bits representation, e4m3fnuz value is half of
    # the e4m3fn value, so we should double the scaling factor to
    # get the same dequantized value.
    # https://onnx.ai/onnx/technical/float8.html
    weight_scale = weight_scale * 2.0
    if input_scale is not None:
        input_scale = input_scale * 2.0
    return weight, weight_scale, input_scale

#Utility function to compute the correct scaling numbers given weight tensor in BF16 or FP16 data type
def compute_fp8_int4_scaling_numbers(
        weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert weight.dtype in [torch.bfloat16, torch.float16]
    qfp8weight, fp8weight_scale = ops.scaled_fp8_quant(weight, scale=None)
    int4_amax = 8
    bias = 8
    int4weight_scales = (torch.max(torch.abs(qfp8weight), dim=1, keepdim=True)[0] / int4_amax) 
    qfp8weight.div_(int4weight_scales)
    qfp8weight.round_()
    qfp8weight.add_(bias)
    qfp8weight.clamp_(-int4_amax + bias, int4_amax -1  + bias) #[0, 15]
    qfp8weight = qfp8weight.to(torch.uint8)
    ###NOTE: need to use torch.uint8 instead of torch.int8; otherwise, for negative numbers, 0xF or <<4 will produce unexpected result or undefined behavior!!
    # pack INT4 values into bytes
    qint4weight = ((qfp8weight[:, ::2] & 0xF) << 4) | (qfp8weight[:, 1::2] & 0xF) 
    return qint4weight, fp8weight_scale, int4weight_scales

def upcast_int4_weight_into_fp8(
    qint4weight: torch.Tensor,
    int4weight_scales: torch.Tensor,
    fp8weight_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    M, N = qint4weight.shape
    bias = 8 #for INT4
    qfp8weight = torch.zeros((M, 2 * N), dtype=torch.uint8)  
    ###NOTE: need to use torch.uint8 instead of torch.int8; otherwise, for negative numbers, 0xF or >>4 will produce unexpected result or undefined behavior!!
    qfp8weight[:, 0::2] =  (qint4weight >> 4) & 0xF
    qfp8weight[:, 1::2] = qint4weight & 0xF
    qfp8weight.subtract_(bias).to(torch.int8)
    qfp8weight.mul_(int4weight_scales).to(torch.float8_e4m3fn)
    return qfp8weight, fp8weight_scale