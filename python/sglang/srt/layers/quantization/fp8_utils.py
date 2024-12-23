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
    # pack INT4 values into bytes
    qint4weight = ((qfp8weight[:, ::2] & 0xF) << 4) | (qfp8weight[:, 1::2] & 0xF) 
    return qint4weight, fp8weight_scale, int4weight_scales