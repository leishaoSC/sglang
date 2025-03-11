import torch
import triton 
import triton.language as tl

# @triton.jit
# def int4_to_fp8_dequant(
#         qweights,  # quantized matrix, K/8 x N
#         scales,  # scales, per channel (N,)
#         K8: tl.constexpr, #K/8
#         N: tl.constexpr
# ):
#     qweights = qweights.trans(1,0) #(N,K8)
#     #use shifts to extract nibbles instead of using tl.interleave
#     # Extract 8 nibbles per int32 element
#     #reverse_order_tensor: [0, 4, 1, 5, 2, 6, 3, 7]
#     reverse_order_tensor = ((tl.arange(0, 2) * 4)[None, :] +
#                                 tl.arange(0, 4)[:, None]).reshape(8)
#     shifts = reverse_order_tensor * 4
#     #reorder and extract in the same step!!
#     # Reorder along axis=2, for weights which has shape (N, K8, 8)
#     weights = (qweights[:, :, None] >> shifts[None, :]) & 0xF  # Shape: (N, K8, 8)  

#     weights = weights.reshape(N, K8*8).trans(1,0)  #(N, K8, 8) to (N,K) to (K, N)
#     weights = tl.where(weights >= 8, weights - 16, weights) # Sign-extend nibble => range [-8..7]

#     #scales = tl.broadcast_to(scales[:], (K8*8,N)) #work in sglang moe, but not unit test
#     scales = tl.broadcast_to(scales[None, :], (K8*8,N)) #not work in sglang moe code
#     #scales = scales + tl.zeros([K8*8, N], dtype=tl.float32)# works but also slow

#     #tl.device_print("scales", scales)
#     dweights = weights * scales
#     dweights = dweights.to(tl.float8e4b8)
    
#     return dweights


@triton.jit
def int4_to_fp8_dequant_kernel(
        output,
        qweights_ptr,  # quantized matrix, K/8 x N
        scales_ptr,  # scales, per channel (N,)
        stride_qw_k,
        stride_qw_n,
        stride_dw_k,
        stride_dw_n,
        K2: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr
):
    #offs_qweights = tl.arange(0, K8)[:, None]*stride_qw_k + tl.arange(0,N)[None, :]
    offs_qweights = tl.arange(0, K)[:, None] //2 *stride_qw_k + tl.arange(0,N)[None, :] 
    b_shifter = (tl.arange(0, K)[:, None] % 2) * 4
    qweights = tl.load(qweights_ptr + offs_qweights)
    qweights = qweights.trans(1,0) #(N,K)
    b_shifter = b_shifter.trans(1,0) #(1, K)


    # offs_scales = tl.arange(0, N)
    offs_scales = tl.arange(0, N)[:, None]
    b_scale_int4 = tl.load(scales_ptr + offs_scales)
    

    #dweights = int4_to_fp8_dequant(qweights, scales, K8, N)

    #new unpacking (INT8 to 2 INT4)
    b = (qweights >> b_shifter) & 0xF #(N, K)
    b = tl.where(b >= 8, b - 16, b) #handle negative values
    #b = ((b.to(tl.float32) - b_zp_num) * b_scale_int4).to(compute_type)#???
    b = (b * b_scale_int4).trans(1,0).to(tl.float8e4b8) #[K,N]


    # offs_dweights = tl.arange(0, K8*8)[:, None]*stride_dw_k + tl.arange(0,N)[None, :]
    offs_dweights = tl.arange(0, K)[:, None]*stride_dw_k + tl.arange(0,N)[None, :]
    tl.store(output + offs_dweights, b)


#Big endian
#286331153  = 0x11111111
#68494903   = 0x04152637(wo reorder 0x01234567) 
#2359144127 = 0x8c9daebf(wo reorder 0x89abcdef)
#4226472392 = 0xfbead9c8(wo reorder 0xfedcba98)

#Little endian
#16777216 = 0x01000000 
#33554432 = 0x02000000 
#117440512 = 0x07000000 
#134217728 = 0x08000000 
#925242628  = 0x37261504 (wo reorder 0x76543210)
#3215891852 = 0xbfae9d8c(wo reorder 0xfedcba98)
#3369724667 = 0xc8d9eafb (wo reorder 0x89abcdef)


# weights = torch.tensor([[1,8,1,0],
#                         [0,0,1,1],
#                         [0,0,1,2],
#                         [0,0,1,3],
#                         [0,0,1,4],
#                         [0,0,1,5],
#                         [0,0,1,6],
#                         [0,0,1,7],
#                         [2,7,8,15],
#                         [0,0,9,14],
#                         [0,0,10,13],
#                         [0,0,11,12],
#                         [0,0,12,11],
#                         [0,0,13,10],
#                         [0,0,14,9],
#                         [0,0,15,8]])
# qweights_big_endian = torch.tensor([[1,8,286331153,68494903],[2,7,2359144127,4226472392] ], dtype=torch.int32, device='cuda') #(K/8, N) with K=8, N=4
# qweights_little_endian = torch.tensor([[16777216,134217728,286331153,925242628],[33554432,117440512,3215891852,3369724667] ], dtype=torch.int32, device='cuda') #(K/8, N) with K=8, N=4

# qweights = qweights_big_endian
# #qweights = qweights_little_endian
# print(f"quantized_unpacked_weights_wo_scale={weights}")
# print(f"quantized_packed_weights_wo_scale={qweights}")

# scales = torch.tensor([1.0,2.5,3.0,4.0], dtype=torch.float32, device='cuda') #(N,) N=4
# print(f"scales={scales}")

#-E 8 -top_k 2 -M 2048 -N 6144 -K 4096
K = 128 #1024 #4096
N = 128 #1024 #6144
M = 64 #2048
K2 = K // 2
#K8 = K // 8
device = 'cuda'
qweights = torch.randint(
    0,
    torch.iinfo(torch.int32).max,
    (K2, N),
    device=device,
    dtype=torch.int32
)
qweights = qweights.to(dtype=torch.int8)

#INT per channel scale: [N]
scales = torch.rand(
    (N,),
    dtype=torch.float32,
    device=device
) 


out = torch.zeros(qweights.shape[0]*2, qweights.shape[1], dtype=torch.float8_e4m3fnuz, device="cuda")
grid = (1,)

import time

elapsed_ns= 0
NUM_ITER=100
for _ in range(NUM_ITER):
    start = time.perf_counter_ns()
    int4_to_fp8_dequant_kernel[grid](
                            out,
                            qweights,
                            scales,
                            qweights.stride(0),
                            qweights.stride(1),
                            out.stride(0),
                            out.stride(1),
                            qweights.shape[0],
                            qweights.shape[1],
                            K,
    )
    end = time.perf_counter_ns()
    elapsed_ns += end - start  # Time in nanoseconds
elapsed_ns = elapsed_ns/NUM_ITER
print(f"Elapsed time with new packing kernel: {elapsed_ns} ns")
# print(f"dequantized_weights={out}")