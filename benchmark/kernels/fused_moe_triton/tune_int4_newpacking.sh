rm -rf ~/.triton/cache
HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" MOE_PADDING=0 python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_int4fp8_newpacking.py \
    --model  "hpcai-tech/grok-1" \
    --tp-size 8 \
    --dtype fp8_w8a8 \
    --use_int4_fp8  \
    --tune  2>&1 | tee triton_int4fp8_tuning.log
