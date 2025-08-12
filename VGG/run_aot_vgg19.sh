# HANS: Dummies for initialization
export NCCL_SKIPS='0'
export NCCL_SHIFT='0'
export NCCL_RANDOM_ID='0'
export NCCL_FIXED_SKIP='1'
export NCCL_CHUNK_INC='0'
export NCCL_MAX_NCHANNELS='1'

###### CHOOSE YOUR RUNNER #####
export RUNNER='python3 aot_vgg19.py --gpus 4 --epochs 151 --lr 1e-1' 
export TOPK_RUNNER='python3 steps_vgg19.py --gpus 4 --epochs 151 --lr 1e-1 --logging_period 659' 
###### END OF RUNNERS #####

# Baseline
export COMM_HOOK="NONE"
$RUNNER --method 'allreduce'

# PowerSGD
export COMM_HOOK="VGG_POWERSGD"
$RUNNER --method 'powersgd'

# Top-1%
export COMM_HOOK="VGG_TOP1"
$TOPK_RUNNER --method 'top1'

# SkipReduce 50%
export COMM_HOOK="NONE"
export LD_LIBRARY_PATH=/app/SkipReduce/test_run/SkipReduce-2.0/build/lib
export NCCL_SHIFT='1'
export NCCL_CHUNK_INC='1'
export NCCL_PROTECT_SIZE_0='5866048' # Protect the CNN layer

export NCCL_SKIPS='4'
$RUNNER --method 'skipreduce_50'

