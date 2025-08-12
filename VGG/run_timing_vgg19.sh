# HANS: Dummies for initialization
export NCCL_SKIPS='0'
export NCCL_SHIFT='0'
export NCCL_RANDOM_ID='0'
export NCCL_FIXED_SKIP='1'
export NCCL_CHUNK_INC='0'
export NCCL_MAX_NCHANNELS='1'

###### CHOOSE YOUR RUNNER #####
export RUNNER='python3 timing_vgg19.py --gpus 4 --epochs 3 --lr 1e-1' 
###### END OF RUNNERS #####

# Top-1%
export COMM_HOOK="VGG_TOP1"
$RUNNER --method 'top1'
