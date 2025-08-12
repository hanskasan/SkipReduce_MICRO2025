# HANS: Dummy for initialization
export NCCL_SKIPS='0'
export NCCL_SHIFT='0'
export NCCL_RANDOM_ID='0'
export NCCL_CHUNK_INC='0'

export SKIPREDUCE_SKIPS='0'
export NCCL_CHUNK_INC='0'
export IS_SKIPREDUCE_RANDOM='0'
export NCCL_FIXED_SKIP='1'
export FSDP_MASK_PROB='0'

# export NCCL_P2P_DISABLE='1'
export NCCL_MAX_NCHANNELS='1'

##### DEFINE RUNNERS #####
export DP_RUNNER_TIMING="torchrun --nnodes=1 --nproc_per_node 4 --master-port 1234 run_swag.py \
--model_name_or_path google-bert/bert-large-uncased  \
--do_train \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--per_device_train_batch_size=4 \
--per_device_eval_batch_size=32 \
--overwrite_output_dir \
--logging_strategy no \
--eval_strategy no \
--save_strategy no \
--max_steps 25 \
--seed 1234 \
--ddp_bucket_cap_mb 25 \
"

##### RUN! #####

# Top-1%
export COMM_HOOK="BERT_TOP1"
$DP_RUNNER_TIMING
