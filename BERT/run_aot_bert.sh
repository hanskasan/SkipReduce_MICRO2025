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


##### DEFINE RUNNERS #####
export DP_RUNNER_TIMING="torchrun --nnodes=1 --nproc_per_node 8 --master-port 1234 run_swag.py \
--model_name_or_path google-bert/bert-large-uncased  \
--do_train \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--per_device_train_batch_size=2 \
--per_device_eval_batch_size=32 \
--overwrite_output_dir \
--logging_strategy no \
--eval_strategy no \
--save_strategy no \
--max_steps 25 \
--seed 0 \
--ddp_bucket_cap_mb 25 \
"

export DP_RUNNER="torchrun --nnodes=1 --nproc_per_node 8 --master-port 1234 run_swag.py \
--model_name_or_path google-bert/bert-large-uncased  \
--do_train \
--do_eval \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--per_device_train_batch_size=2 \
--per_device_eval_batch_size=32 \
--overwrite_output_dir \
--logging_strategy tta \
--eval_strategy tta \
--tta_period 60 \
--save_strategy no \
--logging_first_step \
--eval_on_start \
--seed 0 \
"

export TOP1_DP_RUNNER="torchrun --nnodes=1 --nproc_per_node 8 --master-port 1234 run_swag.py \
--model_name_or_path google-bert/bert-large-uncased  \
--do_train \
--do_eval \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--per_device_train_batch_size=2 \
--per_device_eval_batch_size=32 \
--overwrite_output_dir \
--logging_strategy no \
--eval_strategy steps \
--eval_steps 266 \
--save_strategy no \
--logging_first_step \
--eval_on_start \
--seed 0 \
--data_seed 0 \
"

##### RUN! #####

# Baseline
export COMM_HOOK="NONE"
$DP_RUNNER > aot_reports/baseline.report

# PowerSGD
export COMM_HOOK="BERT_POWERSGD"
$DP_RUNNER > aot_reports/powersgd.report

# Top-1%
export COMM_HOOK="BERT_TOP1"
$TOP1_DP_RUNNER > aot_reports/top1.report

# SkipReduce 50%
export COMM_HOOK="NONE"
export LD_PRELOAD=/workspace/NCCL/random_selective/build/lib/libnccl.so
export NCCL_SHIFT='1'
export NCCL_CHUNK_INC='1'
export NCCL_PROTECT_SIZE_0='31254528' # Protect the embedding layer
export NCCL_SKIPS='4'
$DP_RUNNER > aot_reports/skipreduce_50.report