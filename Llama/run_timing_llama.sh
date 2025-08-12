# For determinism
export CUBLAS_WORKSPACE_CONFIG=:16:8

export TASK_NAME=mnli

# DEFINE LIBRARY
export NCCL_ALGO='Ring'
# export LD_LIBRARY_PATH=/app/SkipReduce/libs/SkipReduce-2.0/build/lib/:$LD_LIBRARY_PATH

# DEFINE LIBRARY (FOR RUNPOD)

# HANS: Dummy for initialization
export NCCL_SKIPS='0'
export NCCL_SHIFT='0'
export NCCL_RANDOM_ID='0'
export NCCL_FIXED_SKIP='0'
export NCCL_CHUNK_INC='0'
export SKIPREDUCE_SKIPS='0'
export IS_SKIPREDUCE_RANDOM='0'
export IS_FINE_GRAIN='0'

# export NCCL_P2P_DISABLE='1'
export NCCL_MAX_NCHANNELS='1'

export MODEL_PATH='meta-llama/Llama-3.2-1B'

# export FSDP_RUNNER="accelerate launch --config_file hans_fsdp_config.yaml llama_glue.py --model_name_or_path $MODEL_PATH --fp16 \
# --task_name $TASK_NAME --do_train --do_eval --do_predict False --max_seq_length 128 --per_device_train_batch_size 8 --learning_rate 5e-6 \
# --num_train_epochs 5 --ddp_find_unused_parameters False --save_strategy no --overwrite_output_dir \
# --eval_strategy steps --logging_strategy steps --logging_steps=0.05 --eval_steps=0.05 --logging_first_step --eval_on_start \
# --seed 1234 --data_seed 1234"


export TIMING_RUNNER="torchrun --nnodes=1 --nproc_per_node 4 llama_glue.py --model_name_or_path $MODEL_PATH \
--task_name $TASK_NAME --do_train --do_eval False --do_predict False --max_seq_length 128 --per_device_train_batch_size 8 --per_device_eval_batch_size 32 --learning_rate 5e-6 \
--num_train_epochs 3 --ddp_find_unused_parameters False --save_strategy no --overwrite_output_dir \
--eval_strategy no --logging_strategy no --logging_steps=100 --eval_steps=100 --max_steps 25 \
--seed 1234 --data_seed 1234 --ddp_bucket_cap_mb 25"

##### COMMANDS FOR DP #####

# Top-1%
export COMM_HOOK="LLAMA_TOP1_NOMEM"
$TIMING_RUNNER
