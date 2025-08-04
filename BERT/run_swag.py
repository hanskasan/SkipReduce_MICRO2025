#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for multiple choice.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    DataCollatorForMultipleChoice,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry

# HANS: Additionals
import os
import time
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks import noop_hook
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import PowerSGDState, powerSGD_hook
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.50.0.dev0")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If passed, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to the maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

# HANS: Additional variables for memory
bucket_idx = 0
num_of_buckets = 38
grad_mem = []
sum_comm_time = 0

# Custom communication hooks
def _allreduce_fut(
    process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future[torch.Tensor]:
    """Average the input gradient tensor by allreduce and returns a future."""
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    # Apply the division first to avoid overflow, especially for FP16.
    tensor.div_(group_to_use.size())

    # To check the bucket size
    # if torch.distributed.get_rank() == 0:
        # print("Numel:", torch.numel(tensor))

    return (
        dist.all_reduce(tensor, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )

def default_allreduce_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:

    return _allreduce_fut(process_group, bucket.buffer())

def _topk_timing_fut_sync(
    process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future[torch.Tensor]:
    "Averages the input gradient tensor by allreduce and returns a future."
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    nelem = tensor.numel()
    pruning_ratio = 0.9
    how_top = 1 - pruning_ratio
    k = int(how_top * nelem)

    # Apply the division first to avoid overflow, especially for FP16.
    tensor.div_(group_to_use.size())

    # Top-k selection
    tensor = torch.abs(tensor)
    topk_val, topk_idx = torch.topk(tensor, k=int(how_top * nelem), sorted=False)
    dummy = torch.cat([topk_val, topk_idx])

    dist.all_reduce(dummy, op=dist.ReduceOp.SUM, group=group_to_use)

    tensor.zero_().index_put([topk_idx], topk_val, accumulate=True)
    
    future = torch.futures.Future()
    future.set_result(tensor)

    return future

def _topk_timing_fut_sync_with_memory(
    process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future[torch.Tensor]:
    "Averages the input gradient tensor by allreduce and returns a future."
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    nelem = tensor.numel()
    pruning_ratio = 0.999
    how_top = 1 - pruning_ratio
    k = int(how_top * nelem)

    global bucket_idx
    global num_of_buckets
    global grad_mem
    global sum_comm_time

    # Calculate iteration
    iter = bucket_idx // num_of_buckets

    # Calculate communication time
    if bucket_idx == 0:
        sum_comm_time = 0

    # if dist.get_rank() == 0:
        # print(bucket_idx, iter)

    # Apply the division first to avoid overflow, especially for FP16.
    tensor.div_(group_to_use.size())

    # Top-k selection
    acc = tensor.clone()
    if iter > 0:
        acc += grad_mem[bucket_idx % num_of_buckets]
    acc = torch.abs(acc)
    topk_val, topk_idx = torch.topk(acc, k=int(how_top * nelem), sorted=False)
    mask = (acc >= topk_val[0]).bool() # Select any value as threshold.. We don't care about the correctness here
    dummy = torch.cat([topk_val, topk_idx])

    dist.barrier()
    start_time = time.time()
    dist.all_reduce(dummy, op=dist.ReduceOp.SUM, group=group_to_use)
    dist.barrier()
    sum_comm_time += time.time() - start_time

    acc.zero_().index_put([topk_idx], topk_val, accumulate=True)

    # Initially, initialize grad_mem
    if iter == 0:
        grad_mem.append(acc * ~mask)
    elif iter > 0:
        grad_mem[bucket_idx % num_of_buckets] = acc * ~mask

    # Print communication time
    if iter % 10 == 0 and iter > 0 and bucket_idx % num_of_buckets == 0:
        if torch.distributed.get_rank() == 0:
            print("Communication time:", sum_comm_time / 10, iter, bucket_idx)
        sum_comm_time = 0

    # Increment bucket index
    bucket_idx += 1
    
    future = torch.futures.Future()
    future.set_result(acc)

    return future

# HANS: Define custom communication hooks
def topk_timing_allreduce(process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    # temp = _topk_timing_fut(process_group, bucket.buffer())
    temp = _topk_timing_fut_sync(process_group, bucket.buffer())

    return temp.then(lambda fut: fut.value())

def topk_timing_allreduce_with_memory(process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    temp = _topk_timing_fut_sync_with_memory(process_group, bucket.buffer())

    return temp.then(lambda fut: fut.value())

def hacking_topk(input):
    pruning_ratio = 0.99
    topk_selection = 1 - pruning_ratio
    k = int(topk_selection * input.size()[0])

    input_abs = torch.abs(input)
    threshold = torch.topk(input_abs, k, largest=True).values[-1]
    mask = (input_abs >= threshold).bool()

    return mask

def topk_allreduce(process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    grads = bucket.buffer()
    mask = hacking_topk(grads)

    return (
        dist.all_reduce(mask * grads, op=dist.ReduceOp.AVG, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )

def topk_allreduce_with_memory(process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    global bucket_idx
    global num_of_buckets
    global grad_mem

    # Calculate iteration
    iter = bucket_idx // num_of_buckets

    grads = bucket.buffer()

    acc = grads
    if iter > 0:
        acc += grad_mem[bucket_idx % num_of_buckets]

    if iter < 200: # HANS: A little bit of warmup to make the training converge..
        mask = torch.ones_like(acc, dtype=torch.bool)
    else:
        mask = hacking_topk(acc)

    # Initially, initialize grad_mem
    if iter == 0:
        grad_mem.append(acc * ~mask)
    elif iter > 0:
        grad_mem[bucket_idx % num_of_buckets] = acc * ~mask

    # Increment bucket index
    bucket_idx += 1

    return (
        dist.all_reduce(acc * mask, op=dist.ReduceOp.AVG, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )

# HANS: Define custom trainer
class CustomTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iter_counter = 0
        self.timestamp = 0
        os.environ["IS_WARMING"] = '0'
        torch.cuda.manual_seed(torch.distributed.get_rank())

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """

        # HANS: Modify hook
        if self.iter_counter == 0:
            # for name, param in model.named_parameters():
                # print(name)

            if os.environ["COMM_HOOK"] == "BERT_TOP1":
                model.register_comm_hook(None, topk_allreduce_with_memory)

            if os.environ["COMM_HOOK"] == "BERT_POWERSGD":
                state = PowerSGDState(process_group=None, matrix_approximation_rank=4, start_powerSGD_iter=1000, use_error_feedback=True)
                model.register_comm_hook(state, powerSGD_hook)

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        # if is_sagemaker_mp_enabled():
            # loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            # return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            # if torch.distributed.get_rank() == 0:
                # print("Before FW")
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            # print("Loss:", loss)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        # if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # Finally we need to normalize the loss for reporting
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            # if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                # kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

            # HANS: To measure iteration time
            # resolution = 10
            # if self.iter_counter == 0:
            #     self.timestamp = time.time()
            # if self.iter_counter > 0 and self.iter_counter % resolution == 0:
            #     torch.cuda.synchronize()
            #     dist.barrier()
            #     if dist.get_rank() == 0:
            #         print("Elapsed time:", (time.time() - self.timestamp) / resolution)
            #     self.timestamp = time.time()

            # HANS: Increment training step
            self.iter_counter += 1

            if self.iter_counter <= 134:
                os.environ["IS_WARMING"] = '1'
            else:
                os.environ["IS_WARMING"] = '0'

            return loss.detach()

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_swag", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.train_file is not None or data_args.validation_file is not None:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    else:
        # Downloading and loading the swag dataset from the hub.
        raw_datasets = load_dataset(
            "swag",
            "regular",
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # When using your own dataset or a different dataset from swag, you will probably need to change this.
    ending_names = [f"ending{i}" for i in range(4)]
    context_name = "sent1"
    question_header_name = "sent2"

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Preprocessing the datasets.
    def preprocess_function(examples):
        first_sentences = [[context] * 4 for context in examples[context_name]]
        question_headers = examples[question_header_name]
        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
        ]

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length" if data_args.pad_to_max_length else False,
        )
        # Un-flatten
        return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    # Data collator
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorForMultipleChoice(
            tokenizer=tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None, return_tensors="pt"
        )
    )

    # Metric
    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "multiple-choice",
        "dataset_tags": "swag",
        "dataset_args": "regular",
        "dataset": "SWAG",
        "language": "en",
    }

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
