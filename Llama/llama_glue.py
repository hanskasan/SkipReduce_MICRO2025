#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
"""Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# HANS: Additionals
import time
import torch
import torch.distributed as dist
from torch import nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from torch.profiler import profile, ProfilerActivity

from torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks import noop_hook
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import PowerSGDState, powerSGD_hook

# For determinism
seed = 1234
# torch.manual_seed(seed)
random.seed(seed)
# torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
os.environ['PYTHONHASHSEED'] = str(seed)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.50.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
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
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


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
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
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

            if os.environ["COMM_HOOK"] == "LLAMA_TOP1_NOMEM":
                model.register_comm_hook(None, topk_allreduce)

            if os.environ["COMM_HOOK"] == "LLAMA_TOP1":
                model.register_comm_hook(None, topk_allreduce_with_memory)

            if os.environ["COMM_HOOK"] == "LLAMA_POWERSGD":
                state = PowerSGDState(process_group=None, matrix_approximation_rank=1, start_powerSGD_iter=5, use_error_feedback=True)
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

            # if torch.distributed.get_rank() == 0:
                # print("End of BW")

            # HANS: Check for NaNs in gradient
            # if torch.distributed.get_rank() == 0:
                # for name, param in model.named_parameters():
                    # print(name, param.numel())
                # if param.grad is not None:
                #     if torch.distributed.get_rank() == 0:
                #         # print(name, "Max:", torch.max(param.grad), "Avg:", torch.mean(param.grad), "zeros:", (param.grad == 0).sum(), "size:", param.grad.size())

                #         if torch.isnan(param.grad).any():
                #             print("Nan at", name, "is", torch.isnan(param.grad).sum(), "size:", param.grad.size()[0])
                #             # assert False
                #         else:
                #             print("Gradient at", name, "mean:", torch.mean(param.grad), ", max:", torch.max(param.grad), ",min:", torch.min(param.grad))

            # HANS: Plot top-k gradient histogram
            # top_ratio = 0.25
            # if torch.distributed.get_rank() == 0 and self.iter_counter == 0 or (self.iter_counter % 100) == 0:
            #     # Define bin_edges (Llama 3.2-1B)
            #     bin_edges = torch.tensor([262668288, 266862592, 267911168, 268959744, 273154048, 289931264, 306708480, 323485696, 323487744, 323489792, 327684096, 328732672, 329781248, 333975552, 350752768, 367529984, 384307200, 384309248, 384311296, 388505600, 389554176, 390602752, 394797056, 411574272, 428351488, 445128704, 445130752, 445132800, 449327104, 450375680, 451424256, 455618560, 472395776, 489172992, 505950208, 505952256, 505954304, 510148608, 511197184, 512245760, 516440064, 533217280, 549994496, 566771712, 566773760, 566775808, 570970112, 572018688, 573067264, 577261568, 594038784, 610816000, 627593216, 627595264, 627597312, 631791616, 632840192, 633888768, 638083072, 654860288, 671637504, 688414720, 688416768, 688418816, 692613120, 693661696, 694710272, 698904576, 715681792, 732459008, 749236224, 749238272, 749240320, 753434624, 754483200, 755531776, 759726080, 776503296, 793280512, 810057728, 810059776, 810061824, 814256128, 815304704, 816353280, 820547584, 837324800, 854102016, 870879232, 870881280, 870883328, 875077632, 876126208, 877174784, 881369088, 898146304, 914923520, 931700736, 931702784, 931704832, 935899136, 936947712, 937996288, 942190592, 958967808, 975745024, 992522240, 992524288, 992526336, 996720640, 997769216, 998817792, 1003012096, 1019789312, 1036566528, 1053343744, 1053345792, 1053347840, 1057542144, 1058590720, 1059639296, 1063833600, 1080610816, 1097388032, 1114165248, 1114167296, 1114169344, 1118363648, 1119412224, 1120460800, 1124655104, 1141432320, 1158209536, 1174986752, 1174988800, 1174990848, 1179185152, 1180233728, 1181282304, 1185476608, 1202253824, 1219031040, 1235808256, 1235810304, 1235812352, 1235814400, 1235818496]).cuda(torch.distributed.get_rank())

            #     grads = []

            #     for params in model.parameters():
            #         temp = torch.flatten(params.grad)
            #         if torch.count_nonzero(temp) > 0:
            #             grads.append(temp) 

            #     del temp
            #     grads = torch.cat(grads)
            #     k = int(top_ratio * grads.size()[0])
            #     grad_abs = torch.abs(grads)
            #     del grads
            #     temp = torch.topk(grad_abs, k, largest=False)
            #     del grad_abs

            #     dumpname = "/workspace/SkipReduce_Models/Llama/GLUE/dumps_top/sst/top_" + str(top_ratio * 100) + "_iter_" + str(self.iter_counter)
            #     bin_indices = torch.bucketize(temp.indices, bin_edges, right=True)
            #     hist = torch.bincount(bin_indices)
            #     hist_at_cpu = hist.cpu().numpy()

            #     with open(dumpname, 'w') as f:
            #         np.savetxt(f, hist_at_cpu)

            #     del temp
            #     del hist
            #     del hist_at_cpu

            # HANS: Dump weights and gradients
            # gpu = torch.distributed.get_rank()
            # bin_edges = torch.arange(-100, 100, 0.1).cuda(gpu)

            # if self.iter_counter % 200 == 0:
            #     params = torch.Tensor().cuda(gpu)
            #     grads = torch.Tensor().cuda(gpu)

            #     for param in model.parameters():
            #         temp_param = param
            #         temp_param = torch.flatten(temp_param)
            #         params = torch.cat((params, temp_param))

            #         temp_grad = param.grad
            #         temp_grad = torch.flatten(temp_grad)
            #         grads = torch.cat((grads, temp_grad))

            #     bucket_indices_param = torch.bucketize(params, bin_edges)
            #     hist_param = torch.bincount(bucket_indices_param)
            #     hist_at_cpu_param = hist_param.cpu().numpy()

            #     bucket_indices_grad = torch.bucketize(grads, bin_edges)
            #     hist_grad = torch.bincount(bucket_indices_grad)
            #     hist_at_cpu_grad = hist_grad.cpu().numpy()

            #     dumpname_param = "./dumps/dump_param/iter-"+str(self.iter_counter)+"_gpu"+str(gpu)
            #     dumpname_grad = "./dumps/dump_grad/iter-"+str(self.iter_counter)+"_gpu"+str(gpu)

            #     with open(dumpname_param, 'w') as f:
            #         np.savetxt(f, hist_at_cpu_param)
                
            #     with open(dumpname_grad, 'w') as f:
            #         np.savetxt(f, hist_at_cpu_grad)

            # HANS: Specifically dump weights and gradients
            # gpu = torch.distributed.get_rank()
            # bin_edges = torch.arange(-100, 100, 0.1).cuda(gpu)

            # if self.iter_counter % 200 == 0:

            #     for name, param in model.named_parameters():
            #         params = param
            #         grads = param.grad

            #         bucket_indices_param = torch.bucketize(params, bin_edges)
            #         hist_param = torch.bincount(bucket_indices_param)
            #         hist_at_cpu_param = hist_param.cpu().numpy()
    
            #         bucket_indices_grad = torch.bucketize(grads, bin_edges)
            #         hist_grad = torch.bincount(bucket_indices_grad)
            #         hist_at_cpu_grad = hist_grad.cpu().numpy()

            #         dumpname_param = "./small_dumps/dump_param/iter-"+str(self.iter_counter)+"_gpu"+str(gpu)+"_"+name
            #         dumpname_grad = "./small_dumps/dump_grad/iter-"+str(self.iter_counter)+"_gpu"+str(gpu)+"_"+name

            #         with open(dumpname_param, 'w') as f:x
            #             np.savetxt(f, hist_at_cpu_param)

            #         with open(dumpname_grad, 'w') as f:
            #             np.savetxt(f, hist_at_cpu_grad)

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

# HANS: Custom hooks

# HANS: Additional variables for memory
bucket_idx = 0
num_of_buckets = 65
grad_mem = []
sum_comm_time = 0

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

def _mask_allreduce_fut(
    process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future[torch.Tensor]:
    """Average the input gradient tensor by allreduce and returns a future."""
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    # Apply the division first to avoid overflow, especially for FP16.
    tensor.div_(group_to_use.size())

    # Generate mask
    skip_prob = 0.0
    device = 'cuda:' + str(torch.distributed.get_rank())
    mask = torch.rand(tensor.size(), device=device) < (1 - skip_prob)
    tensor *= mask

    # if torch.distributed.get_rank() <= 1:
        # print("MASK:", mask)

    return (
        dist.all_reduce(tensor, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )

def mask_allreduce_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:

    return _mask_allreduce_fut(process_group, bucket.buffer())

def _noop_fut(
    process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future[torch.Tensor]:
    """Average the input gradient tensor by allreduce and returns a future."""
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    # Apply the division first to avoid overflow, especially for FP16.
    # tensor.div_(group_to_use.size())

    # To check the bucket size
    # if torch.distributed.get_rank() == 0:
        # print(torch.numel(tensor))

    future = torch.futures.Future()
    future.set_result(tensor)

    # print(dist.all_reduce(tensor, group=group_to_use, async_op=True).get_future())

    return future.then(lambda fut: fut.value()[0])

    # return (
        # dist.all_reduce(tensor, group=group_to_use, async_op=True)
        # .get_future()
        # .then(lambda fut: fut.value()[0])
    # )

def noop_allreduce_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:

    return _noop_fut(process_group, bucket.buffer())

# Top-k that does not work, but fast
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
    iter = (bucket_idx - 1) // num_of_buckets

    # if dist.get_rank() == 0:
        # print(bucket_idx, iter)

    # Apply the division first to avoid overflow, especially for FP16.
    tensor.div_(group_to_use.size())

    # Top-k selection
    acc = tensor.clone()
    if iter > 0:
        acc += grad_mem[(bucket_idx - 1) % num_of_buckets]
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
        grad_mem[(bucket_idx - 1) % num_of_buckets] = acc * ~mask

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

    del input_abs
    del threshold

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
    iter = (bucket_idx - 1) // num_of_buckets

    grads = bucket.buffer()

    acc = grads
    if iter > 0:
        acc += grad_mem[(bucket_idx - 1) % num_of_buckets]

    mask = hacking_topk(acc)

    # Initially, initialize grad_mem
    if iter == 0:
        grad_mem.append(acc * ~mask)
    elif iter > 0:
        grad_mem[(bucket_idx - 1) % num_of_buckets] = acc * ~mask

    # Increment bucket index
    bucket_idx += 1

    return (
        dist.all_reduce(acc * mask, op=dist.ReduceOp.AVG, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )

def _int8_allreduce_fut(
    process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future[torch.Tensor]:
    "Averages the input gradient tensor by allreduce and returns a future."
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    # Apply the division first to avoid overflow, especially for FP16.
    tensor.div_(group_to_use.size())

    # Scale
    scale = tensor.abs().max() / 127
    tensor = (tensor / scale).clamp(-128, 127).round().to(dtype=torch.int8)

    return (
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0].to(dtype=torch.float32) * scale)
    )

def int8_allreduce(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:

    # HANS: For debugging
    # if torch.distributed.get_rank() == 0:
        # print(bucket.buffer().numel())

    temp = _int8_allreduce_fut(process_group, bucket.buffer())
    return temp

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
    send_example_telemetry("run_glue", model_args, data_args)

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

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "nyu-mll/glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
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
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    # HANS: Print parameters
    # if torch.distributed.get_rank() == 0:
    #     for name, param in model.named_parameters():
    #         print(name, "with size", param.numel())
    # assert False

    # HANS: Modifications for Llama
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: "
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = evaluate.load("glue", data_args.task_name, cache_dir=model_args.cache_dir)
    elif is_regression:
        metric = evaluate.load("mse", cache_dir=model_args.cache_dir)
    else:
        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        # if training_args.resume_from_checkpoint is not None:
            # checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
            # checkpoint = last_checkpoint
        timestamp = time.time() # HANS: To measure end-to-end training time
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trace_file = f"trace_{torch.distributed.get_rank()}.json"
        # prof.export_chrome_trace(trace_file)
        if torch.distributed.get_rank() == 0:
            print("End-to-end runtime:", time.time() - timestamp)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # trainer.save_model()  # Saves the tokenizer too for easy upload

        # trainer.log_metrics("train", metrics)
        # trainer.save_metrics("train", metrics)
        # trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            valid_mm_dataset = raw_datasets["validation_mismatched"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(valid_mm_dataset), data_args.max_eval_samples)
                valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
            eval_datasets.append(valid_mm_dataset)
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
