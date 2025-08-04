import os
from datetime import datetime
import argparse
import math
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
import random
import numpy as np
# import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import PowerSGDState, powerSGD_hook

# @nvtx.annotate("topk", color="yellow")
def hacking_topk(input):
    pruning_ratio = 0.99
    topk_selection = 1 - pruning_ratio
    k = int(topk_selection * input.size()[0])

    input_abs = torch.abs(input)
    threshold = torch.topk(input_abs, k, largest=True).values[-1]
    mask = (input_abs >= threshold).bool()

    output = input * mask

    return output

# def debugging_topk(input):
#     bottom_ratio = 0.5

#     # Find threshold
#     k = int(bottom_ratio * input.size()[0])
#     data_abs = torch.abs(input)
#     temp = torch.topk(data_abs, k, largest=False)
#     threshold = temp.values[-1]

#     if True:

def random_pruning(input):
    pruning_ratio = 0.75
    input_size = torch.numel(input)

    offset = int(pruning_ratio * torch.distributed.get_world_size())

    rand = torch.cuda.FloatTensor(input.size()).uniform_(0.0, 1.0)

    lo_thres = torch.distributed.get_rank() * (1 / torch.distributed.get_world_size())
    hi_thres = ((torch.distributed.get_rank() + offset) % torch.distributed.get_world_size()) * (1 / torch.distributed.get_world_size())

    lo_mask = (rand < lo_thres).bool()
    hi_mask = (rand >= hi_thres).bool()

    if torch.distributed.get_rank() >= (torch.distributed.get_world_size() - offset):
        mask = lo_mask * hi_mask # And
    else:
        mask = lo_mask + hi_mask # or

    # HANS: For debugging
    # print("Rank", torch.distributed.get_rank(), "has ratio", torch.count_nonzero(mask) / torch.numel(mask))

    output = input * mask
    
    return output

def random_pruning_protect(input):
    pruning_ratio = 0.375
    input_size = torch.numel(input)

    if input_size != 5866048:
        # offset = int(pruning_ratio * torch.distributed.get_world_size())
        offset = pruning_ratio * torch.distributed.get_world_size()

        rand = torch.cuda.FloatTensor(input_size).uniform_(0.0, 1.0)

        lo_thres = torch.distributed.get_rank() * (1 / torch.distributed.get_world_size())
        hi_thres = ((torch.distributed.get_rank() + offset) % torch.distributed.get_world_size()) * (1 / torch.distributed.get_world_size())

        lo_mask = (rand < lo_thres).bool()
        hi_mask = (rand >= hi_thres).bool()

        if torch.distributed.get_rank() >= (torch.distributed.get_world_size() - offset):
            mask = lo_mask * hi_mask # And
        else:
            mask = lo_mask + hi_mask # or

        # HANS: For debugging
        # print("Rank", torch.distributed.get_rank(), "has ratio", torch.count_nonzero(mask) / torch.numel(mask))

        input *= mask

    # HANS: For debugging
    # if torch.distributed.get_rank() == 0:
        # print(input_size)
    
    return input

def random_pruning_attack(input):
    pruning_ratio = 0.5
    input_size = torch.numel(input)

    if input_size == 5866048:
        # offset = int(pruning_ratio * torch.distributed.get_world_size())
        offset = pruning_ratio * torch.distributed.get_world_size()

        rand = torch.cuda.FloatTensor(input_size).uniform_(0.0, 1.0)

        lo_thres = torch.distributed.get_rank() * (1 / torch.distributed.get_world_size())
        hi_thres = ((torch.distributed.get_rank() + offset) % torch.distributed.get_world_size()) * (1 / torch.distributed.get_world_size())

        lo_mask = (rand < lo_thres).bool()
        hi_mask = (rand >= hi_thres).bool()

        if torch.distributed.get_rank() >= (torch.distributed.get_world_size() - offset):
            mask = lo_mask * hi_mask # And
        else:
            mask = lo_mask + hi_mask # or

        # HANS: For debugging
        # print("Rank", torch.distributed.get_rank(), "has ratio", torch.count_nonzero(mask) / torch.numel(mask))

        input *= mask

    # HANS: For debugging
    # if torch.distributed.get_rank() == 0:
        # print(input_size)
    
    return input

def random_pruning_randomratio(input):
    rand_ratio = torch.cuda.FloatTensor(1).uniform_(0.0, 1.0)
    rand_ratio_int = int(rand_ratio * torch.distributed.get_world_size())
    pruning_ratio = rand_ratio_int / torch.distributed.get_world_size()
    
    offset = int(pruning_ratio * torch.distributed.get_world_size())

    rand = torch.cuda.FloatTensor(input.size()).uniform_(0.0, 1.0)

    lo_thres = torch.distributed.get_rank() * (1 / torch.distributed.get_world_size())
    hi_thres = ((torch.distributed.get_rank() + offset) % torch.distributed.get_world_size()) * (1 / torch.distributed.get_world_size())

    lo_mask = (rand < lo_thres).bool()
    hi_mask = (rand >= hi_thres).bool()

    if torch.distributed.get_rank() >= (torch.distributed.get_world_size() - offset):
        mask = lo_mask * hi_mask # And
    else:
        mask = lo_mask + hi_mask # or

    output = input * mask
    
    return output

def random_pruning_randomratio_protect(input):

    rand_ratio = torch.cuda.FloatTensor(1).uniform_(0.0, 1.0)
    rand_ratio_int = int(rand_ratio * torch.distributed.get_world_size())
    pruning_ratio = rand_ratio_int / torch.distributed.get_world_size()

    input_size = torch.numel(input)

    if input_size != 5866048:
        offset = int(pruning_ratio * torch.distributed.get_world_size())

        rand = torch.cuda.FloatTensor(input_size).uniform_(0.0, 1.0)

        lo_thres = torch.distributed.get_rank() * (1 / torch.distributed.get_world_size())
        hi_thres = ((torch.distributed.get_rank() + offset) % torch.distributed.get_world_size()) * (1 / torch.distributed.get_world_size())

        lo_mask = (rand < lo_thres).bool()
        hi_mask = (rand >= hi_thres).bool()

        if torch.distributed.get_rank() >= (torch.distributed.get_world_size() - offset):
            mask = lo_mask * hi_mask # And
        else:
            mask = lo_mask + hi_mask # or

        # HANS: For debugging
        # if torch.distributed.get_rank() == 0:
            # print("Rank", torch.distributed.get_rank(), "has ratio", torch.count_nonzero(mask) / torch.numel(mask))

        input *= mask

    # HANS: For debugging
    # if torch.distributed.get_rank() == 0:
        # print(input_size)
    
    return input
    

def topk_allreduce(process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    grads = bucket.buffer()
    message = hacking_topk(grads)

    return (
        dist.all_reduce(message, op=dist.ReduceOp.AVG, group=group_to_use, async_op=True)
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
    else:
        grad_mem[(bucket_idx - 1) % num_of_buckets] = acc * ~mask

    # Increment bucket index
    bucket_idx += 1

    return (
        dist.all_reduce(acc * mask, op=dist.ReduceOp.AVG, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )


def random_prune_allreduce(process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    grads = bucket.buffer()
    message = random_pruning(grads)

    return (
        dist.all_reduce(message, op=dist.ReduceOp.AVG, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )

def random_prune_randomratio_allreduce(process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    grads = bucket.buffer()
    message = random_pruning_randomratio(grads)

    return (
        dist.all_reduce(message, op=dist.ReduceOp.AVG, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )

def random_prune_protect_allreduce(process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    grads = bucket.buffer()
    message = random_pruning_protect(grads)

    return (
        dist.all_reduce(message, op=dist.ReduceOp.AVG, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )

def random_prune_attack_allreduce(process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    grads = bucket.buffer()
    message = random_pruning_attack(grads)

    return (
        dist.all_reduce(message, op=dist.ReduceOp.AVG, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )

def random_prune_randomratio_protect_allreduce(process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    grads = bucket.buffer()
    message = random_pruning_randomratio_protect(grads)

    return (
        dist.all_reduce(message, op=dist.ReduceOp.AVG, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )

def _allreduce_fut(
    process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future[torch.Tensor]:
    "Averages the input gradient tensor by allreduce and returns a future."
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    # Apply the division first to avoid overflow, especially for FP16.
    tensor.div_(group_to_use.size())

    return (
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )

def default_allreduce(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:

    # HANS: For debugging
    # if torch.distributed.get_rank() == 0:
        # print(bucket.buffer().numel())

    temp = _allreduce_fut(process_group, bucket.buffer())
    return temp

def _allreduce_fut_hash(
    process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future[torch.Tensor]:
    "Averages the input gradient tensor by allreduce and returns a future."
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    # Apply the division first to avoid overflow, especially for FP16.
    tensor.div_(group_to_use.size())

    # HASH
    perm = torch.randperm(tensor.size(0), device='cuda')
    tensor = tensor[perm]
    inverse_perm = torch.argsort(perm)

    future = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group_to_use, async_op=True).get_future()
    future.wait()

    return (
        future.then(lambda fut: fut.value()[0][inverse_perm])
    )

def default_allreduce_with_hash(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:

    # HANS: For debugging
    # if torch.distributed.get_rank() == 0:
        # print("Numel:", bucket.buffer().numel())

    # HASH!
    # cache = bucket.buffer()
    # perm = torch.randperm(cache.size(0), device='cuda')
    # inverse_perm = torch.argsort(perm)

    # start_time = time.time()
    temp = _allreduce_fut_hash(process_group, bucket.buffer())

    # dist.barrier()
    # torch.cuda.synchronize()

    # elapsed_time = (time.time() - start_time)
    # print(elapsed_time)

    return temp

def _allreduce_fut_pmhash(
    process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future[torch.Tensor]:
    "Averages the input gradient tensor by allreduce and returns a future."
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    # Apply the division first to avoid overflow, especially for FP16.
    tensor.div_(group_to_use.size())

    # HASH
    nelem = tensor.numel()
    grains = nelem
    for i in range(int(math.sqrt(nelem)), nelem):
    # for i in range(int(math.pow(nelem, 1/4)), nelem):
        if nelem % i == 0:
            grains = i
            break

    # grains = 2

    rand = torch.cuda.FloatTensor(grains).uniform_(0.0, 1.0)
    perm = torch.argsort(rand)

    tensor = tensor.view(grains, -1)
    tensor = tensor[perm]
    inverse_perm = torch.argsort(perm)

    future = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group_to_use, async_op=True).get_future()
    future.wait()

    return (
        future.then(lambda fut: fut.value()[0][inverse_perm].flatten())
        # future.then(lambda fut: fut.value()[0].flatten())
    )

def default_allreduce_with_pmhash(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:

    # start_time = time.time()

    # temp = _allreduce_fut(process_group, bucket.buffer())
    temp = _allreduce_fut_pmhash(process_group, bucket.buffer())

    # dist.barrier()
    # torch.cuda.synchronize()
    # elapsed_time = (time.time() - start_time)
    # print(elapsed_time)

    return temp

def _allreduce_fut_rotate(
    process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future[torch.Tensor]:
    "Averages the input gradient tensor by allreduce and returns a future."
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    # Apply the division first to avoid overflow, especially for FP16.
    tensor.div_(group_to_use.size())

    # ROTATE
    nelem = tensor.numel()
    divisor = nelem
    for i in range(int(math.sqrt(nelem)), nelem):
        if nelem % i == 0:
            divisor = i
            break

    tensor = tensor.view(-1, divisor)
    tensor = torch.rot90(tensor, 1)

    future = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group_to_use, async_op=True).get_future()
    future.wait()

    return (
        future.then(lambda fut: torch.rot90(fut.value()[0], -1).flatten())
    )

def default_allreduce_with_rotate(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    temp = _allreduce_fut_rotate(process_group, bucket.buffer())
    return temp

def _allreduce_fut_roll(
    process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future[torch.Tensor]:
    "Averages the input gradient tensor by allreduce and returns a future."
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    # Apply the division first to avoid overflow, especially for FP16.
    tensor.div_(group_to_use.size())

    # ROLL
    nelem = tensor.numel()
    shift = random.randint(1, 64)
    # if torch.distributed.get_rank() == 0:
        # print(shift, "\t", tensor.numel())
    tensor = torch.roll(tensor, shift)

    future = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group_to_use, async_op=True).get_future()
    future.wait()

    return (
        future.then(lambda fut: torch.roll(fut.value()[0], -shift))
    )

def default_allreduce_with_roll(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    temp = _allreduce_fut_roll(process_group, bucket.buffer())
    return temp

preprocess = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

preprocesswithflip = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def setrandom(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)
    # torch.use_deterministic_algorithms(True, warn_only=True)

def main():
    parser = argparse.ArgumentParser()
   
    parser.add_argument('-g', '--gpus', default=4, type=int,
                        help='number of gpus per node')
    parser.add_argument('--epochs', default=3, type=int, metavar='N',
                        help='number of total epochs to run')
    
    parser.add_argument('--datatype', default='F32', type=str)

    parser.add_argument('--lr', default = 1e-3, type=float)

    parser.add_argument('--prune', default = 0, type=int)

    parser.add_argument('--method', default = 'baseline', type=str)

    parser.add_argument('--nol2', default=False, action="store_true")

    parser.add_argument('--noaug', default=False, action="store_true")

    parser.add_argument('--nodrop', default=False, action="store_true")

    parser.add_argument('--dump_grad', default=False, action="store_true")

    parser.add_argument('--flat_lr', default=False, action="store_true")

    parser.add_argument('--timestamp_period', default=60, type=int)

    parser.add_argument('--seed', default=0, type=int)

    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '2025'
    #os.environ['MASTER_PORT'] = '8008'

    os.environ['NCCL_ALGO'] = 'Ring'
    # os.environ['NCCL_MAX_NCHANNELS'] = '1'
    # os.environ['NCCL_MIN_NCHANNELS'] = str(args.rings)
    # os.environ['NCCL_DEBUG'] = "INFO"

    if args.noaug:
        # train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=preprocess, download=True)
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=preprocess, download=True)
    else:
        # train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=preprocesswithflip, download=True)
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=preprocesswithflip, download=True)

    # test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=preprocess, download=True)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=preprocess, download=True)
                                
    mp.spawn(train, nprocs=args.gpus, args=(train_dataset, test_dataset, args,))


def train(gpu, train_dataset, test_dataset, args):
    #DISTRIBUTED
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', world_size=args.gpus, rank=gpu)

    #SETUPS
    setrandom(args.seed)

    # loss_name = "./aot_reports_1ring/vgg19/cifar100/loss_"+str(args.method)
    trainacc_name = "./aot_reports/trainacc_"+str(args.method)
    testacc_name = "./aot_reports/testacc_"+str(args.method)

    print("We are headed to:", trainacc_name)

    # filename = loss_name
    # ext = ".csv"


    #MODEL AND DATATYPE 
    if args.nodrop:
        model = torchvision.models.vgg19(pretrained=False, dropout=0.0)
    else:
        model = torchvision.models.vgg19(pretrained=False)

    #default is float32
    
    if args.datatype=="F16":
        model.half()
    elif args.datatype=="BF16": 
        model.bfloat16()

    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d): #for numerical stability reasons, otherwise occasional NaN
            layer.float()

    model.cuda(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu)


    # MODIFY HOOK    
    if os.environ["COMM_HOOK"] == "VGG_TOP1":
        model.register_comm_hook(None, topk_allreduce_with_memory)

    if os.environ["COMM_HOOK"] == "VGG_POWERSGD":
        state = PowerSGDState(process_group=None, matrix_approximation_rank=4)
        model.register_comm_hook(state, powerSGD_hook)

    #HYPERPARAMETERS

    batch_size = 512 // args.gpus # Local
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    if args.nol2:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    
    if not args.flat_lr:
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=151, steps_per_epoch=98)

    
    #DATASETS                           
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.gpus, rank=gpu)
                                                                    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               sampler=train_sampler)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size = batch_size,
                                              shuffle=False,
                                              pin_memory=True)

    eval_set = torch.utils.data.Subset(train_dataset, [random.randint(0,len(train_dataset)-1) for i in range(len(test_dataset))])
    eval_loader = torch.utils.data.DataLoader(dataset=eval_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              pin_memory=True)

    total_step = len(train_loader)

    dump_grad_epochs = [1, 2, 3, 4, 5, 10, 50, 100, 149, 150, 200]

    bin_edges = torch.tensor([0, 1000, 4097000, 4101096, 20878312, 20882408, 123642856, 126002152, 126002664, 128361960, 128362472, 130721768, 130722280, 133081576, 133082088, 135441384, 135441896, 137801192, 137801704, 140161000, 140161512, 141341160, 141341672, 141931496, 141931752, 142521576, 142521832, 143111656, 143111912, 143406824, 143407080, 143554536, 143554664, 143628392, 143628520, 143665384, 143665448, 143667176, 143667240]).cuda()
    # histogram_zeros = torch.zeros(len(bin_edges) -1, dtype=torch.int64).cuda()

    # Delete report files if exist
    if torch.distributed.get_rank() == 0:
        try:
            # os.remove(loss_name+".csv")
            os.remove(trainacc_name+".txt")
            os.remove(testacc_name+".txt")
        except:
            print("No existing report files found.")

    last_timestamp = time.time()

    idx = 0

    for epoch in range(args.epochs):

        for i, (images, labels) in enumerate(train_loader):

            model.train()

            # if idx >= 1:
                # assert False

            if args.datatype=="F16":
                images = images.cuda(gpu, non_blocking=True).half()
            elif args.datatype=="BF16":
                images = images.cuda(gpu, non_blocking=True).bfloat16()
            else:
                images = images.cuda(gpu, non_blocking=True)

            labels = labels.cuda(gpu, non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            if gpu == 0 and args.dump_grad:
                if epoch in dump_grad_epochs and idx==0:
                    g = torch.Tensor().cuda(gpu)

                    for params in model.parameters():
                        g_temp = params.grad
                        g_temp = torch.flatten(g_temp)                    
                        g = torch.cat((g, g_temp))

                    g = g.cpu().numpy()

                    np.savetxt(gradname+str(epoch), g)
                    print("done extracting")
            
            optimizer.step()
            if not args.flat_lr:
                scheduler.step()
            
            # Dump loss
            # if gpu == 0:
                # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step, loss.item()))
                # with open(loss_name+ext, "a+") as f:
                    # print("{}".format(loss.item()), file=f)

            # To list the parameters
            # for name, param in model.named_parameters():
                # if torch.distributed.get_rank() == 0 and param.requires_grad:
                    # print("Name:", name, "Size:", param.numel())
                    # print(name)
            # assert False

            # Dump gradient histogram
            # if gpu == 0:
            #     if epoch in dump_grad_epochs and idx==0:
            #         g = torch.Tensor().cuda(gpu)
                    
            #         for params in model.parameters():
            #             g_temp = params.grad
            #             g_temp = torch.flatten(g_temp)                    
            #             g = torch.cat((g, g_temp))
                    
            #         # Find bottom-k
            #         bottom_ratio = 0.25
            #         k = int(bottom_ratio * g.size()[0])
            #         grad_abs = torch.abs(g)
            #         temp = torch.topk(grad_abs, k, largest=False)
            #         threshold = temp.values[-1]

            #         dumpname = "./bottom_grads/vgg19/cifar100/bottom_25/dump_epoch"+str(epoch)

            #         bin_indices = torch.bucketize(temp.indices, bin_edges, right=True)
            #         bin_indices -= 1
            #         hist = torch.bincount(bin_indices)

            #         hist_at_cpu = hist.cpu().numpy()

            #         with open(dumpname, 'w') as f:
            #             np.savetxt(f, hist_at_cpu)

            # torch.distributed.barrier()

            # HANS: For debugging
            # print("Delta at GPU", torch.distributed.get_rank(), "is", time.time() - last_timestamp)

            # if (time.time() - last_timestamp) >= args.timestamp_period:
            #     if gpu == 0:
            #         print("Evaluate at iteration", idx, "at time", time.time(), "!")
            #         evaluation(model, gpu, epoch+1, eval_loader, trainacc_name, "Train", args)
            #         evaluation(model, gpu, epoch+1, test_loader, testacc_name, "Test", args)
            
            #     torch.distributed.barrier()
            #     last_timestamp = time.time()


            if gpu == 0:
                if (time.time() - last_timestamp) >= args.timestamp_period:
                    print("Evaluate at iteration", idx, "at time", time.time(), "!")
                    evaluation(model, gpu, epoch+1, eval_loader, trainacc_name, "Train", args)
                    evaluation(model, gpu, epoch+1, test_loader, testacc_name, "Test", args)
            
                    last_timestamp = time.time()

            idx += 1

        # if not args.flat_lr:
            # scheduler.step()

        
            

def evaluation(model, gpu, epoch, dataloader, filename, evalname, args):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataloader:
            if args.datatype=="F16":
                images = images.cuda(gpu, non_blocking=True).half()
            elif args.datatype=="BF16":
                images = images.cuda(gpu, non_blocking=True).bfloat16()
            else:
                images = images.cuda(gpu, non_blocking=True)

            labels = labels.cuda(gpu, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
    with open(filename+".txt", "a+") as f:
        # print("Epoch {}. {} accuracy = {}%".format(epoch, evalname, accuracy), file=f)  
        print("{}%".format(accuracy), file=f)
        # if torch.distributed.get_rank() == 0:
            # print("{}%".format(accuracy))

if __name__ == '__main__':
    main()
