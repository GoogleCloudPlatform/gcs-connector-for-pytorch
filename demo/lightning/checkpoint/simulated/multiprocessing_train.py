import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed.checkpoint as dist_cp
import os
import time

from lightning.pytorch.strategies import FSDPStrategy

from dataflux_pytorch.lightning.gcs_filesystem import GCSDistributedWriter, GCSDistributedReader
import statistics
from typing import Optional, Dict


class BenchmarkStrategy(FSDPStrategy):
    def __init__(self, project: str, path: str, model: nn.Module, **kwargs) -> None:
        super().__init__(**kwargs)
        self.writer = GCSDistributedWriter(path, project, None)
        self.reader = GCSDistributedReader(path, project, None)
        self.model = model

    def save_checkpoint(self, checkpoint: Dict[str, torch.Tensor], filepath: str, storage_options: Optional[Dict] = None) -> None:
        dist_cp.save(state_dict=checkpoint, checkpoint_id=filepath,
                     storage_writer=self.writer)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        empty_state_dict = {}
        dist_cp.load(state_dict=empty_state_dict,
                     checkpoint_id=checkpoint_path, storage_reader=self.reader)


class SimpleModel(nn.Module):
    def __init__(self, size: int, padding_size: int) -> None:
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.dummy_tensors = [torch.randn(size, size)
                              for _ in range(padding_size)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str)
    parser.add_argument("--ckpt-dir-path", type=str)
    parser.add_argument("--model-size", type=int, default=100)
    parser.add_argument("--clear-kernel-cache",
                        action="store_true",
                        default=False)
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--padding-size", type=int, default=1000)
    parser.add_argument("--world-size", type=int)
    return parser.parse_args()


def format_size(size_bytes: int) -> str:
    size_mb = size_bytes / (1024 * 1024)
    return f"{size_mb / 1024:.2f} GB" if size_mb >= 1024 else f"{size_mb:.2f} MB"


def get_tensor_size_bytes(tensor: torch.Tensor) -> int:
    return tensor.element_size() * tensor.numel()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup() -> None:
    dist.destroy_process_group()


def split_tensor(tensor: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
    numel = tensor.numel()
    split_size = numel // world_size
    start_idx = rank * split_size
    end_idx = start_idx + split_size if rank != world_size - 1 else numel
    return tensor.view(-1)[start_idx:end_idx]


def time_checkpoint_operation(benchmark_strategy: BenchmarkStrategy, distributed_state_dict: Dict[str, torch.Tensor], filepath: str, sample_size: int, operation: str) -> list:
    times = []
    for i in range(sample_size):
        start_time = time.time()
        if operation == 'save':
            checkpoint_path = os.path.join(
                filepath, f'checkpoints/ckpt_{i}.ckpt')
            benchmark_strategy.save_checkpoint(
                distributed_state_dict, filepath=checkpoint_path)
        elif operation == 'load':
            checkpoint_path = os.path.join(
                filepath, f'checkpoints/ckpt_{i}.ckpt')
            benchmark_strategy.load_checkpoint(checkpoint_path=checkpoint_path)
        end_time = time.time()
        times.append(end_time - start_time)
        dist.barrier()  # Synchronize processes
    return times


def run_benchmark(rank, world_size: int, model_size: int, project: str, filepath: str, padding_size: int, sample_size: int):
    setup(rank, world_size)

    model = SimpleModel(model_size, padding_size)

    dummy_input = torch.randn(100, model_size)
    _ = model(dummy_input)

    full_state_dict = model.state_dict()
    for i, tensor in enumerate(model.dummy_tensors):
        full_state_dict[f'dummy_tensor_{i}'] = tensor

    benchmark_strategy = BenchmarkStrategy(
        project=project, path=filepath, model=model)

    distributed_state_dict = {f"{key}_shard_{rank}": split_tensor(
        tensor, world_size, rank) for key, tensor in full_state_dict.items()}

    save_checkpoint_times = time_checkpoint_operation(
        benchmark_strategy, distributed_state_dict, filepath, sample_size, 'save')
    load_checkpoint_times = time_checkpoint_operation(
        benchmark_strategy, distributed_state_dict, filepath, sample_size, 'load')

    if rank == 0:
        print("######################")
        print(
            f"Time taken to save checkpoint: {statistics.mean(save_checkpoint_times):.4f} seconds")
        print(
            f"Time taken to load checkpoint: {statistics.mean(load_checkpoint_times):.4f} seconds")
        total_distributed_size_bytes = sum(get_tensor_size_bytes(
            tensor) for tensor in distributed_state_dict.values())
        print(
            f"Size of distributed tensors (rank {rank}): {format_size(total_distributed_size_bytes)}")
        print(
            f"Total size of all tensors (rank {rank}): {format_size(total_distributed_size_bytes * world_size)}")
        print("######################")


def main():
    args = parse_args()

    mp.set_start_method('spawn')
    mp.spawn(run_benchmark, args=(args.world_size, args.model_size, args.project, args.ckpt_dir_path, args.padding_size, args.sample_size),
             nprocs=args.world_size, join=True)


if __name__ == "__main__":
    # world_size = int(os.getenv("WORLD_SIZE"))
    # model_size = int(os.getenv("NUM_LAYERS"))
    # project = os.getenv("PROJECT")
    # path = os.getenv("CKPT_DIR_PATH")
    # sample_size = int(os.getenv("SAMPLE_SIZE", 3))
    # padding_size = int(os.getenv("PADDING_SIZE", 2000))
    main()
