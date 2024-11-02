import argparse
import os
import statistics
import time

import torch
import torch.distributed
import torch.nn as nn
from google.cloud import storage
from lightning import Trainer
from lightning.pytorch.demos import WikiText2
from lightning.pytorch.strategies import FSDPStrategy
from torch.utils.data import DataLoader

from demo.lightning.checkpoint.multinode.strategies import (
    DatafluxFSDPStrategy, FSSpecFSDPStrategy, LoadFromBootDiskFSDP)
from demo.lightning.checkpoint.multinode.train import (DemoTransformer,
                                                       init_processes)

DF_FSDP_STRATEGY = "dataflux_fsdp"
FSSPEC_FSDP_STRATEGY = "fsspec_fsdp"
FSDP_STRATEGY = "fsdp"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_only", action="store_true", default=False)
    parser.add_argument("--load_only", action="store_true", default=False)
    parser.add_argument(
        '--strategy',
        choices=[DF_FSDP_STRATEGY, FSSPEC_FSDP_STRATEGY, FSDP_STRATEGY],
        default=DF_FSDP_STRATEGY)
    parser.add_argument("--distributed_filesystem",
                        action="store_true",
                        default=False)
    return parser.parse_args()


def validate(args):
    if args.distributed_filesystem and args.strategy != FSDP_STRATEGY:
        raise ValueError(
            "Strategy must be set to fsdp when using a distributed filesystem like GCSFuse or Filestore"
        )
    if args.save_only and args.load_only:
        raise ValueError("Either save_only or load_only can be set.")
    if args.save_only and args.strategy != FSDP_STRATEGY:
        raise ValueError(
            "strategy must be set to fsdp if benchmarking only checkpoint saves."
        )
    if args.load_only and args.strategy != FSDP_STRATEGY:
        raise ValueError(
            "strategy must be set to fsdp if benchmarking only checkpoint loads."
        )


def get_strategy(args, project, ckpt_dir_path):
    strategy = None
    policy = {nn.TransformerEncoderLayer, nn.TransformerDecoderLayer}
    if args.strategy == DF_FSDP_STRATEGY:
        print("Using DatafluxFSDPStrategy")
        strategy = DatafluxFSDPStrategy(
            path=ckpt_dir_path,
            project_name=project,
            storage_client=None,
            state_dict_type="sharded",
            use_orig_params=False,
            auto_wrap_policy=policy,
        )
    elif args.strategy == FSSPEC_FSDP_STRATEGY:
        print("Using FSSpecFSDPStrategy")
        strategy = FSSpecFSDPStrategy(
            path=ckpt_dir_path,
            state_dict_type="sharded",
            use_orig_params=False,
            auto_wrap_policy=policy,
        )
    elif args.strategy == FSDP_STRATEGY and args.load_only:
        print("Using CustomFSDPStrategy.")
        strategy = LoadFromBootDiskFSDP(
            ckpt_path=ckpt_dir_path,
            project_name=project,
            state_dict_type="sharded",
            use_orig_params=False,
            auto_wrap_policy=policy,
        )
    elif (args.strategy == FSDP_STRATEGY
          and args.save_only) or args.distributed_filesystem:
        print("Using FSDPStrategy.")
        strategy = FSDPStrategy(
            state_dict_type="sharded",
            use_orig_params=False,
            auto_wrap_policy=policy,
        )
    else:
        raise ValueError("Invalid strategy.")
    return strategy


def copy_bucket_to_local(bucket_name, local_dir):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs()

    for blob in blobs:
        local_path = os.path.join(local_dir, blob.name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)


def print_times(args, avg_save_time, avg_load_time):
    avg_save_time_str = str(avg_save_time) + " seconds"
    avg_load_time_str = str(avg_load_time) + " seconds"
    if args.save_only:
        avg_load_time_str = "skipped"
    elif args.load_only:
        avg_save_time_str = "skipped"

    print("##################################")
    print("Average time to save one checkpoint: " + avg_save_time_str)
    print("Average time to load one checkpoint: " + avg_load_time_str)
    print("##################################")


def main(ckpt_dir_path: str, ckpt_restore_path: str = ""):
    args = parse_args()
    validate(args)

    if os.environ.get("COORDINATOR_ADDRESS"):
        init_processes()
    torch.cuda.empty_cache()
    dataset = WikiText2()
    dataloader = DataLoader(dataset, num_workers=1)

    strategy = get_strategy(args, os.getenv("PROJECT"), ckpt_dir_path)
    num_save_calls = int(os.environ.get("NUM_SAVE_CALLS", 3))
    num_nodes = int(os.environ.get("NUM_NODES", 1))

    trainer = Trainer(
        enable_checkpointing=False,
        logger=False,
        default_root_dir=ckpt_dir_path,
        plugins=[],
        min_epochs=1,
        max_epochs=1,
        max_steps=1,
        accelerator=os.environ.get("ACCELERATOR", "gpu"),
        strategy=strategy,
        devices=os.environ.get("NUM_DEVICES", 'auto'),
        num_nodes=num_nodes,
    )

    init_from_checkpoint = os.environ.get("CKPT_RESTORE_PATH") is not None
    with trainer.init_module(empty_init=init_from_checkpoint):
        if init_from_checkpoint:
            model = DemoTransformer.load_from_checkpoint(
                os.environ.get("CKPT_RESTORE_PATH"),
                nlayers=int(os.environ.get("NUM_LAYERS", 10)))
        else:
            model = DemoTransformer(vocab_size=dataset.vocab_size,
                                    nlayers=int(
                                        os.environ.get("NUM_LAYERS", 10)))
            trainer.fit(model, dataloader)
    trainer.print(torch.cuda.memory_summary())
    print(f"Saving checkpoint to {ckpt_dir_path} {num_save_calls} times.")
    checkpoint_paths = []
    save_checkpoint_times = []
    for i in range(num_save_calls):
        checkpoint_path = os.path.join(ckpt_dir_path,
                                       f'checkpoints/ckpt_{i}.ckpt/')
        checkpoint_paths.append(checkpoint_path)
        start = time.time()
        trainer.save_checkpoint(checkpoint_path)
        end = time.time()
        save_checkpoint_times.append(end - start)
    if torch.distributed.get_rank() == 0:
        print(f"Saved checkpoint to {ckpt_dir_path} {num_save_calls} times.")
    num_load_calls = int(os.environ.get("NUM_LOAD_CALLS", 3))
    load_checkpoint_times = []
    # TODO deal with these
    if args.save_only:
        print("Skipping loads because you set --save_only")
        num_load_calls = 0
        load_checkpoint_times = [0]
    elif args.load_only:
        print(f"Copying contents of {ckpt_dir_path} to {ckpt_restore_path}")
        copy_bucket_to_local(ckpt_dir_path.removeprefix("gs://"),
                             os.path.dirname(ckpt_restore_path))
        avg_save_time = 0
    for i in range(num_load_calls):
        new_ckpt_dir_path = checkpoint_paths[i]
        with trainer.init_module(empty_init=True):
            start = time.time()
            model = DemoTransformer.load_from_checkpoint(new_ckpt_dir_path)
            end = time.time()
        total = end - start
        if torch.distributed.get_rank() == 0:
            print(
                f"Iteration {i}: Loaded checkpoint from {new_ckpt_dir_path} in {total} seconds"
            )
        load_checkpoint_times.append(total)

    if torch.distributed.get_rank() == 0:
        avg_load_time = statistics.mean(load_checkpoint_times)
        avg_save_time = statistics.mean(save_checkpoint_times)
        print_times(args, avg_save_time, avg_load_time)
        print(f"All save times: {save_checkpoint_times}")
        print(f"All load times: {load_checkpoint_times}")


if __name__ == "__main__":
    start = time.time()
    main(
        os.getenv("CKPT_DIR_PATH"),
        os.getenv("CKPT_RESTORE_PATH"),
    )
    end = time.time()
    print(f"Benchmark took {end - start} seconds.")
