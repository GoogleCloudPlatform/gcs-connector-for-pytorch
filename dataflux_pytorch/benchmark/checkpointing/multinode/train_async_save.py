import os
import statistics
import time

import torch
import torch.distributed as dist
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos import (LightningTransformer, Transformer,
                                     WikiText2)
from torch.utils.data import DataLoader

from demo.lightning.checkpoint.multinode.strategies import DatafluxFSDPStrategy
from demo.lightning.checkpoint.multinode.train import init_processes


class DemoTransformer(LightningTransformer):

    def __init__(
        self,
        vocab_size: int = 33278,
        nlayers: int = 2,
        simulated_workload_duration: int = 0,
    ) -> None:
        super().__init__(vocab_size)
        self.simulated_workload_duration = simulated_workload_duration
        self.model = Transformer(vocab_size=vocab_size, nlayers=nlayers)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.trainer.model.parameters(), lr=0.1)

    def forward(self, inputs, target):
        # Sleep for a few seconds to simulate cpu/gpu workload.
        time.sleep(self.simulated_workload_duration)
        return super().forward(inputs, target)


def main():
    project = os.getenv("PROJECT")
    num_nodes = int(os.environ.get("NUM_NODES", 1))
    devices = os.environ.get("NUM_DEVICES", 'auto')

    use_async = bool(os.getenv("USE_ASYNC", "1") == "1")
    ckpt_dir_path = os.getenv("CKPT_DIR_PATH")
    num_layers = int(os.environ.get("NUM_LAYERS", 10))
    min_epochs = int(os.environ.get("MIN_EPOCHS", 4))
    max_epochs = int(os.environ.get("MAX_EPOCHS", 5))
    max_steps = int(os.environ.get("MAX_STEPS", 3))
    steps_per_save = int(os.environ.get("STEPS_PER_SAVE", 1))
    run_count = int(os.getenv("NUM_SAVE_CALLS", 1))
    simulated_workload_duration = int(
        os.getenv("SIMULATED_WORKLOAD_DURATION", 1))

    rank = 0
    if os.environ.get("COORDINATOR_ADDRESS"):
        init_processes()
        dist.init_process_group("gloo",
                                rank=int(os.environ.get("NODE_RANK", 0)),
                                world_size=num_nodes)

    torch.cuda.empty_cache()

    dataset = WikiText2()
    dataloader = DataLoader(dataset, num_workers=1)

    trainer_fit_times = []
    for i in range(run_count):

        model = DemoTransformer(
            vocab_size=dataset.vocab_size,
            nlayers=num_layers,
            simulated_workload_duration=simulated_workload_duration,
        )

        checkpoint_callback = ModelCheckpoint(
            save_top_k=-1,
            every_n_train_steps=steps_per_save,
            filename="checkpoint-{epoch:02d}-{step:02d}",
            enable_version_counter=True,
            dirpath=ckpt_dir_path,
        )

        strategy = DatafluxFSDPStrategy(
            project_name=project,
            storage_client=None,
            state_dict_type="sharded",
            use_orig_params=False,
            use_async=use_async,
        )

        trainer = Trainer(
            default_root_dir=ckpt_dir_path,
            plugins=[],
            callbacks=[checkpoint_callback],
            max_steps=max_steps,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            accelerator="gpu",
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            enable_progress_bar=False,
        )

        init_start_event = torch.cuda.Event(enable_timing=True)
        init_end_event = torch.cuda.Event(enable_timing=True)

        init_start_event.record()
        trainer.fit(model, dataloader)
        init_end_event.record()

        total_time = init_start_event.elapsed_time(init_end_event) / 1000
        rank = int(os.environ.get("NODE_RANK", 0))
        print(
            f"Individual run {i+1} of {run_count} trainer.fit() #{rank} took {total_time} seconds."
        )
        trainer_fit_times.append(total_time)

    # All runs complete.
    if int(os.environ.get("NODE_RANK", 0)) == 0:
        avg_trainer_fit_time = statistics.mean(trainer_fit_times)
        avg_trainer_fit_time_str = str(avg_trainer_fit_time) + " seconds"
        print("##################################")
        print("Average time for trainer.fit(): " + avg_trainer_fit_time_str)
        print("##################################")
        print(f"All trainer.fit() times: {trainer_fit_times}")

    # Cleanup.
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
