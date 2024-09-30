"""
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """
import os
import sys
import socket
import time

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos import (LightningTransformer, Transformer,
                                     WikiText2)
from torch.utils.data import DataLoader

from dataflux_pytorch.lightning import DatafluxLightningCheckpoint


def configure_master_addr():
    """Get coordinator IP Address with retries"""
    coordinator_address = ""
    coordinator_ip_address = ""
    if os.environ.get("COORDINATOR_ADDRESS") is not None:
        coordinator_address = os.environ.get("COORDINATOR_ADDRESS")
        coordinator_found = False
        lookup_attempt = 1
        max_coordinator_lookups = 50
        while not coordinator_found and lookup_attempt <= max_coordinator_lookups:
            try:
                coordinator_ip_address = socket.gethostbyname(
                    coordinator_address)
                coordinator_found = True
            except socket.gaierror:
                print(
                    f"Failed to recognize coordinator address {coordinator_address} on"
                    f" attempt {lookup_attempt}, retrying...")
                lookup_attempt += 1
                time.sleep(5)
    print(f"Coordinator IP address: {coordinator_ip_address}")
    os.environ["MASTER_ADDR"] = str(coordinator_ip_address)


def init_processes():
    """Initializes the distributed environment."""
    # Get the necessary environment variables from the GKE environment
    world_size = int(os.environ["WORLD_SIZE"])

    job_index = int(os.environ.get("JOB_INDEX"))
    job_completion_index = int(os.environ.get("JOB_COMPLETION_INDEX"))
    processes_in_job = int(os.environ.get("PROCESSES_IN_JOB"))
    rank = job_index * processes_in_job + job_completion_index
    os.environ["NODE_RANK"] = str(rank)

    configure_master_addr()


def main(project: str, ckpt_dir_path: str, save_only_latest: bool):
    if os.environ.get("COORDINATOR_ADDRESS"):
        init_processes()
    dataset = WikiText2()
    dataloader = DataLoader(dataset, num_workers=1)

    model = DemoTransformer(vocab_size=dataset.vocab_size,
                            nlayers=int(os.environ.get("NUM_LAYERS", 2)))
    dataflux_ckpt = DatafluxLightningCheckpoint(project_name=project)
    # Save once per step, and if `save_only_latest`, replace the last checkpoint each time.
    # Replacing is implemented by saving the new checkpoint, and then deleting the previous one.
    # If `save_only_latest` is False, a new checkpoint is created for each step.
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1 if save_only_latest else -1,
        every_n_train_steps=1,
        filename="checkpoint-{epoch:02d}-{step:02d}-{self.global_rank}",
        enable_version_counter=True,
    )
    accelerator = os.environ.get("ACCELERATOR", "cpu")
    min_epochs = os.environ.get("MIN_EPOCHS", 4)
    max_epochs = os.environ.get("MAX_EPOCHS", 5)
    max_steps = os.environ.get("MAX_STEPS", 3)
    trainer = Trainer(default_root_dir=ckpt_dir_path,
                      plugins=[dataflux_ckpt],
                      callbacks=[checkpoint_callback],
                      min_epochs=min_epochs,
                      max_epochs=max_epochs,
                      max_steps=max_steps,
                      accelerator=accelerator,
                      devices=4,
                      )
    trainer.fit(model, dataloader)

    start = time.time()

    for i in range(max_steps):
        print(
            f"\n## Writing checkpoint for step {i} on Node {trainer.global_rank}")
        checkpoint_path = os.path.join(
            ckpt_dir_path, f'checkpoints/ckpt_{i}_{trainer.global_rank}.ckpt')
        print(f"\n## Checkpoint path: {checkpoint_path}")
        trainer.save_checkpoint(checkpoint_path)
        print(f"\n## Checkpoint saved: {checkpoint_path}")
    end = time.time()
    print("Average time to save one checkpoint: " +
          str((end - start) / max_steps) + " seconds")
# start = time.time()
# for i in range(max_steps):
#     data = dataflux_ckpt.load_checkpoint(
#         os.path.join(ckpt_dir_path, f'checkpoints/ckpt_{i}_{trainer.global_rank}.ckpt'))
# end = time.time()
# print("Average time to load one checkpoint: " +
#       str((end - start) / max_steps) + " seconds")


class DemoTransformer(LightningTransformer):

    def __init__(
        self,
        vocab_size: int = 33278,
        nlayers: int = 2,
    ) -> None:
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size, nlayers=nlayers)


if __name__ == "__main__":

    main(
        "gcs-tess",
        "gs://yashsha-us-east1-d/",
        os.getenv("SAVE_ONLY_LATEST") == "1",
    )
