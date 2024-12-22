# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

# Borrowed from https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/packed_dataset.py
# with minor modificaitons to use GCS Connector for Pytorch for data loading.

import io
import struct
from typing import Optional, Tuple

import numpy as np
from dataflux_core.download import download_single
from dataflux_pytorch import dataflux_iterable_dataset
from google.cloud import storage
from lit_llama.packed_dataset import (CombinedDataset, PackedDataset,
                                      PackedDatasetIterator)
from torch.utils.data import DataLoader, get_worker_info

# Data proportions from https://arxiv.org/pdf/2302.13971.pdf Table 1
data_config = [
    ("arxiv", 2.5),
    ("book", 4.5),
    ("c4", 15.0),
    ("cc", 67.0),
    ("github", 4.5),
    ("stackexchange", 2.0),
    ("wikipedia", 4.5),
]

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}

HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24  # bytes


class DatafluxPackedDataset(PackedDataset):

    def __init__(self,
                 filenames,
                 bucket_name,
                 n_chunks,
                 block_size,
                 seed=12345,
                 shuffle=True,
                 wrap=False,
                 num_processes=1,
                 process_rank=0):
        super().__init__(filenames, n_chunks, block_size, seed, shuffle, wrap,
                         num_processes, process_rank)
        self.bucket_name = bucket_name

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id

        max_num_files = len(self._filenames) // num_shards * num_shards
        filenames = self._filenames[shard_id:max_num_files:num_shards]

        return DatafluxPackedDatasetIterator(filenames=filenames,
                                             n_chunks=self._n_chunks,
                                             block_size=self._block_size,
                                             seed=self._seed,
                                             shuffle=self._shuffle,
                                             wrap=self._wrap,
                                             bucket_name=self.bucket_name)


class DatafluxPackedDatasetIterator(PackedDatasetIterator):

    def __init__(self, filenames, n_chunks, block_size, seed, shuffle, wrap,
                 bucket_name):
        self.storage_client = storage.Client()
        self.bucket_name = bucket_name
        super().__init__(filenames, n_chunks, block_size, seed, shuffle, wrap)

    def _read(self, path):
        bytes_content = download_single(self.storage_client, self.bucket_name,
                                        path)
        bytes_io = io.BytesIO(bytes_content)
        magic = bytes_io.read(len(HDR_MAGIC))
        assert magic == HDR_MAGIC, "File doesn't match expected format."
        version = struct.unpack("<Q", bytes_io.read(8))
        assert (1, ) == version
        (dtype_code, ) = struct.unpack("<B", bytes_io.read(1))
        dtype = dtypes[dtype_code]
        (chunk_size, ) = struct.unpack("<Q", bytes_io.read(8))
        return dtype, chunk_size, bytes_io

    def _load_n_chunks(self):
        self._close_mmaps()
        self._mmaps = []
        self._buffers = []

        if self._n_chunks > len(self._filenames[self._file_idx:]):
            if not self._wrap:
                raise StopIteration
            else:
                self._file_idx = 0

        for i in range(self._n_chunks):
            filename = self._filenames[self._file_idx + i]
            if self._dtype is None:
                self._dtype, self._chunk_size, bytes_io = self._read(filename)
                self._n_blocks = self._chunk_size // self._block_size

            bytes_io.seek(HDR_SIZE)
            self._buffers.append(bytes_io.read())

        self._file_idx += self._n_chunks
        n_all_blocks = self._n_chunks * self._n_blocks

        self._block_idxs = (self._rng.permutation(n_all_blocks)
                            if self._shuffle else range(n_all_blocks))

        self._curr_idx = 0


def list_with_dataflux(project_name, bucket_name):
    dataset = dataflux_iterable_dataset.DataFluxIterableDataset(
        project_name=project_name, bucket_name=bucket_name)
    filenames = [name for name, _ in dataset.objects]
    return filenames


def create_dataloader(
    project_name: str,
    bucket_name: str,
    batch_size: int,
    block_size: int,
    data_dir: str,
    fabric,
    shuffle: bool = True,
    seed: int = 12345,
) -> DataLoader:
    datasets = []
    filenames = list_with_dataflux(project_name, bucket_name)
    for prefix, _ in data_config:
        files_in_this_dataset = [
            name for name in filenames if name.startswith(prefix)
        ]

        dataset = DatafluxPackedDataset(
            files_in_this_dataset,
            bucket_name=bucket_name,
            n_chunks=4,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets,
                                       seed=seed,
                                       weights=weights)

    return DataLoader(combined_dataset,
                      batch_size=batch_size,
                      shuffle=False,
                      pin_memory=True)


def create_dataloaders(
    project_name: str,
    bucket_name: str,
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: str = "data/lit-redpajama",
    val_data_dir: Optional[str] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        project_name=project_name,
        bucket_name=bucket_name,
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
    )
    val_dataloader = (create_dataloader(
        project_name=project_name,
        bucket_name=bucket_name,
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=val_data_dir,
        shuffle=False,
        seed=seed,
    ) if val_data_dir else None)
    return train_dataloader, val_dataloader
