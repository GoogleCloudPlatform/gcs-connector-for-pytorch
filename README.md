# Google Cloud Storage Connector for PyTorch

Cloud Storage's Connector for PyTorch is an open source product supported by Google that provides a direct Cloud Storage integration with PyTorch. To learn more, see the official documentation [here](https://cloud.google.com/storage/docs/pytorch-connector).

The Connector for PyTorch implements PyTorch’s [dataset primitive](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) that can be used to efficiently load training data from GCS. The library currently supports [map-style datasets](https://pytorch.org/docs/stable/data.html#map-style-datasets) for random data access patterns and [iterable-style datasets](https://pytorch.org/docs/stable/data.html#iterable-style-datasets) for streaming data access patterns.

Furthermore, the Connector for PyTorch provides a checkpointing interface to conveniently save and load checkpoints directly to and from a Google Cloud Storage (GCS) bucket.

All of these features can be used out of the box for performant executions with single-node and multinode ML workflows. Demo code for multinode implementation using FSDP can be found in our [multinode README](https://github.com/GoogleCloudPlatform/dataflux-pytorch/tree/main/demo/lightning/checkpoint/multinode).

## Getting started

### Prerequisites
- Python 3.8 or greater is installed (Note: Using 3.12+ is not recommended as PyTorch does not support).

### Installation

```shell
pip install gcs-torch-dataflux
```

### Configuration
Authentication must be provided to use the Connector for PyTorch via [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials) through one of the following methods:
1. While running this library on a GCE VM, Application Default Credentials will automatically use the VM’s attached service account by default. More details can be found [here](https://cloud.google.com/compute/docs/access/app-authentication-methods).
2. Application Default Credentials can also be configured manually as described [here](https://cloud.google.com/docs/authentication/application-default-credentials). The quickest way is to log in directly using the gcloud CLI:
```shell
gcloud auth application-default login
```

### Examples
Please checkout the `demo` directory for a complete set of examples, which includes a [simple starter Jupyter Notebook (hosted by Google Colab)](demo/simple-walkthrough/Getting%20Started%20with%20Dataflux%20Dataset%20for%20PyTorch%20with%20Google%20Cloud%20Storage.ipynb) and an [end-to-end image segmentation training workload walkthrough](demo/image_segmentation/README.md). Those examples will help you understand how the Connector for PyTorch works and how you can integrate it into your own workload.

#### Sample Examples
Before getting started, please make sure you have installed the library and configured authentication following the instructions above.

##### Data Loading
Both map-style and iterable-style datasets for PyTorch can be constructed by specifying the project name, bucket name and an optional prefix.
```python
from dataflux_pytorch import dataflux_mapstyle_dataset

# Please update these fields.
PROJECT_NAME = "<PROJECT_NAME>"
BUCKET_NAME = "<BUCKET_NAME>"
PREFIX = "<PREFIX>"

# Map-style dataset.
map_dataset = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
  project_name=PROJECT_NAME,
  bucket_name=BUCKET_NAME,
  config=dataflux_mapstyle_dataset.Config(prefix=PREFIX),
)

# Random access to an object.
sample_object = map_dataset.objects[0]

# Learn about the name and the size (in bytes) of the object.
name = sample_object[0]
size = sample_object[1]

# Iterate over the datasets.
for each_object in map_dataset:
  # Raw bytes of the object.
  print(each_object)
```
Similarly, for `DataFluxIterableDataset`:
```python
from dataflux_pytorch import dataflux_iterable_dataset

# Iterable-style dataset.
iterable_dataset = dataflux_iterable_dataset.DataFluxIterableDataset(
  project_name=PROJECT_ID,
  bucket_name=BUCKET_NAME,
  config=dataflux_iterable_dataset.Config(prefix=PREFIX),
)

for each_object in iterable_dataset:
  # Raw bytes of the object.
  print(each_object)
```

The Connector for PyTorch offers the flexibility to transform the downloaded raw bytes of data into any format of choice. This is particularly useful since the `PyTorch DataLoader` works well with Numpy arrays or PyTorch tensors.

```python
# Assume that you have a bucket with image files and you want
# to load them into Numpy arrays for training.
import io
import numpy
from PIL import Image

transform = lambda img_in_bytes : numpy.asarray(Image.open(io.BytesIO(img_in_bytes)))

map_dataset = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
  project_name=PROJECT_NAME,
  bucket_name=BUCKET_NAME,
  config=dataflux_mapstyle_dataset.Config(prefix=PREFIX),
  data_format_fn=transform,
)

for each_object in map_dataset:
  # each_object is now a Numpy array.
  print(each_object)
```

Similarly, for `DataFluxIterableDataset`:
```python
iterable_dataset = dataflux_iterable_dataset.DataFluxIterableDataset(
  project_name=PROJECT_ID,
  bucket_name=BUCKET_NAME,
  config=dataflux_mapstyle_dataset.Config(prefix=PREFIX),
  data_format_fn=transform,
)
for each_object in dataset:
  # each_object is now a Numpy array.
  print(each_object)
```

##### Checkpointing

The Connector for PyTorch supports fast data loading and allows the user to save and load model checkpoints directly to/from a Google Cloud Storage (GCS) bucket. The checkpointing implementation leverages multipart file upload to parallelize checkpoint writes to GCS, greatly increasing performance over single-threaded writes.

```python
import torch
import torchvision

from dataflux_pytorch import dataflux_checkpoint

ckpt = dataflux_checkpoint.DatafluxCheckpoint(
  project_name=PROJECT_NAME, bucket_name=BUCKET_NAME
)
CKPT_PATH = "checkpoints/epoch0.ckpt"

model = torchvision.models.resnet50()

with ckpt.writer(CKPT_PATH) as writer:
  torch.save(model.state_dict(), writer)

with ckpt.reader(CKPT_PATH) as reader:
  read_state_dict = torch.load(reader)

model.load_state_dict(read_state_dict)
```

Note that saving or restoring checkpoint files will stage the checkpoint file in CPU memory during save/restore, requiring additional available CPU memory equal to the size of the checkpoint file.

##### Lightning Checkpointing

The Connector for PyTorch also provides an integration for PyTorch Lightning, featuring a DatafluxLightningCheckpoint, an implementation of PyTorch Lightning's CheckpointIO.

End to end example and the notebook for the PyTorch Lightning integration can be found in the [demo/lightning](https://github.com/GoogleCloudPlatform/dataflux-pytorch/tree/main/demo/lightning) directory.

```python
from lightning import Trainer
from lightning.pytorch.demos import WikiText2, LightningTransformer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from dataflux_pytorch.lightning import DatafluxLightningCheckpoint

CKPT = "gcs://BUCKET_NAME/checkpoints/ckpt.ckpt"
dataflux_ckpt = DatafluxLightningCheckpoint(project_name=PROJECT_NAME)

dataset = WikiText2()
dataloader = DataLoader(dataset, num_workers=1)
model = LightningTransformer(vocab_size=dataset.vocab_size)
trainer = Trainer(
    plugins=[dataflux_ckpt],
    min_epochs=4,
    max_epochs=5,
    max_steps=10,
    accelerator="cpu",
)

trainer.fit(model, dataloader)
trainer.save_checkpoint(CKPT)
```

Note that saving or restoring checkpoint files will stage the checkpoint file in CPU memory during save/restore, requiring additional available CPU memory equal to the size of the checkpoint file.

##### Async Checkpointing

PyTorch Lightning single-node - Our lightning checkpointing implementation has built-in support for the experimental [AsyncCheckpointIO](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.plugins.io.AsyncCheckpointIO.html#asynccheckpointio) featureset. This is an optimzation that allows for non-blocking `save_checkpoint` calls during a training loop. For more details on our support for this feature please see the [checkpoint README](https://github.com/GoogleCloudPlatform/dataflux-pytorch/tree/main/demo/lightning/checkpoint/singlenode#using-asynccheckpointio). This lightning feature is only supported for single-node executions.

PyTorch single-node - An example of how you can perform asynchrenous checkpoint saves can be found [here](https://github.com/GoogleCloudPlatform/dataflux-pytorch/tree/main/demo/checkpointing/). This is useful when training a model that results in a large checkpoint file size and avoid blocking training during saving until the upload has finished.

PyTorch multi-node - To allow for asyncronous saves with multinode executions we utilize PyTorch's [Async Distributed Save](https://pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html) to allow for similar non-blocking checkpoint operations. To leverage this feature, initialize [DatafluxFSDPStrategy](https://github.com/GoogleCloudPlatform/dataflux-pytorch/blob/05ce5af3d2a2efdffcea71292170054652bc80bf/demo/lightning/checkpoint/multinode/strategies.py#L44) with the `use_async=True` parameter. An example usage can be found [here](https://github.com/GoogleCloudPlatform/dataflux-pytorch/blob/main/dataflux_pytorch/benchmark/checkpointing/multinode/train_async_save.py).

> [!NOTE]
> Async checkpointing is experimental and currently does not verify if the save has succeeded. 

Sample code:
```python
import torch.distributed as dist
from torch.distributed.checkpoint import async_save
from dataflux_pytorch.lightning.gcs_filesystem import GCSDistributedWriter

# Initialize the process group.
default_ranks = list(range(dist.get_world_size()))
checkpoint_group = dist.new_group(default_ranks, backend='nccl')

# Create GCS Distributed Writer for checkpoint saves.
writer = GCSDistributedWriter(path, project, storage_client)

# Do work...
checkpoint_future = None
for _ in range(num_epochs):
    # Workload step
    # ...

    # Avoid queuing more then one checkpoint request at a time.
    if checkpoint_future is not None:
        checkpoint_future.result()

    checkpoint_future = async_save(
        state_dict,
        checkpoint_id=path,
        storage_writer=writer,
        process_group=checkpoint_group)
```

## Performance

### Map-style Dataset
We tested the Map-style Dataset's early performance using [DLIO benchmark](https://github.com/argonne-lcf/dlio_benchmark) simulations with standard mean file-sizes and dataset sizes. A total of 5 training epochs were simulated. For small files (100KB, 500KB), the Connector for PyTorch can be **2-3x** faster than using GCS native APIs.

<table>
  <tr>
   <td style="background-color: #d9d2e9"><strong>File size / count</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Tool</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Training time (s)</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" style="background-color: #d9d9d9"><em>100 KiB / 500000 files</em>
   </td>
   <td style="background-color: #d9d9d9">Direct GCS API calls
   </td>
   <td style="background-color: #d9d9d9">1,299
   </td>
  </tr>
  <tr>
   <td style="background-color: #d9d9d9"><strong>Connector Map-style Dataset</strong>
   </td>
   <td style="background-color: #d9d9d9"><strong>515</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" style="background-color: #f3f3f3"><em>500 KiB / 2.2m files</em>
   </td>
   <td style="background-color: #f3f3f3">Direct GCS API calls
   </td>
   <td style="background-color: #f3f3f3">6,499
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3"><strong>Connector Map-style Dataset</strong>
   </td>
   <td style="background-color: #f3f3f3"><strong>2,058</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" style="background-color: #d9d9d9"><em>3 MiB / 50000 files</em>
   </td>
   <td style="background-color: #d9d9d9">Direct GCS API calls
   </td>
   <td style="background-color: #d9d9d9">399
   </td>
  </tr>
  <tr>
   <td style="background-color: #d9d9d9"><strong>Connector Map-style Dataset</strong>
   </td>
   <td style="background-color: #d9d9d9"><strong>277</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" style="background-color: #f3f3f3"><em>150 MiB / 5000 files</em>
   </td>
   <td style="background-color: #f3f3f3">Direct GCS API calls
   </td>
   <td style="background-color: #f3f3f3">1,396
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3"><strong>Connector Map-style Dataset</strong>
   </td>
   <td style="background-color: #f3f3f3"><strong>1,173</strong>
   </td>
  </tr>
</table>

### Iterable-style Dataset
Since the [DLIO benchmark](https://github.com/argonne-lcf/dlio_benchmark) doesn’t easily support an implementation of a PyTorch iterable dataset, we implemented a [simple training loop](demo/list-and-download/iterable/simple_iterable_dataset.py) that has similar IO behaviors as the DLIO benchmark and used that loop to benchmark the Iterable Datasets.

<table>
  <tr>
   <td style="background-color: #d9d2e9"><strong>File size / count</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Tool</strong>
   </td>
   <td style="background-color: #d9d2e9"><strong>Training time (s)</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" style="background-color: #d9d9d9"><em>100 KiB / 500000 files</em>
   </td>
   <td style="background-color: #d9d9d9">Direct GCS API calls
   </td>
   <td style="background-color: #d9d9d9">1,145
   </td>
  </tr>
  <tr>
   <td style="background-color: #d9d9d9"><strong>Connector Iterable-style Dataset</strong>
   </td>
   <td style="background-color: #d9d9d9"><strong>611</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" style="background-color: #f3f3f3"><em>500 KiB / 2.2m files</em>
   </td>
   <td style="background-color: #f3f3f3">Direct GCS API calls
   </td>
   <td style="background-color: #f3f3f3">5,174
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3"><strong>Connector Iterable-style Dataset</strong>
   </td>
   <td style="background-color: #f3f3f3"><strong>2,503</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" style="background-color: #d9d9d9"><em>3 MiB / 50000 files</em>
   </td>
   <td style="background-color: #d9d9d9">Direct GCS API calls
   </td>
   <td style="background-color: #d9d9d9">413
   </td>
  </tr>
  <tr>
   <td style="background-color: #d9d9d9"><strong>Connector Iterable-style Dataset</strong>
   </td>
   <td style="background-color: #d9d9d9"><strong>384</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" style="background-color: #f3f3f3"><em>150 MiB / 5000 files</em>
   </td>
   <td style="background-color: #f3f3f3">Direct GCS API calls
   </td>
   <td style="background-color: #f3f3f3">1,225
   </td>
  </tr>
  <tr>
   <td style="background-color: #f3f3f3"><strong>Connector Iterable-style Dataset</strong>
   </td>
   <td style="background-color: #f3f3f3"><strong>1,143</strong>
   </td>
  </tr>
</table>

*Note: Within each experiment, all training parameters such as batch size and parallelism are consistent. The team is working on publishing a detailed analysis soon.*

### Checkpointing
Checkpoint benchmarks were taken on a single GCE `n2d-standard-48` node co-located with a storage bucket in the `us-west1` region. These results were the average of 10 runs. Checkpoints were tested on [PyTorch Lightning's Transformer demo model](https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/demos/transformer.py) ranging in size from 19.8M to 6.5B parameters (~76MB to ~24GB checkpoints). Checkpoints used PyTorch Lightning's default checkpoint format where a single checkpoint file is produced.

### Checkpoint Save

| Checkpoint Type | Model Parameters | Checkpoint File Size (MB) | Avg Checkpoint Save Time | Write Throughput (MB/s) |
| --- | --- | --- | --- | --- |
| Without Connector | 19.8M  | 75.6  | 0.81    | 93.33   |
| Connector         | 19.8M  | 75.6  | 0.56    | 135.00  |
| Without Connector | 77.9 M | 298   | 2.87    | 103.98  |
| Connector         | 77.9 M | 298   | 1.03    | 289.32  |
| Without Connector | 658 M  | 2,500 | 25.61   | 97.61   |
| Connector         | 658 M  | 2,500 | 6.25    | 400.00  |
| Without Connector | 6.5 B  | 24,200| 757.10  | 31.96   |
| Connector         | 6.5 B  | 24,200| 64.50   | 375.19  |


### Checkpoint Load

| Checkpoint Type | Model Parameters | Checkpoint File Size (MB) | Avg Checkpoint Restore Time | Read Throughput (MB/s) |
| --- | --- | --- | --- | --- |
| Without Connector   | 19.8M  | 75.6  | 2.38      | 31.76   |
| Connector           | 19.8M  | 75.6  | 0.51      | 148.24  |
| Without Connector   | 77.9 M | 298   | 1.69      | 176.33  |
| Connector           | 77.9 M | 298   | 1.03      | 289.32  |
| Without Connector   | 658 M  | 2,500 | 186.57    | 13.40   |
| Connector           | 658 M  | 2,500 | 14.77     | 169.26  |
| Without Connector   | 6.5 B  | 24,200| 2,093.52  | 11.56   |
| Connector           | 6.5 B  | 24,200| 113.14    | 213.89  |

## Additional Technical Limitations

For more detail on technical limitations please see our [official documentation](https://cloud.google.com/storage/docs/pytorch-connector).

### Composite Objects
To optimize the download performance of small files, the Connector for PyTorch library utilizes the [GCS Compose API](https://cloud.google.com/storage/docs/json_api/v1/objects/compose) to concatenate a set of smaller objects into a new and larger one in the same bucket under a folder named “dataflux-composed-objects”. The new composite objects will be removed at the end of your training loop but in rare cases that they don’t, you can run this command to clean the composed files.
``` shell
gcloud storage rm --recursive gs://<my-bucket>/dataflux-composed-objects/
```

You can turn of this behavior by setting the "disable_compose" parameter to true, or by setting the “max_composite_object_size” parameter to 0 when constructing the dataset. Example:
```python
dataset = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
  project_name=PROJECT_NAME,
  bucket_name=BUCKET_NAME,
  config=dataflux_mapstyle_dataset.Config(
    prefix=PREFIX,
    max_composite_object_size=0,
    disable_compose=True,
  ),
)
```

Note that turning off this behavior may cause the training loop to take significantly longer to complete when working with small files. However, composed download will hit QPS and throughput limits at a lower scale than downloading files directly, so you should disable this behavior when running at high multi-node scales where you are able to hit project QPS or throughput limits without composed download.

#### Egress Throughput Limits
429 errors acompanied with messages indicating `This workload is drawing too much egress bandwidth from Google Cloud Storage` or `triggered the Cloud Storage Egress Bandwidth Cap` indicate that the data throughput rate of your workload is exceeding the maximum capacity of your Google Cloud Project. To address these issues, you can take the following steps:

1. Check that other workloads executing within your project are not drawing excess bandwidth. Bandwidth usage can be viewed by following steps [here](https://cloud.google.com/storage/docs/bandwidth-usage#bandwidth-monitoring).
2. Follow [these instructions](https://cloud.google.com/storage/docs/bandwidth-usage#increase) to apply for a quota increae of up to 1 Tbps.
3. If more than 1 Tpbs is required, contact your Technical Account Manager or Google representative to file a request on your behalf. This request should specify that you wish to increase the GCS bandwidth caps on your project.
4. Adjust the `listing_retry_config` and `download_retry_config` options to tune your retry backoff and maximize performance.

    ```python
    from google.cloud.storage.retry import DEFAULT_RETRY

    dataset = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
        project_name=PROJECT_NAME,
        bucket_name=BUCKET_NAME,
        config=dataflux_mapstyle_dataset.Config(
            prefix=PREFIX,
            max_composite_object_size=0,
            list_retry_config=DEFAULT_RETRY.with_deadline(300.0).with_delay(
                initial=1.0, multiplier=1.2, maximum=45.0
            ),
            download_retry_config=DEFAULT_RETRY.with_deadline(600.0).with_delay(
                initial=1.0, multiplier=1.5, maximum=90.0
            ),
        ),
    )
    ```

## Contributing
We welcome your feedback, issues, and bug fixes. If you have a major feature or change in functionality you'd like to contribute, please open a GitHub Issue for discussion prior to sending a pull request. Please see [CONTRIBUTING](docs/contributing.md) for more information on how to report bugs or submit pull requests.

We aim to provide an initial response to all incoming issues, pull requests, etc. within 24 business hours. If it has been a while and you haven't heard from us, feel free to add a new comment to the issue.

### Code of Conduct

This project has adopted the Google Open Source Code of Conduct. Please see [code-of-conduct.md](docs/code-of-conduct.md) for more information.

### License

The Connector for PyTorch library has an Apache License 2.0. Please see the [LICENSE](LICENSE) file for more information.

*PyTorch, the PyTorch logo and any related marks are trademarks of The Linux Foundation.*
