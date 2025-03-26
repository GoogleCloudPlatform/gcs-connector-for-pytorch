import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, Union

import torch
import torch.distributed as dist
from dataflux_core import user_agent
from dataflux_pytorch.lightning import DatafluxLightningCheckpoint
from dataflux_pytorch.lightning.gcs_filesystem import (GCSDistributedReader,
                                                       GCSDistributedWriter)
from google.cloud import storage
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.types import _PATH, _Stateful
from lightning.pytorch.strategies.fsdp import _METADATA_FILENAME
from lightning.pytorch.trainer.states import TrainerFn
from torch.distributed.checkpoint import async_save, load, save
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import TypeGuard


class DatafluxFSDPStrategy(FSDPStrategy):

    def __init__(self,
                 project_name,
                 storage_client,
                 use_async=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.ckpt_io = DatafluxLightningCheckpoint(project_name,
                                                   storage_client)
        self.storage_client = storage.Client(project=project_name)
        user_agent.add_dataflux_user_agent(self.storage_client)

        # Attributes used for async behavior.
        self.use_async = use_async
        self.checkpoint_group = None
        self._checkpoint_future = None

        if self.use_async:
            default_ranks = list(range(dist.get_world_size()))
            self.checkpoint_group = dist.new_group(
                default_ranks, backend=self.process_group_backend)

    def save_checkpoint(
        self,
        path: _PATH,
        state: Dict[str, Union[Module, Optimizer, Any]],
        storage_options: Optional[Any] = None,
        filter: Optional[Dict[str, Callable[[str, Any], bool]]] = None,
    ) -> None:
        if storage_options is not None:
            raise TypeError(
                "`FSDPStrategy.save_checkpoint(..., storage_options=...)` is\
                not supported because`FSDPStrategy` does not use the \
                    `CheckpointIO`.")
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        modules = [
            module for module in state.values() if _has_fsdp_modules(module)
        ]
        if len(modules) == 0:
            raise ValueError(
                "Could not find a FSDP model in the provided checkpoint state. Please provide the model as"
                " part of the state like so: `save_checkpoint(..., state={'model': model, ...})`. Make sure"
                " you set up the model (and optimizers if any) through the strategy before saving the checkpoint."
            )
        if len(modules) > 1:
            raise ValueError(
                "Found multiple FSDP models in the given state. Saving checkpoints with FSDP is"
                " currently limited to a single model per checkpoint. To save multiple models, call the"
                " save method for each model separately with a different path."
            )
        module = modules[0]
        state_dict_ctx = _get_sharded_state_dict_context(module)

        converted_state: Dict[str, Any] = {}
        metadata: Dict[str, Any] = {}
        with state_dict_ctx:
            for key, obj in state.items():
                converted: Any
                if isinstance(obj, Module):
                    converted = obj.state_dict()
                    target_dict = converted_state
                elif isinstance(obj, Optimizer):
                    converted = FSDP.optim_state_dict(module, obj)
                    target_dict = converted_state
                else:  # everything not a module or optimizer is considered metadata
                    converted = obj.state_dict() if isinstance(
                        obj, _Stateful) else obj
                    target_dict = metadata
                _apply_filter(key, filter or {}, converted, target_dict)

        path = Path(self.broadcast(path))
        writer = GCSDistributedWriter(path, self.storage_client.project,
                                      self.storage_client)

        start_time = time.time()
        if self.use_async:
            self._async_save(converted_state, path, writer)
        else:
            self._save(converted_state, path, writer)
        duration_ms = (time.time() - start_time) / 1000
        strategy = "async_save" if self.use_async else "save"
        print(f"Checkpoint rank #{self.global_rank} {strategy} "
              f"duration: {duration_ms:.4f} ms.")

        if self.global_rank == 0:
            self.ckpt_io.save_checkpoint(metadata, path / _METADATA_FILENAME)

    def _save(self, converted_state, path, writer):
        save(converted_state, checkpoint_id=path, storage_writer=writer)

    def _async_save(self, converted_state, path, writer):
        self._resolve_future()
        path = Path(self.broadcast(path))
        self._checkpoint_future = async_save(
            converted_state,
            checkpoint_id=path,
            storage_writer=writer,
            process_group=self.checkpoint_group)

    def _resolve_future(self):
        """Resolve previous async future if one exists.

        If a previous future exists, wait for checkpointing to finish,
        avoiding queuing more then one checkpoint request at a time.
        """
        if self._checkpoint_future is not None:
            self._checkpoint_future.result()

    def get_sharded_state_dict_context(
            self, module: Module) -> Generator[None, None, None]:

        from torch.distributed.fsdp.api import (ShardedOptimStateDictConfig,
                                                ShardedStateDictConfig,
                                                StateDictType)

        state_dict_config = ShardedStateDictConfig(offload_to_cpu=True)
        optim_state_dict_config = ShardedOptimStateDictConfig(
            offload_to_cpu=True)
        state_dict_type_context = FSDP.state_dict_type(
            module=module,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,
            state_dict_config=state_dict_config,
            optim_state_dict_config=optim_state_dict_config,
        )
        return state_dict_type_context  # type: ignore[return-value]

    def load_checkpoint(self, checkpoint_path):
        # broadcast the path from rank 0 to ensure all the states are loaded \
        # from a common path.
        path = Path(self.broadcast(checkpoint_path))

        assert self.model is not None
        assert self.lightning_module is not None

        from torch.distributed.checkpoint.optimizer import \
            load_sharded_optimizer_state_dict

        state_dict_ctx = self.get_sharded_state_dict_context(self.model)

        reader = GCSDistributedReader(path, self.storage_client.project,
                                      self.storage_client)

        with state_dict_ctx:
            module_state = {"model": self.model.state_dict()}
            load(module_state, reader)
            self.model.load_state_dict(
                module_state["model"],
                strict=self.lightning_module.strict_loading)

            if self.lightning_module.trainer.state.fn == TrainerFn.FITTING and self.optimizers:

                for idx, optim in enumerate(self.optimizers):
                    optim_key = f"optimizer_{idx}"
                    optim_state = load_sharded_optimizer_state_dict(
                        model_state_dict=module_state["model"],
                        optimizer_key=optim_key,
                        storage_reader=reader,
                    )
                    flattened_osd = FSDP.optim_state_dict_to_load(
                        optim_state_dict=optim_state[optim_key],
                        model=self.model,
                        optim=optim,
                    )
                    optim.load_state_dict(flattened_osd)

        # Load metadata (anything not a module or optimizer)
        new_path = path / _METADATA_FILENAME
        metadata = None
        with reader.fs.create_stream(path=new_path,
                                     mode='rb') as metadata_file:
            metadata = torch.load(metadata_file)
        return metadata

    def teardown(self):
        # Ensure any async operation completes before shutting down.
        self._resolve_future()
        super().teardown()


def _apply_filter(key: str, filter: Dict[str, Callable[[str, Any], bool]],
                  source_dict: object, target_dict: Dict[str, Any]) -> None:
    # filter out if necessary
    if key in filter and isinstance(source_dict, dict):
        filter_fn = filter[key]
        for k, v in source_dict.items():
            if filter_fn(k, v):
                # save the state
                target_dict.setdefault(key, {})
                target_dict[key][k] = v
    else:
        # save the state
        target_dict[key] = source_dict


def _get_sharded_state_dict_context(
        module: Module) -> Generator[None, None, None]:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.api import (ShardedOptimStateDictConfig,
                                            ShardedStateDictConfig,
                                            StateDictType)

    state_dict_config = ShardedStateDictConfig(offload_to_cpu=True)
    optim_state_dict_config = ShardedOptimStateDictConfig(offload_to_cpu=True)
    state_dict_type_context = FSDP.state_dict_type(
        module=module,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        state_dict_config=state_dict_config,
        optim_state_dict_config=optim_state_dict_config,
    )
    return state_dict_type_context  # type: ignore[return-value]


def _has_fsdp_modules(module: object) -> TypeGuard[Module]:
    from torch.distributed.fsdp import FullyShardedDataParallel

    return isinstance(module, Module) and any(
        isinstance(m, FullyShardedDataParallel) for m in module.modules())
