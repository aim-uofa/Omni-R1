import itertools
from contextlib import contextmanager
from copy import deepcopy
from typing import Union

import deepspeed
from accelerate import Accelerator
from deepspeed.runtime.engine import DeepSpeedEngine
from packaging import version
from torch.nn.parallel.distributed import DistributedDataParallel


def prepare_deepspeed(model, accelerator):
    # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
    stage = config_kwargs["zero_optimization"]["stage"]

    if model is not None:
        hidden_size = (
            max(model.config.hidden_sizes)
            if getattr(model.config, "hidden_sizes", None)
            else getattr(model.config, "hidden_size", None)
        )
        if hidden_size is not None and stage == 3:
            # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache
            # @ step 0: expected module 1, but got module 0`
            # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
            config_kwargs.update(
                {
                    "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                    "zero_optimization.stage3_param_persistence_threshold": 10
                    * hidden_size,
                    "zero_optimization.stage3_prefetch_bucket_size": 0.9
                    * hidden_size
                    * hidden_size,
                }
            )

    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO
    # disabled (stage 0)
    if stage != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model


@contextmanager
def unwrap_model_for_generation(
    model: Union["DistributedDataParallel", "DeepSpeedEngine"],
    accelerator: "Accelerator",
    gather_deepspeed3_params: bool = True,
):
    """
    Context manager to unwrap distributed or accelerated models for generation tasks.

    Args:
        model (`Union[DistributedDataParallel, DeepSpeedEngine]`):
            Model to be unwrapped.
        accelerator (`~accelerate.Accelerator`):
            Accelerator instance managing the model.
        gather_deepspeed3_params (`bool`, *optional*, defaults to `True`):
            Whether to gather weights for DeepSpeed ZeRO Stage 3 models. If `False`, skips parameter gathering, which
            can be more memory-efficient but may lead to slower generation times.

    Yields:
        Unwrapped model.

    Example:
    ```python
    with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
        generated_outputs = unwrapped_model.generate(input_ids)
    ```
    """
    unwrapped_model = accelerator.unwrap_model(model)
    if (
        accelerator.state.deepspeed_plugin is not None
        and accelerator.state.deepspeed_plugin.zero_stage == 3
    ):
        if not gather_deepspeed3_params:
            yield accelerator.unwrap_model(model)
        else:
            with deepspeed.zero.GatheredParameters(model.parameters()):
                remove_hooks(model)
                yield accelerator.unwrap_model(model)
                add_hooks(model)
    else:
        yield unwrapped_model


def get_all_parameters(sub_module, recurse=False):
    return itertools.chain(
        sub_module.named_parameters(recurse=recurse),
        sub_module.ds_external_parameters(),
    )


def iter_params(module, recurse=False):
    return [param for _, param in get_all_parameters(module, recurse)]


def add_hooks(model: "DeepSpeedEngine") -> None:
    """Adds the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if not hasattr(
        model, "optimizer"
    ):  # before the first training step, the model has no optimizer
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError("The model optimizer is None, which is not yet supported.")
    if version.parse(deepspeed.__version__) >= version.parse("0.16.4"):
        # Account for renaming in https://github.com/deepspeedai/DeepSpeed/pull/6847
        optimizer_offload._register_deepspeed_module(optimizer_offload.module)
    else:
        optimizer_offload._register_hooks_recursively(optimizer_offload.module)


def remove_hooks(model: "DeepSpeedEngine") -> None:
    """Removes the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if not hasattr(
        model, "optimizer"
    ):  # before the first training step, the model has no optimizer
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError("The model optimizer is None, which is not yet supported.")

    for param in iter_params(optimizer_offload.module, recurse=True):
        param.ds_active_sub_modules.clear()

    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []
