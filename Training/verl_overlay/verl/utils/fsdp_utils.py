import functools
import itertools
import json
import logging
import math
import os
from abc import ABC
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from typing import Optional, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from packaging import version
from peft.utils.save_and_load import get_peft_model_state_dict
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._runtime_utils import _lazy_init
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from transformers.trainer_pt_utils import get_module_class_from_name

from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.model import check_exclude_modules, check_target_modules

logger = logging.getLogger(__name__)

if version.parse(torch.__version__) >= version.parse("2.6"):
    from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard
    from torch.distributed.fsdp._fully_shard._fsdp_init import _get_post_forward_mesh_info
    from torch.distributed.tensor import DTensor, Shard
    from torch.distributed.tensor._dtensor_spec import DTensorSpec

    fully_shard_module = torch.distributed.fsdp._fully_shard._fully_shard
elif version.parse(torch.__version__) >= version.parse("2.4"):
    from torch.distributed._composable.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard

    fully_shard_module = torch.distributed._composable.fsdp
else:
    fully_shard, MixedPrecisionPolicy, FSDPModule, CPUOffloadPolicy, fully_shard_module = None, None, None, None, None


def init_fn(x: torch.nn.Module):
    if torch.distributed.get_rank() != 0:
        x = x.to_empty(device=get_device_id(), recurse=False)
        get_torch_device().empty_cache()
    return x


def get_init_weight_context_manager(use_meta_tensor=True, mesh: DeviceMesh = None):
    from accelerate import init_empty_weights

    cpu_init_weights = lambda: torch.device("cpu")
    if use_meta_tensor:
        if mesh is None:
            init_context = init_empty_weights if torch.distributed.get_rank() != 0 else cpu_init_weights
        else:
            init_context = init_empty_weights if mesh.get_coordinate()[-1] != 0 else cpu_init_weights
    else:
        init_context = cpu_init_weights
    return init_context


def get_fsdp_wrap_policy(module, config=None, is_lora=False):


    if config is None:
        config = {}


    def _get_attr(attr_name, default_value=None):
        if hasattr(config, "get"):
            return config.get(attr_name, default_value)
        else:
            return config.__getattribute__(attr_name)

    if _get_attr("disable", False):
        return None

    default_transformer_cls_names_to_wrap = getattr(module, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = _get_attr(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )
    min_num_params = _get_attr("min_num_params", 0)
    auto_wrap_policy = None

    policies = []

    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy


    if is_lora and min_num_params == 0:

        def lambda_policy_fn(module):
            return bool(
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )

        lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
        policies.append(lambda_policy)

    if min_num_params > 0:
        size_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
        policies.append(size_policy)
    elif fsdp_transformer_layer_cls_to_wrap is not None:
        transformer_cls_to_wrap = set()
        missing_classes = []
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(module, layer_class)
            if transformer_cls is None:


                missing_classes.append(layer_class)
            else:
                transformer_cls_to_wrap.add(transformer_cls)

        if not transformer_cls_to_wrap:
            raise Exception(
                f"Could not find any of the transformer layer classes to wrap in the model: "
                f"{list(fsdp_transformer_layer_cls_to_wrap)}"
            )
        if missing_classes:
            logger.warning(
                "FSDP wrap policy: skipped missing layer class names %s, wrapping %s",
                missing_classes,
                sorted(c.__name__ for c in transformer_cls_to_wrap),
            )

        transformer_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_cls_to_wrap,
        )
        policies.append(transformer_policy)

    if len(policies) > 0:
        auto_wrap_policy = functools.partial(_or_policy, policies=policies)

    return auto_wrap_policy


@torch.no_grad()
def offload_fsdp_model_to_cpu(model: FSDP, empty_cache: bool = True):
    if fsdp_version(model) == 2 or fsdp_version(model) == 0:
        offload_fsdp2_model_to_cpu(model, empty_cache)
        return

    assert isinstance(model, FSDP)

    _lazy_init(model, model)
    assert model._is_root, "Only support root model offloading to CPU"
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        assert (
            flat_param.data.data_ptr() == flat_param._local_shard.data_ptr()
            and id(flat_param.data) != id(flat_param._local_shard)
            and flat_param.data.size() == flat_param._local_shard.size()
        )
        handle.flat_param_to(torch.device("cpu"), non_blocking=True)

        flat_param._local_shard = flat_param.data
        assert id(flat_param._local_shard) != id(flat_param.data)
    if empty_cache:
        get_torch_device().empty_cache()


@torch.no_grad()
def offload_fsdp2_model_to_cpu(model, empty_cache: bool = True):
    model.cpu()
    if empty_cache:
        get_torch_device().empty_cache()


@torch.no_grad()
def load_fsdp_model_to_gpu(model: FSDP):
    if fsdp_version(model) == 2 or fsdp_version(model) == 0:
        load_fsdp2_model_to_gpu(model)
        return

    assert isinstance(model, FSDP)

    _lazy_init(model, model)
    assert model._is_root, "Only support root model loading to GPU"
    device_id = get_device_id()
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        handle.flat_param_to(torch.device(f"{get_device_name()}:{device_id}"), non_blocking=True)

        flat_param._local_shard = flat_param.data


@torch.no_grad()
def load_fsdp2_model_to_gpu(model):
    device = get_device_id()
    model.to(device)


@torch.no_grad()
def offload_fsdp_optimizer(optimizer):
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)


@torch.no_grad()
def load_fsdp_optimizer(optimizer, device_id):
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device_id, non_blocking=True)


@contextmanager
def meta_device_init():


    device = torch.device("meta")
    old_register_parameter = nn.Module.register_parameter
    registered = set()

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)


        if param is not None and param not in registered:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)
            registered.add(module._parameters[name])

    try:
        nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        registered.clear()
        nn.Module.register_parameter = old_register_parameter


def parallel_load_safetensors(filepath):


    from safetensors.torch import load_file

    safetensors2param = {}

    index_file = os.path.join(filepath, "model.safetensors.index.json")
    if os.path.exists(index_file):
        index = json.load(open(index_file, "rb"))
        for param_name, filename in index["weight_map"].items():
            safetensors2param.setdefault(filename, []).append(param_name)
    else:

        param_file = os.path.join(filepath, "model.safetensors")
        assert os.path.exists(param_file), f"Cannot find {param_file}"
        states = load_file(param_file)
        for param_name in states:
            safetensors2param.setdefault("model.safetensors", []).append(param_name)
        del states

    total_files = len(safetensors2param)
    ckpt_chunks = sorted(safetensors2param.keys())
    world_size = dist.get_world_size()
    size = int(math.ceil(total_files / world_size))
    ckpt_chunks = [ckpt_chunks[rank * size : rank * size + size] for rank in range(world_size)]

    shard_states = {}
    device = get_device_id()
    for rank, files in enumerate(ckpt_chunks):
        if rank == dist.get_rank():
            for file in files:
                file = os.path.join(filepath, file)
                states = load_file(file, device=device)

                shard_states.update(states)
        else:
            for file in files:
                for param_name in safetensors2param[file]:
                    shard_states[param_name] = rank
    return shard_states


def parallel_init_module_fn(module: torch.nn.Module, shard_states: dict[str, torch.nn.Parameter]):


    state2fqn = {}
    for name, state in itertools.chain(
        module.named_parameters(remove_duplicate=False), module.named_buffers(remove_duplicate=False)
    ):
        state2fqn.setdefault(state, []).append(name)

    shared = {s for s, names in state2fqn.items() if len(names) > 1}
    materialized_states = {}

    @torch.no_grad()
    def create_and_sync_state(param_name, state, is_param):
        assert param_name in shard_states, f"{param_name} not loaded"
        device = get_device_id()
        if is_param:
            param = torch.nn.Parameter(torch.empty_like(state.data, device=device), requires_grad=state.requires_grad)
        else:
            param = torch.empty_like(state.data, device=device)
        loaded = shard_states[param_name]
        if isinstance(loaded, torch.nn.Parameter | torch.Tensor):

            param.data.copy_(loaded.data)
            dist.broadcast(param.data, src=dist.get_rank())
        else:
            assert isinstance(loaded, int)
            dist.broadcast(param.data, src=loaded)
        shard_states.pop(param_name)
        del loaded
        return param

    def init_fn(sub_mod: torch.nn.Module, recurse: bool = True):
        param_and_buffers = tuple(sub_mod.named_parameters(recurse=False)) + tuple(sub_mod.named_buffers(recurse=False))

        for name, state in param_and_buffers:
            if not state.is_meta:
                continue
            is_param = name in sub_mod._parameters
            fqn = state2fqn[state].pop(0)

            if (not is_param) and fqn not in shard_states:
                if state.is_meta:
                    raise RuntimeError(
                        f"find a non-persistent buffer ({fqn}) initiated with device meta. Such buffer is not saved "
                        f"in checkpoint and user should guarantee to init in CPU / GPU device."
                    )
                continue

            if state in shared:
                if state not in materialized_states:
                    materialized_states[state] = create_and_sync_state(fqn, state, is_param)
                else:
                    if fqn in shard_states:
                        shard_states.pop(fqn)
                materialize_state = materialized_states[state]

            else:
                materialize_state = create_and_sync_state(fqn, state, is_param)
            if is_param:
                sub_mod._parameters[name] = materialize_state
            else:
                sub_mod._buffers[name] = materialize_state
        if recurse:
            for module in sub_mod.children():
                init_fn(module, recurse=True)


        return sub_mod

    return init_fn


def fsdp_version(model):
    if isinstance(model, FSDP):
        return 1
    elif isinstance(model, FSDPModule):
        return 2
    else:
        return 0


def get_fsdp_state_ctx(model, state_type, state_cfg, optim_cfg):
    if fsdp_version(model) == 1:
        return FSDP.state_dict_type(model, state_type, state_cfg, optim_cfg)
    else:
        return nullcontext()


def get_fsdp_full_state_dict(model: torch.nn.Module, offload_to_cpu: bool = True, rank0_only: bool = True):


    if fsdp_version(model) == 1:
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        state_dict_config = FullStateDictConfig(offload_to_cpu=offload_to_cpu, rank0_only=rank0_only)
        with get_fsdp_state_ctx(
            model, state_type=StateDictType.FULL_STATE_DICT, state_cfg=state_dict_config, optim_cfg=None
        ):
            state_dict = model.state_dict()
        return state_dict
    elif fsdp_version(model) == 2 or fsdp_version(model) == 0:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        state_dict_config = StateDictOptions(
            full_state_dict=True, cpu_offload=offload_to_cpu, broadcast_from_rank0=not rank0_only
        )
        state_dict = get_model_state_dict(model, options=state_dict_config)
        return state_dict
    else:
        raise NotImplementedError(f"Unknown FSDP version {fsdp_version}")


def fsdp2_load_full_state_dict(model: torch.nn.Module, full_state: dict, device_mesh=None, cpu_offload=None):


    if version.parse(torch.__version__) >= version.parse("2.7.0"):
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
    else:


        from verl.third_party.torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict


    if dist.get_rank() == 0:
        model = model.to(device=get_device_id(), non_blocking=True)
    else:
        model = model.to_empty(device=get_device_id())

    cpu_offload = cpu_offload is not None
    options = StateDictOptions(full_state_dict=True, cpu_offload=cpu_offload, broadcast_from_rank0=True)
    set_model_state_dict(model, full_state, options=options)


    bufs = sorted(model.named_buffers(), key=lambda x: x[0])
    for name, buf in bufs:
        dist.broadcast(buf, src=0)

    if cpu_offload:
        model.to("cpu", non_blocking=True)
        for buf in model.buffers():
            buf.data = buf.data.to(get_device_id())


@contextmanager
def maybe_patch_fsdp_module(model):
    if fully_shard_module is None:
        yield
        return

    orig_fsdp_module = fully_shard_module.FSDPModule

    class FSDPModuleABC(ABC, orig_fsdp_module):
        pass

    try:
        if isinstance(model, ABC):
            fully_shard_module.FSDPModule = FSDPModuleABC
        yield
    finally:
        fully_shard_module.FSDPModule = orig_fsdp_module


def _select_fsdp2_wrap_targets(model, fsdp_transformer_layer_cls_to_wrap):


    _tie = getattr(model.config, "tie_word_embeddings", False)
    _wrap_by_name = set() if _tie else {"embed_tokens", "lm_head"}

    modules = []
    for name, module in model.named_modules():
        leaf_name = name.rsplit(".", 1)[-1] if "." in name else name
        if (
            module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap
            or (isinstance(module, nn.Embedding) and not _tie)
            or (leaf_name in _wrap_by_name and hasattr(module, "weight"))
        ):
            modules.append(module)
    return modules


def apply_fsdp2(model, fsdp_kwargs, config):

    assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"

    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.get("wrap_policy", {}).get(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )

    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]

    if isinstance(fsdp_transformer_layer_cls_to_wrap, set):
        fsdp_transformer_layer_cls_to_wrap = list(fsdp_transformer_layer_cls_to_wrap)
    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and fsdp_transformer_layer_cls_to_wrap[0] is not None

    modules = _select_fsdp2_wrap_targets(model, fsdp_transformer_layer_cls_to_wrap)

    for idx, module in enumerate(modules):


        with maybe_patch_fsdp_module(module):
            fully_shard(module, **fsdp_kwargs)


    with maybe_patch_fsdp_module(model):
        fully_shard(model, **fsdp_kwargs)


    if config.get("forward_prefetch", False):
        fsdp_modules = [m for m in modules if isinstance(m, FSDPModule)]
        for i, m in enumerate(fsdp_modules):
            next_targets = fsdp_modules[i + 1 : i + 2]
            if next_targets and hasattr(m, "set_modules_to_forward_prefetch"):
                m.set_modules_to_forward_prefetch(next_targets)


def get_shard_placement_fn(fsdp_size):


    def shard_placement_fn(param):
        shape = list(param.shape)
        for i in range(len(shape)):
            if shape[i] % fsdp_size == 0:
                return Shard(i)
        return Shard(0)

    return shard_placement_fn


def fsdp2_clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):

    from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:

        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    total_norm = total_norm.to(get_device_id(), non_blocking=True)
    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


def layered_summon_lora_params(fsdp_module) -> OrderedDict:


    lora_params = OrderedDict()
    peft_model = getattr(fsdp_module, "_fsdp_wrapped_module", fsdp_module)

    for name, submodule in fsdp_module.named_modules():
        if name == "":

            continue
        if fsdp_version(submodule) == 0:
            continue


        clean_prefix = name.replace("_fsdp_wrapped_module.", "")
        if clean_prefix.endswith(".model") or clean_prefix.endswith(".layers"):
            continue


        is_fsdp1 = fsdp_version(submodule) == 1


        nested_fsdp_names = {n for n, m in submodule.named_modules() if n != "" and fsdp_version(m) > 0}

        if not any(
            "lora_" in n
            for n, _ in submodule.named_parameters()
            if not any(n.startswith(f"{nn}.") for nn in nested_fsdp_names)
        ):
            continue

        if is_fsdp1:
            submodule._is_root = True
        summon_ctx = FSDP.summon_full_params(submodule, writeback=False) if is_fsdp1 else nullcontext()

        with summon_ctx:
            sub_state_dict = {
                n: p
                for n, p in submodule.named_parameters()
                if not any(n.startswith(f"{nn}.") for nn in nested_fsdp_names)
            }
            sub_lora_params = get_peft_model_state_dict(peft_model, state_dict=sub_state_dict)
            if not sub_lora_params:
                continue
            sub_lora_params = {
                f"{clean_prefix}.{key}": (
                    param.full_tensor().detach().cpu() if hasattr(param, "full_tensor") else param.detach().cpu()
                )
                for key, param in sub_lora_params.items()
            }
            lora_params.update(sub_lora_params)
            if is_fsdp1:
                submodule._is_root = False
        get_torch_device().empty_cache()

    return lora_params


def collect_lora_params(module: FSDP, layered_summon: bool, base_sync_done: bool) -> OrderedDict:


    from peft.utils.save_and_load import get_peft_model_state_dict

    lora_params = OrderedDict()
    peft_model = getattr(module, "_fsdp_wrapped_module", module)
    if fsdp_version(module) > 0:
        if layered_summon:
            if not base_sync_done:
                raise ValueError(
                    "To use layered_summon, you must make sure base-model is preloaded in vllm, e.g. let "
                    "rollout.load_format=safetensors"
                )
            lora_params = layered_summon_lora_params(module)
            if not lora_params:
                import logging

                logging.getLogger(__name__).warning("layered_summon returned empty, falling back to full summon")


                is_fsdp1 = fsdp_version(module) == 1
                summon_ctx = (
                    FSDP.summon_full_params(module, writeback=False, offload_to_cpu=True) if is_fsdp1 else nullcontext()
                )
                with summon_ctx:
                    lora_params = get_peft_model_state_dict(peft_model)
                    lora_params = {
                        name: param.full_tensor().detach().cpu()
                        if hasattr(param, "full_tensor")
                        else param.detach().cpu()
                        for name, param in lora_params.items()
                    }
                get_torch_device().empty_cache()
        else:


            is_fsdp1 = fsdp_version(module) == 1
            summon_ctx = FSDP.summon_full_params(module, writeback=False) if is_fsdp1 else nullcontext()
            with summon_ctx:
                if base_sync_done:
                    lora_params = get_peft_model_state_dict(peft_model)
                    lora_params = {
                        name: param.full_tensor().detach().cpu()
                        if hasattr(param, "full_tensor")
                        else param.detach().cpu()
                        for name, param in lora_params.items()
                    }
                else:
                    model = peft_model.base_model.model
                    orig_dev = "cpu" if "cpu" in str(next(model.parameters()).device) else get_device_name()
                    model = model.to("cpu")
                    for name, param in model.state_dict().items():
                        if any(x in name for x in ["_flat_param", "lora_"]):
                            continue
                        name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
                        lora_params[name] = (
                            param.full_tensor().detach().cpu()
                            if hasattr(param, "full_tensor")
                            else param.detach().cpu()
                        )
                    model = model.to(orig_dev)
            get_torch_device().empty_cache()
    else:
        if base_sync_done:
            lora_params = get_peft_model_state_dict(peft_model)
        else:
            model = peft_model.base_model.model
            orig_dev = "cpu" if "cpu" in str(next(model.parameters()).device) else get_device_name()
            model = model.to("cpu")
            for name, param in model.state_dict().items():
                if any(x in name for x in ["_flat_param", "lora_"]):
                    continue
                name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
                lora_params[name] = param.detach().cpu()
            model = model.to(orig_dev)
    return lora_params


def replace_lora_wrapper(k, peft_config):


    stacked_params = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if k.endswith(".weight"):
        module_k = k[: -len(".weight")]
        if check_exclude_modules(peft_config, module_k):
            return k
        elif any([module_k.endswith(s) for s in stacked_params]) or check_target_modules(peft_config, module_k):
            return f"{module_k}.base_layer.weight"
    if k.endswith(".bias"):
        module_k = k[: -len(".bias")]
        if check_exclude_modules(peft_config, module_k):
            return k
        elif any([module_k.endswith(s) for s in stacked_params]) or check_target_modules(peft_config, module_k):
            return f"{module_k}.base_layer.bias"
    return k


def set_reshard_after_forward(module: FSDPModule, reshard_after_forward: bool, recurse: bool = True) -> None:


    if not isinstance(reshard_after_forward, bool):
        raise ValueError(f"reshard_after_forward should be a bool, got {type(reshard_after_forward)}")
    self_module = cast(nn.Module, module)
    modules = list(self_module.modules()) if recurse else [self_module]
    for module in modules:
        if isinstance(module, FSDPModule):
            state = module._get_fsdp_state()
            state._auto_reshard_after_forward = False
            if fsdp_param_group := state._fsdp_param_group:
                fsdp_param_group.post_forward_mesh_info = _get_post_forward_mesh_info(
                    reshard_after_forward, fsdp_param_group.mesh_info
                )


def normalize_peft_param_name(params: dict) -> dict:


    def _normalize_peft_name(name: str) -> str:
        return name.replace("base_model.model.", "").replace("base_model.", "").replace(".base_layer", "")

    def _is_lora_key(name: str) -> bool:

        return ("lora_" in name) or (".adapter_" in name)

    params = [(_normalize_peft_name(k), v) for k, v in params.items()]

    params = {k: v for k, v in params if not _is_lora_key(k)}
    return params


def _merge_or_unmerge_lora_(module, merge: bool):


    from peft.tuners.lora import LoraLayer

    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, LoraLayer):
                is_merged = getattr(m, "merged", False)
                if merge and not is_merged:
                    m.merge()
                elif (not merge) and is_merged:
                    m.unmerge()


def _clean_merged_lora_(module):

    from peft.tuners.lora import LoraLayer

    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, LoraLayer):
                merged_adapters = getattr(m, "merged_adapters", False)
                if merged_adapters:
                    m.merged_adapters = []


def fsdp_merge_unmerge(module: nn.Module, do_merge: bool):


    version = fsdp_version(module)
    assert version in [1, 2], f"fsdp_merge_unmerge requires FSDP module, got version {version}"

    if version == 1:

        with FSDP.summon_full_params(module, writeback=True, with_grads=False):
            _merge_or_unmerge_lora_(module, merge=do_merge)
    else:

        for name, submodule in module.named_modules():
            if isinstance(submodule, FSDPModule) and name != "":
                with FSDP.summon_full_params(submodule, writeback=True, with_grads=False):
                    _merge_or_unmerge_lora_(submodule, merge=do_merge)


def collect_merged_lora_params(module: nn.Module) -> OrderedDict:


    ver = fsdp_version(module)
    assert ver in [1, 2], f"collect_merged_lora_params requires FSDP module, got version {ver}"

    merged_params = OrderedDict()
    peft_model = getattr(module, "_fsdp_wrapped_module", module)

    from peft.tuners.lora import LoraLayer

    def _backup_base_weights(mod):

        backups = {}
        for mname, m in mod.named_modules():
            if isinstance(m, LoraLayer):
                base = m.get_base_layer()
                backups[mname] = base.weight.data.clone()
        return backups

    def _restore_base_weights(mod, backups):

        for mname, m in mod.named_modules():
            if isinstance(m, LoraLayer) and mname in backups:
                base = m.get_base_layer()
                base.weight.data.copy_(backups[mname])
        _clean_merged_lora_(mod)

    def _clean_key(key):
        key = key.replace("_fsdp_wrapped_module.", "")
        key = key.replace("base_model.model.", "")
        key = key.replace(".base_layer", "")
        return key

    if ver == 1:
        with FSDP.summon_full_params(module, writeback=True, with_grads=False):
            backups = _backup_base_weights(module)
            _merge_or_unmerge_lora_(module, merge=True)
            model = peft_model.base_model.model
            for name, param in model.state_dict().items():
                if any(x in name for x in ["_flat_param", "lora_"]):
                    continue
                name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
                merged_params[name] = (
                    param.full_tensor().detach().cpu() if hasattr(param, "full_tensor") else param.detach().cpu()
                )
            _restore_base_weights(module, backups)
        get_torch_device().empty_cache()
    else:


        base_weights_backup = backup_base_model_weights(module)
        fsdp_merge_unmerge(module, do_merge=True)

        for pname, param in module.named_parameters():
            if any(x in pname for x in ["_flat_param", "lora_"]):
                continue
            clean_key = _clean_key(pname)
            val = param.full_tensor().detach().cpu() if hasattr(param, "full_tensor") else param.detach().cpu()
            merged_params[clean_key] = val

        restore_base_model_weights(module, base_weights_backup)
        _clean_merged_lora_(module)
        get_torch_device().empty_cache()

    return merged_params


def backup_base_model_weights(module):


    from peft import PeftModel

    backup = {}
    with torch.no_grad():

        if isinstance(module, PeftModel):

            with module.disable_adapter():

                for name, param in module.named_parameters():
                    if "lora" not in name.lower():
                        backup[name] = param.data.clone().cpu()
        else:

            for name, param in module.named_parameters():
                backup[name] = param.data.clone().cpu()
    return backup


def restore_base_model_weights(module, backup):


    with torch.no_grad():
        for name, param in module.named_parameters():
            if name in backup:
                param.data.copy_(backup[name].to(param.device))


@contextmanager
def merged_lora_context(actor, backup_adapters=False):


    base_weights_backup = None
    if backup_adapters:

        base_weights_backup = backup_base_model_weights(actor)


    fsdp_merge_unmerge(actor, do_merge=True)
    try:

        yield
    finally:
        if backup_adapters and base_weights_backup is not None:

            restore_base_model_weights(actor, base_weights_backup)
            _clean_merged_lora_(actor)
        else:

            fsdp_merge_unmerge(actor, do_merge=False)


def fsdp2_sharded_save_to_cpu(
    model: torch.nn.Module,
) -> tuple[dict[str, tuple[torch.Tensor, DTensorSpec]], DTensorSpec]:


    cpu_sharded_state = {}
    global_spec = None

    for param_name, param in model.named_parameters():

        if not isinstance(param, DTensor):

            cpu_tensor = param.detach().cpu()
            cpu_sharded_state[param_name] = (cpu_tensor, None)
            continue


        if global_spec is None:
            global_spec = param._spec
            assert hasattr(global_spec, "device_mesh"), "DTensorSpec must contain 'device_mesh' attribute"
            assert hasattr(global_spec, "placements"), "DTensorSpec must contain 'placements' attribute"


        local_gpu_tensor = param._local_tensor

        local_cpu_tensor = local_gpu_tensor.detach().cpu()

        cpu_sharded_state[param_name] = (local_cpu_tensor, param._spec)

    assert global_spec is not None, "No DTensor-type parameters found in the model. FSDP2 sharding may not be enabled."
    return cpu_sharded_state, global_spec


def fsdp2_sharded_load_from_cpu(
    model: torch.nn.Module,
    cpu_sharded_state: dict[str, tuple[torch.Tensor, Optional[DTensorSpec]]],
    target_spec: DTensorSpec,
) -> None:


    current_device_mesh = None
    for param in model.parameters():
        if isinstance(param, DTensor):
            current_device_mesh = param._spec.device_mesh
            break
    assert current_device_mesh is not None, "DTensor parameters not initialized in the model to be loaded"
    assert current_device_mesh == target_spec.device_mesh, (
        f"device_mesh mismatch during loading! Original: {target_spec.device_mesh}, Current: {current_device_mesh}"
    )

    for param_name, param in model.named_parameters():

        if param_name not in cpu_sharded_state:
            continue


        local_cpu_tensor, saved_spec = cpu_sharded_state[param_name]


        if isinstance(param, DTensor):

            assert saved_spec is not None, f"DTensorSpec missing in saved state for parameter {param_name}"
            assert saved_spec.placements == target_spec.placements, (
                f"Sharding strategy mismatch for parameter {param_name} (conflicts with global rules)!"
            )


            target_device = param._local_tensor.device
            local_gpu_tensor = local_cpu_tensor.to(target_device)


            param._local_tensor.copy_(local_gpu_tensor)

        else:

            target_device = param.device
            param.data.copy_(local_cpu_tensor.to(target_device))


    dist.barrier()
