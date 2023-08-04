"""
Utilities for randomness.
"""
import ctypes
import os
import sys
import random
import logging
import datetime
import pickle
from copy import deepcopy
from pathlib import Path
from omegaconf import OmegaConf, open_dict
from typing import Any, Dict, List, Tuple, Optional, Union

import GPUtil
import psutil

import numpy as np
import torch as th
from torch import cuda
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from nntrainer.typext import ConstantHolder
# ---------- Multiprocessing ----------

MAP_TYPES: Dict[str, Any] = {
    'int': ctypes.c_int,
    'long': ctypes.c_long,
    'float': ctypes.c_float,
    'double': ctypes.c_double
}

# ---------- Logger ----------
DEFAULT = "default"
REF = "ref"
NONE = "none"
LOGGER_NAME = "trainlog"
LOGGING_FORMATTER = logging.Formatter("%(levelname)5s %(message)s", datefmt="%m%d %H%M%S")


class LogLevelsConst(ConstantHolder):
    """
    Loglevels, same as logging module.
    """
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0


def create_logger_without_file(name: str, log_level: int = LogLevelsConst.INFO, no_parent: bool = False,
                               no_print: bool = False) -> logging.Logger:
    """
    Create a stdout only logger.

    Args:
        name: Name of the logger.
        log_level: Verbosity level.
        no_parent: Disable parents, can be used to avoid duplicate log entries.
        no_print: Do not print a message on creation.
    Returns:
        Created logger.
    """
    return create_logger(name, log_dir="", log_level=log_level, no_parent=no_parent, no_print=no_print)


def create_logger(
        name: str, *, filename: str = "run", log_dir: Union[str, Path] = "", log_level: int = LogLevelsConst.INFO,
        no_parent: bool = False, no_print: bool = False) -> logging.Logger:
    """
    Create a new logger.

    Notes:
        This created stdlib logger can later be retrieved with logging.getLogger(name) with the same name.
        There is no need to pass the logger instance between objects.

    Args:
        name: Name of the logger.
        log_dir: Target logging directory. Empty string will not create files.
        filename: Target filename.
        log_level: Verbosity level.
        no_parent: Disable parents, can be used to avoid duplicate log entries.
        no_print: Do not print a message on creation.

    Returns:
    """
    # create logger, set level
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # remove old handlers to avoid duplicate messages
    remove_handlers_from_logger(logger)

    # file handler
    file_path = None
    if log_dir != "":
        ts = get_timestamp_for_filename()
        file_path = Path(log_dir) / "{}_{}.log".format(filename, ts)
        file_hdlr = logging.FileHandler(str(file_path))
        file_hdlr.setFormatter(LOGGING_FORMATTER)
        logger.addHandler(file_hdlr)

    # stdout handler
    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(LOGGING_FORMATTER)
    logger.addHandler(strm_hdlr)

    # disable propagating to parent to avoid double logs
    if no_parent:
        logger.propagate = False

    if not no_print:
        print(f"Logger: '{name}' to {file_path}")
    return logger


def remove_handlers_from_logger(logger: logging.Logger) -> None:
    """
    Remove handlers from the logger.

    Args:
        logger: Logger.
    """
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.flush()
        handler.close()


def print_logger_info(logger: logging.Logger) -> None:
    """
    Print infos describing the logger: The name and handlers.

    Args:
        logger: Logger.
    """
    print(logger.name)
    x = list(logger.handlers)
    for i in x:
        handler_str = f"Handler {i.name} Type {type(i)}"
        print(handler_str)
        
# ---------- Time utilities ----------

def get_timestamp_for_filename(dtime: Optional[datetime.datetime] = None):
    """
    Convert datetime to timestamp for filenames.

    Args:
        dtime: Optional datetime object, will use now() if not given.

    Returns:
    """
    if dtime is None:
        dtime = datetime.datetime.now()
    ts = str(dtime).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    return ts

# ---------- Random ----------

def set_seed(seed: int, cudnn_deterministic: bool = False, cudnn_benchmark: bool = True):
    """
    Set all relevant seeds for torch, numpy and python

    Args:
        seed: int seed
        cudnn_deterministic: set True for deterministic training..
        cudnn_benchmark: set False for deterministic training.
    """
    th.manual_seed(seed)
    cuda.manual_seed(seed)
    cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cudnn_deterministic:
        cudnn.deterministic = True
    cudnn.benchmark = cudnn_benchmark


def get_truncnorm_tensor(shape: Tuple[int], *, mean: float = 0, std: float = 1, limit: float = 2) -> th.Tensor:
    """
    Create and return normally distributed tensor, except values with too much deviation are discarded.

    Args:
        shape: tensor shape
        mean: normal mean
        std: normal std
        limit: which values to discard

    Returns:
        Filled tensor with shape (*shape)
    """
    assert isinstance(shape, (tuple, list)), f"shape {shape} is not a tuple or list of ints"
    num_examples = 8
    tmp = th.empty(shape + (num_examples,)).normal_()
    valid = (tmp < limit) & (tmp > -limit)
    _, ind = valid.max(-1, keepdim=True)
    return tmp.gather(-1, ind).squeeze(-1).mul_(std).add_(mean)


def fill_tensor_with_truncnorm(input_tensor: th.Tensor, *, mean: float = 0, std: float = 1, limit: float = 2) -> None:
    """
    Fill given input tensor with a truncated normal dist.

    Args:
        input_tensor: tensor to be filled
        mean: normal mean
        std: normal std
        limit: which values to discard
    """
    # get truncnorm values
    tmp = get_truncnorm_tensor(input_tensor.shape, mean=mean, std=std, limit=limit)
    # fill input tensor
    input_tensor[...] = tmp[...]


# ---------- Profiling ----------

def profile_gpu_and_ram() -> Tuple[List[str], List[float], List[float], List[float], float, float, float]:
    """
    Profile GPU and RAM.

    Returns:
        GPU names, total / used memory per GPU, load per GPU, total / used / available RAM.
    """

    # get info from gputil
    _str, dct_ = _get_gputil_info()
    dev_num = os.getenv("CUDA_VISIBLE_DEVICES")
    if dev_num is not None:
        # single GPU set with OS flag
        # gpu_info = [dct_[int(dev_num)]]
        gpu_info = []
        for dev_dict in dct_:
            gpu_info.append(dev_dict)
    else:
        # possibly multiple gpus, aggregate values
        gpu_info = []
        for dev_dict in dct_:
            gpu_info.append(dev_dict)

    # convert to GPU info and MB to GB
    gpu_names: List[str] = [gpu["name"] for gpu in gpu_info]
    total_memory_per: List[float] = [gpu["memoryTotal"] / 1024 for gpu in gpu_info]
    used_memory_per: List[float] = [gpu["memoryUsed"] / 1024 for gpu in gpu_info]
    load_per: List[float] = [gpu["load"] / 100 for gpu in gpu_info]

    # get RAM info and convert to GB
    mem = psutil.virtual_memory()
    ram_total: float = mem.total / 1024 ** 3
    ram_used: float = mem.used / 1024 ** 3
    ram_avail: float = mem.available / 1024 ** 3

    return gpu_names, total_memory_per, used_memory_per, load_per, ram_total, ram_used, ram_avail


def _get_gputil_info():
    """
    Returns info string for printing and list with gpu infos. Better formatting than the original GPUtil.

    Returns:
        gpu info string, List[Dict()] of values. dict example:
            ('id', 1),
            ('name', 'GeForce GTX TITAN X'),
            ('temperature', 41.0),
            ('load', 0.0),
            ('memoryUtil', 0.10645266950540452),
            ('memoryTotal', 12212.0)])]
    """

    gpus = GPUtil.getGPUs()
    attr_list = [
        {'attr': 'id', 'name': 'ID'}, {'attr': 'name', 'name': 'Name'},
        {'attr': 'temperature', 'name': 'Temp', 'suffix': 'C', 'transform': lambda x: x, 'precision': 0},
        {'attr': 'load', 'name': 'GPU util.', 'suffix': '% GPU', 'transform': lambda x: x * 100,
         'precision': 1},
        {'attr': 'memoryUtil', 'name': 'Memory util.', 'suffix': '% MEM', 'transform': lambda x: x * 100,
         'precision': 1}, {'attr': 'memoryTotal', 'name': 'Memory total', 'suffix': 'MB', 'precision': 0},
        {'attr': 'memoryUsed', 'name': 'Memory used', 'suffix': 'MB', 'precision': 0}
    ]
    gpu_strings = [''] * len(gpus)
    gpu_info = []
    for _ in range(len(gpus)):
        gpu_info.append({})

    for attrDict in attr_list:
        attr_precision = '.' + str(attrDict['precision']) if (
                'precision' in attrDict.keys()) else ''
        attr_suffix = str(attrDict['suffix']) if (
                'suffix' in attrDict.keys()) else ''
        attr_transform = attrDict['transform'] if (
                'transform' in attrDict.keys()) else lambda x: x
        for gpu in gpus:
            attr = getattr(gpu, attrDict['attr'])

            attr = attr_transform(attr)

            if isinstance(attr, float):
                attr_str = ('{0:' + attr_precision + 'f}').format(attr)
            elif isinstance(attr, int):
                attr_str = '{0:d}'.format(attr)
            elif isinstance(attr, str):
                attr_str = attr
            else:
                raise TypeError('Unhandled object type (' + str(
                    type(attr)) + ') for attribute \'' + attrDict[
                                    'name'] + '\'')

            attr_str += attr_suffix

        for gpuIdx, gpu in enumerate(gpus):
            attr_name = attrDict['attr']
            attr = getattr(gpu, attr_name)

            attr = attr_transform(attr)

            if isinstance(attr, float):
                attr_str = ('{0:' + attr_precision + 'f}').format(attr)
            elif isinstance(attr, int):
                attr_str = ('{0:' + 'd}').format(attr)
            elif isinstance(attr, str):
                attr_str = ('{0:' + 's}').format(attr)
            else:
                raise TypeError(
                    'Unhandled object type (' + str(
                        type(attr)) + ') for attribute \'' + attrDict[
                        'name'] + '\'')
            attr_str += attr_suffix
            gpu_info[gpuIdx][attr_name] = attr
            gpu_strings[gpuIdx] += '| ' + attr_str + ' '

    return "\n".join(gpu_strings), gpu_info


# ---------- Modeling ----------

def count_parameters(model, part_name, verbose=True):
    """
    Count number of parameters in PyTorch model,
    """
    n_all = sum(p.numel() for p in model.parameters())
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if verbose:
        print(f"{part_name}\'s total parameters: {n_all}, frozen: {n_frozen}")
    return n_all, n_frozen


# ---------- Config ----------
def resolve_sameas_config_recursively(config: Dict, *, root_config: Optional[Dict] = None):
    """
    Recursively resolve config fields described with same_as.

    If any container in the config has the field "same_as" set, find the source identifier and copy all data
    from there to the target container. The source identifier can nest with dots e.g.
    same_as: "net_video_local.input_fc_config" will copy the values from container input_fc_config located inside
    the net_video_local container.

    Args:
        config: Config to modify.
        root_config: Config to get the values from, usually the same as config.

    Returns:
    """
    if root_config is None:
        root_config = config
    # loop the current config and check
    loop_keys = list(config.keys())
    for key in loop_keys:
        value = config[key]
        if not isinstance(value, dict):
            continue
        same_as = value.get("same_as")
        if same_as is not None:
            # current container should be filled with the values from the source container. loop source container
            source_container = get_dict_value_recursively(root_config, same_as)
            for key_source, val_source in source_container.items():
                # only write fields that don't exist yet, don't overwrite everything
                if key_source not in config[key]:
                    # at this point we want a deepcopy to make sure everything is it's own object
                    config[key][key_source] = deepcopy(val_source)
            # at this point, remove the same_as field.
            del value["same_as"]

        # check recursively
        resolve_sameas_config_recursively(config[key], root_config=root_config)


def get_dict_value_recursively(dct: Dict, key: str) -> Any:
    """
    Nest into the dict given a key like root.container.subcontainer

    Args:
        dct: Dict to get the value from.
        key: Key that can describe several nesting steps at one.

    Returns:
        Value.
    """
    key_parts = key.split(".")
    if len(key_parts) == 1:
        # we arrived at the leaf of the dict tree and can return the value
        return dct[key_parts[0]]
    # nest one level deeper
    return get_dict_value_recursively(dct[key_parts[0]], ".".join(key_parts[1:]))


def check_config_dict(name: str, config: Dict[str, Any], strict: bool = True) -> None:
    """
    Make sure config has been read correctly with .pop(), and no fields are left over.

    Args:
        name: config name
        config: config dict
        strict: Throw errors
    """
    remaining_keys, remaining_values = [], []
    for key, value in config.items():
        if key == REF:
            # ignore the reference configurations, they can later be used for copying things with same_as
            continue
        remaining_keys.append(key)
        remaining_values.append(value)
    # check if something is left over
    if len(remaining_keys) > 0:
        if not all(value is None for value in remaining_values):
            err_msg = (
                f"keys and values remaining in config {name}: {remaining_keys}, {remaining_values}. "
                f"Possible sources of this error: Typo in the field name in the yaml config file. "
                f"Incorrect fields given with --config flag. "
                f"Field should be added to the config class so it can be parsed. "
                f"Using 'same_as' and forgot to set these fields to null.")

            if strict:
                print(f"Print config for debugging: {config}")
                raise ValueError(err_msg)
            logging.getLogger(LOGGER_NAME).warning(err_msg)
            
# ---------- Multi-gpu Setting ----------
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(cfg):
    OmegaConf.set_struct(cfg, True)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        with open_dict(cfg):
            cfg.rank = int(os.environ["RANK"])
            cfg.world_size = int(os.environ['WORLD_SIZE'])
            cfg.gpu = int(os.environ['LOCAL_RANK'])
            cfg.data.train_batch_size_per_gpu = int(cfg.train.batch_size / cfg.world_size)
            cfg.data.val_batch_size_per_gpu = int(cfg.val.batch_size / cfg.world_size)
            cfg.distributed = True
            cfg.dist_backend = 'nccl'
    elif 'SLURM_PROCID' in os.environ:
        with open_dict(cfg):
            cfg.rank = int(os.environ['SLURM_PROCID'])
            cfg.gpu = cfg.rank % th.cuda.device_count()
    else:
        print('Not using distributed mode')
        with open_dict(cfg):
            cfg.data.train_batch_size_per_gpu = int(cfg.train.batch_size)
            cfg.data.val_batch_size_per_gpu = int(cfg.val.batch_size)
            cfg.distributed = False
        return

    th.cuda.set_device(cfg.gpu)
    print('| distributed init (rank {}): {}'.format(cfg.rank, cfg.dist_url), flush=True)
    th.distributed.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
                                         world_size=cfg.world_size, rank=cfg.rank)
    th.distributed.barrier()
    setup_for_distributed(cfg.rank == 0)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def is_main_process():
    return get_rank() == 0


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = th.ByteStorage.from_buffer(buffer)
    tensor = th.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = th.LongTensor([tensor.numel()]).to("cuda")
    size_list = [th.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(th.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = th.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = th.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list
