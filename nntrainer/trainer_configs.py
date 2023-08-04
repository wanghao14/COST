"""
Configuration setup for TrainerBase. Moved to separate file to avoid circular imports.
"""
from copy import deepcopy
from typing import Dict, List, Optional

from nntrainer import lr_scheduler, optimizer, typext
from omegaconf import OmegaConf
from nntrainer.utils import resolve_sameas_config_recursively, check_config_dict


class BaseTrainerState(typext.SaveableBaseModel):
    """
    Current trainer state that must be saved for training continuation..
    """
    # total time bookkeeping
    time_total: float = 0
    time_val: float = 0
    # state info TO SAVE
    start_epoch: int = 0
    current_epoch: int = 0
    epoch_step: int = 0
    total_step: int = 0
    det_best_field_current: float = 0
    det_best_field_best: Optional[float] = None

    # state info lists
    infos_val_epochs: List[int] = []
    infos_val_steps: List[int] = []
    infos_val_is_good: List[int] = []

    # logging
    last_grad_norm: int = 0


class BaseExperimentConfig(typext.ConfigClass):
    """
    Base configuration class, loads the dict from yaml config files for an experiment.

    This is where the entire config dict will be loaded into first.

    Args:
        config: Configuration dictionary to be loaded.

    Attributes:
        ...
    """

    def __init__(self, config, strict: bool = True) -> None:
        self.config_orig = deepcopy(config)  # backup original input dict
        config = OmegaConf.to_container(config)  # convert to dict to meet the type requirements of original code
        self.config = config  # bind dict to class
        self.strict = strict
        resolve_sameas_config_recursively(config)  # resolve "same_as" reference fields to dictionary objects.
        self.description: str = config.pop("description", "no description given.")
        self.config_type: str = config.pop("config_type")
        
        config_exp = config.pop("exp")
        self.random_seed: Optional[int] = config_exp.pop("random_seed")
        self.use_cuda: bool = config_exp.pop("use_cuda")
        self.use_multi_gpu: bool = config_exp.pop("use_multi_gpu")
        self.cudnn_enabled: bool = config_exp.pop("cudnn_enabled")
        self.cudnn_benchmark: bool = config_exp.pop("cudnn_benchmark")
        self.cudnn_deterministic: bool = config_exp.pop("cudnn_deterministic")
        self.cuda_non_blocking: bool = config_exp.pop("cuda_non_blocking")
        self.fp16_train: bool = config_exp.pop("fp16_train")
        self.fp16_val: bool = config_exp.pop("fp16_val")

    def post_init(self):
        """
        Check config dict for correctness and raise

        Returns:
        """
        if self.strict:
            check_config_dict(self.__class__.__name__, self.config)


class DefaultExperimentConfig(BaseExperimentConfig):
    """
    Default configuration class.

    Args:
        config: Configuration dictionary to be loaded.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.name = "config_default"
        self.train = BaseTrainConfig(config.pop("train"))
        self.val = BaseValConfig(config.pop("val"))
        self.dataset_train = BaseDatasetConfig(config.pop("dataset_train"))
        self.dataset_val = BaseDatasetConfig(config.pop("dataset_val"))
        self.logging = BaseLoggingConfig(config.pop("logging"))
        self.saving = BaseSavingConfig(config.pop("saving"))
        self.optimizer = optimizer.OptimizerConfig(config.pop("optimizer"))
        self.lr_scheduler = lr_scheduler.SchedulerConfig(config.pop("lr_scheduler"))
        self.branch = BaseBranchConfig(config["model"].pop("branch"))


class BaseTrainConfig(typext.ConfigClass):
    """
    Base configuration class for training.

    Args:
        config: Configuration dictionary to be loaded, training part.
    """

    def __init__(self, config: Dict) -> None:
        self.batch_size: int = config.pop("batch_size")
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        self.num_epochs: int = config.pop("num_epochs")
        assert isinstance(self.num_epochs, int) and self.num_epochs > 0
        self.loss_func: str = config.pop("loss_func")
        assert isinstance(self.loss_func, str)
        self.clip_gradient: float = config.pop("clip_gradient")
        assert isinstance(self.clip_gradient, (int, float)) and self.clip_gradient >= -1
        self.cls_weight: float = config.pop("cls_weight")
        assert isinstance(self.cls_weight, (int, float)) and self.cls_weight >= 0
        self.act_weight: float = config.pop("act_weight")
        assert isinstance(self.act_weight, (int, float)) and self.act_weight >= 0


class BaseValConfig(typext.ConfigClass):
    """
    Base configuration class for validation.

    Args:
        config: Configuration dictionary to be loaded, validation part.
    """

    def __init__(self, config: Dict) -> None:
        self.batch_size: int = config.pop("batch_size")
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        self.val_freq: int = config.pop("val_freq")
        assert isinstance(self.val_freq, int) and self.val_freq > 0
        self.val_start: int = config.pop("val_start")
        assert isinstance(self.val_start, int) and self.val_start >= 0
        self.det_best_field: str = config.pop("det_best_field")
        assert isinstance(self.det_best_field, str)
        self.det_best_compare_mode: str = config.pop("det_best_compare_mode")
        assert isinstance(self.det_best_compare_mode, str) and self.det_best_compare_mode in ["min", "max"]
        self.det_best_threshold_mode: str = config.pop("det_best_threshold_mode")
        assert isinstance(self.det_best_threshold_mode, str) and self.det_best_threshold_mode in ["rel", "abs"]
        self.det_best_threshold_value: float = config.pop("det_best_threshold_value")
        assert isinstance(self.det_best_threshold_value, (int, float)) and self.det_best_threshold_value >= 0
        self.det_best_terminate_after: float = config.pop("det_best_terminate_after")
        assert isinstance(self.det_best_terminate_after, int) and self.det_best_terminate_after >= -1


class BaseBranchConfig(typext.ConfigClass):
    """
    Base configuration class for training.

    Args:
        config: Configuration dictionary to be loaded, training part.
    """

    def __init__(self, config: Dict) -> None:
        self.num: int = config.pop("num")
        assert isinstance(self.num, int) and self.num > 0
        self.detect_input: bool = config.pop("detect_input")
        assert isinstance(self.detect_input, bool)
        self.action_input: bool = config.pop("action_input")
        assert isinstance(self.action_input, bool)


class BaseSavingConfig(typext.ConfigClass):
    """
    Base Saving Configuration Class

    Args:
        config: Configuration dictionary to be loaded, saving part.

    Attributes:
        keep_freq: Frequency to keep epochs. 1: Save after each epoch. Default -1: Keep nothing except best and last.
        save_last: Keep last epoch. Needed to continue training. Default: true
        save_best: Keep best epoch. Default: true
        save_opt_state: Save optimizer and lr scheduler. Needed to continue training. Default: true
    """

    def __init__(self, config: Dict) -> None:
        self.keep_freq: int = config.pop("keep_freq")
        self.save_last: bool = config.pop("save_last")
        self.save_best: bool = config.pop("save_best")
        self.save_opt_state: bool = config.pop("save_opt_state")
        assert self.keep_freq >= -1


class BaseDatasetConfig(typext.ConfigClass):
    """
    Base Dataset Configuration class

    Args:
        config: Configuration dictionary to be loaded, dataset part.
    """

    def __init__(self, config) -> None:
        # general dataset info
        self.name: str = config.pop("name")
        self.data_type: str = config.pop("data_type")
        self.subset: str = config.pop("subset")
        self.split: str = config.pop("split")
        self.max_datapoints: int = config.pop("max_datapoints")
        self.shuffle: bool = config.pop("shuffle")
        # general dataloader configuration
        self.pin_memory: bool = config.pop("pin_memory")
        self.num_workers: int = config.pop("num_workers")
        self.drop_last: bool = config.pop("drop_last")


class BaseLoggingConfig(typext.ConfigClass):
    """
    Base Logging Configuration Class

    Args:
        config: Configuration dictionary to be loaded, logging part.
    """

    def __init__(self, config: Dict) -> None:
        self.step_train: int = config.pop("step_train")
        self.step_val: int = config.pop("step_val")
        self.step_gpu: int = config.pop("step_gpu")
        self.step_gpu_once: int = config.pop("step_gpu_once")
        assert self.step_train >= -1
        assert self.step_val >= -1
        assert self.step_gpu >= -1
        assert self.step_gpu_once >= -1
