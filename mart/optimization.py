"""
PyTorch optimization for BERT model. Required.

References:
    Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
    Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

check if this is needed? Can't I just use regular Adam, AdamW, RAdam, AdaBelief or something
"""

import abc
import logging
import math
import sys
from typing import Dict

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class _LRSchedule(ABC):
    """
    Parent of all LRSchedules here.
    """
    warn_t_total = False  # is set to True for schedules where progressing beyond t_total steps doesn't make sense

    def __init__(self, warmup=0.002, t_total=-1, **kw):
        """
        :param warmup:  what fraction of t_total steps will be used for linear warmup
        :param t_total: how many training steps (updates) are planned
        :param kw:
        """
        super(_LRSchedule, self).__init__(**kw)
        if t_total < 0:
            logger.warning(
                "t_total value of {} results in schedule not being applied".format(
                    t_total))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError(
                "Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(
                    warmup))
        warmup = max(warmup, 0.)
        self.warmup, self.t_total = float(warmup), float(t_total)
        self.warned_for_t_total_at_progress = -1

    def get_lr(self, step, nowarn=False):
        """
        :param step:    which of t_total steps we're on
        :param nowarn:  set to True to suppress warning regarding training beyond specified 't_total' steps
        :return:        learning rate multiplier for current update
        """
        if self.t_total < 0:
            return 1.
        progress = float(step) / self.t_total
        ret = self.get_lr_(progress)
        # warning for exceeding t_total (only active with warmup_linear
        if not nowarn and self.warn_t_total and progress > 1. and progress > self.warned_for_t_total_at_progress:
            logger.warning(
                "Training beyond specified 't_total'. Learning rate multiplier set to {}."
                "Please set 't_total' of {} correctly.".format(ret, self.__class__.__name__))
            self.warned_for_t_total_at_progress = progress
        # end warning
        return ret

    @abc.abstractmethod
    def get_lr_(self, progress):
        """
        :param progress:    value between 0 and 1 (unless going beyond t_total steps) specifying training progress
        :return:            learning rate multiplier for current update
        """
        return 1.


class ConstantLR(_LRSchedule):
    def get_lr_(self, progress):
        return 1.


class WarmupCosineSchedule(_LRSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    Decreases learning rate from 1. to 0. over remaining `1 - warmup` steps following a cosine curve.
    If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    warn_t_total = True

    def __init__(self, warmup=0.002, t_total=-1, cycles=.5, **kw):
        """
        :param warmup:      see LRSchedule
        :param t_total:     see LRSchedule
        :param cycles:      number of cycles. Default: 0.5, corresponding to cosine decay from 1.
            at progress==warmup and 0 at progress==1.
        :param kw:
        """
        super(WarmupCosineSchedule, self).__init__(warmup=warmup,
                                                   t_total=t_total, **kw)
        self.cycles = cycles

    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        else:
            progress = (progress - self.warmup) / (
                    1 - self.warmup)  # progress after warmup
            return 0.5 * (1. + math.cos(math.pi * self.cycles * 2 * progress))


class WarmupCosineWithHardRestartsSchedule(WarmupCosineSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    If `cycles` (default=1.) is different from default, learning rate follows `cycles` times a cosine decaying
    learning rate (with hard restarts).
    """

    def __init__(self, warmup=0.002, t_total=-1, cycles=1., **kw):
        super().__init__(
            warmup=warmup, t_total=t_total, cycles=cycles, **kw)
        assert (cycles >= 1.)

    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        else:
            progress = (progress - self.warmup) / (
                    1 - self.warmup)  # progress after warmup
            ret = 0.5 * (1. + math.cos(
                math.pi * ((self.cycles * progress) % 1)))
            return ret


class WarmupCosineWithWarmupRestartsSchedule(WarmupCosineWithHardRestartsSchedule):
    """
    All training progress is divided in `cycles` (default=1.) parts of equal length.
    Every part follows a schedule with the first `warmup` fraction of the training steps
    linearly increasing from 0. to 1.,
    followed by a learning rate decreasing from 1. to 0. following a cosine curve.
    """

    def __init__(self, warmup=0.002, t_total=-1, cycles=1., **kw):
        assert (warmup * cycles < 1.)
        warmup = warmup * cycles if warmup >= 0 else warmup
        super().__init__(
            warmup=warmup, t_total=t_total, cycles=cycles, **kw)

    def get_lr_(self, progress):
        progress = progress * self.cycles % 1.
        if progress < self.warmup:
            return progress / self.warmup
        else:
            progress = (progress - self.warmup) / (
                    1 - self.warmup)  # progress after warmup
            ret = 0.5 * (1. + math.cos(math.pi * progress))
            return ret


class WarmupConstantSchedule(_LRSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    Keeps learning rate equal to 1. after warmup.
    """

    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return 1.


class WarmupLinearSchedule(_LRSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    Linearly decreases learning rate from 1. to 0. over remaining `1 - warmup` steps.
    """
    warn_t_total = True

    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return max((progress - 1.) / (self.warmup - 1.), 0.)


SCHEDULES = {
    None: ConstantLR,
    "none": ConstantLR,
    "warmup_cosine": WarmupCosineSchedule,
    "warmup_constant": WarmupConstantSchedule,
    "warmup_linear": WarmupLinearSchedule
}


class EMA:
    """
    Exponential Moving Average for model parameters.

    References:
        [1] https://github.com/BangLiu/QANet-PyTorch/blob/master/model/modules/ema.py
        [2] https://github.com/hengruo/QANet-pytorch/blob/e2de07cd2c711d525f5ffee35c3764335d4b501d/main.py
    """

    def __init__(self, cfg):
        self.decay = cfg.optim.ema_decay
        self.distributed = cfg.distributed
        self.shadow = {}
        self.original = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, model, step):
        decay = min(self.decay, (1 + step) / (10.0 + step))
        if self.distributed:
            model = model.module
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"Parameter {name} not found in EMA. shadow has {len(self.shadow)} entries"
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model, update_model: bool = True):                 # 将model的参数加载到self.original中，同时用EMA的参数替换model的参数
        if self.distributed:
            model = model.module
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                if update_model:
                    param.data = self.shadow[name]

    def resume(self, model):
        if self.distributed:
            model = model.module
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"Parameter {name} not found in EMA. shadow has {len(self.shadow)} entries"
                try:
                    param.data = self.original[name]
                except KeyError as e:
                    raise KeyError(f"Parameter {name} not found in model. ({len(self.original)} entries)") from e

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def set_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.shadow = state_dict


from copy import deepcopy
# class EMA:
#     """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
#     Keep a moving average of everything in the model state_dict (parameters and buffers).
#     This is intended to allow functionality like
#     https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
#     A smoothed version of the weights is necessary for some training schemes to perform well.
#     E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
#     RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
#     smoothing of weights to match results. Pay attention to the decay constant you are using
#     relative to your update count per epoch.
#     To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
#     disable validation of the EMA weights. Validation will have to be done manually in a separate
#     process, or after the training stops converging.
#     This class is sensitive where it is initialized in the sequence of model init,
#     GPU assignment and distributed training wrappers.
#     I've tested with the sequence in my own train.py for torch.DataParallel, apex.DDP, and single-GPU.
#     """
#
#     def __init__(self, model, decay=0.9999, device=''):
#         # make a copy of the model for accumulating moving average of weights
#         self.ema = deepcopy(model)
#         self.ema.eval()
#         self.updates = 0  # number of EMA updates
#         self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
#         self.device = device  # perform ema on different device from model if set
#         if device:
#             self.ema.to(device=device)
#         for p in self.ema.parameters():
#             p.requires_grad_(False)
#
#     def update(self, model):
#         self.updates += 1
#         d = self.decay(self.updates)
#         with torch.no_grad():
#             if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel,):
#                 msd, esd = model.module.state_dict(), self.ema.module.state_dict()
#             else:
#                 msd, esd = model.state_dict(), self.ema.state_dict()
#
#             for k, v in esd.items():
#                 if v.dtype.is_floating_point:
#                     v *= d
#                     v += (1. - d) * msd[k].detach()
#
#     def update_attr(self, model):
#         # Update class attributes
#         ema = self.ema.module if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel,) else self.ema
#         for k, v in model.__dict__.items():
#             if not k.startswith('_') and k != 'module':
#                 setattr(ema, k, v)


class BertAdam(Optimizer):
    """
    Implements BERT version of Adam algorithm with weight decay fix.

    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate of 1. (no warmup regardless of warmup setting).Default: -1
        schedule: schedule to use for the warmup (see above).
            Can be `'warmup_linear'`, `'warmup_constant'`, `'warmup_cosine'`, `'none'`, `None` or a
                `_LRSchedule` object (see below).
            If `None` or `'none'`, learning rate is always kept constant.
            Default : `'warmup_linear'`
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """

    def __init__(self, params, lr, warmup=-1, t_total=-1,
                 schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0, **_kwargs):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not isinstance(schedule, _LRSchedule) and schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= b1 < 1.0:
            raise ValueError(
                "Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(
                    b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError(
                "Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(
                    b2))
        if not e >= 0.0:
            raise ValueError(
                "Invalid epsilon value: {} - should be >= 0.0".format(e))
        # initialize schedule object
        if not isinstance(schedule, _LRSchedule):
            schedule_type = SCHEDULES[schedule]
            schedule = schedule_type(warmup=warmup, t_total=t_total)
        else:
            if warmup != -1 or t_total != -1:
                logger.warning(
                    "warmup and t_total on the optimizer are ineffective when _LRSchedule object is provided "
                    "as schedule. Please specify custom warmup and t_total in _LRSchedule object.")
        defaults = dict(lr=lr, schedule=schedule,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    continue
                    # return [0]
                lr_scheduled = group['lr']
                lr_scheduled *= group['schedule'].get_lr(state['step'])
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                # next_m.mul_(beta1).add_(1 - beta1, grad) # deprecated
                next_m.mul_(beta1).add_(grad, alpha=1 - beta1)
                next_v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                lr_scheduled = group['lr']
                lr_scheduled *= group['schedule'].get_lr(state['step'])

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
        return loss
