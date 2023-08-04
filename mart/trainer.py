"""
Trainer for retrieval training and validation. Holds the main training loop.
"""

import json
import logging
import os
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from omegaconf import OmegaConf
from timeit import default_timer as timer
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th
from torch import nn
from torch.cuda.amp import autocast
from torch.utils import data
from tqdm import tqdm
from dataset import CaptionDataset, prepare_batch_inputs

from mart.caption_eval_tools import get_reference_files
from mart.configs_mart import MartMetersConst as MMeters
from mart.evaluate_language import evaluate_language_files
from mart.evaluate_repetition import evaluate_repetition_files
from mart.evaluate_stats import evaluate_stats_files
from mart.optimization import BertAdam, EMA
from mart.translator import Translator
from nntrainer import trainer_base
from nntrainer.experiment_organization import ExperimentFilesHandler
from nntrainer.metric import TRANSLATION_METRICS, TextMetricsConst, TextMetricsConstEvalCap
from nntrainer.trainer_configs import BaseTrainerState

import json
import time
import torch
from nntrainer.utils import is_main_process, synchronize, all_gather

def cal_performance(pred, gold):
    pred = pred.max(2)[1].contiguous().view(-1)
    gold = gold.contiguous().view(-1)
    valid_label_mask = gold.ne(CaptionDataset.IGNORE)
    pred_correct_mask = pred.eq(gold)
    n_correct = pred_correct_mask.masked_select(valid_label_mask).sum().item()
    return n_correct

def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]


# only log the important ones to console
TRANSLATION_METRICS_LOG = ["Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "re4"]


class MartFilesHandler(ExperimentFilesHandler):
    """
    Overwrite default filehandler to add some more paths.
    """

    def __init__(self, exp_group: str, exp_name: str, run_name: str, log_dir: str, annotations_dir: str):
        super().__init__(exp_group, exp_name, run_name, log_dir=log_dir)
        self.annotations_dir = annotations_dir
        self.path_caption = self.path_base / "caption"

    def get_translation_files(self, epoch: Union[int, str], split: str) -> Path:
        """
        Get all file paths for storing translation results and evaluation.

        Args:
            epoch: Epoch.
            split: dataset split (val, test)

        Returns:
            Path to store raw model output and ground truth.
        """
        return self.path_caption / f"translations_{epoch}_{split}.json"

    def setup_dirs(self, *, reset: bool = False) -> None:
        """
        Call super class to setup directories and additionally create the caption folder.

        Args:
            reset:

        Returns:
        """
        super().setup_dirs(reset=reset)
        os.makedirs(self.path_caption, exist_ok=True)


class MartModelManager(trainer_base.BaseModelManager):
    """
    Wrapper for MART models.
    """

    def __init__(self, cfg, model: nn.Module):
        super().__init__(cfg)
        # self.cfg = cfg
        self.model_dict: [str, nn.Module] = {"model": model}


class MartTrainerState(BaseTrainerState):
    prev_best_score = 0.
    es_cnt = 0


class MartTrainer(trainer_base.BaseTrainer):
    """
    This part builds based on COOT
    """

    def __init__(
            self, cfg, model: nn.Module, run_name, train_loader_length: int,
            log_dir: str = "experiments", log_level: Optional[int] = None,
            logger: Optional[logging.Logger] = None, print_graph: bool = False, reset: bool = False,
            load_best: bool = False, load_epoch: Optional[int] = None, load_model: Optional[str] = None,
            inference_only: bool = False, annotations_dir: str = "annotations"):
        # create a wrapper for the model
        model_mgr = MartModelManager(cfg, model)
        exp_group, exp_name = cfg.exp.exp_group, cfg.exp.exp_name
        # overwrite default experiment files handler
        exp = MartFilesHandler(exp_group, exp_name, run_name, log_dir=log_dir, annotations_dir=annotations_dir)
        exp.setup_dirs(reset=reset)

        super().__init__(
            cfg, model_mgr, exp_group, exp_name, run_name, train_loader_length, "caption",
            log_dir=log_dir, log_level=log_level, logger=logger, print_graph=print_graph, reset=reset,
            load_best=load_best, load_epoch=load_epoch, load_model=load_model, is_test=inference_only,
            exp_files_handler=exp)
        self.model = model
        # ---------- setup ----------

        # update type hints from base classes to inherited classes
        # self.cfg = self.cfg
        self.model_mgr: MartModelManager = self.model_mgr
        self.exp: MartFilesHandler = self.exp

        # # overwrite default state with inherited trainer state in case we need additional state fields
        # self.state = RetrievalTrainerState()
        self.logger.info(OmegaConf.to_yaml(self.cfg))
        # ---------- loss ----------

        # loss is created directly in the mart model and not needed here

        # ---------- additional metrics ----------
        # train loss and accuracy
        self.metrics.add_meter(MMeters.TRAIN_LOSS_PER_WORD, use_avg=False)
        self.metrics.add_meter(MMeters.TRAIN_ACC, use_avg=False)
        self.metrics.add_meter(MMeters.VAL_LOSS_PER_WORD, use_avg=False)
        self.metrics.add_meter(MMeters.VAL_ACC, use_avg=False)

        # track gradient clipping manually
        self.metrics.add_meter(MMeters.GRAD, per_step=True, reset_avg_each_epoch=True)

        # translation metrics (bleu etc.)
        for meter_name in TRANSLATION_METRICS.values():
            self.metrics.add_meter(meter_name, use_avg=False)

        # ---------- optimization ----------

        self.optimizer = None
        self.lr_scheduler = None
        if is_main_process():
            self.ema = EMA(cfg)
        else:
            self.ema = None
        # skip optimizer if not training
        if not self.is_test:
            # Prepare optimizer
            if cfg.distributed:
                param_optimizer = list(self.model.module.named_parameters())                     # whether should I add module?
            else:
                param_optimizer = list(self.model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 "weight_decay": 0.01},
                {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
            ]
            if cfg.optim.ema_decay > 0 and self.ema:
                # register EMA params
                self.logger.info(f"Registering {sum(p.numel() for p in model.parameters())} params for EMA")
                all_names = []
                if cfg.distributed:
                    ema_model = self.model.module
                else:
                    ema_model = self.model
                for name, p in ema_model.named_parameters():
                    if p.requires_grad:
                        self.ema.register(name, p.data)
                    all_names.append(name)

            num_train_optimization_steps = train_loader_length * cfg.train.num_epochs
            self.optimizer = BertAdam(optimizer_grouped_parameters, lr=cfg.optim.lr, warmup=cfg.optim.lr_warmup_proportion,
                                      t_total=num_train_optimization_steps, e=cfg.optim.eps,
                                      schedule="warmup_linear")

        # ---------- Translator ----------

        self.translator = Translator(self.model, self.cfg, logger=self.logger)

        # post init hook for checkpoint loading
        self.hook_post_init()

        if self.load and not self.load_model and self.ema:
            # reload EMA weights from checkpoint (the shadow) and save the model parameters (the original)
            ema_file = self.exp.get_models_file_ema(self.load_ep)
            self.logger.info(f"Update EMA from {ema_file}")
            self.ema.set_state_dict(th.load(str(ema_file)))
            self.ema.assign(self.model, update_model=False)

        # disable ema when loading model directly or when decay is 0 / -1
        if self.load_model or cfg.optim.ema_decay <= 0:
            self.ema = None

    def train_model(self, train_loader: data.DataLoader, val_loader: data.DataLoader, epoch_i: int = 0) -> None:
        """
        Train epochs until done.

        Args:
            train_loader: Training dataloader.
            val_loader: Validation dataloader.
        """
        self.hook_pre_train()  # pre-training hook: time book-keeping etc.
        self.steps_per_epoch = len(train_loader)  # save length of epoch

        # ---------- Epoch Loop ----------
        for _epoch in range(self.state.current_epoch, self.cfg.train.num_epochs):
            train_loader.sampler.set_epoch(_epoch)
            if self.check_early_stop():
                break
            self.hook_pre_train_epoch()  # pre-epoch hook: set models to train, time book-keeping

            # check exponential moving average
            if self.ema is not None and self.state.current_epoch != 0 and self.cfg.optim.ema_decay != -1:
                # use normal parameters for training, not EMA model
                self.ema.resume(self.model)
            self.logger.info('resume parameters to ema mode')

            th.autograd.set_detect_anomaly(True)

            total_loss = 0
            n_word_total = 0
            n_word_correct = 0

            # ---------- Dataloader Iteration ----------
            # time.sleep(2)
            for step, batch in enumerate(train_loader):

                self.hook_pre_step_timer()  # hook for step timing

                # ---------- forward pass ----------
                self.optimizer.zero_grad()
                with autocast(enabled=self.cfg.exp.fp16_train):
                    if self.cfg.data.recurrent:
                        raise NotImplementedError("Recurrent processing was not integrated into this version!")
                        # ---------- training step for recurrent models ----------
                        
                    else:
                        # ---------- non-recurrent MART and maybe others ----------
                        meta = batch[2]
                        batched_data = prepare_batch_inputs(batch[0], use_cuda=self.cfg.exp.use_cuda,
                                                            non_blocking=self.cfg.exp.cuda_non_blocking)
                        input_ids = batched_data["input_ids"]
                        video_features = batched_data["video_feature"]
                        input_masks = batched_data["input_mask"]
                        token_type_ids = batched_data["token_type_ids"]
                        input_labels = batched_data["input_labels"]

                        detect_ids = batched_data["detect_ids"]
                        detect_features = batched_data["detect_feature"]
                        detect_masks = batched_data["detect_mask"]
                        detect_token_type_ids = batched_data["detect_token_type_ids"]
                        detect_cates = batched_data["detect_cates"]

                        action_ids = batched_data["action_ids"]
                        action_features = batched_data["action_feature"]
                        action_masks = batched_data["action_mask"]
                        action_token_type_ids = batched_data["action_token_type_ids"]
                        action_cates = batched_data["action_cates"]

                        # forward & backward
                        loss, pred_scores, cls_loss, act_loss = self.model(input_ids, video_features, input_masks, token_type_ids,
                                                       input_labels, detect_ids, detect_features, detect_masks,
                                                       detect_token_type_ids, detect_cates, action_ids, action_features,
                                                       action_masks, action_token_type_ids, action_cates)
                        pred_scores_list = [pred_scores]
                        input_labels_list = [input_labels]

                self.hook_post_forward_step_timer()  # hook for step timing

                # ---------- backward pass ----------
                grad_norm = None
                if self.cfg.exp.fp16_train:                             # setting false because it may cause NaN!
                    # with fp16 amp
                    self.grad_scaler.scale(loss).backward()
                    if self.cfg.train.clip_gradient != -1:
                        # gradient clipping
                        self.grad_scaler.unscale_(self.optimizer)
                        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_gradient)
                    # gradient scaler realizes if gradients have been unscaled already and doesn't do it again.
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    # self.logger.info("captioning loss: {}, cls loss:{}".format(loss, cls_loss))
                    # with regular float32

                    vid_loss = loss
                    if self.cfg.model.branch.num > 2:
                        loss = loss + cls_loss + act_loss
                    elif self.cfg.model.branch.num == 2:
                        if self.cfg.model.branch.detect_input:
                            loss = loss + cls_loss
                        elif self.cfg.model.branch.action_input:
                            loss = loss + act_loss
                    else:
                        loss = loss
                    loss.backward()
                    if self.cfg.train.clip_gradient != -1:
                        # gradient clipping
                        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_gradient)
                    self.optimizer.step()
                # update model parameters with ema
                if self.ema is not None:
                    self.ema(self.model, self.state.total_step)

                # keep track of loss, accuracy, gradient norm
                total_loss += loss.sum().item()
                n_correct = 0
                n_word = 0
                for pred, gold in zip(pred_scores_list, input_labels_list):
                    n_correct += cal_performance(pred, gold)
                    valid_label_mask = gold.ne(CaptionDataset.IGNORE)
                    n_word += valid_label_mask.sum().item()
                n_word_total += n_word
                n_word_correct += n_correct
                if grad_norm is not None:
                    self.metrics.update_meter(MMeters.GRAD, grad_norm)

                if self.cfg.exp.debug:
                    break

                additional_log = f" Grad {self.metrics.meters[MMeters.GRAD].avg:.2f}"
                self.hook_post_backward_step_timer()  # hook for step timing

                # post-step hook: gradient clipping, profile gpu, update metrics, count step, step LR scheduler, log
                # self.logger.info(self.optimizer.get_lr())
                current_lr = self.optimizer.get_lr()[0]
                self.hook_post_step(step, vid_loss.sum(), cls_loss.sum(), act_loss.sum(), current_lr, additional_log=additional_log,
                # self.hook_post_step(step, vid_loss.sum(), cls_loss.sum(), current_lr, additional_log=additional_log,
                                    disable_grad_clip=True)
                # self.hook_post_step(step, loss.sum(), current_lr, additional_log=additional_log, disable_grad_clip=True)

            # log train statistics
            loss_per_word = 1.0 * total_loss / n_word_total
            accuracy = 1.0 * n_word_correct / n_word_total
            self.metrics.update_meter(MMeters.TRAIN_LOSS_PER_WORD, loss_per_word)
            self.metrics.update_meter(MMeters.TRAIN_ACC, accuracy)
            # return loss_per_word, accuracy

            # ---------- validation ----------
            do_val = self.check_is_val_epoch()                  # 判断是否要validation，可以在这里设置开始validation的时间

            is_best = False
            if do_val:
                # run validation including with ground truth tokens and translation without any text
                _val_loss, _val_score, is_best, _metrics = self.validate_epoch(val_loader)
            else:
                if self.ema is not None:
                    self.ema.assign(self.model)

            # save the EMA weights
            if is_main_process():
                ema_file = self.exp.get_models_file_ema(self.state.current_epoch)
                th.save(self.ema.state_dict(), str(ema_file))

            # post-epoch hook: scheduler, save checkpoint, time bookkeeping, feed tensorboard
            self.hook_post_train_and_val_epoch(do_val, is_best)

        # show end of training log message
        self.hook_post_train()

    @th.no_grad()
    def validate_epoch(self, data_loader: data.DataLoader) -> (
            Tuple[float, float, bool, Dict[str, float]]):
        """
        Run both validation and translation.

        Validation: The same setting as training, where ground-truth word x_{t-1} is used to predict next word x_{t},
        not realistic for real inference.

        Translation: Use greedy generated words to predicted next words, the true inference situation.
        eval_mode can only be set to `val` here, as setting to `test` is cheating
        0. run inference, 1. Get METEOR, BLEU1-4, CIDEr scores, 2. Get vocab size, sentence length

        Args:
            data_loader: Dataloader for validation

        Returns:
            Tuple of:
                validation loss
                validation score
                epoch is best
                custom metrics with translation results dictionary
        """
        self.hook_pre_val_epoch()  # pre val epoch hook: set models to val and start timers
        forward_time_total = 0
        total_loss = 0
        n_word_total = 0
        n_word_correct = 0

        # setup ema
        if self.ema is not None:
            self.ema.assign(self.model)        

        # setup translation submission
        batch_res = {"version": "VERSION 1.0", "results": defaultdict(list),
                     "external_data": {"used": "true", "details": "ay"}}

        dataset: CaptionDataset = data_loader.dataset

        # ---------- Dataloader Iteration ----------
        num_steps = 0
        pbar = tqdm(total=len(data_loader), desc=f"Validate epoch {self.state.current_epoch}")
        time.sleep(2)
        for _step, batch in enumerate(data_loader):
            # ---------- forward pass ----------
            self.hook_pre_step_timer()  # hook for step timing

            with autocast(enabled=self.cfg.exp.fp16_val):
                if self.cfg.data.recurrent:
                   raise NotImplementedError("Recurrent processing was not integrated into this version!")
                else:
                    # non-recurrent but also not untied model (?)
                    meta = batch[2]  # list(dict), len == bsz

                    # validate
                    batched_data = prepare_batch_inputs(batch[0], use_cuda=self.cfg.exp.use_cuda,
                                                        non_blocking=self.cfg.exp.cuda_non_blocking)
                    input_ids = batched_data["input_ids"]
                    video_features = batched_data["video_feature"]
                    input_masks = batched_data["input_mask"]
                    token_type_ids = batched_data["token_type_ids"]
                    input_labels = batched_data["input_labels"]

                    detect_ids = batched_data["detect_ids"]
                    detect_features = batched_data["detect_feature"]
                    detect_masks = batched_data["detect_mask"]
                    detect_token_type_ids = batched_data["detect_token_type_ids"]
                    detect_cates = batched_data["detect_cates"]

                    action_ids = batched_data["action_ids"]
                    action_features = batched_data["action_feature"]
                    action_masks = batched_data["action_mask"]
                    action_token_type_ids = batched_data["action_token_type_ids"]
                    action_cates = batched_data["action_cates"]
                    
                    loss, pred_scores, _, _ = self.model(input_ids, video_features, input_masks,
                                                    token_type_ids, input_labels, detect_ids, detect_features,
                                                    detect_masks, detect_token_type_ids, detect_cates,
                                                    action_ids, action_features, action_masks, action_token_type_ids,
                                                    action_cates)
                    pred_scores_list = [pred_scores]
                    input_labels_list = [input_labels]
                    # input_labels_list = [action_labels]
                    # translate
                    model_inputs = [batched_data["input_ids"], batched_data["video_feature"],
                                    batched_data["input_mask"], batched_data["token_type_ids"],
                                    batched_data["detect_ids"], batched_data["detect_feature"],
                                    batched_data["detect_mask"], batched_data["detect_token_type_ids"],
                                    batched_data["detect_cates"], batched_data["action_ids"],
                                    batched_data["action_feature"],
                                    batched_data["action_mask"], batched_data["action_token_type_ids"]]
                    dec_seq = self.translator.translate_batch(
                        model_inputs, use_beam=self.cfg.beam.use_beam, recurrent=False, untied=False)
                    for example_idx, (cur_gen_sen, cur_meta) in enumerate(zip(dec_seq, meta)):
                        # example_idx indicates which example is in the batch
                        cur_data = {
                            "sentence": dataset.convert_ids_to_sentence(cur_gen_sen.cpu().tolist()),
                            "timestamp": cur_meta["timestamp"], "gt_sentence": cur_meta["gt_sentence"]}
                        batch_res["results"][cur_meta["name"]].append(cur_data)

                # keep logs
                n_correct = 0
                n_word = 0
                for pred, gold in zip(pred_scores_list, input_labels_list):
                    n_correct += cal_performance(pred, gold)
                    valid_label_mask = gold.ne(CaptionDataset.IGNORE)
                    n_word += valid_label_mask.sum().item()

                # calculate metrix
                n_word_total += n_word
                n_word_correct += n_correct
                total_loss += loss.item()

            # end of step
            self.hook_post_forward_step_timer()
            forward_time_total += self.timedelta_step_forward
            num_steps += 1

            if self.cfg.exp.debug:
                break

            pbar.update()
        pbar.close()
        synchronize()

        batch_res_list = all_gather(batch_res)
        batch_res ={"version": "VERSION 1.0", "results": defaultdict(list),
                     "external_data": {"used": "true", "details": "ay"}}
        for split_batch_res in batch_res_list:
            batch_res['results'].update(split_batch_res['results'])
        # ---------- validation done ----------

        # sort translation
        batch_res["results"] = self.translator.sort_res(batch_res["results"])

        # write translation results of this epoch to file
        eval_mode = self.cfg.data.val_split  # which dataset split
        file_translation_raw = self.exp.get_translation_files(self.state.current_epoch, eval_mode)
        self.logger.info(file_translation_raw)
        if is_main_process():
            json.dump(batch_res, file_translation_raw.open("wt", encoding="utf8"))
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()

        # get reference files (ground truth captions)
        reference_files_map = get_reference_files(self.cfg.data.name, self.exp.annotations_dir)
        reference_files = reference_files_map[eval_mode]
        reference_file_single = reference_files[0]

        # language evaluation
        res_lang = evaluate_language_files(file_translation_raw, reference_files, verbose=False, all_scorer=True)
        # basic stats
        res_stats = evaluate_stats_files(file_translation_raw, reference_file_single, verbose=False)
        # repetition
        res_rep = evaluate_repetition_files(file_translation_raw, reference_file_single, verbose=False)

        # merge results
        all_metrics = {**res_lang, **res_stats, **res_rep}
        assert len(all_metrics) == len(res_lang) + len(res_stats) + len(res_rep), (
            "Lost infos while merging translation results!")

        # flatten results and make them json compatible
        flat_metrics = {}
        for key, val in all_metrics.items():
            if isinstance(val, Mapping):
                for subkey, subval in val.items():
                    flat_metrics[f"{key}_{subkey}"] = subval
                continue
            flat_metrics[key] = val
        for key, val in flat_metrics.items():
            if isinstance(val, (np.float16, np.float32, np.float64)):
                flat_metrics[key] = float(val)

        # feed meters
        for result_key, meter_name in TRANSLATION_METRICS.items():
            self.metrics.update_meter(meter_name, flat_metrics[result_key])

        # log translation results
        self.logger.info(f"Done with translation, epoch {self.state.current_epoch} split {eval_mode}")
        self.logger.info(", ".join([f"{name} {flat_metrics[name]:.2%}" for name in TRANSLATION_METRICS_LOG]))

        # calculate and output validation metrics
        loss_per_word = 1.0 * total_loss / n_word_total
        accuracy = 1.0 * n_word_correct / n_word_total
        self.metrics.update_meter(MMeters.TRAIN_LOSS_PER_WORD, loss_per_word)
        self.metrics.update_meter(MMeters.TRAIN_ACC, accuracy)
        forward_time_total /= num_steps
        self.logger.info(
            f"Loss {loss_per_word:.5f} Acc {accuracy:.3%} total {timer() - self.timer_val_epoch:.3f}s, "
            f"forward {forward_time_total:.3f}s")

        # find field which determines whether this is a new best epoch
        if self.cfg.val.det_best_field == "cider":
            val_score = flat_metrics["CIDEr"]
        else:
            raise NotImplementedError(f"best field {self.cfg.val.det_best_field} not known")

        # check for a new best epoch and update validation results
        is_best = self.check_is_new_best(val_score)
        self.hook_post_val_epoch(loss_per_word, is_best)

        if self.is_test:
            # for test runs, save the validation results separately to a file
            self.metrics.feed_metrics(False, self.state.total_step, self.state.current_epoch)
            metrics_file = self.exp.path_base / f"val_ep_{self.state.current_epoch}.json"
            self.metrics.save_epoch_to_file(metrics_file)
            self.logger.info(f"Saved validation results to {metrics_file}")

            # update the meteor metric in the result if it's -999 because java crashed. only in some conditions
            best_ep = self.exp.find_best_epoch()
            self.logger.info(f"Dataset split config {self.cfg.data.val_split} loaded {self.load_ep} best {best_ep}")
            if self.cfg.data.val_split == "val" and self.load_ep == best_ep == self.state.current_epoch:
                # load metrics file and write it back with the new meteor IFF meteor is -999
                metrics_file = self.exp.get_metrics_epoch_file(best_ep)
                metrics_data = json.load(metrics_file.open("rt", encoding="utf8"))
                # metrics has stored meteor as a list of tuples (epoch, value). convert to dict, update, convert back.
                meteor_dict = dict(metrics_data[TextMetricsConst.METEOR])
                if ((meteor_dict[best_ep] + 999) ** 2) < 1e-4:
                    meteor_dict[best_ep] = flat_metrics[TextMetricsConstEvalCap.METEOR]
                    metrics_data[TextMetricsConst.METEOR] = list(meteor_dict.items())
                    json.dump(metrics_data, metrics_file.open("wt", encoding="utf8"))
                    self.logger.info(f"Updated meteor in file {metrics_file}")

        return total_loss, val_score, is_best, flat_metrics

    def get_opt_state(self) -> Dict[str, Dict[str, nn.Parameter]]:
        """
        Return the current optimizer and scheduler state.
        Note that the BertAdam optimizer used already includes scheduling.

        Returns:
            Dictionary of optimizer and scheduler state dict.
        """
        return {
            "optimizer": self.optimizer.state_dict()
            # "lr_scheduler": self.lr_scheduler.state_dict()
        }

    def set_opt_state(self, opt_state: Dict[str, Dict[str, nn.Parameter]]) -> None:
        """
        Set the current optimizer and scheduler state from the given state.

        Args:
            opt_state: Dictionary of optimizer and scheduler state dict.
        """
        self.optimizer.load_state_dict(opt_state["optimizer"])
        # self.lr_scheduler.load_state_dict(opt_state["lr_scheduler"])

    def get_files_for_cleanup(self, epoch: int) -> List[Path]:
        """
        Implement this in the child trainer.

        Returns:
            List of files to cleanup.
        """
        return [
            # self.exp.get_translation_files(epoch, split="train"),
            self.exp.get_translation_files(epoch, split="val"),
            self.exp.get_models_file_ema(epoch)]
