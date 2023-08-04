"""
Train captioning. Our code builds heavily on MART(https://github.com/jayleicn/recurrent-transformer) and 
COOT (https://github.com/gingsi/coot-videotext). Thanks for their sharing codes.
"""

import os
import torch
import numpy as np
import hydra

from mart.trainer import MartTrainer
from mart.model import create_mart_model
from dataset import create_datasets_and_loaders
from nntrainer.utils import set_seed, init_distributed_mode, is_main_process
from omegaconf import OmegaConf, open_dict


@hydra.main(version_base=None, config_path="configs", config_name="yc2_non_recurrent")
def main(cfg):
    # for ddp training
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.local_rank = int(os.environ["LOCAL_RANK"])

    # set seed
    verb = "Set seed"
    if cfg.exp.random_seed is None:
        with open_dict(cfg):
            cfg.exp.random_seed = np.random.randint(0, 2**15, dtype=np.int32).item()
        verb = "Randomly generated seed"
    else:
        verb = "Pre setting seed"
    set_seed(cfg.exp.random_seed, cudnn_deterministic=cfg.exp.cudnn_deterministic, 
             cudnn_benchmark=cfg.exp.cudnn_benchmark)

    # init multi-gpu training
    init_distributed_mode(cfg)
    
    if cfg.print.config and is_main_process():
        print(cfg)
    print(f"{verb} {cfg.exp.random_seed} deterministic {cfg.exp.cudnn_deterministic} "
          f"benchmark {cfg.exp.cudnn_benchmark}")

    # create dataset
    train_set, train_loader, val_loader = create_datasets_and_loaders(cfg)

    for i, run_number in enumerate(range(cfg.exp.start_run, cfg.exp.start_run+cfg.exp.num_runs)):
        run_name = f"{cfg.exp.run_name}{run_number}"

        # create model from config
        model = create_mart_model(cfg, len(train_set.word2idx)).cuda()

        if cfg.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])

        # print model for debug if requested
        if cfg.print.model and i == 0:
            print(model)

        # always load best epoch during validation
        load_best = cfg.exp.load_best or cfg.validate

        # create trainer
        trainer = MartTrainer(
            cfg, model, run_name, len(train_loader), log_dir=cfg.exp.log_dir, log_level=cfg.logging.level, 
            print_graph=cfg.print.graph, reset=cfg.exp.reset, load_best=load_best, load_epoch=cfg.exp.load_epoch, 
            load_model=cfg.exp.load_model, inference_only=cfg.validate, annotations_dir=cfg.data.annotations_dir
        )

        if cfg.validate:
            # run validation
            if not trainer.load and not cfg.ignore_untrained:
                raise ValueError("Validating an untrained model! No checkpoints were loaded. Add --ignore_untrained "
                                 "to ignore this error.")
            trainer.validate_epoch(val_loader)
            trainer.close()
            break
        else:
            # run training
            trainer.train_model(train_loader, val_loader)
            trainer.close()
            del model
            del trainer


if __name__ == "__main__":
    main()
