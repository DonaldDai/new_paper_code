import argparse

import os
import torch.distributed as dist
import configuration.opts as opts
from trainer.transformer_trainer import TransformerTrainer
# from trainer.seq2seq_trainer import Seq2SeqTrainer
from common.utils import Data_Type
import datetime
from glob import glob
from pathlib import Path

def data_type(t):
    try:
        return Data_Type[t]
    except KeyError:
        raise argparse.ArgumentTypeError(f"{t} is not a valid data type")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.train_opts(parser)
    opts.train_opts_transformer(parser)
    parser.add_argument("--data-type", type=data_type, default='base')
    parser.add_argument("--bar", type=bool, default=False)
    parser.add_argument("--seq2vec-path", type=str, default='')
    opt = parser.parse_args()

    timeout = datetime.timedelta(days=1)
    dist.init_process_group("nccl", timeout=timeout)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    ckpts = glob(os.path.join(opt.pretrain_path, f'checkpoint/model_*.pt'))
    # 自动识别下一个epoch 这里会忽略命令行传入的
    if len(ckpts) > 0:
        opt.starting_epoch = max([int(Path(ckpt).stem.split('_')[1]) for ckpt in ckpts]) + 1
    trainer = TransformerTrainer(opt, local_rank, rank, world_size)
        
    # elif opt.model_choice == 'seq2seq':
    #     trainer = Seq2SeqTrainer(opt)
    print(f"Starting training on rank {rank} out of {world_size} processes | {opt.bar}")
    trainer.train(opt)
