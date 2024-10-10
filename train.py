import argparse

import os
import torch.distributed as dist
import configuration.opts as opts
from trainer.transformer_trainer import TransformerTrainer
# from trainer.seq2seq_trainer import Seq2SeqTrainer
from common.utils import Data_Type

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

    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    trainer = TransformerTrainer(opt, local_rank, rank, world_size)
        
    # elif opt.model_choice == 'seq2seq':
    #     trainer = Seq2SeqTrainer(opt)
    print(f"Starting training on rank {rank} out of {world_size} processes | {opt.bar}")
    trainer.train(opt)
