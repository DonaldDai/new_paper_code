# uncompyle6 version 3.8.0
# Python bytecode 3.7.0 (3394)
# Decompiled from: Python 3.7.13 (default, Mar 29 2022, 02:18:16) 
# [GCC 7.5.0]
# Embedded file name: /home/user/chenh0/TranSARMer_fine_tune/deep-molecular-optimization-main/trainer/base_trainer.py
# Compiled at: 2022-02-16 02:13:00
# Size of source mod 2**32: 1802 bytes
import os, pandas as pd
from abc import ABC, abstractmethod
import torch
from tensorboardX import SummaryWriter
import utils.log as ul
import models.dataset as md
import preprocess.vocabulary as mv

class BaseTrainer(ABC):

    def __init__(self, opt, rank):
        isMain = rank == 0
        self.save_path = opt.save_directory  # os.path.join('experiments_fine_tune_0215', opt.save_directory)
        if isMain:
            self.summary_writer = SummaryWriter(logdir=(os.path.join(self.save_path, 'tensorboard')))
        LOG = ul.get_logger(name='train_model', log_path=(os.path.join(self.save_path, 'train_model.log')))
        self.LOG = LOG
        self.LOG.info(opt)

    def initialize_dataloader(self, data_path, batch_size, vocab, data_type):
        data = pd.read_csv((os.path.join(data_path, data_type + '.csv')), sep=',')
        dataset = md.Dataset(data=data, vocabulary=vocab, tokenizer=(mv.SMILESTokenizer()), prediction_mode=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
          collate_fn=(md.Dataset.collate_fn))
        return dataloader

    def to_tensorboard(self, train_loss, validation_loss, accuracy, epoch):
        self.summary_writer.add_scalars('loss', {'train':train_loss, 
         'validation':validation_loss}, epoch)
        self.summary_writer.add_scalar('accuracy/validation', accuracy, epoch)
        self.summary_writer.close()

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_optimization(self):
        pass

    @abstractmethod
    def train_epoch(self):
        pass

    @abstractmethod
    def validation_stat(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def train(self):
        pass
# okay decompiling base_trainer.cpython-37.pyc
