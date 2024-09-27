import os
import pickle as pkl
# os.environ["CUDA_VISIBLE_DEVICES"]="5"
import torch
import torch.nn as nn
import configuration.config_default as cfgd
import utils.log as ul
import utils.file as uf
import utils.torch_util as ut
import preprocess.vocabulary as mv
from models.transformer.encode_decode.model import EncoderDecoder
from models.transformer.module.noam_opt import NoamOpt as moptim
from models.transformer.module.decode import decode
from trainer.base_trainer import BaseTrainer
from models.transformer.module.label_smoothing import LabelSmoothing
from models.transformer.module.simpleloss_compute import SimpleLossCompute
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import signal
import models.dataset as md
import pandas as pd

# 全局变量用于控制训练循环
should_stop = False

def setup():
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()

def signal_handler(signum, frame):
    global should_stop
    print(f"Received signal {signum}. Stopping training...", flush=True)
    should_stop = True


class TransformerTrainer(BaseTrainer):

    def __init__(self, opt, rank, world_size):
        super().__init__(opt)
        self.rank = rank
        self.world_size = world_size
        self.data_type = opt.data_type
        print(f'===useing data type {self.data_type}')

    def get_model(self, opt, vocab, device):
        vocab_size = len(vocab.tokens())
        # build a model from scratch or load a model from a given epoch
        if opt.starting_epoch == 1:
            # define model
            model = EncoderDecoder.make_model(vocab_size, vocab_size, N=opt.N,
                                          d_model=opt.d_model, d_ff=opt.d_ff, h=opt.H, dropout=opt.dropout)
        else:
            # Load model
            file_name = os.path.join(opt.pretrain_path, f'checkpoint/model_{opt.starting_epoch-1}.pt')
            model= EncoderDecoder.load_from_file(file_name)
        # move to GPU
        model.to(device)
        return model

    def _initialize_optimizer(self, model, opt):
        optim = moptim(model.src_embed[0].d_model, opt.factor, opt.warmup_steps,
                       torch.optim.Adam(model.parameters(), lr=0, betas=(opt.adam_beta1, opt.adam_beta2),
                                        eps=opt.adam_eps))
        return optim

    def _load_optimizer_from_epoch(self, model, file_name):
        # load optimization
        checkpoint = torch.load(file_name, map_location='cuda:0')
        optim_dict = checkpoint['optimizer_state_dict']
        optim = moptim(optim_dict['model_size'], optim_dict['factor'], optim_dict['warmup'],
                       torch.optim.Adam(model.parameters(), lr=0))
        optim.load_state_dict(optim_dict)
        return optim

    def get_optimization(self, model, opt):
        # optimization
        if opt.starting_epoch == 1:
            optim = self._initialize_optimizer(model, opt)
        else:
            # load optimization
            file_name = os.path.join(opt.pretrain_path,  f'checkpoint/model_{opt.starting_epoch-1}.pt')
            optim = self._load_optimizer_from_epoch(model, file_name)
        return optim

    def initialize_dataloader(self, data_path, batch_size, vocab, data_type, use_random=False):
        data = pd.read_csv((os.path.join(data_path, data_type + '.csv')), sep=',')
        dataset = md.Dataset(data=data, vocabulary=vocab, tokenizer=(mv.SMILESTokenizer()), prediction_mode=False, use_random=use_random)
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank)
        dataloader = DataLoader(dataset, batch_size, sampler=sampler,
          collate_fn=(md.Dataset.collate_fn))
        return dataloader

    def train_epoch(self, dataloader, model, loss_compute, device):

        pad = cfgd.DATA_DEFAULT['padding_value']
        total_loss = 0
        total_tokens = 0
        for i, batch in enumerate(ul.progress_bar(dataloader, total=len(dataloader))):
            if should_stop:
                break
            src, source_length, trg, src_mask, trg_mask, _, _ = batch

            trg_y = trg[:, 1:].to(device)  # skip start token

            # number of tokens without padding
            ntokens = float((trg_y != pad).data.sum())

            # Move to GPU
            src = src.to(device)
            trg = trg[:, :-1].to(device)  # save start token, skip end token
            src_mask = src_mask.to(device)
            trg_mask = trg_mask.to(device)

            # Compute loss
            out = model.forward(src, trg, src_mask, trg_mask)
            loss = loss_compute(out, trg_y, ntokens)
            total_tokens += ntokens
            total_loss += float(loss)

        loss_epoch = total_loss / total_tokens
       # print("total_loss_train",total_loss)
       # print("total_tokens",total_tokens)
        return loss_epoch

    def validation_stat(self, dataloader, model, loss_compute, device, vocab):
        pad = cfgd.DATA_DEFAULT['padding_value']
        total_loss = 0

        n_correct = 0
        total_n_trg = 0
        total_tokens = 0

        tokenizer = mv.SMILESTokenizer()
        for i, batch in enumerate(ul.progress_bar(dataloader, total=len(dataloader))):

            src, source_length, trg, src_mask, trg_mask, _, _ = batch

            trg_y = trg[:, 1:].to(device)  # skip start token

            # number of tokens without padding
            ntokens = float((trg_y != pad).data.sum())

            # Move to GPU
            src = src.to(device)
            trg = trg[:, :-1].to(device)  # save start token, skip end token
            src_mask = src_mask.to(device)
            trg_mask = trg_mask.to(device)

            with torch.no_grad():
                out = model.forward(src, trg, src_mask, trg_mask)
                loss = loss_compute(out, trg_y, ntokens).cuda()
                total_loss += float(loss)
                total_tokens += ntokens
                # Decode
                max_length_target = cfgd.DATA_DEFAULT['max_sequence_length']
                smiles = decode(model, src, src_mask, max_length_target, type='greedy')

                # Compute accuracy
                for j in range(trg.size()[0]):
                    seq = smiles[j, :]
                    target = trg[j]
                    target = tokenizer.untokenize(vocab.decode(target.cpu().numpy()))
                    seq = tokenizer.untokenize(vocab.decode(seq.cpu().numpy()))
                   # print("seq",seq)
                   # print("target",target)
                    if seq == target:
                        n_correct += 1
          #          print("N_CORRECT_1",n_correct)
         #   print("N_CORRECT_2",n_correct)

            # number of samples in current batch
            n_trg = trg.size()[0]
            # total samples
            total_n_trg += n_trg
           # print("n_trg inner:",n_trg)
           # print("total_n_trg inner:",total_n_trg)
           # print("accuracy n_trg inner:",n_correct/n_trg)
           # print("accuracy total n_trg inner:",n_correct/total_n_trg)
           # print("n_correct_num inner:",n_correct)

        # Accuracy
        accuracy = n_correct*1.0 /total_n_trg
        loss_epoch = total_loss / total_tokens
       # print("n_correct_val_final",n_correct)
       # print("n_trg_final",n_trg)
       # print("total_n_trg",total_n_trg)
        return loss_epoch, accuracy

    def _get_model_parameters(self, vocab_size, opt):
        return {
            'vocab_size': vocab_size,
            'N': opt.N,
            'd_model': opt.d_model,
            'd_ff': opt.d_ff,
            'H': opt.H,
            'dropout': opt.dropout
        }

    def save(self, model, optim, epoch, vocab_size, opt):
        """
        Saves the model, optimizer and model hyperparameters
        """
        save_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.save_state_dict(),
            'model_parameters': self._get_model_parameters(vocab_size, opt)
        }

        file_name = os.path.join(self.save_path, f'checkpoint/model_{epoch}.pt')
        uf.make_directory(file_name, is_dir=False)

        torch.save(save_dict, file_name)

    def train(self, opt):
        isMain = self.rank == 0
        global should_stop
        torch.cuda.set_device(self.rank)
        device = torch.device(f"cuda:{self.rank}")
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        # Load vocabulary
        # 加载词表
        if opt.starting_epoch == 1:
            with open(os.path.join(opt.data_path, 'vocab.pkl'), "rb") as input_file:
                vocab = pkl.load(input_file)
        else:
            with open(os.path.join(opt.vocab_path, 'vocab.pkl'), "rb") as input_file:
                vocab = pkl.load(input_file)
        vocab_size = len(vocab.tokens())

        print(f"=====Availablee GPUs: {torch.cuda.device_count()}")
        # Data loader
        dataloader_train = self.initialize_dataloader(opt.data_path, opt.batch_size, vocab, 'train_cut', use_random=True)
        dataloader_validation = self.initialize_dataloader(opt.data_path, opt.batch_size, vocab, 'validation_cut')
        # device = torch.device('cuda')
        #device = ut.allocate_gpu(1)
        #device = ut.allocate_gpu_multi()
        '''
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )
        os.environ['CUDA_VISIBLE_DEVICES']='1'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        '''
        # 重新创建或读取先前model
        model = self.get_model(opt, vocab, device)
        # 重新创建优化器或读取先前optim
        optim = self.get_optimization(model, opt)
        # 分布式
        model = DDP(model,device_ids=[self.rank], output_device=self.rank)

        #model = model.module
        pad_idx = cfgd.DATA_DEFAULT['padding_value']
        criterion = LabelSmoothing(size=len(vocab), padding_idx=pad_idx, smoothing=opt.label_smoothing)

        print(f'=====before train: {self.rank}/{self.world_size}')
        dist.barrier()
        # Train epoch
        for epoch in range(opt.starting_epoch, opt.starting_epoch + opt.num_epoch):
            self.LOG.info("Starting EPOCH #%d", epoch)

            self.LOG.info("Training start")
            model.module.train()
            print(f'=====before train epoch({epoch}): {self.rank}/{self.world_size}')
            dist.barrier()
            loss_epoch_train = self.train_epoch(dataloader_train,
                                                       model.module,
                                                       SimpleLossCompute(
                                                                 model.module.generator,
                                                                 criterion,
                                                                 optim), device)

            if should_stop:
                break
            if not isMain:
                continue
            self.LOG.info("Training end")
            self.save(model.module, optim, epoch, vocab_size, opt)
            self.LOG.info("Validation start")
            model.module.eval()
            loss_epoch_validation, accuracy = self.validation_stat(
                dataloader_validation,
                model.module,
                SimpleLossCompute(
                    model.module.generator, criterion, None),
                device, vocab)


            self.LOG.info("Validation end")

            self.LOG.info(
                "Train loss, Validation loss, accuracy: {}, {}, {}".format(loss_epoch_train, loss_epoch_validation,
                                                                           accuracy))

            self.to_tensorboard(loss_epoch_train, loss_epoch_validation, accuracy, epoch)
