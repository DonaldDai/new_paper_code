import torch
import torch.nn as nn
import copy
import math

from models.transformer.module.positional_encoding import PositionalEncoding
from models.transformer.module.positionwise_feedforward import PositionwiseFeedForward
from models.transformer.module.multi_headed_attention import MultiHeadedAttention
from models.transformer.module.embeddings import Embeddings
from models.transformer.encode_decode.encoder import Encoder
from models.transformer.encode_decode.decoder import Decoder
from models.transformer.encode_decode.encoder_layer import EncoderLayer
from models.transformer.encode_decode.decoder_layer import DecoderLayer
from models.transformer.module.generator import Generator
from common.utils import Data_Type

def show_variable(varName, note=''):
    print(f"{note}:  {varName}")
    # varName=torch.tensor(varName)
    print(f"{note} variable len info: {len(varName)}")
    print(f"{note} variable size info: {varName[0].size()}")
    print(f"{note} variable info: {varName.size()}")
    # show_variable(target_vec, "target_vec")
    # show_variable(collated_arr_source, "collated_arr_source")

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, esm_size, data_type=Data_Type.base):
        super(Embeddings, self).__init__()
        # weight matrix, each row present one word
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        self.linear=nn.Linear(esm_size+d_model,d_model)
        self.data_type = data_type

    def forward(self, x_target_vec):  ## the parallel mechanism require only one variable as input
        # if self.ifSource:
        x=x_target_vec[0]
        target_vec=x_target_vec[1]
        torch_embed = self.lut(x) * math.sqrt(self.d_model)
        if self.data_type != Data_Type.seq_esm:
            return torch_embed
        target_vec=target_vec.unsqueeze(1).expand(-1,x.shape[1],-1)
        embed_seq=torch.concat((torch_embed,target_vec),dim=-1)
        embed_seq=self.linear(embed_seq)
        # show_variable(torch_embed, "torch_embed")
        # show_variable(target_vec, "target_vec")
        # show_variable(x, "x")

        return embed_seq
        # else:
        #     x=x_target_vec
        #     # show_variable(torch_embed, "torch_embed")
        #     # show_variable(target_vec, "target_vec")
        #     # sys.exit()
        #     return self.lut(x) * math.sqrt(self.d_model)

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask, target_vec):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask, target_vec), src_mask,
                           tgt, tgt_mask, target_vec)

    def encode(self, src, src_mask, target_vec):
        return self.encoder(self.src_embed([src, target_vec]), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, target_vec):
        return self.decoder(self.tgt_embed([tgt, target_vec]), memory, src_mask, tgt_mask)

    @classmethod
    def make_model(cls, src_vocab, tgt_vocab, N=6,
                   d_model=256, d_ff=2048, h=8, dropout=0.1, esm_size=1280, data_type=Data_Type.base):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                 c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model, src_vocab, esm_size, data_type=data_type), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab, esm_size, data_type=data_type), c(position)),
            Generator(d_model, tgt_vocab))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

        return model

    @classmethod
    def load_from_file(cls, file_path):
        # Load model
        checkpoint = torch.load(file_path, map_location='cuda:0')
        para_dict = checkpoint['model_parameters']
        vocab_size = para_dict['vocab_size']
        model = EncoderDecoder.make_model(vocab_size, vocab_size, para_dict['N'],
                                  para_dict['d_model'], para_dict['d_ff'],
                                  para_dict['H'], para_dict['dropout'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model