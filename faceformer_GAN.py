import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from wav2vec import Wav2Vec2Model
from Discriminator import Discriminator

# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(device, dataset, T, S):
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    return (mask==1).to(device=device)

# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class FaceformerGAN(nn.Module):
    def __init__(self, args, mode='train'):
        super(FaceformerGAN, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.dataset = args.dataset
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        # wav2vec 2.0 weights initialization
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_feature_map = nn.Linear(768, args.feature_dim)
        # motion encoder
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)
        # periodic positional encoding 
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period = args.period)
        # temporal bias
        self.biased_mask = init_biased_mask(n_head=4, max_seq_len = 600, period=args.period)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=4, dim_feedforward=2*args.feature_dim, batch_first=True)        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        # motion decoder
        self.vertice_map_r = nn.Linear(args.feature_dim, args.vertice_dim)
        # style embedding
        self.obj_vector = nn.Linear(len(args.train_subjects.split()), args.feature_dim, bias=False)
        self.device = args.device
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

        if mode == 'train':
            self.GAN_criterion = nn.MSELoss()
            self.w_GAN = args.w_GAN
            self.w_recon = args.w_recon

    def forward(self, audio, template, vertice, one_hot, teacher_forcing=True):

        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        template = template.unsqueeze(1) # (1,1, V*3)
        obj_embedding = self.obj_vector(one_hot)#(1, feature_dim)
        frame_num = vertice.shape[1]
        hidden_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state
        if self.dataset == "BIWI":
            if hidden_states.shape[1]<frame_num*2:
                vertice = vertice[:, :hidden_states.shape[1]//2]
                frame_num = hidden_states.shape[1]//2
        hidden_states = self.audio_feature_map(hidden_states)

        if teacher_forcing:
            vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
            style_emb = vertice_emb  
            vertice_input = torch.cat((template,vertice[:,:-1]), 1) # shift one position
            vertice_input = vertice_input - template
            vertice_input = self.vertice_map(vertice_input)
            vertice_input = vertice_input + style_emb
            vertice_input = self.PPE(vertice_input)
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            vertice_out = self.vertice_map_r(vertice_out)
        else:
            for i in range(frame_num):
                if i==0:
                    vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
                    style_emb = vertice_emb
                    vertice_input = self.PPE(style_emb)
                else:
                    vertice_input = self.PPE(vertice_emb)
                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
                memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
                vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
                vertice_out = self.vertice_map_r(vertice_out)
                new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
                new_output = new_output + style_emb
                vertice_emb = torch.cat((vertice_emb, new_output), 1)

        vertice_out = vertice_out + template

        return vertice_out, vertice_input

    def forward_G(self, audio, template, vertice, criterion, one_hot, D, teacher_forcing=True):
        vertice_out, vertice_input = self.forward(audio, template, vertice, one_hot, teacher_forcing=teacher_forcing)

        recon_loss = criterion(vertice_out, vertice)  # (batch, seq_len, V*3)
        recon_loss = torch.mean(recon_loss)

        pred = D(torch.cat((vertice_out, vertice_input), dim=-1), one_hot)

        real_label = torch.ones_like(pred) - (0.05 * torch.randn_like(pred)).abs()
        fake_label = torch.zeros_like(pred) + (0.05 * torch.randn_like(pred)).abs()

        G_loss_GAN = ((pred - real_label)**2).mean()
        G_loss = (self.w_recon * recon_loss) + (self.w_GAN * G_loss_GAN)

        return G_loss, recon_loss, G_loss_GAN

    def forward_D(self, audio, template, vertice, criterion, one_hot, D, teacher_forcing=True):
        # TODO: Concat other info for other GAN types
        vertice_out, vertice_input = self.forward(audio, template, vertice, one_hot, teacher_forcing=teacher_forcing)

        D_input_real = torch.cat((vertice, vertice_input), dim=-1)
        D_input_fake = torch.cat((vertice_out, vertice_input), dim=-1)

        # We only care about the prediction of the last timestep
        D_real = D(D_input_real, one_hot)
        D_fake = D(D_input_fake.detach(), one_hot)

        real_label = torch.ones_like(D_real) - (0.05 * torch.randn_like(D_real)).abs()
        fake_label = torch.zeros_like(D_fake) + (0.05 * torch.randn_like(D_real)).abs()

        D_loss_real = ((D_real - real_label) ** 2).mean()
        D_loss_fake = ((D_fake - fake_label) ** 2).mean()
        D_loss = self.w_GAN * (D_loss_real + D_loss_fake)

        return D_loss, D_loss_real, D_loss_fake

    def predict(self, audio, template, one_hot):
        template = template.unsqueeze(1) # (1,1, V*3)
        obj_embedding = self.obj_vector(one_hot)
        hidden_states = self.audio_encoder(audio, self.dataset).last_hidden_state
        if self.dataset == "BIWI":
            frame_num = hidden_states.shape[1]//2
        elif self.dataset == "vocaset":
            frame_num = hidden_states.shape[1]
        hidden_states = self.audio_feature_map(hidden_states)

        for i in range(frame_num):
            if i==0:
                vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
                style_emb = vertice_emb
                vertice_input = self.PPE(style_emb)
            else:
                vertice_input = self.PPE(vertice_emb)

            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            vertice_out = self.vertice_map_r(vertice_out)
            new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
            new_output = new_output + style_emb
            vertice_emb = torch.cat((vertice_emb, new_output), 1)

        vertice_out = vertice_out + template
        return vertice_out
