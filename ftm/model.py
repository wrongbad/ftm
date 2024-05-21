import torch
from dataclasses import dataclass, field
import numpy as np

@dataclass
class ModelArgs:
    n_tokens: int = 256
    n_ivecs: int = 0
    n_outputs: int = 256
    n_embed: int = 512
    n_mem_key: int = 64
    n_mem_val: int = 64
    n_mem_dim: int = 1
    n_mlp: int = 1024
    n_layers: int = 4
    decays: list = field(default_factory=lambda:[0.999, 0.99, 0.9, 0.8, 0.6])
    dropout: float = 0.02


class GatedProj(torch.nn.Module):
    def __init__(self, isize, osize):
        super().__init__()
        self.w = torch.nn.Linear(isize, osize, bias=False)
        self.w_gate = torch.nn.Linear(isize, osize, bias=True)

    def forward(self, x):
        gate = self.w_gate(x).sigmoid()
        return self.w(x) * gate


class FeedForward(torch.nn.Module):
    def __init__(self, isize, msize, osize):
        super().__init__()
        self.wp = GatedProj(isize, msize)
        self.wo = torch.nn.Linear(msize, osize, bias=False)
    
    def forward(self, x):
        return self.wo(self.wp(x))


class RMSNorm(torch.nn.Module):
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(n_embed))

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


# def smear(x, decay, prev_mem=None):

#     hop = 64
#     pad = hop - 1
#     if prev_mem is not None:
#         # y = torch.cat((torch.zeros(hop-1), x))
#         y = torch.cat((prev_mem[:, None, :], x), dim=1)
#         pad = hop - 2
#         # y = torch.cat((torch.zeros(hop-2), prev_mem[:, None, :], x))

#     y = torch.nn.functional.pad(x, [0,0,pad,0])

#     y = y.transpose(-2, -1) # -> Batch Feat Time

#     k0 = (decay ** torch.arange(hop)).flip(0)[None, None, :]

#     y = torch.nn.functional.conv1d(y, k0, groups=y.shape[-2])

#     decay = decay ** hop
#     while hop < y.shape[-1]:
#         y = torch.cat((y[...,:hop], y[...,hop:] + y[...,:-hop] * decay))
#         decay *= decay
#         hop *= 2

#     return y.transpose(-2, -1) # -> Batch Time Feat


def smear(x, decay, prev_mem=None):
    x0 = x[:,:1]
    if prev_mem is not None:
        x0 = x0 + prev_mem * decay
    y = torch.cat((
        x0, 
        x[:,1:] + x[:,:-1] * decay
    ), dim=1)
    hop = 2
    while hop < y.shape[1]:
        y = torch.cat((
            y[:,:hop], 
            y[:,hop:] + y[:,:-hop] * decay
        ), dim=1)
        decay *= decay
        hop *= 2
    return y



class Decay(torch.nn.Module):
    def __init__(self, config, decay):
        super().__init__()
        self.decay = decay
        # self.w_val = torch.nn.Linear(config.n_embed, config.n_mem_val)
        self.w_val = GatedProj(config.n_embed, config.n_mem_val)
        self.w_que = torch.nn.Linear(config.n_embed, config.n_mem_val)
        # self.w_load = torch.nn.Linear(msize, osize)
        self.w_out = torch.nn.Linear(config.n_mem_val, config.n_embed, bias=False)
        self.prev_mem = None

    def forward(self, x):
        scale = np.sqrt(1 - self.decay)
        store = self.w_val(x) * scale
        mem = smear(store, self.decay, self.prev_mem)
        self.prev_mem = mem[:, -1:].detach().clone()
        que = self.w_que(x).sigmoid()
        load = mem * que * scale
        # gate = self.w_gate(x)
        # return load * gate.sigmoid()
        return self.w_out(load)


class Decay2D(torch.nn.Module):
    def __init__(self, config, decay):
        super().__init__()
        self.decay = decay
        # self.w_store = GatedProj(isize, msize)
        self.w_key = torch.nn.Linear(config.n_embed, config.n_mem_key)
        self.w_val = torch.nn.Linear(config.n_embed, config.n_mem_val)
        # self.w_store = torch.nn.Linear(isize, msize)
        self.w_que = torch.nn.Linear(config.n_embed, config.n_mem_key)
        # self.w_load = torch.nn.Linear(msize, osize)
        self.w_out = torch.nn.Linear(config.n_mem_val, config.n_embed, bias=False)
        self.prev_mem = None

    def forward(self, x):
        scale = np.sqrt(1 - self.decay)

        # store = self.w_store(x) * scale
        # key = self.w_key(x)[:, :, :, None].softmax(dim=-2)
        
        v = self.w_val(x)[:, :, None, :]
        k = self.w_key(x)[:, :, :, None].sigmoid()
        q = self.w_que(x)[:, :, None, :].sigmoid()

        store = k * v * scale
        mem = smear(store, self.decay, self.prev_mem)
        
        # que = self.w_que(x)[:, :, None, :].softmax(dim=-1)
        load = q @ mem * scale
        # load = self.w_load(mem) * scale
        # gate = self.w_gate(x)
        # return load * gate.sigmoid()
        return self.w_out(load[:,:,0])


class Decay2DBlk(torch.nn.Module):
    def __init__(self, config, decay):
        super().__init__()
        self.decay = decay
        self.w_val = torch.nn.Linear(config.n_embed, config.n_mem_val, bias=False)
        self.w_key = torch.nn.Linear(config.n_embed, config.n_mem_key)
        self.w_que = torch.nn.Linear(config.n_embed, config.n_mem_key)
        # self.w_gate = torch.nn.Linear(config.n_embed, config.n_mem_val)
        self.w_out = torch.nn.Linear(config.n_mem_val, config.n_embed, bias=False)
        self.prev_mem = None

        self.block_size = 128
        
        i = torch.arange(self.block_size)[:, None]
        j = torch.arange(self.block_size)[None, :]
        decays = (torch.tensor(decay) ** (i - j)).tril_(0)
        self.register_buffer('decays', decays)
        
        # block_decay = decay ** self.block_size
        # block_decays = (decay * torch.tensor(block_decay) ** (i - j - 1)).tril_(-1)
        # self.register_buffer('block_decays', block_decays)

        block_decay = decay ** self.block_size
        block_decays = (decay * torch.tensor(block_decay) ** (i - j)).tril_(0)
        self.register_buffer('block_decays', block_decays)


    def forward(self, x):
        B, T, _ = x.shape
        T_blk = self.block_size
        N_blk = T // T_blk

        v = self.w_val(x) * (1 - self.decay)
        k = self.w_key(x).sigmoid()
        q = self.w_que(x).sigmoid()
        D_v, D_k = v.shape[-1], k.shape[-1]

        v = v.view(B, N_blk, T_blk, D_v)
        k = k.view(B, N_blk, T_blk, D_k).transpose(-1, -2)
        q = q.view(B, N_blk, T_blk, D_k)

        a = q @ k # B, N_blk, T(dst), T(src)
        a = a * self.decays[:T_blk, :T_blk]
        y = a @ v # B, N_blk, T(dst), D_v

        kv = k @ (v * self.decays[T_blk-1, :T_blk, None]) # B, N_blk, D_k, D_v
        
        kv = kv.permute(0,2,3,1).unsqueeze(-1) # B, D_k, D_v, N(src), 1
        
        if self.prev_mem is None:
            self.prev_mem = torch.zeros_like(kv[:,:,:,-1:])

        kv2 = torch.cat((self.prev_mem, kv[:,:,:,:-1]), dim=3)
        self.prev_mem = kv[:,:,:,-1:].detach().clone()

        kv = self.block_decays[:N_blk, :N_blk] @ kv2 # B, D_k, D_v, N(dst), 1
        kv = kv.squeeze(-1).permute(0,3,1,2) # B, N_blk, D_k, D_v

        y2 = q @ kv # B, N_blk, T(dst), D_v
        y2 = y2 * self.decays[:, :1]
        y = y + y2
        y = y.view(B, T, D_v)

        # y = y * self.w_gate(x).sigmoid()
        return self.w_out(y)

        

class ResidBranch(torch.nn.Module):
    def __init__(self, n_embed, dropout, block):
        super().__init__()
        self.norm = RMSNorm(n_embed)
        self.block = block
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.norm(x)
        x = self.block(x)
        return self.dropout(x)



class MultiDecay(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        blocks = []
        decay_cls = Decay if config.n_mem_dim == 1 else Decay2DBlk
        # for d in config.decays:
        #     blocks += [
        #         ResidBranch(
        #             config.n_embed, 
        #             config.dropout,
        #             Decay(
        #                 config.n_embed, 
        #                 config.n_mem, 
        #                 config.n_embed,
        #                 d
        #             )
        #         ),
        #         ResidBranch(
        #             config.n_embed, 
        #             config.dropout,
        #             FeedForward(
        #                 config.n_embed, 
        #                 config.n_mlp, 
        #                 config.n_embed,
        #             )
        #         )
        #     ]
        blocks = [
            ResidBranch(
                config.n_embed, 
                config.dropout,
                decay_cls(config, d)
            ) for d in config.decays
        ]
        blocks += [
            ResidBranch(
                config.n_embed, 
                config.dropout,
                FeedForward(
                    config.n_embed, 
                    config.n_mlp, 
                    config.n_embed,
                )
            )
        ]
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x

class Decformer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.n_tokens:
            self.w_itok = torch.nn.Embedding(config.n_tokens, config.n_embed)
        if config.n_ivecs:
            self.w_ivec = torch.nn.Linear(config.n_ivecs, config.n_embed)
        self.blocks = torch.nn.ModuleList([
            MultiDecay(config)
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.n_embed)
        self.wo = torch.nn.Linear(config.n_embed, config.n_outputs, bias=True)

    def forward(self, tokens=None, ivecs=None):
        x = None
        if tokens is not None:
            x = self.w_itok(tokens)
        if ivecs is not None:
            y = self.w_ivec(ivecs)
            x = y if x is None else x + y
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.wo(x)
    
    def reset_states(self):
        for b0 in self.blocks:
            for b1 in b0.blocks:
                b2 = b1.block
                if hasattr(b2, "prev_mem"):
                    b2.prev_mem = None
