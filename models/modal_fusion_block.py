'''
Author: Yunxiang Liu u7191378@anu.edu.au
Date: 2022-10-10 22:29:55
LastEditors: Yunxiang Liu u7191378@anu.edu.au
LastEditTime: 2022-10-16 20:17:30
FilePath: \HoiTransformer\models\modal_fusion_block
Description: Grouping_block
'''
import torch
import numpy as np
from torch import nn 


class Word_fusion_block(nn.Module):
    
    def __init__(self, queries_dim, emb_dim, world_embedding_path, gumbel=True, tau=1., num_heads=8, device='cuda', drop_out=.1):
        super().__init__()
        self.emb_dim = emb_dim
        glove_word_embedding = torch.from_numpy(np.load(world_embedding_path)).to(device)
        self.register_buffer('glove_word_embedding', glove_word_embedding)
        self.word_dim = self.glove_word_embedding.size(1)
        self.atten = word_attention(queries_dim, self.emb_dim, self.glove_word_embedding, gumbel, tau, num_heads, drop_out)
        self.connection = SubLayerConnection(queries_dim)
        self.glove_word_dim = self.glove_word_embedding.size(1)
        
    def forward(self, queries):
        return self.connection(queries, lambda queries: self.atten(queries))
    
    

class word_attention(nn.Module):
    
    def __init__(self, queries_dim, emb_dim, word_embedding, gumbel=True, tau=1., num_heads=8, drop_out=.1):
        super().__init__()
        assert emb_dim % num_heads == 0, 'The embedded dimension should be divisible by number of heads'
        self.emb_dim = emb_dim
        self.glove_word_embedding = word_embedding
        self.num_words, self.word_dim = self.glove_word_embedding.size()
        self.num_heads = num_heads
        self.dim_head = emb_dim // self.num_heads
        self.scale = self.dim_head ** -0.5
        self.query_dim = queries_dim
        self.to_q = nn.Linear(self.query_dim, emb_dim, bias=False)
        self.to_k = nn.Linear(self.word_dim, emb_dim, bias=False)
        self.to_v = nn.Linear(self.word_dim, emb_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(emb_dim, self.query_dim), nn.Dropout(drop_out))
        self.gumbel = gumbel
        self.tau = tau
    
    def forward(self, queries):
        b, num_queries, _ = queries.size()
        glove_word_embedding =  self.glove_word_embedding.expand(b, -1, -1)
        q = self.to_q(queries)
        k = self.to_k(glove_word_embedding)
        v = self.to_v(glove_word_embedding)
        
        q = q.reshape(b, num_queries, self.num_heads, self.dim_head).transpose(1, 2).reshape(b*self.num_heads, num_queries, self.dim_head)
        k = k.reshape(b, self.num_words, self.num_heads, self.dim_head).transpose(1, 2).reshape(b*self.num_heads, self.num_words, self.dim_head)
        v = v.reshape(b, self.num_words, self.num_heads, self.dim_head).transpose(1, 2)
        
        dots = torch.matmul(q, k.transpose(1, 2)) * self.scale
        dots = dots.reshape(b, self.num_heads, num_queries, self.num_words)
        if self.gumbel:
            attn = gumbel_softmax(dots, hard=True)
        else:
            attn = dots.softmax(-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, num_queries, -1)
        return self.to_out(out)
            
        
class SubLayerConnection(nn.Module):

    def __init__(self, size):
        super(SubLayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)


    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))

    
def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False) -> torch.Tensor:
    # _gumbels = (-torch.empty_like(
    #     logits,
    #     memory_format=torch.legacy_contiguous_format).exponential_().log()
    #             )  # ~Gumbel(0,1)
    # more stable https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(-1)

    if hard:
        # Straight through.
        index = y_soft.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

def build_fusion_block(args):
    return Word_fusion_block(queries_dim=args.hidden_dim, 
                             emb_dim=args.fuse_dim, 
                             world_embedding_path=args.word_representation_path, 
                             gumbel=args.gumbel, 
                             tau=args.tau, 
                             num_heads=args.fusion_heads, 
                             device=args.device, 
                             drop_out=args.fusion_drop_out)