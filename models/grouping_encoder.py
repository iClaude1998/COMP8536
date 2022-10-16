'''
Author: Yunxiang Liu u7191378@anu.edu.au
Date: 2022-10-15 16:18:37
LastEditors: Yunxiang Liu u7191378@anu.edu.au
LastEditTime: 2022-10-17 00:36:38
FilePath: \HoiTransformer\models\grouping_encoder.py
Description: 
'''
import math
import torch
from torch import nn 
from timm.models.layers import DropPath

def gelu(x):
  return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))


class Region_Proposal_Encoder(nn.Module):
    def __init__(self, num_queries, dim, heads, dim_head, mlp_dim, e_attn_dropout, e_dropout,
                 grouping_heads, d_grouping_head, mlp_ratio=(0.5, 4.0)):
        super().__init__()
        self.layers = nn.ModuleList(list(map(lambda q: group_encoder(q, dim, heads, dim_head, mlp_dim, e_attn_dropout, e_dropout,
                 grouping_heads, d_grouping_head, mlp_ratio), num_queries)))
        self.out_encoder = Transformer_Encoder(dim, heads, dim_head, mlp_dim, e_attn_dropout, e_dropout)
        
    def forward(self, x, mask):
        out = self.layers[0](x, mask)
        for layer in self.layers[1:]:
            out = layer(out)
        return self.out_encoder(out, None)
            
    
class group_encoder(nn.Module):
    """A Transformer encoder followed by a grouping block

    Args:
        nn (_type_): _description_
    """
    
    def __init__(self, num_queries, dim, heads, dim_head, mlp_dim, e_attn_dropout, e_dropout,
                 grouping_heads, d_grouping_head, mlp_ratio=(0.5, 4.0)):
        super().__init__()
        self.num_queries = num_queries
        self.queries_embedding = nn.Embedding(num_queries, dim)
        self.encoder = Transformer_Encoder(dim, heads, dim_head, mlp_dim, e_attn_dropout, e_dropout)
        self.group_block = GroupingBlock(dim, grouping_heads, d_grouping_head, self.num_queries, mlp_ratio)
    
    def forward(self, x, mask=None):
        batch, num, _ = x.size()
        if mask is not None:
            encoding_mask = self.generate_encoder_mask(mask)
            decoding_mask = self.generate_decoder_mask(mask)
        else:
            encoding_mask = None
            decoding_mask = None
        queries = self.queries_embedding.weight.expand(batch, -1, -1)
        x_queries = torch.cat([x, queries], dim=1)
        encoded_x_queries = self.encoder(x_queries, encoding_mask)
        # x, queries = encoded_x_queries[:, :num, :], encoded_x_queries[:, num:, :]
        return self.group_block(encoded_x_queries[:, :num, :], encoded_x_queries[:, num:, :], decoding_mask)
          
    def generate_encoder_mask(self, mask):
        batch = mask.size(0)
        new_mask = mask.new_ones((batch, self.num_queries))
        mask = torch.cat([mask, new_mask], dim=1)
        return (mask[:, :, None] * mask[:, None, :]).unsqueeze(1)
    
    def generate_decoder_mask(self, mask):
        batch = mask.size(0)
        q_mask = mask.new_ones((batch, self.num_queries, 1))
        new_mask = q_mask * mask.unsqueeze(1)
        return new_mask.unsqueeze(1)
    
class Transformer_Encoder(nn.Module):
    r"""
        A single multi-heads Transformer Encoder layer
        view the classical paper 'Attention is All You Need': https://arxiv.org/pdf/1706.03762.pdf for details
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, attn_dropout, dropout):
        super().__init__()
        self.att_layer = SelfAttention(dim, heads, dim_head, attn_dropout, dropout)
        self.feedforward = FeedForward(dim, mlp_dim, dropout)
        self.connections = nn.ModuleList([])
        for _ in range(2):
            connetion = SubLayerConnection(dim)
            self.connections.append(connetion)

    def forward(self, x, mask):
        x = self.connections[0](x, lambda x: self.att_layer(x, mask))
        x = self.connections[1](x, lambda x: self.feedforward(x))
        return x
        
        
class SubLayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size):
        super(SubLayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)


    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0):
        super().__init__()
        self.net0 = nn.Sequential(nn.Linear(dim, hidden_dim))

        self.net1 = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(hidden_dim, dim),
                                  nn.Dropout(dropout))

    def forward(self, x):
        x = self.net0(x)
        x = self.net1(gelu(x))
        return x



class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, attn_dropout=0., dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim),
                                    nn.Dropout(dropout))

    def forward(self, x, mask=None):
        batch, num, _ = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = q.reshape(batch, num, self.heads, self.dim_head).transpose(1, 2).reshape(batch * self.heads, num, self.dim_head)
        k = k.reshape(batch, num, self.heads, self.dim_head).transpose(1, 2).reshape(batch * self.heads, num, self.dim_head)
        v = v.reshape(batch, num, self.heads, self.dim_head).transpose(1, 2).reshape(batch * self.heads, num, self.dim_head)

        dots = torch.matmul(q, k.transpose(1, 2)) * self.scale
        dots = dots.reshape(batch, self.heads, num, num)

        if mask is not None:
            # mask = (mask[:, None, :] * mask[:, :, None]).unsqueeze(1)
            # dots = dots * mask
            dots = dots.masked_fill(mask == 0, -6.55e4)
        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn.reshape(batch * self.heads, num, num)
        out = torch.matmul(attn, v)
        out = out.reshape(batch, self.heads, num, self.dim_head).transpose(1, 2).reshape(batch, num, self.heads * self.dim_head)
        out = self.to_out(out)
        return out       
        
class GroupingBlock(nn.Module):
    """Grouping Block to group similar segments together.

    Args:
        dim (int): Dimension of the input.
        out_dim (int): Dimension of the output.
        num_heads (int): Number of heads in the grouping attention.
        num_output_group (int): Number of output groups.
        norm_layer (nn.Module): Normalization layer to use.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        hard (bool): Whether to use hard or soft assignment. Default: True
        gumbel (bool): Whether to use gumbel softmax. Default: True
        sum_assign (bool): Whether to sum assignment or average. Default: False
        assign_eps (float): Epsilon to avoid divide by zero. Default: 1
        gum_tau (float): Temperature for gumbel softmax. Default: 1
    """

    def __init__(self,
                 dim,
                 num_heads,
                 dim_head,
                 num_group_token,
                 mlp_ratio=(0.5, 4.0)):
        
        super(GroupingBlock, self).__init__()
        self.dim = dim
        # norm on group_tokens
        self.norm_tokens = nn.LayerNorm(dim)
        tokens_dim, channels_dim = [int(x * dim) for x in mlp_ratio]
        self.mlp_inter = FeedForward(num_group_token, tokens_dim, 0.)
        self.norm_post_tokens = nn.LayerNorm(dim)
        # norm on x
        self.norm_x = nn.LayerNorm(dim)
        self.pre_assign_attn = CrossAttnBlock(
            dim=dim, num_heads=num_heads, dim_head=dim_head, mlp_ratio=4, norm_layer=nn.LayerNorm, post_norm=True)

        # self.assign = AssignAttention(
        #     dim=dim,
        #     num_heads=1,
        #     qkv_bias=True)
        self.assign = CrossAttention(dim, dim, 1, dim)
        self.norm_new_x = nn.LayerNorm(dim)
        self.mlp_channels = FeedForward(dim, channels_dim)



    def project_group_token(self, group_tokens):
        """
        Args:
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]

        inter_weight (torch.Tensor): [B, S_2, S_1], S_2 is the new number of
            group tokens, it's already softmaxed along dim=-1

        Returns:
            projected_group_tokens (torch.Tensor): [B, S_2, C]
        """
        # [B, S_2, C] <- [B, S_1, C]
        projected_group_tokens = self.mlp_inter(group_tokens.transpose(1, 2)).transpose(1, 2)
        projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
        return projected_group_tokens

    def forward(self, x, group_tokens, mask=None):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]
            return_attn (bool): whether to return attention map

        Returns:
            new_x (torch.Tensor): [B, S_2, C], S_2 is the new number of
                group tokens
        """
        group_tokens = self.norm_tokens(group_tokens)
        x = self.norm_x(x)
        # [B, S_2, C]
        projected_group_tokens = self.project_group_token(group_tokens)
        projected_group_tokens = self.pre_assign_attn(projected_group_tokens, x, x, mask)
        new_x  = self.assign(projected_group_tokens, x, x, mask)
        new_x += projected_group_tokens

        new_x = new_x + self.mlp_channels(self.norm_new_x(new_x))

        return new_x   



    
class CrossAttnBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 dim_head,
                 mlp_ratio=4.,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 post_norm=False):
        super().__init__()
        if post_norm:
            self.norm_post = norm_layer(dim)
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        else:
            self.norm_q = norm_layer(dim)
            self.norm_k = norm_layer(dim)
            self.norm_post = nn.Identity()
        # self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn = CrossAttention(dim, dim, num_heads, dim_head, attn_dropout=attn_drop, dropout=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim=dim, hidden_dim=mlp_hidden_dim, dropout=drop)

    def forward(self, query, key, values, mask=None):
        x = query
        x = x + self.drop_path(self.attn(self.norm_q(query), self.norm_k(key), self.norm_k(values), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.norm_post(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, heads=8, dim_head=64, attn_dropout=0., dropout=0.):
        super().__init__()
        inner_dim = int(dim_head * heads)
        self.heads = heads
        self.dim_head = dim_head
        self.scale = self.dim_head ** -0.5

        self.to_q = nn.Linear(dim_q, inner_dim)
        self.to_k = nn.Linear(dim_kv, inner_dim)
        self.to_v = nn.Linear(dim_kv, inner_dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim_q),
                                    nn.Dropout(dropout))


    def forward(self, queries, keys, values, mask=None):
        batch, num_q, _ = queries.shape
        num_k, num_v = keys.shape[1], values.shape[1]
        assert num_k == num_v, 'mismatch between keys and values(the number of keys and values should be the same).'

        q = self.to_q(queries)
        k = self.to_k(keys)
        v = self.to_v(values)
        q = q.reshape(batch, num_q, self.heads, self.dim_head).transpose(1, 2).reshape(batch * self.heads, num_q,
                                                                                       self.dim_head)
        k = k.reshape(batch, num_k, self.heads, self.dim_head).transpose(1, 2).reshape(batch * self.heads, num_k,
                                                                                       self.dim_head)
        v = v.reshape(batch, num_k, self.heads, self.dim_head).transpose(1, 2).reshape(batch * self.heads, num_v,
                                                                                       self.dim_head)
        dots = torch.matmul(q, k.transpose(1, 2)) * self.scale
        dots = dots.reshape(batch, self.heads, num_q, num_k)

        if mask is not None:
            dots = dots.masked_fill(mask == 0, -6.55e4)
            # dots = dots * mask
        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn.reshape(batch * self.heads, num_q, num_k)
        out = torch.matmul(attn, v)
        out = out.reshape(batch, self.heads, num_q, self.dim_head).transpose(1, 2).reshape(batch, num_q,
                                                                                           self.heads * self.dim_head)
        out = self.to_out(out)
        return out
    
def build_RPE(args):
    num_queries = list(map(int, args.n_queries))
    mlp_ratio = list(map(float, args.mlp_ratio))
    return Region_Proposal_Encoder(num_queries, 
                                   args.hidden_dim, 
                                   args.e_num_heads, 
                                   args.e_dim_head, 
                                   args.e_mlp_dim, 
                                   args.e_attn_dropout, 
                                   args.e_dropout, 
                                   args.grouping_heads, 
                                   args.d_grouping_head, 
                                   mlp_ratio=mlp_ratio)