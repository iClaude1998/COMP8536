# ------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License")
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .hoi_matcher import build_matcher as build_hoi_matcher
from .transformer import build_transformer, build_Transformer_encoders, build_Transformer_decoders
from .GC_block import build_GC_block
from .modal_fusion_block import build_fusion_block
from .grouping_encoder import build_RPE

num_humans = 2


class HoiTR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, T_encoders, T_decoders, GC_block ,word_fusion_block, RPE, num_classes, num_actions, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.T_encoders = T_encoders
        self.T_decoders = T_decoders
        # self.transformer = transformer
        hidden_dim = T_decoders.d_model

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.word_fusion_block = word_fusion_block
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.GC_block = GC_block
        self.RPE = RPE
        self.aux_loss = aux_loss

        self.human_cls_embed = nn.Linear(hidden_dim, num_humans + 1)
        self.human_box_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.object_cls_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.object_box_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.action_cls_embed = nn.Linear(hidden_dim, num_actions + 1)
        if self.word_fusion_block is not None: # hidden_dim, self.word_fusion_block.glove_word_dim
            self.action_word_embed = Fusion_head(hidden_dim, self.word_fusion_block.emb_dim, self.word_fusion_block.word_dim)
        

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None

        src = self.input_proj(src)
        bs = src.size(0)
        if self.GC_block is not None:
            src = self.GC_block(src, mask)
        
        pos_embed = pos[-1].flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.T_encoders(src, mask, pos_embed)

        if self.RPE is not None:
            queries = self.RPE(memory.transpose(0, 1), mask)
            # queries = queries.transpose(0, 1)
        else:
            queries = self.query_embed.weight
            queries = queries.unsqueeze(0).repeat(bs, 1, 1)
            
        if self.word_fusion_block is not None:
            queries = self.word_fusion_block(queries)
        hs = self.T_decoders(memory, queries.transpose(0, 1), mask, pos_embed)
        # print(hs.size())
        # raise NotImplementedError

        # hs = self.transformer(src, mask, queries, pos[-1])[0]

        human_outputs_class = self.human_cls_embed(hs)
        human_outputs_coord = self.human_box_embed(hs).sigmoid()
        object_outputs_class = self.object_cls_embed(hs)
        object_outputs_coord = self.object_box_embed(hs).sigmoid()
        action_outputs_class = self.action_cls_embed(hs)

        out = {
            'human_pred_logits': human_outputs_class[-1],
            'human_pred_boxes': human_outputs_coord[-1],
            'object_pred_logits': object_outputs_class[-1],
            'object_pred_boxes': object_outputs_coord[-1],
            'action_pred_logits': action_outputs_class[-1],
        }
        
        if self.word_fusion_block is not None:
            action_represent = self.action_word_embed(hs)
            out['action_pred_embedding'] = action_represent[-1]
        else:
            action_represent = None
            
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                human_outputs_class,
                human_outputs_coord,
                object_outputs_class,
                object_outputs_coord,
                action_outputs_class,
                action_represent,
            )

        return out

    @torch.jit.unused
    def _set_aux_loss(self,
                      human_outputs_class,
                      human_outputs_coord,
                      object_outputs_class,
                      object_outputs_coord,
                      action_outputs_class,
                      action_outputs_pre
                      ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if action_outputs_pre is None:
            return [{
                'human_pred_logits': a,
                'human_pred_boxes': b,
                'object_pred_logits': c,
                'object_pred_boxes': d,
                'action_pred_logits': e,
            } for
                a,
                b,
                c,
                d,
                e,
                in zip(
                human_outputs_class[:-1],
                human_outputs_coord[:-1],
                object_outputs_class[:-1],
                object_outputs_coord[:-1],
                action_outputs_class[:-1],
            )]
        else:
            return [{
                'human_pred_logits': a,
                'human_pred_boxes': b,
                'object_pred_logits': c,
                'object_pred_boxes': d,
                'action_pred_logits': e,
                'action_pred_embedding': f
            } for
                a,
                b,
                c,
                d,
                e,
                f
                in zip(
                human_outputs_class[:-1],
                human_outputs_coord[:-1],
                object_outputs_class[:-1],
                object_outputs_coord[:-1],
                action_outputs_class[:-1],
                action_outputs_pre[:-1]
            )]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, num_actions, matcher, weight_dict, eos_coef, 
                 losses, language_temperature=0.5, device='cuda', word_embedding_path=None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes  # 91
        self.num_actions = num_actions
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        if 'language' in self.losses:
            self.language_temperature = torch.nn.Parameter(torch.tensor(language_temperature, device=device), requires_grad=True)
        else:
            self.language_temperature = None
        
        if word_embedding_path is not None:
            glove_word_embedding = torch.from_numpy(np.load(word_embedding_path)).to(device)
            self.register_buffer('glove_word_embedding', glove_word_embedding)
            self.num_words = self.glove_word_embedding.size(0)


        human_empty_weight = torch.ones(num_humans + 1)
        human_empty_weight[-1] = self.eos_coef
        self.register_buffer('human_empty_weight', human_empty_weight)

        object_empty_weight = torch.ones(num_classes + 1)
        object_empty_weight[-1] = self.eos_coef
        self.register_buffer('object_empty_weight', object_empty_weight)

        action_empty_weight = torch.ones(num_actions + 1)
        action_empty_weight[-1] = self.eos_coef
        self.register_buffer('action_empty_weight', action_empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'human_pred_logits' in outputs
        assert 'object_pred_logits' in outputs
        assert 'action_pred_logits' in outputs
        human_src_logits = outputs['human_pred_logits']
        object_src_logits = outputs['object_pred_logits']
        action_src_logits = outputs['action_pred_logits']

        idx = self._get_src_permutation_idx(indices)



        human_target_classes_o = torch.cat([t["human_labels"][J] for t, (_, J) in zip(targets, indices)])
        object_target_classes_o = torch.cat([t["object_labels"][J] for t, (_, J) in zip(targets, indices)])
        action_target_classes_o = torch.cat([t["action_labels"][J] for t, (_, J) in zip(targets, indices)])



        human_target_classes = torch.full(human_src_logits.shape[:2], num_humans,
                                          dtype=torch.int64, device=human_src_logits.device)
        human_target_classes[idx] = human_target_classes_o

        object_target_classes = torch.full(object_src_logits.shape[:2], self.num_classes,
                                           dtype=torch.int64, device=object_src_logits.device)
        object_target_classes[idx] = object_target_classes_o

        action_target_classes = torch.full(action_src_logits.shape[:2], self.num_actions,
                                           dtype=torch.int64, device=action_src_logits.device)
        action_target_classes[idx] = action_target_classes_o

        human_loss_ce = F.cross_entropy(human_src_logits.transpose(1, 2),
                                        human_target_classes, self.human_empty_weight)
        object_loss_ce = F.cross_entropy(object_src_logits.transpose(1, 2),
                                         object_target_classes, self.object_empty_weight)
        action_loss_ce = F.cross_entropy(action_src_logits.transpose(1, 2),
                                         action_target_classes, self.action_empty_weight)
        loss_ce = human_loss_ce + object_loss_ce + 2 * action_loss_ce
        losses = {
            'loss_ce': loss_ce,
            'human_loss_ce': human_loss_ce,
            'object_loss_ce': object_loss_ce,
            'action_loss_ce': action_loss_ce
        }

        if log:
            losses['class_error'] = 100 - accuracy(action_src_logits[idx], action_target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['action_pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["action_labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'human_pred_boxes' in outputs
        assert 'object_pred_boxes' in outputs

        idx = self._get_src_permutation_idx(indices)

        human_src_boxes = outputs['human_pred_boxes'][idx]
        human_target_boxes = torch.cat([t['human_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        object_src_boxes = outputs['object_pred_boxes'][idx]
        object_target_boxes = torch.cat([t['object_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        human_loss_bbox = F.l1_loss(human_src_boxes, human_target_boxes, reduction='none')
        object_loss_bbox = F.l1_loss(object_src_boxes, object_target_boxes, reduction='none')

        losses = dict()
        losses['human_loss_bbox'] = human_loss_bbox.sum() / num_boxes
        losses['object_loss_bbox'] = object_loss_bbox.sum() / num_boxes
        losses['loss_bbox'] = losses['human_loss_bbox'] + losses['object_loss_bbox']


        human_loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(human_src_boxes),
            box_ops.box_cxcywh_to_xyxy(human_target_boxes)))
        object_loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(object_src_boxes),
            box_ops.box_cxcywh_to_xyxy(object_target_boxes)))
        losses['human_loss_giou'] = human_loss_giou.sum() / num_boxes
        losses['object_loss_giou'] = object_loss_giou.sum() / num_boxes

        losses['loss_giou'] = losses['human_loss_giou'] + losses['object_loss_giou']
        return losses
    
    def loss_language(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        pred_embedding = outputs['action_pred_embedding']
        matched_embedding = pred_embedding[idx]
        action_embedding_o = torch.cat([t["interaction_representations"][J] for t, (_, J) in zip(targets, indices)])
        action_target_classes_o = torch.cat([t["action_labels"][J] for t, (_, J) in zip(targets, indices)]) - 1
        
        loss_queries2words = self.loss_language_queires2words(pred_embedding, action_embedding_o, indices)
        loss_words2instance = self.loss_language_words2instance(matched_embedding, action_target_classes_o)
        loss_one2one = self.loss_one2one_match(matched_embedding, action_embedding_o)
        loss_language = loss_queries2words + loss_words2instance + loss_one2one
        return {'loss_queries2words': loss_queries2words, 
                'loss_words2instance': loss_words2instance, 
                'loss_language': loss_language,
                'loss_one2one': loss_one2one}
    
    def loss_language_queires2words(self, pred_embedding, action_embedding_o, indices):
        b, num_queries, _ = pred_embedding.size()
        pred_embedding = pred_embedding.view(b*num_queries, 1, -1)
        queries_idcs = torch.cat([idx[0] + (i * num_queries) for i, idx in enumerate(indices)])
        similarity_scores = F.cosine_similarity(pred_embedding, action_embedding_o.unsqueeze(0), dim=-1)

        positives = torch.exp(similarity_scores[(queries_idcs, torch.arange(action_embedding_o.size(0)))] / self.language_temperature)
        negatives = torch.exp(similarity_scores / self.language_temperature)

        return torch.sum(-torch.log(positives / negatives.sum(dim=0))) / (num_queries * b)
        # return torch.mean(-torch.log(positives / negatives.sum(dim=0)))
    
    def loss_language_words2instance(self, matched_embedding, action_target_classes_o):
        matched_embedding = matched_embedding.unsqueeze(1)
        num_matched = matched_embedding.size(0)
        
        similarity_scores = F.cosine_similarity(matched_embedding, self.glove_word_embedding.unsqueeze(0), dim=-1)
        positives = torch.exp(similarity_scores[(torch.arange(num_matched), action_target_classes_o)] / self.language_temperature)
        negatives = torch.exp(similarity_scores / self.language_temperature)
        # return torch.sum(-torch.log(positives / negatives.sum(dim=1))) / self.num_words
        return torch.mean(-torch.log(positives / negatives.sum(dim=1)))
    
    def loss_one2one_match(self, matched_embedding, action_embedding_o):
        similarity_scores = F.cosine_similarity(matched_embedding.unsqueeze(1), action_embedding_o.unsqueeze(0), dim=-1)
        positives = torch.exp(torch.diag(similarity_scores) / self.language_temperature)
        negatives = torch.exp(similarity_scores / self.language_temperature)
        
        return torch.mean(-torch.log(positives / negatives.sum(dim=1))) + torch.mean(-torch.log(positives / negatives.sum(dim=0)))
        
        

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'language': self.loss_language
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["human_labels"]) for t in targets)

        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
class Fusion_head(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                   nn.ReLU(True),
                                   nn.LayerNorm(hidden_dim),
                                   nn.Linear(hidden_dim, output_dim))
    def forward(self, x):
        return self.layer(x)
        


def build(args):
    assert args.dataset_file in ['hico', 'vcoco', 'hoia'], args.dataset_file
    if args.dataset_file in ['hico']:
        num_classes = 91
        num_actions = 118
    elif args.dataset_file in ['vcoco']:
        num_classes = 91
        num_actions = 30
    else:
        num_classes = 12
        num_actions = 11

    device = torch.device(args.device)

    if args.backbone == 'swin':
        from .backbone_swin import build_backbone_swin
        backbone = build_backbone_swin(args)
    else:
        backbone = build_backbone(args)

    # transformer = build_transformer(args)
    transformer_encoders = build_Transformer_encoders(args)
    transformer_decoders = build_Transformer_decoders(args)
    
    if args.have_GC_block:
        in_channels = transformer_encoders.d_model
        gc_block = build_GC_block(in_channels, args)
    else:
        gc_block = None
        
    if args.have_fusion_block:
        word_fusion_block = build_fusion_block(args)
    else:
        word_fusion_block=None
        
    if args.have_RPE:
        RPE = build_RPE(args)
    else:
        RPE = None
    model = HoiTR(
        backbone,
        transformer_encoders,
        transformer_decoders,
        gc_block,
        word_fusion_block,
        RPE,
        num_classes=num_classes,
        num_actions=num_actions,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss
    )

    matcher = build_hoi_matcher(args)

    weight_dict = dict(loss_ce=1, loss_bbox=args.bbox_loss_coef, loss_giou=args.giou_loss_coef)
    if word_fusion_block is not None:
        weight_dict['loss_language'] = args.loss_language_coef

    
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if word_fusion_block is not None:
        losses.append('language')

    criterion = SetCriterion(num_classes=num_classes, num_actions=num_actions, matcher=matcher,
                             weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses, 
                             language_temperature=args.lang_T, device=device, word_embedding_path=args.word_representation_path)
    criterion.to(device)

    return model, criterion
