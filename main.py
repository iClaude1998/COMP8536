# ------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License")
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import argparse
import datetime
import getpass
import json
import random
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from datasets import build_dataset
from engine import train_one_epoch
from models import build_model


def create_log_dir(checkpoint='checkpoint', log_path='/data/LOG/train_log'):
    base_dir = os.path.join(log_path, getpass.getuser())
    exp_name = os.path.basename(os.path.abspath('.'))
    log_dir = os.path.join(base_dir, exp_name)
    print(log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(checkpoint):
        cmd = "ln -s {} {}".format(log_dir, checkpoint)
        os.system(cmd)


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

    # Backbone.
    parser.add_argument('--backbone', choices=['resnet50', 'resnet101', 'swin', 'resnet50-hico'], required=True,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'sine-2d'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--freeze_backbone', action='store_true',
                        help="freeze the backbones")                   

    # GC_Block
    parser.add_argument('--have_GC_block', action='store_true')
    parser.add_argument('--ratio', default=1, type=int,
                        help="The ratio of GC block")
    parser.add_argument('--headers', default=1, type=int,
                        help="Number of heads in GC block")
    parser.add_argument('--pooling_type', default='att', type=str,
                        help="Pooling type of GC_block")
    parser.add_argument('--atten_scale', default=False, type=bool,
                        help="whether scaling the attention score")
    parser.add_argument('--fusion_type', default='channel_add', type=str,
                        help="the way of fusing attention scores on feature maps")
    
    # Modal Fusion Blocks
    parser.add_argument('--have_fusion_block', action='store_true')
    parser.add_argument('--word_representation_path', default='./HOI_verb_GloveEmbbeding/HOI_Verb_wordVectors.npy')
    parser.add_argument('--fuse_dim', default=512, type=int,
                        help="The fused dimension for attentional fusion")
    parser.add_argument('--gumbel', default=False, type=bool,
                        help="Whether implement gumbel attention")
    parser.add_argument('--tau', default=1, type=float,
                        help="the tau of gumbel attention, activate when --gumbel is true")
    parser.add_argument('--fusion_heads', default=8, type=int,
                        help="The number of heads for model fusion")
    parser.add_argument('--fusion_drop_out', default=0.1, type=float,
                        help="the drop out score for attention fusion")
    
    # RPE
    parser.add_argument('--have_RPE', action='store_true')
    parser.add_argument('--n_queries', type=list, default=[256, 100], 
                        help="The number of queries for each RPE layer")
    parser.add_argument('--mlp_ratio', type=list, default=[0.5, 4.],
                        help="The hidden dimension ratio on two feedforward layers in grouping blocks")
    parser.add_argument('--e_num_heads', default=6, type=int,
                        help="The number of heads in transformer encoder")
    parser.add_argument('--e_dim_head', default=48, type=int,
                        help="The dimension for each head in transformer encoder")
    parser.add_argument('--e_mlp_dim', default=2048, type=int,
                        help="The hidden dimension in transformer encoder")
    parser.add_argument('--e_attn_dropout', default=0., type=float,
                        help="the drop out score for transformer encoder")
    parser.add_argument('--e_dropout', default=0., type=float,
                        help="the drop out score for transformer encoder")
    parser.add_argument('--grouping_heads', default=6, type=int,
                        help="number of heads in grouping layer")
    parser.add_argument('--d_grouping_head', default=48, type=int,
                        help="The dimension for each head in grouping layer")

    

    # Transformer.
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss.
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # Matcher.
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_word', default=0.6, type=float,
                        help="representation similarity coefficient in the matching cost, activate auto when implement word fusion block")

    # Loss coefficients.
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--loss_language_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.02, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--lang_T', default=.5, type=float,
                        help="The temperature of contrast loss for language fusion loss, activate auto when implement word fusion block")

    # Dataset parameters.
    parser.add_argument('--dataset_file', choices=['hico', 'vcoco', 'hoia'], required=True)

    # Modify to your log path ******************************* !!!
    exp_time = datetime.datetime.now().strftime('%Y%m%d%H%M')
    # create_log_dir(checkpoint='checkpoint', log_path='/home')
    work_dir = 'checkpoint/p_{}'.format(exp_time)

    parser.add_argument('--output_dir', default=work_dir,
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)

    # Distributed training parameters.
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--swin_model', default='base_cascade',
                        choices=['base_cascade', 'tiny_cascade', 'tiny_maskrcnn', 'small_cascade', 'small_maskrcnn'])
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)

    # Fix the seed for reproducibility.
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    if args.freeze_backbone:
        for p in model_without_ddp.backbone.parameters():
            p.requires_grad = False
    

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    if criterion.language_temperature is not None:
        optimizer.add_param_group({'params': criterion.language_temperature, 'lr': args.lr, 'name': 'Temperature'})
        
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    

    dataset_train = build_dataset(image_set='train', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # Load from pretrained DETR model.
    assert args.num_queries == 100, args.num_queries
    assert args.enc_layers == 6 and args.dec_layers == 6
    assert args.backbone in ['resnet50', 'resnet101', 'swin', 'resnet50-hico'], args.backbone
    if args.backbone == 'resnet50':
        pretrain_model = './data/detr_coco/detr-r50-e632da11.pth'
    elif args.backbone == 'resnet101':
        pretrain_model = './data/detr_coco/detr-r101-2c7b67e5.pth'
    elif args.backbone == 'resnet50-hico':
        pretrain_model = './data/detr_hicodet/res50_hico_1cf00bb.pth'
    else:
        pretrain_model = None
    if pretrain_model is not None:
        pretrain_dict = torch.load(pretrain_model, map_location='cpu')['model']
        if args.backbone != 'resnet50-hico':
            resume(model_without_ddp, pretrain_dict)
        else:
            resume(model_without_ddp, pretrain_dict, backbone_only=True)
    
        # my_model_dict = model_without_ddp.state_dict()
        # pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in my_model_dict}
        # my_model_dict.update(pretrain_dict)
        # model_without_ddp.load_state_dict(my_model_dict)

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        # model_without_ddp.load_state_dict(checkpoint['model'])
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    
    if args.freeze_backbone:
        for p in model_without_ddp.backbone.parameters():
            p.requires_grad = False

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 10 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            if (epoch + 1) > args.lr_drop and (epoch + 1) % 10 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def resume(model, checkpoint, backbone_only=False):

    instances_model = set(map(lambda x: x.split('.', 1)[0], model.state_dict().keys()))
    instances_weights = set(map(lambda x: x.split('.', 1)[0], checkpoint.keys()))
    if not backbone_only:
        instances_intersec = instances_model & instances_weights
    else:
        instances_intersec = set(['backbone'])
        print('we load the backbone only')
        for ins in instances_intersec:
            new_dict = {k.split('.', 1)[1]: checkpoint[k] for k in checkpoint.keys() if k.startswith(ins)}
            getattr(model, ins).load_state_dict(new_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('HOI Transformer training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
