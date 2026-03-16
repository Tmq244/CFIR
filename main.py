import os
import sys
import time
import random
import argparse
import json
from datetime import datetime
from pprint import pprint
import torchvision
import torch
from irbench.irbench import IRBench
from irbench.evals.eval_helper import EvalHelper
from preprocess.dataset_tag import FashionIQTrainValDataset, FashionIQTestDataset
from Model.text_only import TextOnlyModel
from Model.TIRG import TIRG
from Model.image_only import ImageOnlyModel
# from Model.match import MatchTextOnly
from Model.match import MatchTIRG
from tensorboardX import SummaryWriter
from preprocess.runner import Trainer,Evaluator
from Model.cross_attention import Combine


def normalize_args(args):
    if isinstance(args.lr_decay_steps, str):
        args.lr_decay_steps = [int(x) for x in args.lr_decay_steps.strip().split(',') if x]
    return args


def _safe_name(value):
    return ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '-' for ch in str(value))


def create_results_state(args):
    started_at = datetime.now()
    results_dir = os.path.join('.', 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    filename = '{}_{}_{}_{}_{}_s{}_b{}_e{}_lr{}_gpu{}_{}.json'.format(
        started_at.strftime('%Y%m%d_%H%M%S'),
        _safe_name(args.method),
        _safe_name(args.target),
        _safe_name(args.backbone),
        _safe_name(args.text_method),
        args.stack_num,
        args.batch_size,
        args.epochs,
        _safe_name(args.lr),
        _safe_name(args.gpu_id.replace(',', '-')),
        _safe_name(args.expr_name),
    )
    args.results_file = os.path.join(results_dir, filename)
    args_dict = {k: v for k, v in vars(args).items()}
    state = {
        'started_at': started_at.isoformat(),
        'results_file': args.results_file,
        'args': args_dict,
        'epochs': [],
    }
    write_results_state(state)
    return state


def write_results_state(state):
    with open(state['results_file'], 'w') as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write('\n')


def init_env(args):
    state = {k:v for k,v in args._get_kwargs()}
    pprint(state)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    if args.manualSeed is None:
        args.manualSeed = random.randint(1,10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manualSeed)
        torch.backends.cudnn.benchmark = True


def main():
    args = normalize_args(parser.parse_args())
    results_state = create_results_state(args)
    init_env(args)
    print('Results file: {}'.format(args.results_file))
    train_dataset = FashionIQTrainValDataset(
        data_root = args.data_root,
        image_root=args.image_root,
        image_size = args.image_size,
        split = 'train',
        target = args.target,
        max_turn_len=args.max_turn_len
    )
    train_loader = train_dataset.get_loader(batch_size=args.batch_size,)
    if (args.target=='all') or (args.target==None):
        targets = ['dress','toptee','shirt']
    else:
        targets = [args.target]

    test_loader = dict()
    for target in targets:
        test_dataset = FashionIQTestDataset(
            data_root = args.data_root,
            image_root=args.image_root,
            image_size = args.image_size,
            split = 'val',
            target = target,
            max_turn_len=args.max_turn_len
        )
        test_loader[target] = test_dataset.get_loader(batch_size=args.batch_size)

    if args.method == 'text-only':
        model = TextOnlyModel(args=args,backbone=args.backbone,texts=train_dataset.get_all_texts(),
        stack_num=args.stack_num,max_turn_len=args.max_turn_len,normalize_scale=args.normalize_scale,
                              text_method=args.text_method,fdims=args.fdims,fc_arch='A',init_with_glove=False)
    elif args.method == 'image-only':
        model = ImageOnlyModel(args=args,backbone=args.backbone,texts=train_dataset.get_all_texts(),
        stack_num=args.stack_num,max_turn_len=args.max_turn_len,normalize_scale=args.normalize_scale,
                     text_method=args.text_method,fdims=args.fdims,fc_arch='A',init_with_glove=False)
    elif args.method == 'tirg':
        model = TIRG(args=args,backbone=args.backbone,texts=train_dataset.get_all_texts(),
        stack_num=args.stack_num,max_turn_len=args.max_turn_len,normalize_scale=args.normalize_scale,
                     text_method=args.text_method,fdims=args.fdims,fc_arch='A',init_with_glove=False)
    # elif args.method == 'match-text-only':
    #     model = MatchTextOnly(args=args,backbone=args.backbone,texts=train_dataset.get_all_texts(),
    #                           text_method=args.text_method,fdims=args.fdims,fc_arch='A',init_with_glove=False)
    elif args.method == 'match-tirg':
        model = MatchTIRG(args=args,backbone=args.backbone,texts=train_dataset.get_all_texts(),
                              text_method=args.text_method,fdims=args.fdims,fc_arch='A',init_with_glove=False)
    elif args.method == 'combine':
        model = Combine(args=args,backbone=args.backbone,texts=train_dataset.get_all_texts(),
        stack_num=args.stack_num,max_turn_len=args.max_turn_len,normalize_scale=args.normalize_scale,
                              text_method=args.text_method,fdims=args.fdims,fc_arch='A',init_with_glove=False)
    else:
        raise NotImplementedError()


    model = model.cuda()
    log_path = os.path.join('logs',args.expr_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    summary_writer = SummaryWriter(log_path)

    trainer = Trainer(args=args,data_loader=train_loader,
                      model=model,summary_writer=summary_writer)
    evaluator = Evaluator(args=args, data_loader=test_loader,
                          model=model, summary_writer=summary_writer, eval_freq=1)
    print("start training")
    for epoch in range(args.epochs):
        epoch += 1
        train_summary = trainer.train(epoch)
        eval_summary = evaluator.test(epoch)
        results_state['epochs'].append({
            'epoch': epoch,
            'recorded_at': datetime.now().isoformat(),
            'train': train_summary,
            'eval': eval_summary,
        })
        write_results_state(results_state)
    # model = torch.load('model.pkl')
    results_state['finished_at'] = datetime.now().isoformat()
    write_results_state(results_state)
    print("Congrats! You just finished training.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train')

    parser.add_argument('--gpu_id',default='1',type=str,help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--manualSeed',type=int,default=int(time.time()),help='manual seed')
    parser.add_argument('--warmup',action='store_true',help='warmup?')
    parser.add_argument('--expr_name',default='devel',type=str,help='experiment name')
    parser.add_argument('--data_root',required=False,type=str,default='data/')
    parser.add_argument('--text_method',default='encode',choices=['lstm','swem','lstm-gru','encode'],type=str)
    parser.add_argument('--fdims',default=2048,type=int,help='output feature dimensions')
    parser.add_argument('--max_turn_len',default=4)
    parser.add_argument('--method',default='combine',type=str,help='method')
    parser.add_argument('--target',default='dress',type=str,help='target (dress | shirt | toptee | all)')
    parser.add_argument('--epochs',default=100,type=int)
    parser.add_argument('--print_freq',default=1,type=int)
    parser.add_argument('--batch_size',default=16,type=int,help='train batchsize')
    parser.add_argument('--image_size',default=224,type=int,help='image size (default:16)')
    parser.add_argument('--backbone',default='resnet18',type=str)
    parser.add_argument('--normalize_scale',default=5.0,type=float)
    parser.add_argument('--lr',default=0.00011148,type=float,help='initial learning rate')
    parser.add_argument('--lrp',default=0.48,type=float,help='lrp')
    parser.add_argument('--lr_decay_factor',default=0.4747,type=float)
    parser.add_argument('--lr_decay_steps',default='10,20,30,40,50,60,70',type=str)
    # parser.add_argument('--lr_decay_steps',default='30,60,90,120',type=str)
    parser.add_argument('--image_root',type=str,default='/projdata1/info_fil/yfyuan/task202008/CVPR/data/')
    parser.add_argument('--attention_type',type=str,default='dot')
    parser.add_argument('--stack_num',type=int,default=1)
    main()
