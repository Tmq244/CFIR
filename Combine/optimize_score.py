import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
import pickle
import argparse
import numpy as np
import tqdm as tqdm
from datetime import datetime
from hyperopt import hp, tpe, fmin
from irbench.evals.eval_helper import EvalHelper


def objective_fn(W,args):
    W = np.array(W) / sum(W)
    final_matrix = {}
    for i, repo in enumerate(args.repos):
        pkl_path = os.path.join('output_score',repo,'hyperopt.val.pkl')
        print(pkl_path)
        try:
            assert os.path.exists(pkl_path)
            saved_scores = pickle.load(open(pkl_path,'rb'))
        except Exception as err:
            raise OSError('{}, {}'.format(pkl_path,err))

        for target, node in saved_scores.items():
            tag = '{}_score'.format(target)
            if tag in final_matrix:
                final_matrix[tag] += node['score']*W[i]
            else:
                final_matrix[tag] = node['score']*W[i]
    r1 = 0.
    r5 = 0.
    r8 = 0.
    r10 = 0.
    r20 = 0.
    r50 = 0.
    r10r50 = 0.
    mrr = 0.
    for target in saved_scores.keys():
        y_score = final_matrix['{}_score'.format(target)]
        y_indices_full = np.argsort(-1 * y_score, axis=1)
        y_indices = y_indices_full[:, :args.topk]

        res = {}
        for qidx, query_id in enumerate(saved_scores[target]['query_ids']):
            res[query_id] = []
            for j in range(args.topk):
                index = y_indices[qidx, j]
                jth_rank_id = saved_scores[target]['index_ids'][index]
                res[query_id].append(jth_rank_id)
        from preprocess.dataset_tag import FashionIQTestDataset
        val_dataset = FashionIQTestDataset(
            data_root=args.data_root,
            split = 'val',
            target=target,
            max_turn_len=args.max_turn_len,
            image_root=args.image_root,
            image_size=args.image_size,
        )
        val_loader = val_dataset.get_loader(batch_size=8)
        val_loader.dataset.set_mode('query')
        eval_helper = EvalHelper()
        gt_dict = {}
        for bidx, input in enumerate(val_dataset):
            _w_key = input[args.max_turn_len+1][0]
            _tid = input[args.max_turn_len][2]
            eval_helper.feed_gt([_w_key,[_tid]])
            gt_dict[_w_key] = _tid
        eval_helper.feed_rank_from_dict(res)
        score = eval_helper.evaluate(metric=['top_k_acc'],kappa=[1,5,8,10,20,50])
        _r1 = score[0][str(1)]['top_k_acc']
        _r8 = score[0][str(8)]['top_k_acc']
        _r10 = score[0][str(10)]['top_k_acc']
        _r20 = score[0][str(20)]['top_k_acc']
        _r50 = score[0][str(50)]['top_k_acc']
        _r5 = score[0][str(5)]['top_k_acc']
        _r10_r50 = 0.5 * (_r10 + _r50)
        r1 += _r1
        r8 += _r8
        r10+=_r10
        r20 += _r20
        r50+=_r50
        r5+=_r5
        r10r50+=_r10_r50
        _pos = 0.
        query_ids = saved_scores[target]['query_ids']
        index_ids = saved_scores[target]['index_ids']
        for qidx, query_id in enumerate(query_ids):
            target_id = gt_dict.get(query_id)
            if target_id is None:
                continue
            ranked_index_ids = [index_ids[idx] for idx in y_indices_full[qidx]]
            pos = ranked_index_ids.index(target_id) if target_id in ranked_index_ids else len(ranked_index_ids)
            _pos += 1.0 / (pos + 1)
        _pos = _pos / float(len(query_ids)) if query_ids else 0.0
        mrr += _pos
    ntargets = float(len(saved_scores)) if saved_scores else 1.0
    r1 /= ntargets
    r5 /= ntargets
    r8 /= ntargets
    r10/=ntargets
    r20 /= ntargets
    r50/=ntargets
    r10r50/=ntargets
    mrr /= ntargets
    print('[{}] R1: {:.4f}, R5: {:.4f}, R8: {:.4f}, R10: {:.4f}, R20: {:.4f}, R50: {:.4f}, R10R50: {:.4f}, MRR: {:.4f}'.format(W, r1, r5, r8, r10, r20, r50, r10r50, mrr))
    return  -1 * r10r50

def main(args):
    args.repos = args.repos.strip().split(',')
    space = [hp.uniform('w{}'.format(i),0,1) for i in range(len(args.repos))]
    best = fmin(fn=lambda W: objective_fn(W, args),
                space=space,
                algo=tpe.suggest,
                max_evals=args.max_eval)
   # best = fmin(
    #    fn=lambda W: objective_fn(W, args),
     #   space=space,
      #  algo=tpe.suggest,
       # max_evals=args.max_eval
    #)
    print('best: {}'.format(best))
    date_key = str(datetime.now().strftime('%Y%m%d%H%M'))[2:]
    SPLITS = ['val']
    for SPLIT in SPLITS:
        print('Save final results for {}'.format(SPLIT))
        os.system('mkdir -p output_optimize')
        final_score = dict()
        for idx, repo in enumerate(args.repos):
            try:
                pkl_path = os.path.join('output_score',repo,'hyperopt.{}.pkl'.format(SPLIT))
                assert os.path.exists(pkl_path)
                saved_scores = pickle.load(open(pkl_path,'rb'))
            except Exception as err:
                raise OSError('{}, {}'.format(pkl_path,err))

            for target, node in saved_scores.items():
                if not target in final_score:
                    final_score[target] = {
                        'score': None,
                        'query_ids': [],
                        'index_ids': []
                    }
                w = float(best['w{}'.format(idx)])
                if idx == 0:
                    final_score[target]['score'] = w*node['score']
                else:
                    final_score[target]['score'] += w * node['score']
                final_score[target]['query_ids'] = node['query_ids']
                final_score[target]['index_ids'] = node['index_ids']
        if not os.path.exists('./output_optimize/{}'.format(date_key)):
            os.makedirs('./output_optimize/{}'.format(date_key))
        with open('./output_optimize/{}/hyperopt.{}.pkl'.format(date_key,SPLIT),'wb') as f:
            pickle.dump(final_score,f)
        TOP_K = 50
        result = dict()
        for target, v in final_score.items():
            _result = []
            query_ids = v['query_ids']
            index_ids = v['index_ids']
            y_score = v['score']
            y_indices = np.argsort(-1*y_score,axis=1)
            for qidx, query_id in enumerate(query_ids):
                _r = []
                for j in range(min(TOP_K,y_score.shape[1])):
                    index = y_indices[qidx, j]
                    _r.append([
                        str(index_ids[index]),
                        float(y_score[qidx,index])
                    ])
                _result.append([query_id,_r])
            result[target] = _result
        with open('./output_optimize/{}/score.{}.pkl'.format(date_key,SPLIT),'wb') as f:
            pickle.dump(result,f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Bayes Optimization on the scores')
    parser.add_argument('--data_root',default='/projdata17/infofil/yfyuan/task202008/Multiturn/data/',type=str)
    parser.add_argument('--repos',default='210209123020_combine_three,210209131222_sum_three',type=str)
    parser.add_argument('--topk',default='50',type=int)
    parser.add_argument('--max_eval',default=10,type=int)
    parser.add_argument('--image_size',default=224,type=int,help='image size (default:16)')
    parser.add_argument('--max_turn_len',default=4,type=int)
    parser.add_argument('--image_root',type=str,default='/projdata1/info_fil/yfyuan/task202008/CVPR/data/')
    args,_ = parser.parse_known_args()
    main(args)




