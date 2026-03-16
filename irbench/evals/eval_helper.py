class EvalHelper(object):
    def __init__(self):
        self.gt_dict = {}
        self.rank_dict = {}

    def feed_gt(self, data):
        unique_id, target_ids = data
        self.gt_dict[unique_id] = list(target_ids)

    def feed_rank_from_dict(self, rank_dict):
        self.rank_dict = {key: list(value) for key, value in rank_dict.items()}

    def evaluate(self, metric=None, kappa=None):
        metric = metric or ['top_k_acc']
        kappa = kappa or []
        results = {}
        if 'top_k_acc' in metric:
            for k in kappa:
                hit = 0
                total = 0
                for query_id, gt_ids in self.gt_dict.items():
                    if not gt_ids:
                        continue
                    ranked_ids = self.rank_dict.get(query_id, [])
                    if k is not None:
                        ranked_ids = ranked_ids[:k]
                    total += 1
                    if set(ranked_ids).intersection(gt_ids):
                        hit += 1
                top_k_acc = hit / float(total) if total else 0.0
                results[str(k)] = {'top_k_acc': top_k_acc}
        return [results]