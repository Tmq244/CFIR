import numpy as np


def _to_vector(feature):
    array = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(array)
    if norm > 0:
        array = array / norm
    return array


class IRBench(object):
    def __init__(self, config=None):
        self.config = config or {}
        self.clean()

    def clean(self):
        self._index_ids = []
        self._index_features = []
        self._query_ids = []
        self._query_features = []
        self._last_rankings = {}

    def feed_index(self, data):
        unique_id, feature = data[:2]
        self._index_ids.append(unique_id)
        self._index_features.append(_to_vector(feature))

    def feed_query(self, data):
        unique_id, feature = data[:2]
        self._query_ids.append(unique_id)
        self._query_features.append(_to_vector(feature))

    def search_all(self, top_k=None):
        if not self._query_features or not self._index_features:
            self._last_rankings = {query_id: [] for query_id in self._query_ids}
            return np.empty((len(self._query_ids), 0), dtype=np.int64)

        query_matrix = np.stack(self._query_features, axis=0)
        index_matrix = np.stack(self._index_features, axis=0)
        scores = np.matmul(query_matrix, index_matrix.T)
        ranking = np.argsort(-scores, axis=1)
        if top_k is not None:
            ranking = ranking[:, :top_k]

        self._last_rankings = {
            query_id: [self._index_ids[idx] for idx in ranking[row].tolist()]
            for row, query_id in enumerate(self._query_ids)
        }
        return ranking

    def render_result(self, result):
        return dict(self._last_rankings)
