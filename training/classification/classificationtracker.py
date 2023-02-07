import os
import numpy as np
from training.common.stattracker import StatTracker


class ClassifcationTracker(StatTracker):
    def __init__(self, tracking_size: int, tracking_type: str = 'all'):
        super(ClassifcationTracker, self).__init__()
        configured = False
        self._tracking_type = tracking_type
        self._prev_acc = np.zeros(tracking_size, dtype=int)
        self._prev_pred = np.zeros(tracking_size, dtype=int)
        if tracking_type in ['FE', 'all']:
            self._forgetting_events = np.zeros(tracking_size, dtype=int)
            configured = True

        if tracking_type in ['SE', 'all']:
            self._switch_events = np.zeros(tracking_size, dtype=int)
            configured = True

        # raise error if wrong config
        if configured == False:
            raise Exception('Invalid tracking type!')

    def update(self, acc: np.ndarray, pred: np.ndarray, idxs: np.ndarray):
        delta = np.clip(self._prev_acc[idxs] - acc, a_min=0, a_max=1)
        if self._tracking_type in ['FE', 'all']:
            self._forgetting_events[idxs] += delta
        if self._tracking_type in ['SE', 'all']:
            change = self._prev_pred[idxs] != pred
            self._switch_events[idxs] += change

        # update previous accuracy
        self._prev_acc[idxs] = acc
        self._prev_pred[idxs] = pred

    def save_statistics(self, directory: str, ld_type: str):
        path = directory + '/ld_statistics/' + ld_type
        if not os.path.exists(path):
            os.makedirs(path)
        if self._tracking_type in ['FE', 'all']:
            np.save(path + '/forgetting_events.npy', self._forgetting_events)
        if self._tracking_type in ['SE', 'all']:
            np.save(path + '/switch_events.npy', self._switch_events)

    def get_stats(self, idxs: np.array, tracking_type: str):
        if tracking_type == 'SE':
            return self._switch_events[idxs]
        elif tracking_type == 'FE':
            return self._forgetting_events[idxs]
        else:
            raise Exception('Tracking type not implemented yet!')

    def get_all_stats(self, tracking_type: str):
        if tracking_type == 'SE':
            return self._switch_events
        elif tracking_type == 'FE':
            return self._forgetting_events
        else:
            raise Exception('Tracking type not implemented yet!')

