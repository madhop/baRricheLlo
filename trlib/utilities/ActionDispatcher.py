import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.cluster import DBSCAN
import multiprocessing


class ActionDispatcher(object):
    """ActionDispatcher Class is used by the FQI algorithm to retrieve
    the action set for eachs state in a transparent way."""

    def __init__(self, actions, state_mask):
        self.actions = actions
        self.state_mask = state_mask

    def get_actions(self, s):
        raise NotImplemented


class ConstantActionDispatcher(ActionDispatcher):
    def __init__(self, actions):
        super(ConstantActionDispatcher, self).__init__(actions, [])

    def get_actions(self, s):
        a = [list(self.actions) for i in range(s.shape[0])]


class KDTActionDispatcher(ActionDispatcher):
    """Base class for Action Dispatceher based on KDTree.
    """

    def __init__(self, actions, state_mask, kdtree, state_normalizer, action_normalizer, filter_outliers, n_jobs):
        """
        Input:
        ------
        actions: list of n-dimensional actions
        state_mask: ids of the state features to consider
        kdtree: istance of the kdtree used to find the neighbors
        state_normalizer: instance of the normalizer of the states
        action_normalizer: instance of the normalizer of the actions (use only for outlier removal)
        filter_outliers: boolean, True to remove the outliers from each action set
        n_jobs: number of parallel jobs
        """

        super(KDTActionDispatcher, self).__init__(actions, state_mask)
        self.kdtree = kdtree
        self.s_normalizer = state_normalizer
        self.a_normalizer = action_normalizer
        self.filter_outliers = filter_outliers
        self.n_jobs = n_jobs
        
    def get_actions(self, s):
        """Given a state s, it returns its actions set. First get the ids
        of the nearest states and then retrieves the actions used on them.
        """
        if self.s_normalizer:
            s = self.s_normalizer.transform(s)

        # if s is a list of states then ids is a list of lists where
        # each of them contains the list of indices of the neighbors of each state.
        # if s is only one state then ids is a list with the ids of the
        # neighbors
        ids = self.find_neighbors(s)

        # Map the indices of the neighbors to actions
        if s.shape[0] > 1:
            a = list(map(lambda x: list(map(lambda i: self.actions[i], x)), ids))
        else:
            a = [self.actions[i] for i in ids[0]]
            
        # Filter the outlier actions
        if self.filter_outliers:
            if s.shape[0] > 1:
                pool = multiprocessing.Pool(self.n_jobs)
                a = list(pool.map(self.remove_outliers, a))
            else:
                a = self.remove_outliers(a)
        
        return a
    
    def remove_outliers(self, action_set):
        """Given a list of actions it performs DBSCAN clustering
        and returns the actions of the biggest cluster.
        """
        if not action_set:
            return action_set
        norm_actions = self.a_normalizer.transform(action_set)

        mdl = DBSCAN(eps=0.05, min_samples=5, metric='minkowski', p=2)
        mdl.fit(norm_actions)
        
        labels = mdl.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        clusters = [x for x in np.unique(labels) if x != -1]
        
        n_elems = [list(labels).count(i) for i in clusters]
        
        if len(n_elems) > 0:
            # if there are clusters
            biggest_cluster = np.argmax(n_elems)
            return [a for i, a in enumerate(action_set) if labels[i] == biggest_cluster] 
        else:
            # if all the actions are considered noise points then return all of them
            return action_set

    def find_neighbors(self, s):
        raise NotImplemented


class FixedKDTActionDispatcher(KDTActionDispatcher):
    """Implementation of the KDTreeActionDispatcher where are returned K neighbors.
    """

    def __init__(self, actions, state_mask, kdtree, state_normalizer=None, action_normalizer=None,
                 filter_outliers=False, n_jobs=1, k=100):
        # state_mask is a np.array containing the ordered indices of the used states
        # the indices consider the variable state_cols in dataset_preprocessing.py
        # n_actions is equal to n_states * k

        super(FixedKDTActionDispatcher, self).__init__(actions, state_mask, kdtree, state_normalizer,
                                                       action_normalizer, filter_outliers, n_jobs)
        self.k = int(k)

    def find_neighbors(self, s):
        ids = self.kdtree.query(s[:, self.state_mask], k=self.k, return_distance=False)
        #ids = ids.squeeze()
        return ids


class RadialKDTActionDispatcher(KDTActionDispatcher):
    """Implementation of the KDTreeActionDispatcher where are returned all the neighbors
    with a distance lower or equal than the radious.
    """
    def __init__(self, actions, state_mask, kdtree, state_normalizer=None, action_normalizer=None,
                 filter_outliers=False, n_jobs=1, radius=10):
        super(RadialKDTActionDispatcher, self).__init__(actions, state_mask, kdtree, state_normalizer, 
                                                        action_normalizer, filter_outliers, n_jobs)
        self.radius = radius

    def find_neighbors(self, s):
        ids = self.kdtree.query_radius(s[:, self.state_mask], r=self.radius)
        #ids = ids.squeeze()
        return ids


class ThresholdActionDispatcher(ActionDispatcher):

    def __init__(self, actions, state_mask, states, thresholds, max_actions, n_jobs):
        super().__init__(actions, state_mask)
        self.states = states
        self.thresholds = thresholds
        self.n_jobs = n_jobs
        self.max_actions = max_actions
        
    def get_actions(self, s):
        """Given a states it computes for each state dimension the distances and returns the actions of the states that
        for each dimension have the distances below the thresholds

        :param s: list of states to retrieve the actions
        :return: list of actions for each state
        """
        
        # 1- compute the distances for each state dimension
        # 2- Filter out all the states that have distances greater than at least a threshold
        # 3- Return the actions of the remained states
        with parallel_backend('threading', n_jobs=self.n_jobs):
            a = Parallel()(delayed(self._get_actions)(x) for x in s[:, self.state_mask])
        
        return a
        """
        # compute the distances for each state dimension
        # d = abs(np.array(s) - self.states[:, self.state_mask])
        d = list(map(lambda x: abs(x - self.states[:, self.state_mask]), s[:, self.state_mask]))

        # Filter out all the states that have distances greater than the thresholds
        # valid_states = ~((d > self.thresholds).any(axis=1))
        valid_states = ~(np.array(list(map(lambda x: (x > self.thresholds[self.state_mask]).any(axis=1), d))))

        return list(map(lambda x: np.array(self.actions)[x].tolist(), valid_states))
        """
        
    def _get_actions(self, x):
        
        vs = ~(np.array((abs(x - self.states) > self.thresholds[self.state_mask]).any(axis=1)))
        
        actions = np.array(self.actions)[vs]
        a_ids = range(len(actions))
        
        if np.count_nonzero(vs) > self.max_actions:
            actions = actions[np.random.choice(a_ids, self.max_actions, False)]
        
        return actions.tolist()
