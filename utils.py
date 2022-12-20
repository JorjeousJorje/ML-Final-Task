import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import preprocessing
from numpy.random import default_rng
from scipy.stats import multivariate_normal


def draw_barplot(x, y, data):
  df = data[[x]+y].melt(id_vars=[x], value_name='values', var_name='metric_type')
  sns.catplot(data=df, x=x, y='values', hue='metric_type', kind="bar", height=10, aspect=1.5)
  
  
class DataGenerator:
    def __init__(self, random_state: int = 10000, max_dim:int = 10) -> None:
        self.random_state = random_state
        self.max_mean = max_dim
        self.min_mean = -max_dim
        self.rng = default_rng()
        self.cov_matrix_storage = {
            0: [[8, -5], [0.2, 0.2]],
            1: [[2, 1], [1, 2]],
            2: [[4, 1], [1, 2]],
            3: [[7, 0], [-1, 2]],
            4: [[1, 0], [0, 100]],
            5: [[6, -3], [-3, 3.5]],
            6: [[12, -6], [-6, 7]],
            7: [[4, 2], [2, 4]],
            8: [[-4, 2], [2, -4]],
        }
        self.scaler = preprocessing.MinMaxScaler(feature_range=(self.min_mean, self.max_mean))
        self.grid_x, self.grid_y = np.mgrid[self.min_mean:self.max_mean:0.01, self.min_mean:self.max_mean:0.01]
        self.scaler.fit(np.arange(0, len(self.grid_x)).reshape(-1, 1))
        self.grid = np.dstack((self.grid_x, self.grid_y))
        
        self.cached_means = {1: {}, 2: {}}
        self.cached_covs = {1: {}, 2: {}}
        self.use_cached = False
    
    
    def _generate_field(self, coords: np.ndarray, cnt: int, target: int = 1):
        if coords.ndim > 2:
            dist = np.zeros(shape=(coords.shape[0], coords.shape[1]))
        else:
            dist = np.zeros_like(coords)
        
        mean, cov = None, None
        for i in range(cnt):
            if (self.use_cached):
                mean = self.cached_means[target][i]
                cov = self.cached_covs[target][i]
            else:
                cov_matrix_index = np.random.randint(low=0, high=len(self.cov_matrix_storage) - 1)
                mean = self.rng.uniform(low=self.min_mean, high=self.max_mean, size=(2,))
                cov = self.cov_matrix_storage[cov_matrix_index]
                self.cached_means[target][i] = mean
                self.cached_covs[target][i] = cov
            dist_gen = multivariate_normal(mean.tolist(), cov)
            dist += dist_gen.pdf(coords)
        return dist
    
    def __get_field(self, cnt, target = 1):
        return self._generate_field(self.grid, cnt, target)
    
    def get_fields(self, cnt1: int, cnt2: int):
        return (self._generate_field(self.grid, cnt1, 1), self._generate_field(self.grid, cnt2, 2))
    
    def generate_train_data_for_class(self, count: int, cnt: int, target:int = 1):
        y = np.full(shape=(count,), fill_value=target)
        X = []
        field = self.__get_field(cnt, target)
        max_prob = np.max(field)
        while len(X) != count:
            randX = np.random.randint(0, field.shape[0] - 1, 1)[0]
            randY = np.random.randint(0, field.shape[1] - 1, 1)[0]
            randProb = np.random.uniform(1e-1, 0.5, 1)[0]
            sample_prob = field[randX, randY]
            if (sample_prob / max_prob > randProb):
                scaled = self.scaler.transform(np.array([randX, randY]).reshape(-1, 1))
                X.append(scaled.squeeze())
        
        return X, y, field
                
    def generate_train_data_for_first_class(self, count: int, cnt: int):
        return self.generate_train_data_for_class(count, cnt, 1)
    
    def generate_train_data_for_second_class(self, count: int, cnt: int):
        return self.generate_train_data_for_class(count, cnt, 2)
    
    
    def generate_train_data(self, count, cnt1: int, cnt2: int):
        X1, y1, field1 = self.generate_train_data_for_first_class(count[0], cnt1)
        X2, y2, field2 = self.generate_train_data_for_second_class(count[1], cnt2)
        X = np.concatenate([X1, X2])
        y = np.concatenate([y1, y2])
        return X, y, (field1, field2, self.grid)
        