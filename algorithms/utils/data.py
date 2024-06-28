import numpy as np
import pandas as pd
import xarray as xr

INIT_TYPE = 0
TEST_TYPE = 1
OOD_TYPE = 2
GEN_TYPE = 3


class DataBase():
    '''
    Saving and looking up the data of the environments
    '''

    def __init__(
        self,
        train_indices,
        test_indices,
        ood_indices,
        laten_shape,
    ):
        '''
        Initialize the database with the indices of training, test, and OOD environments
        id: index value 0~S-1
        type: 0 for train, 1 for test, 2 for OOD test, and 3 for generated
        latent: latent variable [N,T,E] E:=latent_dim  or [E]
        label: cost label [1]
        '''
        #NOTE label_shape = 1
        self.laten_shape = laten_shape

        data = {'id': [], 'type': [], 'latent': [], 'label': []}
        for idx in train_indices:
            data['id'].append(idx)  # assume the index is the same as the env_id
            data['type'].append(INIT_TYPE)  # type=0
            data['latent'].append(np.random.rand(*self.laten_shape))
            data['label'].append(np.random.rand(1))
        for idx in ood_indices:
            data['id'].append(idx)
            data['type'].append(OOD_TYPE)
            data['latent'].append(np.nan)
            data['label'].append(np.nan)
        for idx in test_indices:
            data['id'].append(idx)
            data['type'].append(TEST_TYPE)
            data['latent'].append(np.nan)
            data['label'].append(np.nan)
        self.table = pd.DataFrame(data)

    def update_label(self, env_id, label):
        '''
        After generating data, update the label of the generated data
        '''
        for idx, lab in zip(env_id, label):
            self.table.loc[self.table['id'] == idx, 'label'] = lab

    def update_latent(self, env_id, latent):
        '''
        After generating data, update the latent variable of the generated data
        '''
        latent_list = [latent[i] for i in range(latent.shape[0])]
        for idx, lat in zip(env_id, latent_list):
            row_index = self.table.index[self.table['id'] == idx][0] 
            self.table.at[row_index, 'latent'] = lat

        return latent_list

    def add_gen_data(self, len_gen_data):
        '''
        After generating data, add the generated data to the database
        '''
        new_data = {
            'id': range(len(self.table),
                        len(self.table) + len_gen_data),
            'type': [GEN_TYPE] * len_gen_data,
            'latent': [np.random.rand(*self.laten_shape)] * len_gen_data,
            'label': [np.random.rand(1)] * len_gen_data
        }
        new_data_df = pd.DataFrame(new_data)
        self.table = pd.concat([self.table, new_data_df], ignore_index=True)


class Normalizer():

    def __init__(self, normalization_method='min-max'):
        self.method = normalization_method
        self.params = None

    def fit(self, data):
        if self.method == 'min-max':
            self.params = {'min': data.min(), 'max': data.max()}
        elif self.method == 'z-score':
            self.params = {'mean': data.mean(), 'std': data.std()}
        else:
            raise ValueError('Invalid normalization method')

    def transform(self, data):
        if self.method == 'min-max':
            return (data - self.params['min']) / (self.params['max'] -
                                                  self.params['min'])
        elif self.method == 'z-score':
            return (data - self.params['mean']) / self.params['std']
        else:
            raise ValueError('Invalid normalization method')

    def inverse_transform(self, data):
        if self.method == 'min-max':
            return data * (self.params['max'] -
                           self.params['min']) + self.params['min']
        elif self.method == 'z-score':
            return data * self.params['std'] + self.params['mean']
        else:
            raise ValueError('Invalid normalization method')


def split_scenario_indices(num_S,
                           train_ratio=0.8,
                           seed=None,
                           shuffle=True,
                           save_path=None):

    all_indices = np.arange(num_S)
    if seed is not None:
        np.random.seed(seed)
    if shuffle:
        np.random.shuffle(all_indices)
    train_size = int(num_S * train_ratio)
    train_indices = all_indices[:train_size]
    test_indices = all_indices[train_size:]
    train_indices = sorted(train_indices)
    test_indices = sorted(test_indices)
    if save_path is not None:
        # Save the indices to csv files
        np.savetxt(save_path / 'train_indices.csv', train_indices, fmt='%d')
        np.savetxt(save_path / 'test_indices.csv', test_indices, fmt='%d')

    return train_indices, test_indices


def random_sample(data_list,
                  sample_ratio=0.2,
                  seed=None,
                  fixed=False,
                  threshold=25):
    '''
    Randomly sample a subset of data_list
    '''
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(data_list)
    if fixed:
        sample_size = threshold
    else:
        sample_size = int(len(data_list) * sample_ratio)
        if sample_size > threshold:
            sample_size = threshold

    sample = data_list[:sample_size]
    sample = sorted(sample)
    return sample


def save_to_netcdf(gen_latent, gen_pv, gen_wp, gen_disruption, gen_load,
                   save_path):
    # Create a xarray dataset
    if gen_latent.ndim == 4:
        latent_dataarray = xr.DataArray(gen_latent, dims=('S', 'N', 'T', 'E'))
    elif gen_latent.ndim == 2:
        latent_dataarray = xr.DataArray(gen_latent, dims=('S', 'E'))
    
    pv_dataarray = xr.DataArray(gen_pv, dims=('S', 'PV_ID', 'T'))
    wp_dataarray = xr.DataArray(gen_wp, dims=('S', 'WP_ID', 'T'))
    disruption_dataarray = xr.DataArray(gen_disruption,
                                        dims=('S', 'Vulne_ID', 'T'))
    load_dataarray = xr.DataArray(gen_load, dims=('S', 'T'))

    ds = xr.Dataset({
        "gen_latent": latent_dataarray,
        "gen_pv": pv_dataarray,
        "gen_wp": wp_dataarray,
        "gen_disruption": disruption_dataarray,
        "gen_load": load_dataarray
    })

    # Save the dataset to a netcdf file
    ds.to_netcdf(save_path)


def load_from_netcdf(file_path):
    '''
    You can use ds['gen_latent'].values to get the numpy array of gen_latent
    
    '''
    ds = xr.open_dataset(file_path)

    return ds


def MSE(true, pred):
    return np.mean(np.square(true - pred))


def MAE(true, pred):
    return np.mean(np.abs(true - pred))


def calculate_MMD(true, pred):
    #
    B, N, T = true.shape

    def Gaussian_gram_matrix(s1, s2):
        gamma = 1.
        ones = np.ones(shape=[B, T])

        alpha = np.einsum('bnt, bnt -> bnt', s1, s1)
        alpha = np.einsum('bnt -> bt', alpha)

        beta = np.einsum('bnt, bnt -> bnt', s2, s2)
        beta = np.einsum('bnt -> bt', beta)

        amo = np.einsum('bi, bj -> bij', alpha, ones)
        omb = np.einsum('bi, bj -> bij', ones, beta)

        diff2 = 2 * np.einsum('bni, bnj -> bij', s1, s2) - amo - omb

        return np.exp(diff2 / gamma)

    # Calculate MMD distance
    K_xx = Gaussian_gram_matrix(true, true)
    K_xy = Gaussian_gram_matrix(true, pred)
    K_yy = Gaussian_gram_matrix(pred, pred)

    # calculate mmd
    ones = np.ones(shape=[B, T])
    kxxkyy = K_xx + K_yy
    kxxkyy = np.einsum('bi, bij -> bj', ones, kxxkyy)
    kxxkyy = np.einsum('bi, bi -> b', kxxkyy, ones)
    kxy = np.einsum('bi, bij -> bj', ones, K_xy)
    kxy = np.einsum('bi, bi -> b', kxy, ones)

    T2 = np.full(shape=[B], fill_value=2 * T)
    mmd = (1 / (T * (T - 1))) * (kxxkyy - T2) - (2 / T**2) * kxy
    return mmd
