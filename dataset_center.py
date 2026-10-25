import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class VEFeaturesEnergyDataset(Dataset):
    """
    Volume Element features-Energy Dataset Modified
    The __getitem__ method returns a volume element of a structure containing n_ve VEs
    f_df (pandas dataframe): features dataframe (can be vertices coords, PCA)
    label_df (pandas dataframe): energy symmetry
    n_col (int): takes the first n columns of f_df only as X (used only if col_indices is None)
    n_ve (int): number of volume elements per structure, default is 20
    shuffle_X (bool): shuffle the volume element order every time fetching X
    col_indices (list or numpy array, optional): specific column indices to select from f_df.
                                                If None, uses first n_col columns.
                                                Examples: [0,1,2,3] for first 4 columns,
                                                         np.arange(30,42) for columns 30-41

    Iterating it returns (X, y), where X first column is angular, X second column is radial, y is energy
    """

    def __init__(self, f_df, label_df, n_col=30, n_ve=20, shuffle_X=False, col_indices=None):
        if torch.cuda.device_count() == 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')
        
        # Select columns based on col_indices if provided, otherwise use first n_col columns
        if col_indices is not None:
            # Convert to numpy array if it's a list
            if isinstance(col_indices, list):
                col_indices = np.array(col_indices)
            selected_features = f_df.iloc[:, col_indices]
        else:
            # Default behavior: take the first n_col columns
            selected_features = f_df.iloc[:, 0:n_col]
        # self.center=['cx0', 'cy0', 'cz0', 'cx1', 'cy1', 'cz1', 'cx2', 'cy2', 'cz2', 'cx3', 'cy3', 'cz3']
        # self.f_df = f_df[self.center]
        print("Load data: ", selected_features.shape)
        self.f_col = ['fx0', 'fy0', 'fz0', 'fx1', 'fy1', 'fz1']
        self.features = np.ascontiguousarray(selected_features.to_numpy(dtype=np.float32, copy=True))
        self.forces = np.ascontiguousarray(f_df[self.f_col].to_numpy(dtype=np.float32, copy=True))
        # self.eigen = label_df['eigenval'].values
        self.energy = label_df['energy'].to_numpy(dtype=np.float32, copy=True)
        self.n_ve = n_ve
        self.length = len(self.energy)
        self.shuffle_X = shuffle_X


    def normalize_X(self, mode='standard', pre_mean=None, pre_std=None, pre_max=None, pre_min=None):
        '''
        mode: Determines the normalization method (standard or minmax).
        standard: Normalizes features to have zero mean and a standard variance of 1.
        minmax: Scales features to a [0, 1] range with a small shift to avoid division by zero.
        The purposese are different, usually the minmax mode are used to scale the variables into a specified range. 
        Because the standardization schemes for x and y should be different.
        '''
        if mode == "standard":
            self.pca_mean = np.asarray(pre_mean, dtype=np.float32) if pre_mean is not None else self.features.mean(axis=0, dtype=np.float64).astype(np.float32)
            self.pca_std = np.asarray(pre_std, dtype=np.float32) if pre_std is not None else self.features.std(axis=0, ddof=1, dtype=np.float64).astype(np.float32)
            safe_std = np.where(self.pca_std == 0, 1.0, self.pca_std)
            self.pca_std = safe_std.astype(np.float32, copy=False)
            self.features = ((self.features - self.pca_mean) / self.pca_std).astype(np.float32, copy=False)
        elif mode == "minmax":
            self.min_shift = 1e-5
            self.pca_max = np.asarray(pre_max, dtype=np.float32) if pre_max is not None else self.features.max(axis=0).astype(np.float32)
            self.pca_min = np.asarray(pre_min, dtype=np.float32) if pre_min is not None else self.features.min(axis=0).astype(np.float32)
            scale = np.where((self.pca_max - self.pca_min) == 0, 1.0, (self.pca_max - self.pca_min))
            self.features = ((self.features - self.pca_min) / scale + self.min_shift).astype(np.float32, copy=False)
        # elif mode == ""
        else:
            print("Unrecognized method, not normalization applied")

    def normalize_y(self, mode='mix', offset_energy=0, offset_eigen=0, negate=False):
        if negate:
            self.energy = -self.energy

        if mode == "standard":
            self.en_std = np.nanstd(self.energy)
            self.en_mean = np.nanmean(self.energy)
            self.energy = (self.energy - self.en_mean) / self.en_std
        elif mode == "minmax":
            self.min_shift = 1e-5
            self.en_min = np.nanmin(self.energy)
            self.en_max = np.nanmax(self.energy)
            self.energy = (self.energy - self.en_min) / \
                (self.en_max - self.en_min) + self.min_shift
        elif mode == "mix":
            self.energy += offset_energy
            # self.eigen += offset_eigen
            # self.ei_std = np.nanstd(self.eigen)
            # self.ei_mean = np.nanmean(self.eigen)
            # self.eigen = (self.eigen - self.ei_mean) / self.ei_std
        elif mode == "offset":
            self.eigen += offset_eigen
        else:
            print("Unrecognized method, not normalization applied")

    def __getitem__(self, index):
        energy_label = self.energy[index]
        # eigen_label = self.eigen[index]
        start = index * self.n_ve
        stop = (index + 1) * self.n_ve
        f_tmp = self.features[start:stop]
        forces = self.forces[start:stop]
        # if self.shuffle_X:
        #     np.random.shuffle(f_tmp)
        return f_tmp, forces, energy_label


    def __len__(self):
        return self.length


if __name__ == "__main__":
    # Load dataset
    file_path= './dataset_tot1_rev/dataset_normal/'
    ve_train = pd.read_csv(file_path+'train_ve.csv')
    ve_test = pd.read_csv(file_path+'test_ve.csv')
    label_train = pd.read_csv(file_path+'train_label.csv')
    label_test = pd.read_csv(file_path+'test_label.csv')

    # num of vertices as inputs
    n_clo = 30

    # Example 1: Using default behavior (first n_col columns)
    ds_train = VEFeaturesEnergyDataset(ve_train, label_train, n_col=n_clo)
    
    # Example 2: Using specific column indices (first 30 columns)
    # ds_train = VEFeaturesEnergyDataset(ve_train, label_train, 
    #                                   col_indices=list(range(30)))
    
    # Example 3: Using columns 31-42 (indices 30-41)
    # ds_train = VEFeaturesEnergyDataset(ve_train, label_train, 
    #                                   col_indices=np.arange(30, 42))
    
    # Example 4: Using specific non-contiguous columns
    # ds_train = VEFeaturesEnergyDataset(ve_train, label_train, 
    #                                   col_indices=[0, 2, 4, 6, 8, 10, 15, 20, 25])
    
    # Example 5: Put columns 30-41 before columns 0-29 (reorder without changing CSV)
    # col_indices_reordered = list(range(30, 42)) + list(range(0, 30))
    # ds_train = VEFeaturesEnergyDataset(ve_train, label_train, 
    #                                   col_indices=col_indices_reordered)
    # ds_train.normalize_y(mode='offset', offset=3)
    ds_train.normalize_X(mode='standard')
    # print("ds_mean:{}".format(ds_train.pca_mean))
    # print("ds_std:{}".format(ds_train.pca_std))

    print(ve_train.describe().iloc[:,0:n_clo].loc[['mean','std']])
    ds_train_stat_df=ve_train.describe().iloc[:,0:n_clo].loc[['mean','std']]
    ds_train_stat_df.to_csv("./data1/ds_train_stat.csv", index=True)
    # prepare the testing set.

    # ds_test = VEFeaturesEnergyDataset(ve_test, label_test, n_col=n_clo)
    # ds_test.normalize_y(mode='offset', offset=3)
    # ds_test.normalize_X(mode='standard')
    # print("The data has been normalized.")
