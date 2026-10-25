import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

'''
Only 3 paths are needed for this program, and they are passed explicitly
instead of using global variables:
1.  Data path: from where we load the BOE, bond indices and label data
2.  Dataset path: to which we save the shuffled dataset and labels.
3.  Result path: to save the the model.pth and figures in run.py
'''


def gen_pca_features(BOE_df, output_dir, pca_fname='pca.csv', ncomponent=20):
    pca = PCA(n_components=ncomponent)
    # Normalization before pca fitting
    scaler = StandardScaler()
    scaler.fit(BOE_df)
    BOE_df_scaled = scaler.transform(BOE_df)
    # print(type(BOE_df_scaled))
    BOE_df_scaled = pd.DataFrame(BOE_df_scaled, columns=BOE_df.columns)
    # print(type(BOE_df_scaled))
    # PCA fitting
    pca.fit(BOE_df_scaled)
    pca_df = pd.DataFrame(
        pca.transform(BOE_df), columns=[f'PC{i}' for i in range(ncomponent)]
    )
    os.makedirs(output_dir, exist_ok=True)
    pca_df.to_csv(os.path.join(output_dir, pca_fname), index=False)
    return pca_df


def energy_volele_pca_train_test_split(volele_df, label_df, test=0.2, n_ve=20, length=4000, seed=42,
                                       ve_fname='ve.csv',  label_fname='label.csv', path=None):
    '''
    Train-test split of energy, volume elements coordinates into separate csv files
    Train set volume elements are used to fit PCA, stores transformed train-test PCA features into separate csv files

    ve_fname (str): The filename of storing the train-test splitted volume elements csv
    pca_fname (str): The filename of storing the train-test splitted pca features csv
    label_fname (str): The filename of storing the train-test splitted energy labels csv
    Prefix of train and test will be added at front of filename

    test (float): ratio of test set, range: (0, 1)
    n_ve (int): number of volume element per structure
    length (int): length of your dataset, i.e. number of total structures
    seed (int): RNG seed

    train and test idx are idx for structures, while vol_ele_train/test_idx are idx for BOE indx.

    Returns train and test indices for other uses
    '''
    idx = np.arange(length)
    np.random.seed(seed)
    np.random.shuffle(idx)
    train_idx, test_idx = idx[:-int(length*test)], idx[-int(length*test):]

    vol_ele_train_idx = np.array(
        [np.arange(i*n_ve, (i+1)*n_ve) for i in train_idx]).flatten()
    vol_ele_test_idx = np.array(
        [np.arange(i*n_ve, (i+1)*n_ve) for i in test_idx]).flatten()
    train_volele_df = volele_df.loc[volele_df.index[vol_ele_train_idx]]
    test_volele_df = volele_df.loc[volele_df.index[vol_ele_test_idx]]

    if path is None:
        raise ValueError("Output path must be provided for train/test split.")
    os.makedirs(path, exist_ok=True)

    train_volele_df.to_csv(os.path.join(path, 'train_' + ve_fname), index=False)
    test_volele_df.to_csv(os.path.join(path, 'test_' + ve_fname), index=False)

    label_df.loc[label_df.index[train_idx]].to_csv(
        os.path.join(path, 'train_' + label_fname), index=False
    )
    label_df.loc[label_df.index[test_idx]].to_csv(
        os.path.join(path, 'test_' + label_fname), index=False
    )
    print("The training and testing sets have been splitted.")
    return train_idx, test_idx

def align_force(structure_dir: str, center_dir: str, ref_dir: str):
    """Align forces for a given structure directory and center-data directory.

    structure_dir: directory containing forces_only.dat (e.g. "300")
    center_dir: directory containing standar_pair_list.npy and bond_center_indices.npy
                (e.g. "300/center1") and where df_forces.csv will be written.
    ref_dir: directory containing standar_pair_list.npy and bond_center_indices.npy for the reference structure (e.g. "data1" for center1, "data2"
    """
    # Load while handling possible empty/invalid lines
    data = np.genfromtxt(os.path.join(structure_dir, "forces_only.dat"), invalid_raise=False)
    # print(data.shape)
    # Reshape to (N_steps, 80, 3) with NaN protection
    n_steps = len(data) // 80
    # print(n_steps)
    # Truncate to complete frames
    forces = data[:n_steps*80].reshape(n_steps, 80, 3)
    forces = forces[-4000:, :, :]
    # np.savetxt("4000forces.txt", forces.reshape(forces.shape[0], -1))
    # print(forces.shape)

    # Find the corresponding forces by indices from the pair list and bond center indices
    standard_pair_list = np.load(os.path.join(ref_dir, 'standar_pair_list.npy'))
    bond_center_indices = np.load(os.path.join(ref_dir, 'bond_center_indices.npy'))
    # print("standard_pair_list:{}".format(standard_pair_list))
    # print("bond_center_indices:{}".format(bond_center_indices))
    center_pairs = standard_pair_list[bond_center_indices]
    # print("center_pair_Shape:{}".format(center_pairs.shape))
    # print(center_pairs)
    # The center pairs have been verified by vesta.
    selected_forces = np.zeros((len(forces)*len(center_pairs), 6))
    for i, force in enumerate(forces):
        for j, pair in enumerate(center_pairs):
            idx = i*len(center_pairs)+j
            selected_forces[idx] = force[pair].flatten()
    # print(forces[0][center_pairs[0]].flatten())
    # print("force shape:{}".format(selected_forces.shape))
    # Put the forces into df
    force_columns = []
    for i in range(2):
        force_columns.extend([f'fx{i}', f'fy{i}', f'fz{i}'])
    df_forces = pd.DataFrame(selected_forces, columns=force_columns)
    df_forces.to_csv(os.path.join(center_dir, "df_forces.csv"), index=False)
    print("The forces dataframe has been generated and saved!")

    return df_forces

def gen_dataset(df_BOE, df_forces, df_label, mode: str, dataset_root: str, data_dir: str):
    """Generate a dataset for a given mode under dataset_root.

    dataset_root: base directory where dataset_* subfolders will be created
    data_dir: directory used for saving PCA-related outputs (e.g. center1 path)
    """
    # Create output directory with mode name
    outpath = os.path.join(dataset_root, f"dataset_{mode}")
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    if mode == "pca":
        # pca to BOE df to reduce variables
        df = gen_pca_features(df_BOE, output_dir=data_dir)
    elif mode == "normal":
        df = df_BOE
    elif mode == "reorganize":
        # This part is to re arrange the BOE all coords orders so that the geometry can be better presented.
        original_columns = df_BOE.columns
        groups = [original_columns[i*3: (i+1)*3] for i in range(10)]
        new_order = [8, 9, 2, 3, 6, 7, 8, 9, 0, 1, 4, 5]
        reordered_groups = [groups[i] for i in new_order]
        new_columns = [col for group in reordered_groups for col in group]
        df = df_BOE[new_columns]
    else:
        raise ValueError(
            "Invalid mode. Choose from 'pca', 'normal', or 'reorganize'.")
    print(mode+" BOE data shape:{}".format(df.shape))
    merged_data = pd.concat([df, df_forces], axis=1)
    print("The merged data shape of "+ mode + " is :{}".format(merged_data.shape))

    # merged_data.to_csv(outpath+"/BOE_force_all_coords.csv", index=False)
    # Next we split the data into training and testing set.
    train_idx, test_idx = energy_volele_pca_train_test_split(
        merged_data, df_label, path=outpath)
    
    return

def energy_labels(file_name:str, label_fname:str, config_list:np.array, column: int):
    '''
    config_list: the rows yuo want to choose
    column: if the input file has multiple lines, choose the line that you want. column = 1 in eigen energy and = 0 in toten.
    '''
    with open(file_name) as file_object:
        lines = file_object.readlines()
        eigenvals = []
        for line in lines:
            eigenvals.append(float(line.strip().split()[column]))
        eigenvals = np.array(eigenvals)[config_list]
        # print(eigenvals[1800:2000])
    label_df = pd.DataFrame(eigenvals, columns=[label_fname])
    label_df.to_csv(label_fname+".csv", index=False)
    return label_df

if __name__ == "__main__":
    # Use the same window of configurations as in gen_center.py
    start = 0

    # Process each temperature directory separately
    # for temp_dir in ["300", "350", "400","500","600"]:
    # for temp_dir in [ "500","600"]:
    for temp_dir in ["400_veri"]:
        print(f"\n=== Processing directory {temp_dir} ===")

        #TODO: change b/t data1 and data2 according to the center mode
        # Paths for this temperature directory
        data_path = os.path.join(temp_dir, "center1")
        dataset_path = os.path.join(temp_dir, "dataset1")
        ref_path = os.path.join("300", "center1")  # Assuming ref data is in center2 for now
        
        if not os.path.isdir(data_path): 
            print(f"Data path {data_path} not found, skipping {temp_dir}.")
            continue

        os.makedirs(dataset_path, exist_ok=True)

        # Read BOE features for this directory
        boe_file = os.path.join(data_path, "all_BOE_features_mode.csv")
        if not os.path.isfile(boe_file):
            print(f"BOE feature file {boe_file} not found, skipping {temp_dir}.")
            continue
        df_BOE = pd.read_csv(boe_file)

        # Read and align forces (expects forces_only.dat in temp_dir and
        # pair-list data in ./data1 for center1)

        df_forces = align_force(
            structure_dir=temp_dir,
            center_dir=data_path,
            ref_dir=ref_path,
        )

        # Read energies for this directory
        energy_file = os.path.join(temp_dir, "energy.txt")
        if not os.path.isfile(energy_file):
            print(f"Energy file {energy_file} not found, skipping {temp_dir}.")
            continue
        energy = np.genfromtxt(energy_file, invalid_raise=False)
        print(energy.shape)
        energy_df = pd.DataFrame(energy, columns=['energy'])
        print("Energy data shape:{}".format(energy_df.shape))
        # Generate train/test split and save under temp_dir/dataset/... for mode "normal"
        gen_dataset(
            df_BOE=df_BOE,
            df_forces=df_forces,
            df_label=energy_df,
            mode="normal",
            dataset_root=dataset_path,
            data_dir=data_path,
        )




