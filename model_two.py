import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math
import os
import time


    
class Attention(nn.Module):
    def __init__(self, input_size, c, n_pos=10):
        super().__init__()
        self.output_size = 8  # fx, fy, fz, energy, eigenvalue
        self.n_pos = n_pos
        self.coord_dim = input_size // n_pos  # Should be 3 for xyz
        # Attention across positions (before feature extraction)
        self.attn = nn.MultiheadAttention(embed_dim=self.coord_dim, num_heads=1, batch_first=True).double()
        # self.attn = nn.MultiheadAttention(embed_dim=10, num_heads=2, batch_first=True).double()
        # Per-position feature extraction
        self.FC1 = nn.Linear(self.coord_dim, c).double()
        self.FC2 = nn.Linear(c, 4*c).double()
        self.FC3 = nn.Linear(4*c, 4*c).double()
        # Post-attention processing
        self.FC4 = nn.Linear(4*c, 2*c).double()
        self.FC5 = nn.Linear(2*c, 2*c).double()
        self.FC6 = nn.Linear(2*c, c).double()
        self.FC7 = nn.Linear(c, self.output_size).double()

    def forward(self, x):
        # x: (batch, n_pos * coord_dim)
        batch_size = x.size(0)
        # Reshape to (batch, n_pos, coord_dim)
        x = x.view(batch_size, self.n_pos, self.coord_dim)
        # Attention across positions (input: (batch, n_pos, coord_dim))
        attn_out, _ = self.attn(x, x, x)  # (batch, n_pos, coord_dim)
        x = attn_out
        x = self.FC1(x)  # (batch, n_pos, c)
        x = F.leaky_relu(x)
        x = self.FC2(x)
        x = F.leaky_relu(x)
        x = self.FC3(x)
        x = F.leaky_relu(x)
        x = self.FC4(x)
        x = F.leaky_relu(x)
        x = self.FC5(x)
        x = F.leaky_relu(x)
        x = self.FC6(x)
        x = F.leaky_relu(x)
        # Pooling: mean over positions
        x = x.mean(dim=1)  # (batch, c)
        x = self.FC7(x)    # (batch, output_size)
        return x

# class Attention(nn.Module):
#     def __init__(self, input_size, c, n_pos=10):
#         super().__init__()
#         self.output_size = 8  # fx, fy, fz, energy, eigenvalue
#         self.n_pos = n_pos
#         self.coord_dim = input_size // n_pos  # Should be 3 for xyz
#         # Per-position feature extraction
#         self.FC1 = nn.Linear(self.coord_dim, c).double()
#         self.FC2 = nn.Linear(c, 2*c).double()
#         self.FC3 = nn.Linear(2*c, c).double()
#         # Attention across positions
#         self.attn = nn.MultiheadAttention(embed_dim=c, num_heads=2, batch_first=True).double()
#         # Post-attention processing
#         self.FC4 = nn.Linear(c, c).double()
#         self.FC5 = nn.Linear(c, c).double()
#         self.FC6 = nn.Linear(c, c).double()
#         self.FC7 = nn.Linear(c, self.output_size).double()

#     def forward(self, x):
#         # x: (batch, n_pos * coord_dim)
#         batch_size = x.size(0)
#         # if x.numel() != 0:
#         #     print(x.shape)
#         # Reshape to (batch, n_pos, coord_dim)
#         x = x.view(batch_size, self.n_pos, self.coord_dim)
#         x = self.FC1(x)  # (batch, n_pos, c)
#         x = F.leaky_relu(x)
#         x = self.FC2(x)
#         x = F.leaky_relu(x)
#         x = self.FC3(x)
#         x = F.leaky_relu(x)
#         # Attention across positions
#         attn_out, _ = self.attn(x, x, x)  # (batch, n_pos, c)
#         x = attn_out
#         x = self.FC4(x)
#         x = F.leaky_relu(x)
#         x = self.FC5(x)
#         x = F.leaky_relu(x)
#         x = self.FC6(x)
#         x = F.leaky_relu(x)
#         # Pooling: mean over positions
#         x = x.mean(dim=1)  # (batch, c)
#         x = self.FC7(x)    # (batch, output_size)
#         return x
    
class SubNet(nn.Module):
    def __init__(self, input_size, output_size, c, extra_hidden: bool = True):
        super().__init__()
        self.output_size = output_size  # fx, fy, fz, energy, eigenvalue
        self.extra_hidden = extra_hidden
        # self.FC1 = nn.Linear(input_size, c).double()
        # self.FC2 = nn.Linear(c, 4*c).double()
        # # Add more complexity
        # self.FC3 = nn.Linear(4*c, 2*c).double()
        # self.FC4 = nn.Linear(2*c, c).double()
        # self.FC5 = nn.Linear(c, self.output_size).double()
        self.FC1 = nn.Linear(input_size, c)
        self.FC2 = nn.Linear(c, 4*c)
        # Add more complexity
        self.FC3 = nn.Linear(4*c, 2*c)
        self.FC4 = nn.Linear(2*c, c)
        if self.extra_hidden:
            self.FC5 = nn.Linear(c, c)
            self.FC6 = nn.Linear(c, self.output_size)
        else:
            self.FC5 = nn.Linear(c, self.output_size)

    def forward(self, x):
        x = self.FC1(x)
        x = F.leaky_relu(x)

        x = self.FC2(x)
        x = F.leaky_relu(x)

        x = self.FC3(x)
        x = F.leaky_relu(x)

        x = self.FC4(x)
        x = F.leaky_relu(x)

        x = self.FC5(x)
        if self.extra_hidden:
            x = F.leaky_relu(x)
            x = self.FC6(x)

        return x


def infer_subnet_config_from_state_dict(state_dict: dict) -> dict:
    """Infer SubNet dimensions and whether the checkpoint uses the legacy output layer."""
    input_size = state_dict['FC1.weight'].shape[1]
    c = state_dict['FC1.weight'].shape[0]
    extra_hidden = 'FC6.weight' in state_dict
    output_size = state_dict['FC6.weight'].shape[0] if extra_hidden else state_dict['FC5.weight'].shape[0]
    return {
        'input_size': input_size,
        'output_size': output_size,
        'c': c,
        'extra_hidden': extra_hidden,
    }


def build_subnet_from_state_dict(state_dict: dict) -> SubNet:
    """Construct a compatible SubNet directly from a saved state dict."""
    config = infer_subnet_config_from_state_dict(state_dict)
    model = SubNet(**config)
    model.load_state_dict(state_dict)
    return model

class EnergyModelContainer():
    def __init__(self, model, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.criterion = torch.nn.MSELoss()
        self.device = device

    def save(self, path):
        """
        method save(path)

        saves the model to a given path as a state dict
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """
        method load(path)

        loads the model from the saved path, the number of layers and neurons per layer should be the same as the saved model
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def fit(self, loader1, loader2, ep: int, learning_rate=1e-3):
        '''
        Args:
        f_loss_sum: Cumulative force loss (MSE) for the epoch.
        e_loss_sum: Cumulative energy loss (MSE) for the epoch.
        loss_sum: Cumulative total loss (force + energy) for the epoch.
        f_loss: Temporary storage for force loss in each batch.
        X: Input features (e.g., atomic coordinates).
        forces: Target forces (vector quantities).
        energy: Target energy (scalar quantity).
        '''
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.97)
        self.model.train()
        # ep is the total epoch number
        for epoch in range(ep):
            f_loss_sum, e_loss_sum, loss_sum = 0, 0, 0
            f_loss = 0
            # X is a 3D tensor of shape (batch_size, ve_length, features)for example 8*20*20:
            for X, forces, energy in loader1:
                # print(X.shape, forces.shape, energy.shape)
                X = X.double().to(self.device)
                forces = forces.double().to(self.device)
                energy = energy.double().to(self.device)
                # Resets gradients (avoids accumulation from previous batches).
                self.optimizer.zero_grad()
                tot_e = torch.zeros(energy.size(), device=self.device)
                out_forces=[]
                
                for i in range(len(X[0])):
                    # Predicts forces and energy for the i-th VE.
                    res = self.model(X[:, i, :])
                    # Sums energy predictions across ves (for comparison with total energy).
                    tot_e += res[:, 6]
                    out_forces.append(res[:, 0:6])  # Collects force predictions
                out_forces = torch.stack(out_forces, dim=1)
                # Forces prediction. Extracts force components (0:6 assumes the model outputs)
                #TODO: If only FCNN is used, then out_forces should be a single tensor
                # out_forces = self.model(X)[:, :, 0:6].double()
                tot_e = tot_e.double()
                f_loss = torch.nn.functional.mse_loss(out_forces, forces)
                e_loss = self.criterion(tot_e.squeeze(), energy)

                # Total Loss & Backpropagation
                # f_loss = f_loss.to(self.device)
                # e_loss = e_loss.to(self.device)
                loss = f_loss + e_loss 
                loss= loss.double().to(self.device)
                loss.backward()
                f_loss_sum += f_loss.item()
                e_loss_sum += e_loss.item()
                loss_sum += loss.item()
                # Updates model weights using computed gradients (Adam optimizer).
                self.optimizer.step()
            # Adjusts learning rate (exponential decay with gamma=0.9).
            self.scheduler.step()
            print("Epoch: ", epoch, "Energy Loss (MSE): ", np.round(e_loss_sum / len(loader1), 6),
                    "Forces Loss (MSE): ", np.round(f_loss_sum / len(loader1), 6))
    
    def predict(self, loader):
            """
            Run validation of a trained model over data in a loader
            Handles 2 pairs of forces (x,y,z) and energy prediction

            Inputs:
                loader - Data loader

            Returns:
                If fit_force=True: TrueForces, PredForces, TrueEnergy, PredEnergy
                If fit_force=False: TrueEnergy, PredEnergy
            """
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.eval()


            with torch.no_grad():
                TrueEnergy, PredEnergy = [], []
                # Initialize force containers for 2 pairs of (x,y,z) forces
                TrueForces_pair1 = []
                TrueForces_pair2 = []
                PredForces_pair1 = []
                PredForces_pair2 = []

                for xs, true_forces, true_energy  in loader:
                    # Move data to device
                    xs = xs.to(device)
                    true_forces = true_forces.to(device)
                    true_energy = true_energy.to(device)
                    

                    batch_size = xs.size(0) #batch size
                    num_ve = xs.size(1)  # Number of volume elements

                    # Initialize energy predictions
                    pred_energies = torch.zeros(batch_size, device='cpu')
            
                    for i in range(num_ve):
                        # Model outputs: [fx1, fy1, fz1, fx2, fy2, fz2, energy]
                        out_res = self.model(xs[:, i, :])

                        # Split into force pairs and energy
                        pred_fx1 = out_res[:, 0]
                        pred_fy1 = out_res[:, 1]
                        pred_fz1 = out_res[:, 2]
                        pred_fx2 = out_res[:, 3]
                        pred_fy2 = out_res[:, 4]
                        pred_fz2 = out_res[:, 5]
                        pred_energy = out_res[:, 6]

                        # Store force predictions
                        PredForces_pair1.append(torch.stack(
                            [pred_fx1, pred_fy1, pred_fz1], dim=1).cpu().numpy())
                        #if i == 0: print(PredForces_pair1)
                        PredForces_pair2.append(torch.stack(
                            [pred_fx2, pred_fy2, pred_fz2], dim=1).cpu().numpy())

                        # Store true forces (assuming loader provides both pairs)
                        TrueForces_pair1.append(
                            true_forces[:, i, 0:3].cpu().numpy())
                        TrueForces_pair2.append(
                            true_forces[:, i, 3:6].cpu().numpy())

                        # Accumulate energy & eigenvalue predictions
                        pred_energies += pred_energy.cpu()
                  

                    # Store energy predictions
                    PredEnergy.append(pred_energies.cpu().numpy())
                    TrueEnergy.append(true_energy.cpu().numpy())


                # Concatenate all batches
                return (
                    np.concatenate(TrueForces_pair1, axis=0),
                    np.concatenate(TrueForces_pair2, axis=0),
                    np.concatenate(PredForces_pair1, axis=0),
                    np.concatenate(PredForces_pair2, axis=0),
                    np.concatenate(TrueEnergy, axis=0),
                    np.concatenate(PredEnergy, axis=0),

                )

                
    def predict_BOE(self, BOEs:np.array):
        PredEnergy=0
        PredForces_pair1 = []
        PredForces_pair2 = []
        for i in range(len(BOEs)):
            out_res = self.model(BOEs[i])
            # print(out_res)
            pred_fx1,pred_fy1,pred_fz1,pred_fx2, pred_fy2, pred_fz2, pred_energy, pred_eigen= out_res
            # Store force predictions
            PredForces_pair1.append(torch.tensor(
                [pred_fx1, pred_fy1, pred_fz1]).cpu().detach().numpy())
            #if i == 0: print(PredForces_pair1)
            PredForces_pair2.append(torch.tensor(
                [pred_fx2, pred_fy2, pred_fz2]).cpu().detach().numpy())
            PredEnergy += pred_energy.cpu().detach().numpy()
        return  PredForces_pair1, PredForces_pair2, PredEnergy
      
    


    def evaluate(self,test_loader):
        """
        Evaluate model performance with 2 pairs of (x,y,z) forces and energy

        Args:
            test_loader: Data loader for test set
            model: Trained model

        Returns:
            If fit_force=True: (ForceStats_pair1, ForceStats_pair2, EnergyStats)
            If fit_force=False: EnergyStats
        """

        # Get predictions for both force pairs and energy
        f1_true, f2_true, f1_pred, f2_pred, e_true, e_pred = self.predict(
            test_loader)

        # Create DataFrames for each force pair
        ForceStats_pair1 = pd.DataFrame({
            'pred_fx': f1_pred[:, 0].flatten(),
            'pred_fy': f1_pred[:, 1].flatten(),
            'pred_fz': f1_pred[:, 2].flatten(),
            'true_fx': f1_true[:, 0].flatten(),
            'true_fy': f1_true[:, 1].flatten(),
            'true_fz': f1_true[:, 2].flatten()
        })

        ForceStats_pair2 = pd.DataFrame({
            'pred_fx': f2_pred[:, 0].flatten(),
            'pred_fy': f2_pred[:, 1].flatten(),
            'pred_fz': f2_pred[:, 2].flatten(),
            'true_fx': f2_true[:, 0].flatten(),
            'true_fy': f2_true[:, 1].flatten(),
            'true_fz': f2_true[:, 2].flatten()
        })

        EnergyStats = pd.DataFrame({
            'pred_energy': e_pred.flatten(),
            'true_energy': e_true.flatten()
        })

        # Process DataFrames
        ForceStats_pair1, ForceStats_pair2, EnergyStats = process_force_stats(
            ForceStats_pair1, ForceStats_pair2, EnergyStats
        )

        # Print metrics
        print_metrics(ForceStats_pair1, ForceStats_pair2, EnergyStats)

        return ForceStats_pair1, ForceStats_pair2, EnergyStats

        
        

def plot_force_comparisons(df1, df2, energy_df, save_path: str, fname: str):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Pair 1 components
    axes[0,0].scatter(df1['true_fx'], df1['pred_fx'], alpha=0.3)
    axes[0,0].set_title('Pair 1 X Component')
    # axes[0,0].plot([df1['true_fx'].min(), df1['true_fx'].max()], 
    #             [df1['true_fx'].min(), df1['true_fx'].max()], 'r--')
    
    axes[0,1].scatter(df1['true_fy'], df1['pred_fy'], alpha=0.3)
    axes[0,1].set_title('Pair 1 Y Component')
    
    axes[0,2].scatter(df1['true_fz'], df1['pred_fz'], alpha=0.3)
    axes[0,2].set_title('Pair 1 Z Component')
    
    # Pair 2 components
    axes[1,0].scatter(df2['true_fx'], df2['pred_fx'], alpha=0.3)
    axes[1,0].set_title('Pair 2 X Component')
    
    axes[1,1].scatter(df2['true_fy'], df2['pred_fy'], alpha=0.3)
    axes[1,1].set_title('Pair 2 Y Component')
    
    axes[1,2].scatter(df2['true_fz'], df2['pred_fz'], alpha=0.3)
    axes[1,2].set_title('Pair 2 Z Component')

    axes[0,3].scatter(energy_df['true_energy'], energy_df['pred_energy'], alpha=0.3)
    axes[0,3].set_title('Energy Prediction')

    axes[1,3].scatter(energy_df['true_energy'], energy_df['pred_energy'], alpha=0.3)
    axes[1,3].set_title('Prediction')
    
    # First handle the 2x3 force plots
    for i in range(2):
        for j in range(3):
            ax = axes[i,j]
            ax.set_xlabel('True Force')
            ax.set_ylabel('Predicted Force')
            ax.axline((0,0), slope=1, color='r', linestyle='--')

    # Then handle the right column
    # Top right (energy)
    axes[0,3].set_xlabel('True Energy')
    axes[0,3].set_ylabel('Predicted Energy')
    axes[0,3].axline((0,0), slope=1, color='r', linestyle='--')

    # Bottom right (energy)
    axes[1,3].set_xlabel('True Energy')
    axes[1,3].set_ylabel('Predicted Energy')
    axes[1,3].axline((0,0), slope=1, color='r', linestyle='--')
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path+fname)
    print("RMSE plot has been saved.")

def print_metrics(df1, df2, energy_df, n_atom = 80):
    """Print evaluation metrics for both force pairs and energy"""
    # Force pair 1 metrics
    rmse_fx1 = np.sqrt(df1['sqerr_fx'].mean())
    rmse_fy1 = np.sqrt(df1['sqerr_fy'].mean())
    rmse_fz1 = np.sqrt(df1['sqerr_fz'].mean())
    rmse_force1 = np.sqrt(df1['sqerr_force'].mean())

    # Force pair 2 metrics
    rmse_fx2 = np.sqrt(df2['sqerr_fx'].mean())
    rmse_fy2 = np.sqrt(df2['sqerr_fy'].mean())
    rmse_fz2 = np.sqrt(df2['sqerr_fz'].mean())
    rmse_force2 = np.sqrt(df2['sqerr_force'].mean())

    # Energy metrics
   
    rmse_energy = np.sqrt(energy_df['sqerr_energy'].mean()) / n_atom * 1000
    print(f"""Energy RMSE: {np.round(rmse_energy,6)} meV/Atom""")

    print(f"""
    Force Pair 1 RMSE:
    X: {np.round(rmse_fx1*1000,6)} meV/Å
    Y: {np.round(rmse_fy1*1000,6)} meV/Å
    Z: {np.round(rmse_fz1*1000,6)} meV/Å
    Magnitude: {np.round(rmse_force1*1000,6)} meV/Å
    
    Force Pair 2 RMSE:
    X: {np.round(rmse_fx2*1000,6)} meV/Å
    Y: {np.round(rmse_fy2*1000,6)} meV/Å
    Z: {np.round(rmse_fz2*1000,6)} meV/Å
    Magnitude: {np.round(rmse_force2*1000,6)} meV/Å
    """)

def process_force_stats(df1, df2, energy_df):
    """
    Calculate error metrics for force pairs and optionally energy.

    Args:
        df1: DataFrame for first force pair with columns ['pred_fx', 'pred_fy', 'pred_fz', 
             'true_fx', 'true_fy', 'true_fz']
        df2: DataFrame for second force pair (same columns as df1)
        energy_df: Optional DataFrame with ['pred_energy', 'true_energy'] columns.
                  If None, energy processing is skipped.

    Returns:
        If energy_df provided: returns df1, df2, energy_df with calculated metrics
        If energy_df not provided: returns df1, df2 with calculated metrics
    """
    # Process first force pair
    df1['pred_force_mag'] = np.sqrt(
        df1['pred_fx']**2 + df1['pred_fy']**2 + df1['pred_fz']**2)
    df1['true_force_mag'] = np.sqrt(
        df1['true_fx']**2 + df1['true_fy']**2 + df1['true_fz']**2)

    # Process second force pair
    df2['pred_force_mag'] = np.sqrt(
        df2['pred_fx']**2 + df2['pred_fy']**2 + df2['pred_fz']**2)
    df2['true_force_mag'] = np.sqrt(
        df2['true_fx']**2 + df2['true_fy']**2 + df2['true_fz']**2)

    # Calculate squared errors for forces
    for df in [df1, df2]:
        df['sqerr_fx'] = (df['pred_fx'] - df['true_fx'])**2
        df['sqerr_fy'] = (df['pred_fy'] - df['true_fy'])**2
        df['sqerr_fz'] = (df['pred_fz'] - df['true_fz'])**2
        df['sqerr_force'] = (df['pred_force_mag'] - df['true_force_mag'])**2

    # Process energy if energy_df is provided
    energy_df['sqerr_energy'] = (
        energy_df['pred_energy'] - energy_df['true_energy'])**2

    return df1, df2, energy_df


class DualEnergyModelContainer():
    """
    Jointly trains two SubNet models (one per BOE mode) with a combined loss.
    Energy: averaged across both models. Forces: all forces from both models contribute to loss.
    """
    def __init__(self, model1, model2, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model1 = model1.to(device)
        self.model2 = model2.to(device)
        self.criterion = torch.nn.MSELoss()
        self.device = device

    def save(self, path1, path2):
        torch.save(self.model1.state_dict(), path1)
        torch.save(self.model2.state_dict(), path2)

    def load(self, path1, path2):
        state_dict1 = torch.load(path1, map_location=self.device)
        state_dict2 = torch.load(path2, map_location=self.device)

        if isinstance(self.model1, SubNet):
            config1 = infer_subnet_config_from_state_dict(state_dict1)
            current_config1 = {
                'input_size': self.model1.FC1.in_features,
                'output_size': self.model1.output_size,
                'c': self.model1.FC1.out_features,
                'extra_hidden': self.model1.extra_hidden,
            }
            if config1 != current_config1:
                self.model1 = SubNet(**config1).to(self.device)

        if isinstance(self.model2, SubNet):
            config2 = infer_subnet_config_from_state_dict(state_dict2)
            current_config2 = {
                'input_size': self.model2.FC1.in_features,
                'output_size': self.model2.output_size,
                'c': self.model2.FC1.out_features,
                'extra_hidden': self.model2.extra_hidden,
            }
            if config2 != current_config2:
                self.model2 = SubNet(**config2).to(self.device)

        self.model1.load_state_dict(state_dict1)
        self.model2.load_state_dict(state_dict2)
        self.model1.eval()
        self.model2.eval()

    def _summarize_results(self, results, n_atom: int = 80):
        """Compute compact scalar summaries from evaluate()/predict() outputs."""
        m1_p1, m1_p2 = results['model1']
        m2_p1, m2_p2 = results['model2']
        energy_df = results['energy']

        force_frames = [m1_p1, m1_p2, m2_p1, m2_p2]
        pooled_sqerr = np.concatenate([frame['sqerr_force'].to_numpy() for frame in force_frames])
        pooled_force_rmse = float(np.sqrt(pooled_sqerr.mean()))

        summary = {
            'pooled_force_rmse': pooled_force_rmse,
            'energy_rmse_mev_atom': float(np.sqrt(energy_df['sqerr_energy'].mean()) / n_atom * 1000),
            'model1_pair1_rmse_mev': float(np.sqrt(m1_p1['sqerr_force'].mean()) * 1000),
            'model1_pair2_rmse_mev': float(np.sqrt(m1_p2['sqerr_force'].mean()) * 1000),
            'model2_pair1_rmse_mev': float(np.sqrt(m2_p1['sqerr_force'].mean()) * 1000),
            'model2_pair2_rmse_mev': float(np.sqrt(m2_p2['sqerr_force'].mean()) * 1000),
        }
        return summary

    def _append_details(self, details_path: str | None, text: str):
        if details_path is None:
            return
        with open(details_path, 'a', encoding='utf-8') as handle:
            handle.write(text + '\n')

    def fit(
        self,
        loader1,
        loader2,
        ep: int,
        learning_rate=1e-3,
        force_weight=1.0,
        energy_weight=0.1,
        val_loaders=None,
        per_temp_val_loaders=None,
        checkpoint_paths=None,
        details_path=None,
        val_frequency: int = 1,
        per_temp_frequency: int = 5,
    ):
        """
        Jointly train model1 and model2.
        loader1: DataLoader for BOE mode-1 descriptors (X1, forces1, energy)
        loader2: DataLoader for BOE mode-2 descriptors (X2, forces2, energy)
        Both loaders must yield the same energy targets (same structures, same split).
        """
        # Single optimizer over BOTH models' parameters
        self.optimizer = torch.optim.Adam(
            list(self.model1.parameters()) + list(self.model2.parameters()),
            lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.97)
        self.model1.train()
        self.model2.train()
        best_val_force_rmse = float('inf')
        best_epoch = None

        if details_path is not None:
            with open(details_path, 'w', encoding='utf-8') as handle:
                handle.write('Dual-model training details\n')

        for epoch in range(ep):
            epoch_start = time.perf_counter()
            f_loss_sum, e_loss_sum, loss_sum = 0, 0, 0
            n_batches = 0

            for (X1, forces1, energy1), (X2, forces2, energy2) in zip(loader1, loader2):
                X1 = X1.to(self.device, dtype=torch.float32, non_blocking=True)
                forces1 = forces1.to(self.device, dtype=torch.float32, non_blocking=True)
                energy1 = energy1.to(self.device, dtype=torch.float32, non_blocking=True)

                X2 = X2.to(self.device, dtype=torch.float32, non_blocking=True)
                forces2 = forces2.to(self.device, dtype=torch.float32, non_blocking=True)
                # energy2 should be the same as energy1 (same structures)

                self.optimizer.zero_grad()

                batch_size, num_ve, feature_dim = X1.shape
                out1 = self.model1(X1.reshape(batch_size * num_ve, feature_dim)).reshape(batch_size, num_ve, -1)
                out2 = self.model2(X2.reshape(batch_size * num_ve, feature_dim)).reshape(batch_size, num_ve, -1)
                out_forces1 = out1[:, :, 0:6]
                out_forces2 = out2[:, :, 0:6]
                tot_e1 = out1[:, :, 6].sum(dim=1)
                tot_e2 = out2[:, :, 6].sum(dim=1)

                # Averaged energy from both models
                avg_energy = (tot_e1 + tot_e2) / 2.0

                # Force loss: both models' forces contribute
                f_loss1 = torch.nn.functional.mse_loss(out_forces1, forces1)
                f_loss2 = torch.nn.functional.mse_loss(out_forces2, forces2)
                f_loss = f_loss1 + f_loss2

                # Energy loss: averaged prediction vs ground truth
                e_loss = self.criterion(avg_energy.squeeze(), energy1)

                loss = force_weight * f_loss + energy_weight * e_loss
                loss.backward()
                self.optimizer.step()

                f_loss_sum += f_loss.item()
                e_loss_sum += e_loss.item()
                loss_sum += loss.item()
                n_batches += 1

            self.scheduler.step()
            train_e_loss = e_loss_sum / n_batches
            train_f_loss = f_loss_sum / n_batches
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_seconds = time.perf_counter() - epoch_start
            compact_line = (
                f"Ep {epoch:03d} trainE={train_e_loss:.6f} trainF={train_f_loss:.6f} "
                f"lr={current_lr:.2e} time={epoch_seconds:.2f}s"
            )
            detail_lines = [
                f"Epoch {epoch}",
                f"  train_energy_mse={train_e_loss:.10f}",
                f"  train_force_mse={train_f_loss:.10f}",
                f"  train_total_loss={loss_sum / n_batches:.10f}",
                f"  learning_rate={current_lr:.10e}",
                f"  epoch_seconds={epoch_seconds:.10f}",
            ]

            if val_loaders is not None and (epoch + 1) % val_frequency == 0:
                val_results = self.evaluate(val_loaders[0], val_loaders[1], verbose=False)
                val_summary = self._summarize_results(val_results)
                compact_line += (
                    f" valF={val_summary['pooled_force_rmse'] * 1000:.3f} "
                    f"valE={val_summary['energy_rmse_mev_atom']:.3f}"
                )
                detail_lines.extend([
                    f"  pooled_val_force_rmse_mevA={val_summary['pooled_force_rmse'] * 1000:.10f}",
                    f"  pooled_val_energy_rmse_mev_atom={val_summary['energy_rmse_mev_atom']:.10f}",
                    f"  val_model1_pair1_rmse_mevA={val_summary['model1_pair1_rmse_mev']:.10f}",
                    f"  val_model1_pair2_rmse_mevA={val_summary['model1_pair2_rmse_mev']:.10f}",
                    f"  val_model2_pair1_rmse_mevA={val_summary['model2_pair1_rmse_mev']:.10f}",
                    f"  val_model2_pair2_rmse_mevA={val_summary['model2_pair2_rmse_mev']:.10f}",
                ])

                if val_summary['pooled_force_rmse'] < best_val_force_rmse:
                    best_val_force_rmse = val_summary['pooled_force_rmse']
                    best_epoch = epoch
                    compact_line += ' best'
                    detail_lines.append('  checkpoint=updated')
                    if checkpoint_paths is not None:
                        self.save(checkpoint_paths[0], checkpoint_paths[1])

            print(compact_line)

            if per_temp_val_loaders and (epoch + 1) % per_temp_frequency == 0:
                temp_parts = []
                detail_lines.append('  per_temperature_validation:')
                for temp, loaders in per_temp_val_loaders.items():
                    temp_results = self.evaluate(loaders[0], loaders[1], verbose=False)
                    temp_summary = self._summarize_results(temp_results)
                    temp_parts.append(f"{temp}:{temp_summary['pooled_force_rmse'] * 1000:.1f}")
                    detail_lines.append(
                        f"    {temp}K force_rmse_mevA={temp_summary['pooled_force_rmse'] * 1000:.10f} "
                        f"energy_rmse_mev_atom={temp_summary['energy_rmse_mev_atom']:.10f}"
                    )
                print('  valT ' + ' '.join(temp_parts))

            self._append_details(details_path, '\n'.join(detail_lines) + '\n')

        if best_epoch is not None:
            print(
                f"Best validation checkpoint: epoch {best_epoch} "
                f"pooled_force_rmse={best_val_force_rmse * 1000:.3f} meV/A"
            )
            self._append_details(
                details_path,
                f"Best validation checkpoint: epoch {best_epoch}, "
                f"pooled_force_rmse_mevA={best_val_force_rmse * 1000:.10f}",
            )

    def predict(self, loader1, loader2):
        """
        Run prediction using both models jointly.
        Returns combined force/energy predictions.
        """
        self.model1.eval()
        self.model2.eval()

        with torch.inference_mode():
            TrueEnergy, PredEnergy = [], []
            TrueForces1_pair1, TrueForces1_pair2 = [], []
            PredForces1_pair1, PredForces1_pair2 = [], []
            TrueForces2_pair1, TrueForces2_pair2 = [], []
            PredForces2_pair1, PredForces2_pair2 = [], []

            for (X1, forces1, energy1), (X2, forces2, energy2) in zip(loader1, loader2):
                X1 = X1.to(self.device, dtype=torch.float32, non_blocking=True)
                forces1 = forces1.to(self.device, dtype=torch.float32, non_blocking=True)
                X2 = X2.to(self.device, dtype=torch.float32, non_blocking=True)
                forces2 = forces2.to(self.device, dtype=torch.float32, non_blocking=True)
                energy1 = energy1.to(self.device, dtype=torch.float32, non_blocking=True)

                batch_size, num_ve, feature_dim = X1.shape
                out1 = self.model1(X1.reshape(batch_size * num_ve, feature_dim)).reshape(batch_size, num_ve, -1)
                out2 = self.model2(X2.reshape(batch_size * num_ve, feature_dim)).reshape(batch_size, num_ve, -1)

                PredForces1_pair1.append(out1[:, :, 0:3].reshape(-1, 3).cpu().numpy())
                PredForces1_pair2.append(out1[:, :, 3:6].reshape(-1, 3).cpu().numpy())
                TrueForces1_pair1.append(forces1[:, :, 0:3].reshape(-1, 3).cpu().numpy())
                TrueForces1_pair2.append(forces1[:, :, 3:6].reshape(-1, 3).cpu().numpy())

                PredForces2_pair1.append(out2[:, :, 0:3].reshape(-1, 3).cpu().numpy())
                PredForces2_pair2.append(out2[:, :, 3:6].reshape(-1, 3).cpu().numpy())
                TrueForces2_pair1.append(forces2[:, :, 0:3].reshape(-1, 3).cpu().numpy())
                TrueForces2_pair2.append(forces2[:, :, 3:6].reshape(-1, 3).cpu().numpy())

                pred_e1 = out1[:, :, 6].sum(dim=1).cpu()
                pred_e2 = out2[:, :, 6].sum(dim=1).cpu()

                avg_pred_e = ((pred_e1 + pred_e2) / 2.0).numpy()
                PredEnergy.append(avg_pred_e)
                TrueEnergy.append(energy1.cpu().numpy())

        return {
            'model1': (
                np.concatenate(TrueForces1_pair1, axis=0),
                np.concatenate(TrueForces1_pair2, axis=0),
                np.concatenate(PredForces1_pair1, axis=0),
                np.concatenate(PredForces1_pair2, axis=0),
            ),
            'model2': (
                np.concatenate(TrueForces2_pair1, axis=0),
                np.concatenate(TrueForces2_pair2, axis=0),
                np.concatenate(PredForces2_pair1, axis=0),
                np.concatenate(PredForces2_pair2, axis=0),
            ),
            'energy': (
                np.concatenate(TrueEnergy, axis=0),
                np.concatenate(PredEnergy, axis=0),
            ),
        }

    def evaluate(self, test_loader1, test_loader2, verbose: bool = True):
        results = self.predict(test_loader1, test_loader2)

        f1_true_p1, f1_true_p2, f1_pred_p1, f1_pred_p2 = results['model1']
        f2_true_p1, f2_true_p2, f2_pred_p1, f2_pred_p2 = results['model2']
        e_true, e_pred = results['energy']

        # Model-1 force stats
        ForceStats_m1_pair1 = pd.DataFrame({
            'pred_fx': f1_pred_p1[:, 0], 'pred_fy': f1_pred_p1[:, 1], 'pred_fz': f1_pred_p1[:, 2],
            'true_fx': f1_true_p1[:, 0], 'true_fy': f1_true_p1[:, 1], 'true_fz': f1_true_p1[:, 2],
        })
        ForceStats_m1_pair2 = pd.DataFrame({
            'pred_fx': f1_pred_p2[:, 0], 'pred_fy': f1_pred_p2[:, 1], 'pred_fz': f1_pred_p2[:, 2],
            'true_fx': f1_true_p2[:, 0], 'true_fy': f1_true_p2[:, 1], 'true_fz': f1_true_p2[:, 2],
        })
        # Model-2 force stats
        ForceStats_m2_pair1 = pd.DataFrame({
            'pred_fx': f2_pred_p1[:, 0], 'pred_fy': f2_pred_p1[:, 1], 'pred_fz': f2_pred_p1[:, 2],
            'true_fx': f2_true_p1[:, 0], 'true_fy': f2_true_p1[:, 1], 'true_fz': f2_true_p1[:, 2],
        })
        ForceStats_m2_pair2 = pd.DataFrame({
            'pred_fx': f2_pred_p2[:, 0], 'pred_fy': f2_pred_p2[:, 1], 'pred_fz': f2_pred_p2[:, 2],
            'true_fx': f2_true_p2[:, 0], 'true_fy': f2_true_p2[:, 1], 'true_fz': f2_true_p2[:, 2],
        })

        EnergyStats = pd.DataFrame({
            'pred_energy': e_pred.flatten(),
            'true_energy': e_true.flatten(),
        })

        ForceStats_m1_pair1, ForceStats_m1_pair2, EnergyStats = process_force_stats(
            ForceStats_m1_pair1, ForceStats_m1_pair2, EnergyStats)
        ForceStats_m2_pair1, ForceStats_m2_pair2, _ = process_force_stats(
            ForceStats_m2_pair1, ForceStats_m2_pair2, EnergyStats.copy())

        if verbose:
            print("=== Model 1 Forces ===")
            print_metrics(ForceStats_m1_pair1, ForceStats_m1_pair2, EnergyStats)
            print("=== Model 2 Forces ===")
            print_metrics(ForceStats_m2_pair1, ForceStats_m2_pair2, EnergyStats)

        return {
            'model1': (ForceStats_m1_pair1, ForceStats_m1_pair2),
            'model2': (ForceStats_m2_pair1, ForceStats_m2_pair2),
            'energy': EnergyStats,
        }

