from pymatgen.core import Structure, Lattice
from pymatgen.util.coord import pbc_shortest_vectors
import numpy as np
import pandas as pd


class BPProcessor:
    # Class-level cache for reference data - now stores both modes
    _ref_cache = {}  # Changed to dict to store both modes
    BOE_NEIGHBOR_CUTOFF = 3
    # compat: preserve historical BOE slot semantics used by existing training CSV/model
    # geometric: deterministic bond-center sorting without legacy slot remapping
    ORDERING_POLICY = "compat"

    @classmethod
    def _cache_key(cls, mode: str):
        return (
            mode,
            round(float(cls.BOE_NEIGHBOR_CUTOFF), 8),
            str(cls.ORDERING_POLICY),
        )

    @classmethod
    def set_ordering_policy(cls, policy: str):
        policy = str(policy).lower().strip()
        if policy not in ("compat", "geometric"):
            raise ValueError("ORDERING_POLICY must be one of: compat, geometric")
        cls.ORDERING_POLICY = policy

    def __init__(self, primitive: Structure, mode: str, ref_struct: Structure = None, use_cache: bool = True):
        self.e_z = np.array([0, 0, 1])
        self.e_y = np.array([0, 1, 0])
        self.original_structure = primitive
        self.mode = mode
        cache_key = BPProcessor._cache_key(mode)
        # Check if this specific mode is cached
        if use_cache and cache_key in BPProcessor._ref_cache:
            # Use cached reference data for this mode
            cache = BPProcessor._ref_cache[cache_key]
            self.ref_struct = cache['ref_struct']
            self.flattened_structure = cache['flattened_structure']
            self.bond_pair = cache['bond_pair']
            self.sorted_bond_pair = cache['sorted_bond_pair']
            self.ref_center_struct = cache['ref_center_struct']
            self.BOE_center_indices = cache['BOE_center_indices']
            self.BOE_indices = cache['BOE_indices']
            self.flat_indices = cache['flat_indices']
            self.ref_indices = cache['ref_indices']
        else:
            # Need to recalculate for this mode
            if ref_struct is None:
                self.ref_struct = primitive
            else:
                self.ref_struct = ref_struct
            self.flattened_structure = self.to_2D()
            self.bond_pair = self.get_pair_list()
            self.sorted_bond_pair = self.sort_pairs()
            self.ref_center_struct = self.get_center_struct(self.ref_struct)
            self.BOE_center_indices = self.get_BOE_center(mode=mode)
            raw_boe_indices = self.find_BOE_indices()
            self.BOE_indices = self._align_indices_via_reference(raw_boe_indices, mode)
            self.flat_indices = np.hstack((self.sorted_bond_pair[self.get_BOE_center(mode='1')].flatten(),
                                           self.sorted_bond_pair[self.get_BOE_center(mode='2')].flatten()))
            self.ref_indices = np.hstack((self.sorted_bond_pair[self.get_BOE_center(mode='1')][:,0],self.sorted_bond_pair[self.get_BOE_center(mode='1')][:,1],
                                          self.sorted_bond_pair[self.get_BOE_center(mode='2')][:,0],self.sorted_bond_pair[self.get_BOE_center(mode='2')][:,1]))
            # Cache this mode
            BPProcessor._ref_cache[cache_key] = {
                'ref_struct': self.ref_struct.copy(),  # Deep copy to protect from corruption
                'flattened_structure': self.flattened_structure,
                'bond_pair': self.bond_pair,
                'sorted_bond_pair': self.sorted_bond_pair,
                'ref_center_struct': self.ref_center_struct,
                'BOE_center_indices': self.BOE_center_indices,
                'BOE_indices': self.BOE_indices,
                'flat_indices': self.flat_indices,
                'ref_indices': self.ref_indices
            }
        # ori_center_struct is no longer needed by get_BOE_any (vectorized),
        # but kept available as a lazy property for backward compatibility
        self._ori_center_struct = None
        self.BOEs = self.get_BOE_any()

    @classmethod
    def fast_BOE(cls, frac_coords: np.ndarray, lat_matrix: np.ndarray, mode: str) -> np.ndarray:
        """
        Compute BOEs from raw fractional coords and lattice matrix,
        using cached indices. No Structure objects created.
        Requires that the cache for this mode was already populated by a prior __init__ call.
        """
        cache = cls._ref_cache[cls._cache_key(mode)]
        bond_list = cache['sorted_bond_pair']
        center_indices = cache['BOE_center_indices']
        BOE_indices = cache['BOE_indices'].astype(int)

        fcoords1 = frac_coords[bond_list[:, 0]]
        fcoords2 = frac_coords[bond_list[:, 1]]
        frac_diff = fcoords2 - fcoords1
        #keyt step: Convert fractional differences to the range [-0.5, 0.5] to handle PBC correctly
        frac_diff -= np.round(frac_diff)
        center_frac = fcoords1 + frac_diff / 2
        # Keep only the BOE centers
        boe_center_frac = center_frac[center_indices]
        #Gather BOE atom fractional coords using the cached BOE_indices
        boe_atom_frac = frac_coords[BOE_indices]
        # Compute PBC shortest vectors from center to each atom, broadcasted over all centers and atoms
        diffs = boe_atom_frac - boe_center_frac[:, np.newaxis, :]
        #apply PBC correction to the diffs
        diffs -= np.round(diffs)

        return np.einsum('ijk,kl->ijl', diffs, lat_matrix)

    @property
    def ori_center_struct(self):
        if self._ori_center_struct is None:
            self._ori_center_struct = self.get_center_struct(self.original_structure)
        return self._ori_center_struct

    def to_2D(self) -> Structure:
        '''
        Get 2D struct for the standard ref struct
        '''
        lattice = self.ref_struct.lattice
        z = lattice.abc[2] / 2
        coords = self.ref_struct.cart_coords.copy()
        coords[:, 2] = z
        atomlist = ["C"] * len(coords)
        return Structure(lattice, atomlist, coords, coords_are_cartesian=True)

    def get_pair_list(self) -> np.ndarray:
        '''
        Get pair list for ref struct
        '''
        radius = 1.0
        neighbor_list = self.flattened_structure.get_all_neighbors(radius)
        neighbor_indices = [
            [i, neighbor.index]
            for i, neighbors in enumerate(neighbor_list)
            for neighbor in neighbors if neighbor.index > i
        ]
        return np.unique(np.array(neighbor_indices), axis=0)

    def sort_pairs(self) -> np.ndarray:
        '''
        Sort the pair indices(N*2) so that the coord with second index always has a larger z coord.
        Because when training the NN, the output of 2nd force is with larger z.
        So in order to align, we need to sort the pairs.
        '''
        coords = self.flattened_structure.cart_coords
        bond_pairs = self.bond_pair
        for pair in bond_pairs:
            if coords[pair[0]][2] > coords[pair[1]][2]:
                pair[0], pair[1] = pair[1], pair[0]
        return bond_pairs

    def get_center_struct(self, structure: Structure) -> Structure:
        '''
        Get the center atoms and put them into a struct, work for both standard and non standard struct.
        Vectorized version: replaces per-pair pbc_shortest_vectors calls with batch numpy ops.
        '''
        coords = structure.frac_coords
        cart_coords = structure.cart_coords
        lattice = structure.lattice
        bond_list = self.sorted_bond_pair

        fcoords1 = coords[bond_list[:, 0]]
        fcoords2 = coords[bond_list[:, 1]]
        frac_diff = fcoords2 - fcoords1
        frac_diff -= np.round(frac_diff)
        cart_diff = frac_diff @ lattice.matrix
        centers = cart_coords[bond_list[:, 0]] + cart_diff / 2

        atomlist = ["C"] * len(centers)
        return Structure(lattice, atomlist, centers, coords_are_cartesian=True)

    def get_BOE_center(self, mode: str) -> np.ndarray:
        '''
        Identify the BOE centers as indices in the pair list for ref struct.
        '''
        layer_cell = self.ref_struct
        lattice = layer_cell.lattice
        layer_coords = layer_cell.frac_coords
        BOE_center_indices = []
        bond_list = self.sorted_bond_pair
        for i, (index1, index2) in enumerate(bond_list):
            vect = np.squeeze(
                pbc_shortest_vectors(
                    lattice, layer_coords[index1], layer_coords[index2])
            )
            dot_z = np.dot(vect, self.e_z)
            dot_y = np.dot(vect, self.e_y)
            if mode == "1":
                if (dot_z > 0 and dot_y > 0) or (dot_z < 0 and dot_y < 0):
                    BOE_center_indices.append(i)
            elif mode == "2":
                if (dot_z < 0 and dot_y > 0) or (dot_z > 0 and dot_y < 0):
                    BOE_center_indices.append(i)
        return np.array(BOE_center_indices)

    def _align_indices_via_reference(self, raw_boe_indices: np.ndarray, mode: str) -> np.ndarray:
        """Map BOE atom indices onto a stable slot layout for model compatibility."""
        if self.ORDERING_POLICY == "geometric":
            return np.asarray(raw_boe_indices, dtype=int)

        coords = self.ref_struct.frac_coords
        lattice = self.ref_struct.lattice
        center_struct = self.ref_center_struct

        M = len(raw_boe_indices)
        N= 10
        aligned_indices = np.zeros((M, N), dtype=int)

        for i, center_idx in enumerate(self.BOE_center_indices):
            center_frac = center_struct[center_idx].frac_coords
            atoms_idx = np.asarray(raw_boe_indices[i], dtype=int)
            local_vecs = np.zeros((N, 3), dtype=float)
            for j, a_idx in enumerate(atoms_idx):
                local_vecs[j] = np.squeeze(pbc_shortest_vectors(lattice, center_frac, coords[a_idx]))

            sorted_vecs = _legacy_sort_atoms(np.array([local_vecs]), mode=mode)[0]

            for target_slot in range(N):
                target_v = sorted_vecs[target_slot]
                if np.all(target_v == 0):
                    continue
                for j in range(N):
                    if np.allclose(target_v, local_vecs[j], atol=1e-4):
                        aligned_indices[i, target_slot] = atoms_idx[j]
                        break

        return aligned_indices

    @staticmethod
    def _bond_center_sort_key(vec: np.ndarray, tol: float = 1e-8):
        radius = float(np.linalg.norm(vec))
        is_center = radius < tol
        azimuth = float(np.arctan2(vec[1], vec[0])) if radius >= tol else 0.0
        elevation = float(np.arctan2(vec[2], np.hypot(vec[0], vec[1]))) if radius >= tol else 0.0
        return (
            is_center,
            round(radius, 8),
            round(azimuth, 8),
            round(elevation, 8),
            round(float(vec[0]), 8),
            round(float(vec[1]), 8),
            round(float(vec[2]), 8),
        )

    def _sorted_bond_center_indices(self, center_idx_list: list[int] | np.ndarray, boe_center_idx: int) -> np.ndarray:
        """Sort neighboring bond-center indices deterministically and keep the BOE center bond last."""
        center_struct = self.ref_center_struct
        lattice = self.ref_struct.lattice
        boe_center_frac = center_struct[boe_center_idx].frac_coords

        keyed_indices = []
        for idx in np.asarray(center_idx_list, dtype=int):
            local_vec = np.squeeze(pbc_shortest_vectors(lattice, boe_center_frac, center_struct[idx].frac_coords))
            keyed_indices.append((self._bond_center_sort_key(local_vec), idx))

        keyed_indices.sort(key=lambda item: item[0])
        return np.array([idx for _, idx in keyed_indices], dtype=int)

    def find_BOE_indices(self) -> np.ndarray:
        '''
        Identify BOE indices for ref struct so that we can use it to get BOE for any structure even perturbed.

        '''
        center_cell = self.ref_center_struct
        BOE_bond_center_indices = []
        BOE_center_indices = self.BOE_center_indices
        bond_list = self.sorted_bond_pair
        neighbor_count_issues = []
        
        for i, idx in enumerate(BOE_center_indices):
            # here we only get neighbors for the BOE centers
            #TODO: The cutoff can be expanded to get larger neighbor shells if needed.
            cutoff = float(self.BOE_NEIGHBOR_CUTOFF)
            # Center cell include all BO centers, but we need to get neighbors in the center structure and only loop for BOE_centers
            center_neighbors = center_cell.get_neighbors(
                center_cell[idx], cutoff)
            # print(len(center_neighbors))
            if i == 0:
                expected_neighbor_count = len(center_neighbors)
            if len(center_neighbors) != expected_neighbor_count:
                neighbor_count_issues.append((i, len(center_neighbors)))
                print("WARNING: The center at index {} has {} neighbors (expected {})".format(
                    idx, len(center_neighbors), expected_neighbor_count))
            indices_list = [center_neighbor.index for center_neighbor in center_neighbors] + [idx]
            if self.ORDERING_POLICY == "geometric":
                ordered = self._sorted_bond_center_indices(indices_list, idx)
            else:
                ordered = np.asarray(indices_list, dtype=int)
            BOE_bond_center_indices.append(ordered)
        
        # Use dtype=object to handle inhomogeneous shapes safely
        BOE_bond_center_indices = np.array(BOE_bond_center_indices, dtype=object)
        
        if neighbor_count_issues:
            print(f"WARNING: {len(neighbor_count_issues)} centers have incorrect neighbor counts. "
                  f"This may indicate reference structure geometry issues.")
        
        M = len(BOE_bond_center_indices)
        if M == 0:
            return np.zeros((0, 0), dtype=int)

        n_bonds_per_boe = len(BOE_bond_center_indices[0])
        if any(len(indices) != n_bonds_per_boe for indices in BOE_bond_center_indices):
            raise ValueError(
                "Inconsistent BOE bond-center counts were found across BOE centers. "
                "Use a cutoff that yields the same number of neighboring bond centers for each BOE center."
            )

        BOE_indices = np.zeros((M, 2 * n_bonds_per_boe), dtype=int)
        for i in range(M):
            idxs = np.asarray(BOE_bond_center_indices[i], dtype=int)
            selected = bond_list[idxs]
            BOE_indices[i] = np.asarray(selected, dtype=int).reshape(-1)
        return BOE_indices

    def get_BOE_any(self) -> np.ndarray:
        '''
        Use the standard pair list, BOE indices to derive the BOE for any structure.
        Vectorized version: replaces nested per-atom pbc_shortest_vectors calls with batch numpy ops.
        '''
        center_indices = self.BOE_center_indices
        BOE_indices = self.BOE_indices.astype(int)
        structure = self.original_structure
        coords = structure.frac_coords
        lattice = structure.lattice
        bond_list = self.sorted_bond_pair
        lat_matrix = lattice.matrix

        # Vectorized center computation in fractional coords
        fcoords1 = coords[bond_list[:, 0]]
        fcoords2 = coords[bond_list[:, 1]]
        frac_diff = fcoords2 - fcoords1
        frac_diff -= np.round(frac_diff)
        center_frac = fcoords1 + frac_diff / 2  # (N_bonds, 3)

        # Gather BOE center and atom fractional coords
        boe_center_frac = center_frac[center_indices]          # (n, 3)
        boe_atom_frac = coords[BOE_indices]                    # (n, 10, 3)

        # PBC shortest vectors from center to each atom
        diffs = boe_atom_frac - boe_center_frac[:, np.newaxis, :]  # (n, 10, 3)
        diffs -= np.round(diffs)

        # Convert fractional diffs to Cartesian
        layer_BOEs = np.einsum('ijk,kl->ijl', diffs, lat_matrix)

        return layer_BOEs
    
def get_centers(BOE_atoms: np.ndarray) -> np.ndarray:
    '''
    Derive bond-center coordinates for each BOE.

    The BOE atoms are expected to be ordered as 2 * N atoms corresponding to N bonds:
    (0,1), (2,3), ..., (2N-2, 2N-1). The final bond is the BOE center bond itself,
    so its midpoint is at the origin and omitted from the returned center list.
    '''
    if BOE_atoms.ndim != 3 or BOE_atoms.shape[1] % 2 != 0:
        raise ValueError("BOE_atoms must have shape (n_boe, 2 * n_bonds, 3).")

    n_pairs = BOE_atoms.shape[1] // 2
    neighbor_pair_count = max(n_pairs - 1, 0)
    centers = np.zeros((len(BOE_atoms), neighbor_pair_count, 3))

    for i, atom_group in enumerate(BOE_atoms):
        for j in range(neighbor_pair_count):
            idx1 = 2 * j
            idx2 = idx1 + 1
            c1 = atom_group[idx1]
            c2 = atom_group[idx2]
            if np.all(c1 == 0) or np.all(c2 == 0):
                continue
            centers[i, j] = (c1 + c2) / 2

    return centers


def _legacy_sort_atoms(BOE_atoms: np.ndarray, mode: str) -> np.ndarray:
    """Historical slot assignment used by older training CSV/model semantics."""
    sorted_atoms = np.zeros((len(BOE_atoms), 10, 3))
    for i, atom_group in enumerate(BOE_atoms):
        for coords in atom_group:
            x, y, z = coords
            if x < 0 and y < 0 and z < 0 and abs(x) > 1:
                sorted_atoms[i][0] = coords
            elif x < 0 and y < 0 and z > 0 and abs(x) > 1:
                sorted_atoms[i][1] = coords
            elif x < 0 and y > 0 and z < 0 and abs(x) > 1:
                sorted_atoms[i][2] = coords
            elif x < 0 and y > 0 and z > 0 and abs(x) > 1:
                sorted_atoms[i][3] = coords
            elif x > 0 and y < 0 and z < 0 and abs(x) > 1:
                sorted_atoms[i][4] = coords
            elif x > 0 and y < 0 and z > 0 and abs(x) > 1:
                sorted_atoms[i][5] = coords
            elif x > 0 and y > 0 and z < 0 and abs(x) > 1:
                sorted_atoms[i][6] = coords
            elif x > 0 and y > 0 and z > 0 and abs(x) > 1:
                sorted_atoms[i][7] = coords
            else:
                if mode == "1":
                    if y < 0 and z < 0:
                        sorted_atoms[i][8] = coords
                    elif y > 0 and z > 0:
                        sorted_atoms[i][9] = coords
                elif mode == "2":
                    if abs(x) < 0.5 and y > 0 and z < 0:
                        sorted_atoms[i][8] = coords
                    elif abs(x) < 0.5 and y < 0 and z > 0:
                        sorted_atoms[i][9] = coords
    return sorted_atoms

def sort_atoms(BOE_atoms: np.ndarray, mode: str) -> np.ndarray:
    '''
    Align BOE atoms by sorting whole bond pairs according to their bond-center positions.
    '''
    if BOE_atoms.ndim != 3 or BOE_atoms.shape[1] % 2 != 0:
        raise ValueError("BOE_atoms must have shape (n_boe, 2 * n_bonds, 3).")

    sorted_atoms = np.zeros_like(BOE_atoms)
    for i, atom_group in enumerate(BOE_atoms):
        bond_pairs = atom_group.reshape(-1, 2, 3)
        pair_centers = bond_pairs.mean(axis=1)
        order = sorted(
            range(len(pair_centers)),
            key=lambda idx: BPProcessor._bond_center_sort_key(pair_centers[idx]),
        )
        sorted_atoms[i] = bond_pairs[order].reshape(-1, 3)
    return sorted_atoms

if __name__ == "__main__":
    ref_struct =Structure.from_file("POSCAR_REV.vasp").copy()
    # test_struct.perturb(0.2, 0.01)
    test_struct = Structure.from_file("final_struct.vasp").copy()
    BP_1 = BPProcessor(primitive=test_struct, mode="1",
                        ref_struct=ref_struct)
    BP_2 = BPProcessor(primitive=test_struct, mode="2",
                        ref_struct=ref_struct)
    # print(BOEs[0:3])
    BOEs_1_np = BP_1.get_BOE_any()
    BOEs_2_np = BP_2.get_BOE_any()
    # centers = get_centers(BOE_atoms=BOEs_1_np)
    # print(centers.shape)
    print(BOEs_1_np.shape, BOEs_1_np[0])

