
#import os
#os.chdir('/media/fred/Local Disk/Bioinfo/code')

import sys
sys.path.append("./MaSIF_colab")
sys.path.append("./MaSIF_colab/data_preprocessing")

import torch
import utils
import pykeops
import numpy as np

from Bio.PDB import *
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
from data_preprocessing.download_pdb import convert_to_npy

# Custom data loader and model:
from helper import *
from model import dMaSIF
from data_iteration import iterate
from data import ProteinPairsSurfaces, PairData, CenterPairAtoms, load_protein_pair
from data import RandomRotationPairAtoms, NormalizeChemFeatures, iface_valid_filter


def generate_point_cloud_binding(pdb_fn, pdb_nm, chains, args):

    parser = PDBParser(PERMISSIVE=True)
    structure = parser.get_structure("structure", pdb_fn)
    convert_to_npy(pdb_fn, args.chains_dir, args.npy_dir, chains)
    single_pdb = "{n}_{c}_{c}".format(n=pdb_nm, c=chains[0])

    # Ensure reproducability:
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Load the train and test datasets:
    transformations = (
        Compose([NormalizeChemFeatures(), CenterPairAtoms(), RandomRotationPairAtoms()])
        if args.random_rotation
        else Compose([NormalizeChemFeatures()])
    )

    if single_pdb != "":
        single_data_dir = Path(args.npy_dir)
        test_dataset = [load_protein_pair(single_pdb, single_data_dir, single_pdb=True)]
        test_pdb_ids = [single_pdb]

    # PyTorch geometric expects an explicit list of "batched variables":
    batch_vars = ["xyz_p1", "xyz_p2", "atom_coords_p1", "atom_coords_p2"]
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, follow_batch=batch_vars
    )

    net = dMaSIF(args)
    net.load_state_dict(torch.load(args.model_path, map_location=args.device)["model_state_dict"])
    net = net.to(args.device)

    # Perform one pass through the data:
    info = iterate(net, test_loader, None, args, test=True,
                   save_path=Path(args.preds_dir), pdb_ids=test_pdb_ids)
    return info


''' esimate bfactor and predict binding for each residue
'''
def generate_residue_binding\
    (pdb_fn, smask_fn, embd_fn, ptcld_fn,
     atom_binding_fn, resid_binding_fn, args):

    structure, atom_coords, ptcld_coords, bfactors = \
        utils.load_results(pdb_fn, embd_fn, ptcld_fn)

    atom_bfs, atom_clases = utils.point_cloud_to_atom\
        (atom_coords, ptcld_coords, bfactors, smask_fn, args)

    resid_clases, resid_ids = utils.atom_to_residue(atom_clases, structure)
    resid_clases = utils.classify_residues\
        (resid_clases, resid_ids, args.resid_clas_cho, args.resid_clas_ratio)

    utils.save_atom_binding(atom_bfs, structure, atom_binding_fn)

    return resid_clases


''' run dmasif on given pdb and return  binding for each residue
'''
def run_dmasif(pdb_nm, chains, pdb_fn, smask_fn, embd_fn, ptcld_fn,
               atom_bd_fn, resid_bd_fn, args):

    #_ = generate_point_cloud_binding(pdb_fn, pdb_nm, chains, args)

    resid_binding = generate_residue_binding\
        (pdb_fn, smask_fn, embd_fn, ptcld_fn, atom_bd_fn, resid_bd_fn, args)

    return resid_binding
