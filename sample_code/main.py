
import sys
sys.path.append("MaSIF_colab")
sys.path.append("MaSIF_colab/data_preprocessing")

import os
import glob
import torch
import shutil
import pykeops
import argparse
import numpy as np

from Bio.PDB import *
#from google.colab import files
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
from data_preprocessing.download_pdb import convert_to_npy

# Custom data loader and model:
from helper import *
from model import dMaSIF
from data_iteration import iterate
from data import ProteinPairsSurfaces, PairData, CenterPairAtoms, load_protein_pair
from data import RandomRotationPairAtoms, NormalizeChemFeatures, iface_valid_filter

# For showing the plot in nglview
#from google.colab import output
#output.enable_custom_widget_manager()
#import nglview as ng
#import ipywidgets as widgets
#from pdbparser.pdbparser import pdbparser

# For downloading files
#from google.colab import files


def generate_descr(model_path, output_path, pdb_file,
                   npy_directory, radius, resolution,supsampling):

    parser = argparse.ArgumentParser(description="Network parameters")

    parser.add_argument("--curvature_scales",type=list,
                        default=[1.0, 2.0, 3.0, 5.0, 10.0])

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--single_pdb",type=str,default=pdb_file)
    parser.add_argument("--embedding_layer",type=str,default="dMaSIF")
    parser.add_argument("--experiment_name", type=str, default=model_path)

    parser.add_argument("--dropout",type=float,default=0.0)
    parser.add_argument("--variance",type=float,default=0.1)
    parser.add_argument("--distance",type=float,default=1.05)
    parser.add_argument("--radius", type=float, default=radius)
    parser.add_argument("--resolution",type=float,default=resolution)

    parser.add_argument("--k",type=int,default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atom_dims",type=int,default=6)
    parser.add_argument("--emb_dims",type=int,default=16)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--post_units",type=int,default=8)

    parser.add_argument("--in_channels",type=int,default=16)
    parser.add_argument("--orientation_units",type=int,default=16)
    parser.add_argument("--unet_hidden_channels",type=int,default=8)
    parser.add_argument("--sup_sampling", type=int, default=supsampling)

    parser.add_argument("--site", type=bool, default=True,
                        help='set to true for site model')
    parser.add_argument("--search",type=bool,default=False,
                        help='Set to true for search model')
    parser.add_argument("--no_chem", type=bool, default=False)
    parser.add_argument("--no_geom", type=bool, default=False)
    parser.add_argument("--single_protein",type=bool,default=True,
                        help='set to false for site')
    parser.add_argument("--use_mesh", type=bool, default=False)
    #parser.add_argument("--single_protein",type=bool,default=True)
    parser.add_argument("--random_rotation",type=bool,default=False)

    args = parser.parse_args("")

    model_path = args.experiment_name
    save_predictions_path = Path(output_path)

    # Ensure reproducability:
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Load the train and test datasets:
    transformations = (
        Compose([NormalizeChemFeatures(), CenterPairAtoms(),
                 RandomRotationPairAtoms()])
        if args.random_rotation else
        Compose([NormalizeChemFeatures()])
    )

    if args.single_pdb != "":
        single_data_dir = Path(npy_directory)
        test_dataset = [load_protein_pair\
                        (args.single_pdb, single_data_dir, single_pdb=True)]
        test_pdb_ids = [args.single_pdb]

    # PyTorch geometric expects an explicit list of "batched variables":
    batch_vars = ["xyz_p1", "xyz_p2", "atom_coords_p1", "atom_coords_p2"]
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, follow_batch=batch_vars
    )

    net = dMaSIF(args)
    checkpoint = torch.load(model_path, map_location=args.device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net = net.to(args.device)

    # Perform one pass through the data:
    info = iterate(net, test_loader, None, args, test=True,
                   save_path=save_predictions_path, pdb_ids=test_pdb_ids)
    return info


def run_dMaSIF():
    # Protonate the pdb file using reduce
    tmp_pdb = '/content/pdbs/tmp_1.pdb'
    shutil.copyfile(target_pdb, tmp_pdb)

    # Remove protons if there are any
    !reduce -Trim -Quiet /content/pdbs/tmp_1.pdb > /content/pdbs/tmp_2.pdb
    # Add protons
    !reduce -HIS -Quiet /content/pdbs/tmp_2.pdb > /content/pdbs/tmp_3.pdb

    tmp_pdb = '/content/pdbs/tmp_3.pdb'
    shutil.copyfile(tmp_pdb, target_pdb)

    #!rm /content/pdbs/tmp_1.pdb /content/pdbs/tmp_2.pdb /content/pdbs/tmp_3.pdb

    # Generate the surface features
    convert_to_npy(target_pdb, chains_dir, npy_dir, chains)

    # Generate the embeddings
    pdb_name = "{n}_{c}_{c}".format(n= target_name, c=chain_name)
    info = generate_descr(model_path, pred_dir, pdb_name, npy_dir,
                          radius, resolution, supsampling)
    # In info I hardcoded memory usage to 0 so MaSIF would run on the CPU. We might want to change this.


def generate_pdb_hotspot_residue():
    list_hotspot_residues = False
    from Bio.PDB.PDBParser import PDBParser
    from scipy.spatial.distance import cdist

    parser=PDBParser(PERMISSIVE=1)
    structure=parser.get_structure("structure", target_pdb)

    coord = np.load("preds/{n}_{c}_predcoords.npy".format
                    (n= target_name, c=chain_name)) # [15386,3]

    embedding = np.load("/content/preds/{n}_{c}_predfeatures_emb1.npy".
                        format(n= target_name, c=chain_name)) # [15386,34]

    atom_coords = np.stack([atom.get_coord() for atom in structure.get_atoms()]) # [1191,3]

    b_factor = embedding[:, -2] # [15386,]
    # b_factor = (b_factor - min(b_factor)) / (max(b_factor) - min(b_factor))

    dists = cdist(atom_coords, coord) # [1191,15386]
    nn_ind = np.argmin(dists, axis=1)# [1191]
    dists = dists[np.arange(len(dists)), nn_ind] # [1191,]
    atom_b_factor = b_factor[nn_ind] # [1191,]
    dist_thresh = 2.0
    atom_b_factor[dists > dist_thresh] = 0.0

    for i, atom in enumerate(structure.get_atoms()):
        atom.set_bfactor(atom_b_factor[i] * 100)

    # Create folder for the embeddings
    pred_dir = '/content/output'
    os.makedirs(pred_dir, exist_ok=True)

    # Save pdb file with per-atom b-factors
    io = PDBIO()
    io.set_structure(structure)
    io.save("/content/output/per_atom_binding.pdb")

    # residuce id of each atom
    atom_residues = np.array([atom.get_parent().id[1]
                              for atom in structure.get_atoms()])
    hotspot_res = {}
    for residue in structure.get_residues():
        res_id = residue.id[1]
        res_b_factor = np.max(atom_b_factor[atom_residues == res_id])
        hotspot_res[res_id] = res_b_factor
        for atom in residue.get_atoms():
            atom.set_bfactor(res_b_factor * 100)

    # Save pdb file with per-residue b-factors
    io = PDBIO()
    io.set_structure(structure)
    io.save("/content/output/per_resi_binding.pdb")

    if list_hotspot_residues:
        print('Sorted on residue contribution (high to low')
        for w in sorted(hotspot_res, key=hotspot_res.get, reverse=True):
            print(w, hotspot_res[w])



def main():
    create_paths()
    run_dMaSIF()
    generate_pdb_hotspot_residue()
    plot_output()
    get_pdb()
