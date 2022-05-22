
# MaSIF, pykeops,plyfile,pyvtk

#import os
#os.chdir('/media/fred/Local Disk/Bioinfo/code')

import sys
sys.path.append("./MaSIF_colab")
sys.path.append("./MaSIF_colab/data_preprocessing")

import os
import glob
import torch
import utils
import shutil
import pykeops
import argparse
import warnings
import numpy as np

from Bio.PDB import *
from os.path import isfile, join
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
#from tqdm.notebook import tqdm


def generate_point_cloud_binding(pdb_fn, pdb_nm, chains, args):

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
    # net.load_state_dict(torch.load(args.model_path, map_location=args.device))
    net.load_state_dict(torch.load(args.model_path, map_location=args.device)["model_state_dict"])
    net = net.to(args.device)

    # Perform one pass through the data:
    info = iterate(net, test_loader, None, args, test=True,
                   save_path=Path(output_path), pdb_ids=test_pdb_ids)
    return info


''' predict binding for each residue
'''
def generate_residue_binding(pdb_fn, smask_fn, embd_fn, ptcld_fn, dist_thresh, resid_thresh):

    structure, atom_coords, ptcld_coords, bfactors =\
        utils.load_results(pdb_fn, embd_fn, ptcld_fn)

    atom_bfs, structure = utils.point_cloud_to_atom\
        (atom_coords, ptcld_coords, bfactors, structure, dist_thresh)

    resid_bfs, resid_ids = utils.atom_to_residue(atom_bfs, smask_fn, structure)
    pred_classes = utils.classify_residues(resid_bfs, resid_ids, resid_thresh)
    return pred_classes

''' run dmasif on given pdb and return  binding for each residue
'''
def run_dmasif(pdb_nm, chains, pdb_fn, smask_fn, embd_fn,
               ptcld_fn, dist_thresh, resid_thresh, args):

    generate_point_cloud_binding(pdb_fn, pdb_nm, chains, args)
    resid_binding = generate_residue_binding\
        (pdb_fn, smask_fn, embd_fn, ptcld_fn, dist_thresh, resid_thresh)

    return resid_binding

'''
def run(target_pdb, target_name, chain_name, chains):
  # Protonate the pdb file using reduce
  tmp_pdb = '/content/pdbs/tmp_1.pdb'
  shutil.copyfile(target_pdb, tmp_pdb)

  # Remove protons if there are any
  !reduce -Trim -Quiet /content/pdbs/tmp_1.pdb > /content/pdbs/tmp_2.pdb
  # Add protons
  !reduce -HIS -Quiet /content/pdbs/tmp_2.pdb > /content/pdbs/tmp_3.pdb

  tmp_pdb = '/content/pdbs/tmp_3.pdb'
  shutil.copyfile(tmp_pdb, target_pdb)

  # Generate the surface features
  convert_to_npy(target_pdb, chains_dir, npy_dir, chains)

  # Generate the embeddings
  pdb_name = "{n}_{c}_{c}".format(n= target_name, c=chain_name)
  info = generate_point_cloud_binding(model_path, pred_dir, pdb_name, npy_dir, radius, resolution, supsampling)
  # In info I hardcoded memory usage to 0 so MaSIF would run on the CPU. We might want to change this.
  generate_residue_binding()

def run_all():
  fns = os.listdir('/content/pdbs')

  ids = []
  for fn in fns:
    if isfile(join('/content/pdbs',fn)) and not 'tmp' in fn:
      ids.append(fn.split('-')[0])
  ids = np.array(ids)

  chains = []
  for fn in fns:
     if isfile(join('/content/pdbs',fn)) and not 'tmp' in fn:
       chains.append(fn.split('-')[1].split('.')[0])
  chains = np.array(chains)

  for id, chain in zip(ids, chains):
    print(id, chain)
    target_pdb = 'pdbs/'+id+'-'+chain+'.pdb'
    target_pdb = "pdbs/5VAY-A.pdb"

    chains = [chain]
    run_each(target_pdb, id, chain, chains)
'''
