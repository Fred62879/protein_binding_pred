
import numpy as np

from Bio.PDB import *
from scipy.spatial import distance
from collections import defaultdict


def atom_valid(atom):
    return atom.get_parent().id[1] >= 0

''' masking out non-surface atoms
'''
def mask_atom(smask_fn, atom_b_factor):
    if smask_fn is None:
        return

    smask = np.load(smask_fn)

    #l, n = len(smask), len(atom_b_factor)
    #if n > l:
    #    warnings.warn('more atoms than mask entries')
    #elif n < l:
    #    warnings.warn('more mask entries than atoms')
    #smask = smask[:len]
    #atom_b_factor = atom_b_factor[:len]

    atom_b_factor[~smask] = 0.0
    return atom_b_factor #, min(n,l)

def load_results(pdb_fn, embd_fn, ptclds_fn):
    # load original atoms
    parser = PDBParser(PERMISSIVE=True)
    structure = parser.get_structure("structure", pdb_fn)

    # load point cloud predictions
    bfactors = np.load(embd_fn)[:,-2]  # [m,]
    ptcld_coords = np.load(ptclds_fn) # [m,3]

    atom_coords = np.stack\
        ([atom.get_coord() for atom in structure.get_atoms()]) # [n,3]
    return structure, atom_coords, ptcld_coords, bfactors


''' find nearest point for each atom and set b factor accordingly
'''
def point_cloud_to_atom(atom_coords, ptcld_coords, bfactors, structure, threshold):
    dists = distance.cdist(atom_coords, ptcld_coords) # [n,m]
    nn_ind = np.argmin(dists, axis=1) # [n,]

    atom_b_factor = bfactors[nn_ind] # [n,]
    dists = dists[np.arange(len(dists)), nn_ind] # [n,]
    atom_b_factor[dists > threshold] = 0.0
    #for i, atom in enumerate(structure.get_atoms()):
    #    atom.set_bfactor(atom_b_factor[i] * 100)
    return atom_b_factor, structure


''' collect b factor of all atoms for each residue
'''
def atom_to_residue(atom_bfs, smask_fn, structure):
    resid_ids = set()
    resid_bfs = defaultdict(lambda : [])
    atom_bfs = mask_atom(smask_fn, atom_bfs)

    for i, atom in enumerate(structure.get_atoms()):
        #if i >= len: break # end of atoms and/or mask
        #if not atom_valid(atom): continue

        resid_id = atom.get_parent().id[1]
        resid_ids.add(resid_id)
        resid_bfs[resid_id].append(atom_bfs[i])

    resid_ids = np.array(list(resid_ids))
    return resid_bfs, resid_ids


''' classify residuce using bfactos from all its atoms
'''
def classify_one_residue(atoms_bf, threshold):
    # atoms_bf: bfactor for each atom in current residue
    n_atoms = len(atoms_bf)
    atoms_bf = np.array(atoms_bf)

    # no atoms predicted for residue
    if n_atoms == 0: return -1

    atoms_bf = atoms_bf.astype(np.bool8)
    if threshold == 0: # >0 intrfce atom -> intrfce residue
        return int(atoms_bf.any())

    n_intrfce_atoms = sum(atoms_bf)
    return int(n_intrfce_atoms > threshold*n_atoms)

def classify_residues(resid_bfs, resid_ids, threshold):
    return [classify_one_residue(resid_bfs[resid_id], threshold)
            for resid_id in resid_ids]

''' get prediction for each residue of a single pdb id
'''
def get_dmasif(pdb_fn, embd_fn, ptclds_fn, smask_fn, dist_thresh, resid_thresh):

    structure, atom_coords, ptcld_coords, bfactors = load_results\
        (pdb_fn, embd_fn, ptclds_fn)

    atom_bfs, structure = point_cloud_to_atom\
        (atom_coords, ptcld_coords, bfactors, structure, dist_thresh)

    resid_bfs, resid_ids = atom_to_residue(atom_bfs, smask_fn, structure)
    return classify_residues(resid_bfs, resid_ids, resid_thresh)




'''
def get_len(structure):
    return sum([1 for _ in structure.get_atoms()])

def get_resid_ids(pdb_fn):
    parser = PDBParser(PERMISSIVE=True)
    structure = parser.get_structure("structure", pdb_fn)

    resid_ids = set()
    for i, atom in enumerate(structure.get_atoms()):
      id = atom.get_parent().id[1]
      resid_ids.add(id)
    return np.array(list(resid_ids))

def get(pdb, get_len=False, get_atom_ids=False, get_resid_ids=False, get_atom_bfactors=False, get_resid_bfactors=False, get_resids=False):
  parser = PDBParser(PERMISSIVE=True)
  structure = parser.get_structure("structure", pdb)
  atoms = structure.get_atoms()
  if get_len: print(sum(1 for _ in atoms)) #

  if get_resid_bfactors:
    resid_bfactors = []
    for i, resid in enumerate(structure.get_residues()):
      resid_bfactors.append(resid.get_bfactor())
    return np.array(resid_bfactors)

  if get_atom_bfactors:
    atom_bfactors = []
    for i, atom in enumerate(structure.get_atoms()):
      atom_bfactors.append(atom.get_bfactor())
    return np.array(atom_bfactors)

  if get_atom_ids:
    atom_ids = []
    for i, atom in enumerate(structure.get_atoms()):
      atom_ids.append(atom.get_id())
    return np.array(atom_ids)


  if get_resids:
    resids = []
    for i, resid in enumerate(structure.get_residues()):
      resids.append(resid.get_id())
    return np.array(resids)
'''
