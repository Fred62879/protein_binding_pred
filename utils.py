
import csv
import numpy as np

from Bio.PDB import *
from scipy.spatial import distance
from collections import defaultdict, OrderedDict


###################
# general utilities

def gather_fns(pdb_dir):
    ids = []
    for fn in fns:
        if isfile(join('/content/pdbs',fn)):
            ids.append(fn.split('-')[0])
    ids = np.array(ids)

    chains = []
    for fn in fns:
        if isfile(join('/content/pdbs',fn)):
            chains.append(fn.split('-')[1].split('.')[0])
    chains = np.array(chains)

    pdb_fns = [id+'-'+chain+'.pdb'
               for id,chain in zip(ids,chains)]

    ptcld_fns = [pdf_fn+'_'+chain+'_predcoords.npy'
                 for pdf_fn, chain in zip(pdf_fns, chains)]

    embd_fns = [pdf_fn+'_'+chain+'_predfeatures_emb1.npy'
                for pdf_fn, chain in zip(pdf_fns, chains)]
    return pdb_fns, ptcld_fns, embd_fns


###########################
# atom and residue utilies

def atom_valid(atom):
    return atom.get_parent().id[1] >= 0

''' masking out non-surface atoms
'''
def mask_atom(smask_fn, atom_b_factor):
    if smask_fn is None:
        return atom_b_factor

    smask = np.load(smask_fn)
    atom_b_factor[~smask] = 0.0
    return atom_b_factor

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

def gt_atom_to_residue(pdb_fn, imask_fn, smask_fn):
    imask = np.load(imask_fn)
    smask = np.load(smask_fn)
    parser = PDBParser(PERMISSIVE=True)
    structure = parser.get_structure("structure", pdb_fn)

    imask[~smask] = 0
    resid_ids = set()
    resid_bfs = defaultdict(lambda : [])
    for i, atom in enumerate(structure.get_atoms()):
        resid_id = atom.get_parent().id[1]
        resid_ids.add(resid_id)
        resid_bfs[resid_id].append(int(imask[i]))

    resid_ids = np.array(list(resid_ids))
    resid_classes = classify_residues(resid_bfs, resid_ids, threshold=0)
    return resid_ids, resid_classes

def get_IDR(idr_pred_fn):
    preds, resid_ids = [], []

    with open(idr_pred_fn) as fp:
        reader = csv.reader(fp, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0: continue
            preds.append(int(row[-1]))
            resid_ids.append(int(row[0].split()[-1]))

    return preds, resid_ids

def sort_IDR(preds, resid_ids, gt_resid_ids):
    res = {}
    for gt_resid_id in gt_resid_ids:
        res[gt_resid_id] = -1

    for pred, resid_id in zip(preds, resid_ids):
        # ignore ids not in gt
        if not resid_id in gt_resid_ids:
            continue

        # convert idr result to 0/1/-1
        if pred == -99.88: pred = -1
        elif pred == 2:    pred = 1
        res[resid_id] = pred

    res = OrderedDict(sorted(res.items()))
    preds = []
    for k, v in res.items():
        preds.append(v)
    return preds


#######################
# evaluation utilities

def evaluate(idr_resid_bind, dmasif_resid_bind, gt_resid_bind, gt_resid_ids):

    valid_idx1 = set( np.where(idr_resid_bind != -1)[0] )
    valid_idx2 = set( np.where(dmasif_resid_bind != -1)[0] )
    valid_idxs = np.array(list(valid_idx1.intersection(valid_idx2)))

    residue_ids = gt_resid_ids[valid_idxs]
    gt_resid_bind = gt_resid_bind[valid_idxs]
    idr_resid_bind = idr_resid_bind[valid_idxs]
    dmasif_resid_bind = dmasif_resid_bind[valid_idxs]

    print(f1(idr_resid_bind, gt_resid_bind))
    print(f1(dmasif_resid_bind, gt_resid_bind))
    #print(f1(dmasif_preds, gt_classes))
    #print(np.where(np.array(gt_classes)==1))
    #print(np.where(np.array(idr_preds)==1))
    #print(np.where(np.array(dmasif_preds)==1))


def confusion_score(pred, gt, v1, v2):
    gt_ids = set(np.where(gt==v1)[0])
    pred_ids = set(np.where(pred==v2)[0])
    return len(gt_ids.intersection(pred_ids))

def f1(pred, gt):
    tp = confusion_score(pred,gt,1,1)
    fp = confusion_score(pred,gt,0,1)
    fn = confusion_score(pred,gt,1,0)
    return tp / (tp + .5*(fp+fn))

#def plot_roc():



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
