
import csv
import numpy as np
import matplotlib.pyplot as plt

from Bio.PDB import *
from scipy.spatial import distance
from collections import defaultdict, OrderedDict
from sklearn.metrics import roc_curve, roc_auc_score


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


''' estimate bfactor for each atom based on point cloud points.
    average k nearest neighbour
    TODO: find all nearest points with dist below threshold and average
    at most k of these
'''
def estimate_atom_bfactor(dists, bfactors, threshold, k=1):
    knn_id = np.argsort(dists, axis=1)[:,:k] # [n,k]
    atom_b_factor = np.mean(bfactors[knn_id], axis=1) # [n,]
    return atom_b_factor


''' convert point cloud bfactor to atom bfactor
'''
def point_cloud_to_atom(atom_coords, ptcld_coords, bfactors, threshold, k):

    dists = distance.cdist(atom_coords, ptcld_coords) # [n,m]
    atom_bfs = estimate_atom_bfactor(dists, bfactors, threshold, k)

    '''
    nn_ind = np.argmin(dists, axis=1) # [n,]
    atom_b_factor = bfactors[nn_ind] # [n,]
    dists = dists[np.arange(len(dists)), nn_ind] # [n,]
    atom_b_factor[dists > threshold] = 0.0
    '''
    #for i, atom in enumerate(structure.get_atoms()):
    #    atom.set_bfactor(atom_b_factor[i] * 100)
    return atom_bfs


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


''' estimate bfactor for residuce based on bfactor for all its atoms
'''
def estimate_resid_bf(atoms_bf, cho, k=1):
    if len(atoms_bf) == 0:
        res = -1
    elif cho == 0:
        res = max(atoms_bf)
    elif cho == 1:
        res = np.mean(atoms_bf)
    elif cho == 2: # averge of largest k
        atoms = np.sort(atoms_bf)
        res = np.mean(atoms[-k:])
    else:
        raise Exception('Unsupported residue bfactor estimation choice')
    return res

def classify_residue(resid_bf, cho, threshold=0.0):
    if resid_bf == -1: # no atoms predicted for residue
        res = -1
    elif cho == 0: # >0 intrfce atom -> intrfce residue
        res = resid_bf > 0
    elif cho == 1:
        res = resid_bf > threshold
    else:
        raise Exception('Unsupported residue classification choice')
    return res

def classify_residues(resid_bfs, resid_ids, bf_cho, clas_cho, threshold, k):
    classes, est_bfs = [], []

    for i, resid_id in enumerate(resid_ids):
        est_bf = estimate_resid_bf(resid_bfs[resid_id], bf_cho, k)
        clas = classify_residue(est_bf, clas_cho, threshold)
        classes.append(clas)
        est_bfs.append(est_bf)

    return classes, est_bfs

def save_atom_binding(atom_bfs, structure, atom_binding_fn):
    for i, atom in enumerate(structure.get_atoms()):
        atom.set_bfactor(atom_bfs[i])
    io = PDBIO()
    io.set_structure(structure)
    io.save(atom_binding_fn)

def save_resid_binding(resid_bfs, structure, resid_binding_fn):
    for i, resid in enumerate(structure.get_residues()):
        for atom in structure.get_atoms():
            atom.set_bfactor(resid_bfs[i])

    io = PDBIO()
    io.set_structure(structure)
    io.save(resid_binding_fn)

def gt_atom_to_residue(pdb_fn, imask_fn, smask_fn, args):
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
    resid_classes, _ = classify_residues\
        (resid_bfs, resid_ids, bf_cho=0, clas_cho=0, threshold=None, k=None)

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

def evaluate(idr_resid_bind, dmasif_resid_bind, gt_resid_bind,
             gt_resid_ids, idr_roc_fn, dmasif_roc_fn):

    gt_resid_bind = np.array(gt_resid_bind)
    idr_resid_bind = np.array(idr_resid_bind)
    dmasif_resid_bind = np.array(dmasif_resid_bind)

    valid_idx1 = set( np.where(idr_resid_bind != -1)[0] )
    valid_idx2 = set( np.where(dmasif_resid_bind != -1)[0] )
    valid_idxs = np.array(list(valid_idx1.intersection(valid_idx2)))

    residue_ids = gt_resid_ids[valid_idxs]
    gt_resid_bind = gt_resid_bind[valid_idxs]
    idr_resid_bind = idr_resid_bind[valid_idxs]
    dmasif_resid_bind = dmasif_resid_bind[valid_idxs]

    print(f1(idr_resid_bind, gt_resid_bind))
    print(f1(dmasif_resid_bind, gt_resid_bind))

    print(roc_auc(idr_resid_bind, gt_resid_bind))
    print(roc_auc(dmasif_resid_bind, gt_resid_bind))

    roc_curves(idr_resid_bind, gt_resid_bind, idr_roc_fn)
    roc_curves(dmasif_resid_bind, gt_resid_bind, dmasif_roc_fn)


def confusion_score(pred, gt, v1, v2):
    gt_ids = set(np.where(gt==v1)[0])
    pred_ids = set(np.where(pred==v2)[0])
    return len(gt_ids.intersection(pred_ids))

def f1(pred, gt):
    tp = confusion_score(pred,gt,1,1)
    fp = confusion_score(pred,gt,0,1)
    fn = confusion_score(pred,gt,1,0)
    return tp / (tp + .5*(fp+fn))

def roc_auc(pred, gt):
    return roc_auc_score(gt, pred)

def roc_curves(pred, gt, fn):
    fpr, tpr, _ = roc_curve(gt, pred)
    plt.plot(fpr, tpr)
    plt.savefig(fn)
    plt.close()
