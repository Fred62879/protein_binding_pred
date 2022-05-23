
import csv
import numpy as np
import matplotlib.pyplot as plt

from Bio.PDB import *
from os.path import exists
from os import remove, makedirs
from scipy.spatial import distance
from collections import defaultdict, OrderedDict
from sklearn.metrics import roc_curve, roc_auc_score

'''
def upload_pdb_to_dataset(args):
  assert(exists(args.dataset_dir))

  dir = args.dataset_dir
  %cd -q {dir}
  uploaded = files.upload()
  %cd -q /content
'''

def make_dirs(dir):
    isExist = exists(dir)
    if not isExist:
      makedirs(dir)
    else:
      files = glob.glob(dir + '/*')
      for f in files:
        remove(f)

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
    #print('smask', smask.astype(np.int8)[:100])
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
    find all nearest points with dist below threshold and average
    at most k of these
'''
def estimate_atom_bfactor(atom_coords, ptcld_coords, bfactors, smask_fn, args):

    dists = distance.cdist(atom_coords, ptcld_coords) # [n,m]
    n, m = dists.shape

    # find k nearest point for each atom
    knn_id = np.argsort(dists, axis=1)[:,:args.atom_k] # [n,k]
    rids = np.arange(0, n).repeat(args.atom_k).reshape((n,-1))
    knn_dists = dists[rids, knn_id] # [n,k]
    knn_bfactors = bfactors[knn_id] # [n,k]

    # find points from above that are within distance threshold for each atom
    print(np.round(knn_dists.T[:50,0],2))
    knn_valid = (knn_dists < args.atom_bf_thresh).astype(np.int8)
    knn_bfactors *= knn_valid

    # average over bfactors for these points
    knn_counts = np.sum(knn_valid, axis=1)
    knn_bfactors = np.sum(knn_bfactors,axis=1)

    knn_counts[knn_counts==0] = 1    # avoid division by zero
    knn_bfactors[knn_counts==0] = -1 # atoms with no estimations
    atom_bfactors = knn_bfactors / knn_counts

    # mask out non-surface atoms
    print('before mask', np.count_nonzero(atom_bfactors))
    atom_bfactors = mask_atom(smask_fn, atom_bfactors)
    print('after mask', np.count_nonzero(atom_bfactors))
    np.save('../data/phase1/pred_atom.npy',atom_bfactors)
    return atom_bfactors

''' classify atom as interface(1)/non-interface(0)/unknown(-1)
    based on estimated bfactor
'''
def classify_atom(atom_bfs, args):
    no_pred_ids = atom_bfs==-1
    atom_classes = (atom_bfs > args.atom_clas_thresh).astype(np.int8)
    atom_classes[no_pred_ids] = -1
    return atom_classes


''' convert point cloud bfactor to atom bfactor
'''
def point_cloud_to_atom(atom_coords, ptcld_coords, bfactors, smask_fn, args):
    atom_bfs = estimate_atom_bfactor\
        (atom_coords, ptcld_coords, bfactors, smask_fn)
    atom_classes = classify_atom(atom_bfs, args)
    return atom_bfs


''' collect b factor of all atoms for each residue
'''
def atom_to_residue(atom_bfs, structure):
    resid_ids = set()
    resid_bfs = defaultdict(lambda : [])

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
def estimate_resid_bf(atoms_bf, args):
    cho = args.resid_bf_cho

    if len(atoms_bf) == 0: # no atoms estimated for cur residue
        res = -1
    elif cho == 0: # max of all atom bfactor
        res = max(atoms_bf)
    elif cho == 1: # mean of all atom bfactor
        res = np.mean(atoms_bf)
    elif cho == 2: # averge of largest k
        atoms = np.sort(atoms_bf)
        res = np.mean(atoms[-args.resid_bf_k:])
    else:
        raise Exception('Unsupported residue bfactor estimation choice')
    return res


''' classify one residue as interface(1)/non-interface(0)/unknown(-1)
    based on its estimated bfactor
'''
def classify_residue(resid_bf, args):
    cho = args.resid_clas_cho

    if resid_bf == -1: # no atoms predicted for currentresidue
        res = -1
    elif cho == 0: # >0 intrfce atom -> intrfce residue
        res = int(resid_bf > 0)
    elif cho == 1: # resid bfactor larger than thresh -> inttrfce residue
        res = int(resid_bf > args.resid_thresh)
    else:
        raise Exception('Unsupported residue classification choice')
    return res


''' estimate bfactors and classify all given residues
'''
def classify_residues(resid_bfs, resid_ids, args):

    classes, est_bfs = [], []

    for i, resid_id in enumerate(resid_ids):
        est_bf = estimate_resid_bf(resid_bfs[resid_id], args)
        clas = classify_residue(est_bf, args)
        classes.append(clas)
        est_bfs.append(est_bf)

    print('=====bfactors', np.round(est_bfs,1))
    print()
    print('=====classes', classes)
    print()
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
        (resid_bfs, resid_ids, resid_bf_cho=0,
         resid_bf_k=None, clas_cho=0, resid_thresh=None)

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
