
import csv
import utils
import numpy as np

from Bio.PDB import *
from collections import defaultdict, OrderedDict


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
    resid_classes = utils.classify_residues(resid_bfs, resid_ids, threshold=0)
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
        elif pred == 2:    pred = 0
        res[resid_id] = pred

    res = OrderedDict(sorted(res.items()))
    preds = []
    for k, v in res.items():
        preds.append(v)
    return preds

def score(pred, gt, v1, v2):
    gt_ids = set(np.where(gt==v1)[0])
    pred_ids = set(np.where(pred==v2)[0])
    return len(gt_ids.intersection(pred_ids))

def f1(pred, gt):
    tp = score(pred,gt,1,1)
    fp = score(pred,gt,0,1)
    fn = score(pred,gt,1,0)
    return tp / (tp + .5*(fp+fn))

#def plot_roc():

def evaluate(idr_fn, imask_fn, smask_fn, pdb_fn, ptcld_fn, embd_fn, dist_thresh, resid_thresh):
    gt_ids, gt_classes = gt_atom_to_residue(pdb_fn, imask_fn, smask_fn)
    idr_preds, idr_ids = get_IDR(idr_fn)
    idr_preds = sort_IDR(idr_preds, idr_ids, gt_ids)
    dmasif_preds = utils.get_dmasif(pdb_fn, embd_fn, ptcld_fn, smask_fn, dist_thresh, resid_thresh)

    idr_preds = np.array(idr_preds)
    gt_classes = np.array(gt_classes)
    dmasif_preds = np.array(dmasif_preds)

    print(f1(idr_preds, gt_classes))
    print(f1(dmasif_preds, gt_classes))
    #print(np.where(np.array(gt_classes)==1))
    #print(np.where(np.array(idr_preds)==1))
    #print(np.where(np.array(dmasif_preds)==1))


if __name__ == '__main__':
    idr_fn='data/phase1/IDR.csv'
    imask_fn='data/phase1/5VAY-A-interface-atom-mask.npy'
    smask_fn='data/phase1/5VAY-A-surface-atom-mask.npy'
    pdb_fn='data/phase1/5VAY-A.pdb'
    ptcld_fn='data/phase1/5VAY-A_A_predcoords.npy'
    embd_fn='data/phase1/5VAY-A_A_predfeatures_emb1.npy'
    dist_thresh = 0.7
    resid_thresh=0
    evaluate(idr_fn, imask_fn, smask_fn, pdb_fn, ptcld_fn, embd_fn, dist_thresh, resid_thresh)
