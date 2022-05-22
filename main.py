
import csv
import utils
import argparse
import numpy as np
import custom_parser
import configargparse

from Bio.PDB import *
from dMaSIF import run_dmasif
from os.path import join, isfile
from collections import defaultdict, OrderedDict


''' run both dmasif and idr on the same pdb and compare with gt
'''
def run_phase1(args):
    dmasif_resid_bind = run_dmasif\
        (args.pdb_nms[0], args.chains[0], args.pdb_fns[0],
         args.smask_fns[0], args.embd_fns[0], args.ptcld_fns[0],
         args.dist_thresh, args.resid_thresh, args)

    gt_ids, gt_resid_bind = utils.gt_atom_to_residue(args.pdb_fns[0], args.imask_fns[0], args.smask_fns[0])
    idr_resid_bind, idr_ids = utils.get_IDR(args.idr_fn)
    idr_resid_bind = utils.sort_IDR(idr_resid_bind, idr_ids, gt_ids)

    gt_resid_bind = np.array(gt_resid_bind)
    idr_resid_bind = np.array(idr_resid_bind)
    dmasif_resid_bind = np.array(dmasif_resid_bind)
    utils.evaluate(idr_resid_bind, dmasif_resid_bind, gt_resid_bind, gt_ids)


''' predict residue binding for all specified pdb using dmasif
'''
def run_phase2(args):

    for i, pdb_fn in enumerate(args.pdb_fns):
        dmasif_resid_bind = run_dmasif\
            (pdb_nm, chain_nm, pdb_fn, smask_fn, embd_fn,
             ptcld_fn, args.dist_thresh, args.resid_thresh, args)



if __name__ == '__main__':

    parser = configargparse.ArgumentParser()
    config = custom_parser.parse_general(parser)
    args = argparse.Namespace(**config)

    if args.phase == 1:
        run_phase1(args)
    elif args.phase == 2:
        run_phase2(args)
    else:
        raise Exception(' Unsupported phase')
