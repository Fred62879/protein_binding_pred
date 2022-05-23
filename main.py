
import utils
import argparse
import numpy as np
import custom_parser
import configargparse

from dMaSIF import run_dmasif


''' run both dmasif and idr on the same pdb and compare with gt
'''
def run_phase1(args):

    dmasif_resid_bind = run_dmasif\
        (args.pdb_nms[0], args.chains[0], args.pdb_fns[0],
         args.smask_fns[0], args.embd_fns[0], args.ptcld_fns[0],
         args.atom_binding_fns[0], args.resid_binding_fns[0], args)

    gt_ids, gt_resid_bind = utils.gt_atom_to_residue\
        (args.pdb_fns[0], args.imask_fns[0], args.smask_fns[0], args)

    idr_resid_bind, idr_ids = utils.get_IDR(args.idr_fn)
    idr_resid_bind = utils.sort_IDR(idr_resid_bind, idr_ids, gt_ids)

    utils.evaluate_phase1(idr_resid_bind, dmasif_resid_bind, gt_resid_bind,
                          gt_ids, args.idr_roc_fn, args.dmasif_roc_fn)


''' predict residue binding for all specified pdb using dmasif
'''
def run_phase2(args):
    f1s, aucs = {}, {}

    for i, pdb_fn in enumerate(args.pdb_fns):
        dmasif_resid_bind = run_dmasif\
            (args.pdb_nms[i], args.chains[i], pdb_fn, args.smask_fns[i],
             args.embd_fns[i], args.ptcld_fns[i], args.atom_binding_fns[i],
             args.resid_binding_fns[i], args)

        gt_resid_ids, gt_resid_bind = utils.gt_atom_to_residue\
            (args.pdb_fns[i], args.imask_fns[i], args.smask_fns[i], args)

        f1, auc = utils.evaluate_phase2\
            (dmasif_resid_bind, gt_resid_bind, gt_resid_ids,
             args.roc_fns[i])

        f1s[pdb_fn] = f1
        aucs[pdb_fn] = auc

    utils.save_metrics(f1s, args.f1_fn)
    utils.save_metrics(aucs, args.auc_fn)


if __name__ == '__main__':

    parser = configargparse.ArgumentParser()
    config = custom_parser.parse_general(parser)
    args = argparse.Namespace(**config)

    print('=== phase{} experiment {} ==='.
          format(args.phase, args.experiment_id))

    if args.phase == 1: #5, 1.5, 0.5, 1, 0.5
        run_phase1(args)
    elif args.phase == 2:
        run_phase2(args)
    else:
        raise Exception(' Unsupported phase')
