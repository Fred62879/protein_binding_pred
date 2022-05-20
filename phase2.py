
import utils
import numpy as np

from os.path import join, isfile


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


def batch_analysis(pdb_dir):
    pdb_fns, ptcld_fns, embd_fns = gather_fns(pdb_dir)


if __name__ == '__main__':
    batch_analysis('data/phase2')
