
from os import listdir
from utils import make_dirs
from os.path import join, isfile


def parse_all_pdbs(config):

    roc_nm = 'roc.png'
    atom_bd_nm = '_atom_binding.pdb'
    embd_nm = '_predfeatures_emb1.npy'
    point_cloud_nm = '_predcoords.npy'
    resid_bd_nm = '_residue_binding.pdb'
    sufce_mask_sufx = '-surface-atom-mask.npy'
    itfce_mask_sufx = '-interface-atom-mask.npy'

    roc_fns = []
    atom_bd_fns, resid_bd_fns = [], []
    pdb_fns, chains, pdb_nms = [], [], set()
    smask_fns, imask_fns, embd_fns, ptcld_fns = [], [], [], []

    fns = listdir(config['dataset_dir']) # include all pdb files

    for pdb_fn in fns:

        # check file validity
        if not isfile(join(config['dataset_dir'], pdb_fn)) \
           or 'tmp' in pdb_fn: continue

        # check if current pdb nm is visited before
        parts = pdb_fn.split('.')
        pdb_nm = parts[0].split('-')[0]
        pdb_chains = parts[0].split('-')[1] # all chains for current pdb
        pdb_nm += '-' + pdb_chains
        if pdb_nm in pdb_nms: continue

        # add all filenames
        fn_prefx = join(config['dataset_dir'], pdb_nm)
        roc_fn = join(config['roc_dir'], pdb_nm + '_' + roc_nm)
        embd_fn = join(config['preds_dir'], pdb_nm + '_' + pdb_chains + embd_nm)
        ptcld_fn = join(config['preds_dir'], pdb_nm + '_' + pdb_chains + point_cloud_nm)
        atom_bd_fn = join(config['outputs_dir'], pdb_nm + '_' + pdb_chains + atom_bd_nm)
        resid_bd_fn = join(config['outputs_dir'], pdb_nm + '_' + pdb_chains + resid_bd_nm)

        pdb_nms.add(pdb_nm)
        roc_fns.append(roc_fn)
        embd_fns.append(embd_fn)
        ptcld_fns.append(ptcld_fn)
        chains.append([pdb_chains])
        atom_bd_fns.append(atom_bd_fn)
        resid_bd_fns.append(resid_bd_fn)
        pdb_fns.append(fn_prefx + '.pdb')
        smask_fns.append(fn_prefx + sufce_mask_sufx)
        imask_fns.append(fn_prefx + itfce_mask_sufx)

    config['chains'] = chains
    config['pdb_fns'] = pdb_fns
    config['roc_fns'] = roc_fns
    config['embd_fns'] = embd_fns
    config['ptcld_fns'] = ptcld_fns
    config['smask_fns'] = smask_fns
    config['imask_fns'] = imask_fns
    config['pdb_nms'] = list(pdb_nms)
    config['atom_binding_fns'] = atom_bd_fns
    config['resid_binding_fns'] = resid_bd_fns


def parse_general(parser, experiment_id):

    # EITHER load either all arguments from a config file
    parser.add('-c', '--config', required=False, is_config_file=True)

    # OR specify each as script arguments
    parser.add_argument('--phase', type=int, required=True)
    parser.add_argument('--experiment_id', type=int, required=experiment_id)

    # I) original dmasif arguments
    parser.add_argument('--device', type=str, default='cpu')
    #parser.add_argument('--single_pdb',type=str, default='5VAY-A_A_A.pdb')
    #parser.add_argument('--target_pdb',type=str, default='5VAY-A_A_A.pdb')
    parser.add_argument('--embedding_layer',type=str, default='dMaSIF')

    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--variance', type=float, default=0.1)
    parser.add_argument('--distance', type=float, default=1.05)

    parser.add_argument('--k', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--atom_dims', type=int, default=6)
    parser.add_argument('--emb_dims', type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--post_units', type=int, default=8)
    parser.add_argument('--in_channels', type=int, default=16)
    parser.add_argument('--orientation_units', type=int, default=16)
    parser.add_argument('--unet_hidden_channels', type=int, default=8)
    parser.add_argument("--curvature_scales",type=list,default=[1.0, 2.0, 3.0, 5.0, 10.0])

    parser.add_argument('--site', type=bool, default=True,
                        help='set to true for site model')
    parser.add_argument('--search',type=bool,default=False,
                        help='Set to true for search model')
    parser.add_argument('--no_chem', type=bool, default=False)
    parser.add_argument('--no_geom', type=bool, default=False)
    parser.add_argument('--single_protein',type=bool, default=True,
                        help='set to false for site')
    parser.add_argument('--use_mesh', type=bool, default=False)
    parser.add_argument('--random_rotation', type=bool, default=False)


    # II) threshold and choices for point cloud -> atom & atom -> residue coversion

    ## atom b factor estimation parameter
    parser.add_argument('--atom_k', type=float, default=5,
                        help='knn to estimate atom bfactor')
    parser.add_argument('--atom_bf_thresh', type=float, default=1.5,
                        help='dist threshold for point cloud->atom')
    parser.add_argument('--atom_clas_thresh', type=float, default=0.5,
                        help='threshold to classify atoms')

    ## residue interface classification parameter
    parser.add_argument('--resid_clas_cho', type=float, default=1,
                        help='choice to classify residue')
    parser.add_argument('--resid_clas_ratio', type=float, default=0.5,
                        help='ratio*100% interface atoms -> interface residuce ')

    args = parser.parse_args()

    # add new arguments
    config = vars(args)

    # hardcoded filenames and dirs
    f1_fn_str = 'f1.npy'
    auc_fn_str = 'auc.npy'

    roc_dir_str = 'roc'
    npy_dir_str = 'npys'
    preds_dir_str = 'preds'
    chains_dir_str = 'chains'
    idr_output_str = 'IDR.csv'
    outputs_dir_str = 'outputs'
    dataset_dir_str = 'dataset'
    model_dir_str = './MaSIF_colab/models'

    # setup current experiment
    radii = [9,9,12,12]
    resolutions = [1, 0.7, 1, 0.7]
    sup_samplings = [100, 150, 100, 150]
    models = ['dMaSIF_site_3layer_16dims_9A_100sup_epoch64',
              'dMaSIF_site_3layer_16dims_9A_0.7res_150sup_epoch85',
              'dMaSIF_site_3layer_16dims_12A_100sup_epoch71',
              'dMaSIF_site_3layer_16dims_12A_0.7res_150sup_epoch59']

    config['radius'] = radii[experiment_id]
    config['resolution'] = resolutions[experiment_id]
    config['sup_sampling'] = sup_samplings[experiment_id]
    config['model_path'] = join(model_dir_str, models[experiment_id])

    experiment_name = models[experiment_id][26:]

    # add filename and directories
    data_dir = join('/content/data', 'phase' + str(args.phase))
    exp_dir = join(data_dir, experiment_name)

    config['f1_fn'] = join(exp_dir, f1_fn_str)
    config['auc_fn'] = join(exp_dir, auc_fn_str)

    config['roc_dir'] = join(exp_dir, roc_dir_str)
    config['npy_dir'] = join(exp_dir, npy_dir_str)
    config['preds_dir'] = join(exp_dir, preds_dir_str)
    config['chains_dir'] = join(exp_dir, chains_dir_str)
    config['outputs_dir'] = join(exp_dir, outputs_dir_str)
    config['dataset_dir'] = join(data_dir, dataset_dir_str)

    make_dirs(data_dir, overwrite=False)
    make_dirs(config['roc_dir'], overwrite=False)
    make_dirs(config['npy_dir'], overwrite=False)
    make_dirs(config['preds_dir'], overwrite=False)
    make_dirs(config['chains_dir'], overwrite=False)
    make_dirs(config['outputs_dir'], overwrite=False)
    make_dirs(config['dataset_dir'], overwrite=False)

    if config['phase'] == 1:
        config['idr_fn'] = join(data_dir, idr_output_str)
        config['idr_roc_fn'] = join(data_dir, 'idr_roc.png')
        config['dmasif_roc_fn'] = join(data_dir, 'dmasif_roc.png')

    parse_all_pdbs(config)
    return config
