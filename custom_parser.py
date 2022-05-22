
from os import listdir
from os.path import join, isfile


def parse_all_pdbs(config):

    atom_bd_nm = '_atom_binding.pdb'
    embd_nm = '_predfeatures_emb1.npy'
    point_cloud_nm = '_predcoords.npy'
    resid_bd_nm = '_residue_binding.pdb'
    sufce_mask_sufx = '-surface-atom-mask.npy'
    itfce_mask_sufx = '-interface-atom-mask.npy'

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
        embd_fn = join(config['preds_dir'], pdb_nm + '_' + pdb_chains + embd_nm)
        ptcld_fn = join(config['preds_dir'], pdb_nm + '_' + pdb_chains + point_cloud_nm)
        atom_bd_fn = join(config['outputs_dir'], pdb_nm + '_' + pdb_chains + atom_bd_nm)
        resid_bd_fn = join(config['outputs_dir'], pdb_nm + '_' + pdb_chains + resid_bd_nm)

        pdb_nms.add(pdb_nm)
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
    config['embd_fns'] = embd_fns
    config['ptcld_fns'] = ptcld_fns
    config['smask_fns'] = smask_fns
    config['imask_fns'] = imask_fns
    config['pdb_nms'] = list(pdb_nms)
    config['atom_binding_fns'] = atom_bd_fns
    config['resid_binding_fns'] = resid_bd_fns


def parse_general(parser):
    # load either all arguments from a config file
    parser.add('-c', '--config', required=False, is_config_file=True)

    # or specify each as script arguments
    parser.add_argument('--phase', type=int, required=True)

    parser.add_argument('--device', type=str, default='cpu')
    #parser.add_argument('--single_pdb',type=str, default='5VAY-A_A_A.pdb')
    #parser.add_argument('--target_pdb',type=str, default='5VAY-A_A_A.pdb')
    parser.add_argument('--embedding_layer',type=str, default='dMaSIF')

    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--variance', type=float, default=0.1)
    parser.add_argument('--distance', type=float, default=1.05)
    parser.add_argument('--resolution', type=float, default=0.7,
                        help='0.7 -> higher point cloud density & performance') # /1

    # thresholding and choices
    parser.add_argument('--resid_bf_cho', type=float, default=0.0,
                        help='0-max, 1-mean, 2-mean of k largest')
    parser.add_argument('--atom_k', type=float, default=5,
                        help='knn for atom bfactor')
    parser.add_argument('--resid_bf_k', type=float, default=0.0,
                        help='k to calculate residuce bfactor, resid_bf_cho==2')
    parser.add_argument('--clas_cho', type=float, default=0.0,
                        help='choice to classify residue')
    parser.add_argument('--atom_thresh', type=float, default=1.0,
                        help='dist threshold to for point cloud->atom, TODO')
    parser.add_argument('--resid_thresh', type=float, default=0.0,
                        help='threshold to classify residue, when clas_cho==1')

    parser.add_argument('--k', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--radius', type=int,  default=9,
                        help='patch radius, settings doesnt impact performance?')
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
    #parser.add_argument('--single_protein',type=bool,default=True)
    parser.add_argument('--random_rotation', type=bool, default=False)

    args = parser.parse_args()

    # add new arguments
    config = vars(args)

    # hardcode strings
    npy_dir_str = 'npys'
    preds_dir_str = 'preds'
    chains_dir_str = 'chains'
    idr_output_str = 'IDR.csv'
    outputs_dir_str = 'outputs'
    dataset_dir_str = 'dataset'
    model_dir_str = './MaSIF_colab/models'
    data_dir = '../data/phase' + str(args.phase)

    if args.phase == 1:
        config['idr_fn'] = join(data_dir, idr_output_str)
    elif args.phase == 2:
        pass
    else:
        raise Exception('Unsupported phase')

    if args.radius == 9:
        if args.resolution == 1:
            model_path = join(model_dir_str, 'dMaSIF_site_3layer_16dims_9A_100sup_epoch64')
            sup_sampling = 100
        else:
            model_path = join(model_dir_str, 'dMaSIF_site_3layer_16dims_9A_0.7res_150sup_epoch85')
            supsampling = 150

    elif args.radius == 12:
        if args.resolution == 1:
            model_path = join(model_dir_str, 'dMaSIF_site_3layer_16dims_12A_100sup_epoch71')
            supsampling = 100
        else:
            model_path = join(model_dir_str, 'dMaSIF_site_3layer_16dims_12A_0.7res_150sup_epoch59')
            supsampling = 100

    config['model_path'] = model_path
    config['sup_sampling'] = supsampling

    config['npy_dir'] = join(data_dir, npy_dir_str)
    config['preds_dir'] = join(data_dir, preds_dir_str)
    config['chains_dir'] = join(data_dir, chains_dir_str)
    config['outputs_dir'] = join(data_dir, outputs_dir_str)
    config['dataset_dir'] = join(data_dir, dataset_dir_str)

    config['idr_roc_fn'] = join(data_dir, 'idr_roc.png')
    config['dmasif_roc_fn'] = join(data_dir, 'dmasif_roc.png')

    parse_all_pdbs(config)

    return config
