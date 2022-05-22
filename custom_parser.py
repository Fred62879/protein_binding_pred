
from os import listdir
from os.path import join, isfile


def parse_all_pdbs(config):

    embd_nm = '_predfeatures_emb1.npy'
    point_cloud_nm = '_predcoords.npy'
    sufce_mask_sufx = '-surface-atom-mask.npy'
    itfce_mask_sufx = '-interface-atom-mask.npy'

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
        if pdb_nm in pdb_nms: continue

        # add relevant file names for current pdb
        pdb_chains = parts[0].split('-')[1] # all chains for current pdb
        cur_chains = [pdb_chains]
        pdb_nm += '-' + pdb_chains

        fn_prefx = join(config['dataset_dir'], pdb_nm)

        embd_fn = join(config['preds_dir'], pdb_nm + '_' + pdb_chains + embd_nm)
        ptcld_fn = join(config['preds_dir'], pdb_nm + '_' + pdb_chains + point_cloud_nm)

        pdb_nms.add(pdb_nm)
        embd_fns.append(embd_fn)
        chains.append(cur_chains)
        ptcld_fns.append(ptcld_fn)
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

    '''
    target_name = args.target_pdb_fn.split('/')
    target_name = target_name[-1].split('.')[0]
    config['target_name'] = target_name
    config['target_pdb_fn'] = join(data_dir, args.target_pdb)
    config['smask_fn'] = join(data_dir, target_name + sufce_mask_sufx)
    config['imask_fn'] = join(data_dir, target_name + itfce_mask_sufx)

    chain_name = 'A'
    chains = [chain_name]
    config['chains'] = chains
    '''



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
    parser.add_argument('--dist_thresh', type=float, default=1.0)
    parser.add_argument('--resid_thresh', type=float, default=0.0)

    parser.add_argument('--k', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--radius', type=int,  default=9,
                        help='patch radius, settings doesnt impact performance?')
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--atom_dims', type=int, default=6)
    parser.add_argument('--emb_dims', type=int, default=16)
    parser.add_argument('--post_units', type=int, default=8)
    parser.add_argument('--in_channels', type=int, default=16)
    parser.add_argument('--orientation_units', type=int, default=16)
    parser.add_argument('--unet_hidden_channels', type=int, default=8)

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
    model_dir_str = './MaSIF_colab/model'
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

    parse_all_pdbs(config)

    return config
