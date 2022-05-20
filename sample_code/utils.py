import os
import glob


def get_pdb(pdb_code=""):
  if pdb_code is None or pdb_code == "":
    upload_dict = files.upload()
    pdb_string = upload_dict[list(upload_dict.keys())[0]]
    with open("tmp.pdb","wb") as out: out.write(pdb_string)
    return "tmp.pdb"
  else:
    os.system(f"wget -qnc -O /content/pdbs/{pdb_code}.pdb https://files.rcsb.org/view/{pdb_code}.pdb ")
    # return f"{pdb_code}.pdb"

def create_paths():
    pred_dir = '/content/pdbs'
    isExist = os.path.exists(pred_dir)
    if not isExist:
        os.makedirs(pred_dir)

    upload_file = True #@param {type:"boolean"}

    %cd -q /content/pdbs
    if upload_file:
        uploaded = files.upload()
    %cd -q /content

    target_pdb = "5VAY"
    get_pdb(target_pdb)

    # target pdb
    target_pdb = "pdbs/5VAY-A.pdb" #@param {type:"string"}
    target_name = target_pdb.split('/')
    target_name = target_name[-1].split('.')

    if target_name[-1] == 'pdb':
        target_name = target_name[0]
    else:
        print('Please upload a valid .pdb file!')

    chain_name = 'A' #@param {type:"string"}
    chains = [chain_name]

    # Path to MaSIF weights
    ''' A resolution of 0.7 Angstrom gives a higher point cloud density and
        a higher performance. Different radii settings do not seem to impact performance.
    '''
    model_resolution = '0.7 Angstrom' #@param ["1 Angstrom", "0.7 Angstrom"]
    patch_radius = '9 Angstrom' #@param ["9 Angstrom", "12 Angstrom"]

    if patch_radius == '9 Angstrom':
        if model_resolution == '1 Angstrom':
            model_path = '/content/MaSIF_colab/models/dMaSIF_site_3layer_16dims_9A_100sup_epoch64'
            resolution = 1.0
            radius = 9
            sup_sampling = 100
        else:
            model_path = '/content/MaSIF_colab/models/dMaSIF_site_3layer_16dims_9A_0.7res_150sup_epoch85'
            resolution = 0.7
            radius = 9
            supsampling = 150

    elif patch_radius == '12 Angstrom':
        if model_resolution == '1 Angstrom':
            model_path = '/content/MaSIF_colab/models/dMaSIF_site_3layer_16dims_12A_100sup_epoch71'
            resolution = 1.0
            radius = 12
            supsampling = 100
        else:
            model_path = '/content/MaSIF_colab/models/dMaSIF_site_3layer_16dims_12A_0.7res_150sup_epoch59'
            resolution = 0.7
            radius = 12
            supsampling = 100

    # create folder for chain
    chains_dir = '/content/chains'
    isExist = os.path.exists(chains_dir)
    if not isExist:
        os.makedirs(chains_dir)
    else:
        files = glob.glob(chains_dir + '/*')
        for f in files:
            os.remove(f)

    # create folder for npy
    npy_dir = '/content/npys'
    isExist = os.path.exists(npy_dir)
    if not isExist:
        os.makedirs(npy_dir)
    else:
        files = glob.glob(npy_dir + '/*')
        for f in files:
            os.remove(f)

    # Create folder for the embeddings
    pred_dir = '/content/preds'
    isExist = os.path.exists(pred_dir)
    if not isExist:
        os.makedirs(pred_dir)
    else:
        files = glob.glob(pred_dir + '/*')
        for f in files:
            os.remove(f)
