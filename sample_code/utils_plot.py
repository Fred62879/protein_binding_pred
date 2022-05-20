


def show_pointcloud(main_pdb, coord_file, emb_file):
  # Normalize embedding to represent a b-factor value between 0-100
  b_factor = []
  for emb in emb_file:
      b_factor.append(emb[-2])
  # b_factor = [(float(i)-min(b_factor))/(max(b_factor)-min(b_factor)) for i in b_factor]

  # writing a psudo pdb of all points using their coordinates and H atom.
  records = []

  for i in range(len(coord_file)):
      points = coord_file[i]
      x_coord = points[0]
      y_coord = points[1]
      z_coord = points[2]

      records.append( { "record_name"       : 'ATOM',
                    "serial_number"     : len(records)+1,
                    "atom_name"         : 'H',
                    "location_indicator": '',
                    "residue_name"      : 'XYZ',
                    "chain_identifier"  : '',
                    "sequence_number"   : len(records)+1,
                    "code_of_insertion" : '',
                    "coordinates_x"     : x_coord,
                    "coordinates_y"     : y_coord,
                    "coordinates_z"     : z_coord,
                    "occupancy"         : 1.0,
                    "temperature_factor": b_factor[i]*100,
                    "segment_identifier": '',
                    "element_symbol"    : 'H',
                    "charge"            : '',
                    } )
  pdb = pdbparser()
  pdb.records = records

  pdb.export_pdb("pointcloud.pdb")

  # reading the psudo PDB we generated above for the point cloud.
  coordPDB = "pointcloud.pdb"
  view = ng.NGLWidget()
  coord_fn = os.path.join("/content", coordPDB))
  view.add_component(ng.FileStructure(coord_fn, defaultRepresentation=False))

  # representation with our customized colorscheme.
  view.add_representation\
      ('point', useTexture = 1, pointSize = 2, colorScheme = "bfactor",
       colorDomain = [100.0, 0.0], colorScale = 'rwb', selection='_H')

  view.add_component(ng.FileStructure(os.path.join("/content", main_pdb)))
  view.background = 'black'
  return view


def show_structure(main_pdb):
  # reading the psudo PDB we generated above for the point cloud.
  view = ng.NGLWidget()

  view.add_component(ng.FileStructure(main_pdb), defaultRepresentation=False)
  view.add_representation("cartoon", colorScheme = "bfactor", colorScale = 'rwb', colorDomain = [100.0, 0.0])
  view.add_representation("ball+stick", colorScheme = "bfactor", colorScale = 'rwb', colorDomain = [100.0, 0.0])
  view.background = 'black'
  return view

#plot_structure = ["Pointcloud", "Residues", "Atoms"]
def plot_output(plot_structure):
    if plot_structure == 'Pointcloud':
        view = show_pointcloud(target_pdb, coord, embedding)
    elif plot_structure == "Residues":
        view = show_structure('/content/output/per_resi_binding.pdb')
    elif plot_structure == "Atoms":
        view = show_structure('/content/output/per_atom_binding.pdb')
