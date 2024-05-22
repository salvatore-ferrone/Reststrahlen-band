"""
This script handles the paths to the data files used in this analysis

It is important since the localtion of the files is computer specific

"""
import json
import os

class PathHandler:
    def __init__(self):
        json_file="basepaths.json"
        with open(json_file, 'r') as f:
            self.basepaths = json.load(f)

    def build_path(self, base_path, *args):
        """
        Build a path by joining base_path with all other arguments.
        """
        return os.path.join(base_path, *args)

    def get_data(self, key):
        """
        Get data from the dictionary using the key.
        """
        return self.data.get(key)

    def set_data(self, key, value):
        """
        Set a value in the dictionary using the key.
        """
        self.data[key] = value


# def obtain_file_paths(ii,shape_model="50K_palmer_v20"):
#     # set up the paths
#     get_spots_path_base="../data/getspots_shapeModels/"+shape_model+"/OTES/"+get_spots_paths[ii]+"/"
#     get_spots_file_name = get_spots_file_names[get_spots_paths[ii]]+"_"+shape_model_switcher[shape_model]+"_smf.txt"
#     spectra_path="../data/datiotescomplete/"+spectra_file_names[ii]
#     get_spots_path=get_spots_path_base+get_spots_file_name
#     ancillary_file_path = "../data/getspots_shapeModels/"+shape_model+"/g_06310mm_spc_tes_0000n00000_v020.fits"
#     wave_channel_path = "../data/datiotescomplete/OTES_wnb.csv"
#     paths = {"get_spots_path":get_spots_path,
#             "spectra_path":spectra_path,
#             "ancillary_file_path":ancillary_file_path,
#             "wave_channel_path":wave_channel_path}

#     return paths