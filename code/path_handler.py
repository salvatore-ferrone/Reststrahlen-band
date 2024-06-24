"""
This script handles the paths to the data files used in this analysis

It is important since the localtion of the files is computer specific

"""
import json
import os

class PathHandler:
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        json_file = os.path.join(dir_path, "basepaths.json")
        with open(json_file, 'r') as f:
            self.basepaths = json.load(f)
            
        self.wavenumbers= self.basepaths['wavenumbers']
        self.shapemodel="palmer50k"
    
    def build_path(self, base_path, *args):
        """
        Build a path by joining base_path with all other arguments.
        """
        return os.path.join(base_path, *args)

    def otes_csv(self, survey_name):
        return self.build_path(self.basepaths["OTES_csv"], survey_name+"_data_spectra.csv")

    def bayes_folder(self, survey_name,model_name="two_gauss"):
        directory=self.build_path(self.basepaths["bayes_fits"], survey_name, model_name)
        os.makedirs(directory, exist_ok=True)
        return directory
    
    def bayes_fits_fname(self, survey_name, row_index,model_name="two_gauss"):
        directory=self.bayes_folder(survey_name,model_name)
        return self.build_path(directory+"/"+survey_name+"_OTES_"+str(row_index).zfill(4)+".h5")

    def json_getspots(self, survey_name):
        get_spots_eq = {
            "EQ1": "OTES_DTS_EQ3pm_facet_sclks.json",
            "EQ2": "OTES_DTS_EQ320am_facet_sclks.json",
            "EQ3": "OTES_DTS_EQ1230pm_facet_sclks.json",
            "EQ4": "OTES_DTS_EQ10am_facet_sclks.json",
            "EQ6": "OTES_DTS_EQ840pm_facet_sclks.json",
        }

        return self.build_path(self.basepaths["json_getspots"], get_spots_eq[survey_name])
    
    
    def facet_spectra(self,survey_name,model_name,face_number_str):
        dir=self.build_path(self.basepaths["facet_spectra"],self.shapemodel,survey_name,model_name)
        os.makedirs(dir,exist_ok=True)
        return self.build_path(dir,face_number_str+".h5")

