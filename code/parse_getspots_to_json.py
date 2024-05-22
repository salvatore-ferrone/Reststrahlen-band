import average_facet_spectra as AFS
import json 
import os 



paths=AFS.obtain_file_paths(ii,shape_model=shape_model)


def main(ii,shape_model="50K_palmer_v20"):
    json_getspots=convert_to_json(ii,shape_model=shape_model)
    save_json(json_getspots,ii,shape_model=shape_model)

def save_json(json_getspots,ii,shape_model="50K_palmer_v20"):
    paths=AFS.obtain_file_paths(ii,shape_model=shape_model)
    
    new_out_path="../data-intermediate/"
    os.makedirs(new_out_path, exist_ok=True)
    new_out_path="../data-intermediate/json-getspots/"
    os.makedirs(new_out_path, exist_ok=True)
    new_out_path="../data-intermediate/json-getspots/"+shape_model
    os.makedirs(new_out_path, exist_ok=True)
    new_out_path="../data-intermediate/json-getspots/"+shape_model+"/"+paths['get_spots_path'].split('/')[-2]+"_facet_sclks.json"
    
    with open(new_out_path, 'w') as fp:
        json.dump(json_getspots, fp)

def convert_to_json(ii,shape_model="50K_palmer_v20"):
    
    paths=AFS.obtain_file_paths(ii,shape_model=shape_model)
    # initialize output
    json_getspots={}
    fp = open(paths['get_spots_path'], 'r')
    for myline in fp:
        if myline[0]=="F":
            current_facet=myline.split('\n')[0]
            json_getspots[current_facet]={}
            json_getspots[current_facet]["sclks"]=[]
        else:
            my_my_sclk = myline.split(' ')[0]
            json_getspots[current_facet]["sclks"].append(my_my_sclk)
    return json_getspots


if __name__=="__main__":
    shape_model="50K_palmer_v20"
    for ii in range(0,5):
        main(ii,shape_model=shape_model)
        print("Done with ", ii)