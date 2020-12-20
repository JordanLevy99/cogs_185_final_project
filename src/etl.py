import json
import sys
sys.path.insert(0, '../src/data/cocoapi/PythonAPI')
from pycocotools.coco import COCO
import pandas as pd
import os

def load_data(annFile):
    # import json
    with open(annFile, 'r') as f:
        data = json.load(f)
    coco = COCO(annFile)
    # _, data['categories'] = fix_ids(data)
    return coco, data

def fix_ids(data):
    '''
    Takes in 'categories' key of a COCO dataset, returns new IDs for those categories (properly mapped from 0 to len(categories)-1)
    '''
    categories = data['categories']
    global id_dict
    id_dict = {}
    for idx, cat in enumerate(categories):
        id_dict[cat['id']] = idx
        cat['id'] = idx
    return id_dict, categories

def drop_null_annotations(coco, data, dataDir, dataType, annFile, overwrite, map_ids=True, save=True, tmpDataDir='data/temp'):
    """
    Takes in a json.load(f) of an annotation file, finds all image ids without an annotation, then drops those images from the file.
    Writes to a new json file without those images.
    """
    # Writes new data to a cleaned json instances file
    fname = tmpDataDir+f"/annotations/clean_instances_{dataType}.json"
    if os.path.isfile(fname) and overwrite==False: print(fname, 'already exists')
    elif os.path.isfile(fname) and overwrite==True: print(f'{fname} already exists, overwriting anyways b/c overwrite=True')
        # if map_ids:
        #     new_data = data.copy()
        #     id_dict = {}
        #     for i, cat in enumerate(new_data['categories']):
        #         id_dict[cat['id']] = i
        #         cat['id'] = i

        # return id_dict


    images_pd = pd.Series(data['images'])
    new_images_pd = images_pd.copy()
    new_data = data.copy()
    new_data['images'] = \
        new_images_pd.loc[~images_pd.apply(lambda x: len(coco.getAnnIds(x['id']))==0)].tolist()
    # sets new_data['images'] to only the list of images with one or more annotations
#     if os.path.isdir(tmpDataF+"/annotations") == False: os.mkdir(dataDir+"/annotations")
    if map_ids: id_dict, _ = fix_ids(new_data)
    print(os.getcwd())

    # print(fname)
    if save:
        try:
            open(fname, 'w')
        except FileNotFoundError:
            os.mkdir(tmpDataDir+'/annotations')
            print('annotation directory created')
        with open(fname, 'w') as f:
            json.dump(new_data, f)
    else:
        print("file saving skipped")
    print("Start images:", len(data['images']))
    print("Images remaining:",len(new_data['images']))
    print("Number of images with no annotations:",len(data['images'])-len(new_data['images']))
    if map_ids: return id_dict
    return new_data

# def convert_ids(target):
#     return id_dict[target]