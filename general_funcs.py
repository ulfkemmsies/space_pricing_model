from sim_classes import *
from graph_classes import *
import json
import os


def load_dict_from_json(filename):
        with open(f'{filename}.json') as json_file:
            data = json.load(json_file)
            return data

def get_filepaths_from_dir(foldername):
        root = "."
        filepaths = []
        for path, subdirs, files in os.walk(root):
                for subdir in subdirs:
                    if subdir == foldername:
                        total_path = os.path.join(path, subdir)
        for path, subdirs, files in os.walk(total_path):
            for file in files:
                filepaths.append(os.path.join(total_path,file))
            
        return filepaths

def get_filenames_from_dir(foldername, extension=None):
        root = "."
        filepaths = []
        for path, subdirs, files in os.walk(root):
                for subdir in subdirs:
                    if subdir == foldername:
                        total_path = os.path.join(path, subdir)
        for path, subdirs, files in os.walk(total_path):
            for file in files:
                if extension:
                    if extension in file:
                        filepaths.append(extension_remover(file))
                else:
                    filepaths.append(extension_remover(file))
            
        return filepaths

def get_subfolder_path(foldername):
    root = "."
    filepaths = []
    for path, subdirs, files in os.walk(root):
            for subdir in subdirs:
                if subdir == foldername:
                    total_path = os.path.join(path, subdir)
    return total_path

def extension_remover(filename):
    out= filename.split(".", 1)
    return out[0]


def attr_dict_to_str(attrs):
    return json.dumps(attrs).replace(",", "\n")