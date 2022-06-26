from sim_classes import *
from graph_classes import *
import json
import os
from st_aggrid import AgGrid, GridOptionsBuilder
import streamlit as st


def load_dict_from_json(filename):
        with open(f'{filename}.json') as json_file:
            data = json.load(json_file)
            return data

def set_instance_attrs_from_dict(instance, dict, keys=None):

    if keys != None:
        for key in keys:
            setattr(instance, key, dict[key])

    elif keys == None:
        for key in dict.keys():
            setattr(instance, key, dict[key])

def create_class_instances_from_dict(classname, dict):
    out = []
    pass


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

def dfs_to_dict(tables_in:dict):

    out = {}
    names = tables_in.keys()
    name = (list(names)[0])
    nodes = list(tables_in[f"{name}"].columns)

    for name in names:
        df = tables_in[name]

        table_dict = {}

        for source in nodes:
            for target in nodes:
                if source != target:
                    
                    if source not in table_dict.keys():
                        table_dict[source] = {}
                        
                    val = df.loc[source,target]

                    if pd.notna(val):
                        
                        if target not in table_dict[source].keys():
                            table_dict[source][target] = {}
                        
                        table_dict[source][target][name] = val

        if len(out.keys()) == 0:
            out.update(table_dict)

        for key_0_old, val_0_old in out.items():
            for key_0_new, val_0_new in table_dict.items():
                if key_0_old == key_0_new:
                    for key_1_old, val_1_old in val_0_old.items():
                        for key_1_new, val_1_new in val_0_new.items():
                            if key_1_old == key_1_new:
                                val_1_old.update(val_1_new)
    
    return out

def folder_tables_to_dict(subfolder):

    filepaths = get_filepaths_from_dir(subfolder)
    filenames = get_filenames_from_dir(subfolder, extension="csv")

    out = {}

    dfs = [pd.read_csv(filepath) for filepath in filepaths]

    for i in range(len(dfs)):
        df = dfs[i]
        df = df.set_index(df["Unnamed: 0"].values)
        df = df.drop(columns=["Unnamed: 0"])
        dfs[i] = df
    
    for i in range(len(filenames)):
        out.update({f"{filenames[i]}": dfs[i]})
    
    return out

def entire_table_folder_to_dict(subfolder):
    tables_in = folder_tables_to_dict(subfolder)
    final = dfs_to_dict(tables_in)
    return final

def dict_depth(dic, level = 1):
       
    str_dic = str(dic)
    counter = 0
    for i in str_dic:
        if i == "}": break
        elif i == "{":
            counter += 1
    return(counter)

def list_depth(list_of_lists):
    if not isinstance(list_of_lists, list):
        return 0
    return max(map(list_depth, list_of_lists), default=0) + 1     

def draw_editable_table(dict_in):
    setattr(st.session_state.sim, "current_edited_var_df", None)

    #Ensuring this doesn't carry over
    how_deep = None

    #Depending on the variable type, must transform to df in different way
    key_in, var_in = list(dict_in.keys())[0], list(dict_in.values())[0]

    if isinstance(var_in, pd.DataFrame):
        df = var_in

        if set(df.index) == set(df.columns): df.insert(loc = 0, column = 'index', value = df.index)

    elif (isinstance(var_in,dict)):

        how_deep = dict_depth(var_in)

        if how_deep <= 2:
            if how_deep == 1: df = pd.DataFrame.from_dict(var_in, orient="index", columns=["values"])
            
            else: df = pd.DataFrame.from_dict(var_in, orient="index")

            if not("name" in df.columns): df.insert(loc = 0, column = 'index', value = df.index)

        else:
            st.write("This dictionary is too deep!")
            return None

    elif isinstance(var_in,list) or isinstance(var_in,set):
        if isinstance(var_in,set): var_in = list(var_in)

        how_deep = list_depth(var_in)

        if how_deep <= 2:
            if how_deep == 1: df = pd.DataFrame({"values":var_in})
            elif how_deep == 2: df = pd.DataFrame(var_in, columns=[i for i in range(len(var_in[0]))]) 
        else:
            st.write("This list is too deep!")
            return None

    else:
        st.write("This variable cannot be transformed into a table! Try again.")
        return None
    
    #builds a gridOptions dictionary using a GridOptionsBuilder instance.
    builder = GridOptionsBuilder.from_dataframe(df)
    go = builder.build()

    #uses the gridOptions dictionary to configure AgGrid behavior.
    new_df = AgGrid(df, gridOptions=go, editable=True, update_mode="MANUAL")['data']
    if "index" in new_df.columns: df = df.set_index('index')
    
    setattr(st.session_state.sim, "current_edited_var_df", new_df)