import pandas as pd
import numpy as np
import feature_calculations as fc
import cv2
import commands
import os

"""
The file is intended to be used to add full slides to the df
"""
#myDB = pd.read_pickle('my_data_table.pkl')
#feature_dict = {'feature_1': fc.calc_feature_1}
#base_path = '/Users/prose/myDrive/dev/insight/histowiz/data/slide_data/'

base_path = os.path.join(os.environ['PWD'], '../data/slide_data/')
white_thresh = 0.95

def add_feature(name, function, df):
    # need to first check in column name exists
    if name in df.columns:
        print "Column exists: please pick a new name"
        return df

    df[name] = [-1]*np.ones_like(df[df.columns[0]])
    df_len = len(df[df.columns[0]])
    for idx, i in enumerate(df.index):
        if idx%100==0: print "Calculating feature for frame", idx, "out of", df_len
        #frame_path = get_frame_path_from_index(i, df)
        frame_path = get_frame_path_from_classification_file(i, df)
        frame = cv2.imread(frame_path)
        #print frame_path, frame
        #df[name][i] = function(frame)
        df.loc[i,name] = function(frame)
    return df

def get_frame_path_from_classification_file(index, df):
    classification = df['classification'][index]
    file_name = df['file_id'][index] + '.jpeg'
    frame_path = (base_path + 'camelyon_' + classification + '/' + file_name)

    return frame_path

def add_slide(slide_path, feature_dict, df):
    # if there are more features here than in
    #   the original df, let's first add those
    #   feature to the existing slides!
    for i in feature_dict:
        if not i in df.columns:
            df = add_feature(i, feature_dict[i], df)
    
    # now let's work with the current slide :)
    if not slide_path.endswith('/') : slide_path += '/'
    #output = commands.getoutput('ls ' + slide_path)
    #files = output.split('\n')
    files = os.listdir(slide_path)
    
    for i,iFile in enumerate(files):
        if not '.jpeg' in iFile: continue
        if i%100==0: print "Calculating feature for frame", i, "out of", len(files)
        frame_path = slide_path + iFile
        df = add_frame(frame_path, feature_dict, df)

    return df

def add_frame(frame_path, feature_dict, df):
    fp_split = frame_path.split('/')
    frame_dict = {'classification': fp_split[-2].split('amit')[1],
                  'file_id' : fp_split[-1].split('.')[1]
    }
    index = fp_split[-1].split('.')[1]
    if fp_split[-2].split('camelyon_')[1] == 'metastatic': index += '_m'
    else: index += '_n'
    img = cv2.imread(frame_path)
    # skip blanks!
    if fc.compute_white_area_1(img) > white_thresh:
        return df
    for idx in feature_dict:
        frame_dict[idx] = feature_dict[idx](img)

    new_row = pd.DataFrame(frame_dict, index=[index])
    df = df.append(new_row)

    return df
