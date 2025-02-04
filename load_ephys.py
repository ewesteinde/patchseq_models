import os
import pandas as pd
from ipfx.dataset.create import create_ephys_data_set
from ipfx.data_set_features import extract_data_set_features
from ipfx.utilities import drop_failed_sweeps
import json

def find_files(substring, directory):
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if substring in file:
                matching_files.append(os.path.join(root, file))
    return matching_files


def load_data(ID, 
              data_path = '/Users/elena.westeinde/Datasets/patch_seq/electrophysiology', 
              metadata_path = '/Users/elena.westeinde/Datasets/patch_seq/specimen_metadata/20200711_patchseq_metadata_mouse.csv',
              save_dir = '/Users/elena.westeinde/Datasets/raw_ephys'
              ):
    save_dir = os.path.join(save_dir, str(ID))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    metadata = pd.read_csv(metadata_path)
    metadata_cell = metadata[metadata['cell_specimen_id'] == ID]
    ephys_ID = metadata_cell['ephys_session_id'].values[0]
    matching_files = find_files(str(ephys_ID), data_path)
    stim_dict = {}
    if len(matching_files) == 0:
        print('No matching files found')
    # return None
    elif len(matching_files) > 1:
        print('Multiple matching files found')
        #return None
    else:
        data_set = create_ephys_data_set(nwb_file = matching_files[0])
        drop_failed_sweeps(data_set)
        cell_features, sweep_features, cell_record, sweep_records, _, _ = \
        extract_data_set_features(data_set)#, subthresh_min_amp=-100.0)
        stim_types = cell_features.keys()
        stim_dict = {stim_type: {} for stim_type in stim_types}

        #for stim_type in stim_types:
        sweeps = cell_features['long_squares']['sweeps']
        for sweep_num in range(len(sweeps)):
            swp = data_set.sweep(cell_features['long_squares']['sweeps'][sweep_num]['sweep_number'])
            stim_dur = 1000 # ms & delay
            v_baseline = cell_features['long_squares']['sweeps'][sweep_num]['v_baseline']
            stim_amp = cell_features['long_squares']['sweeps'][sweep_num]['stim_amp']
            time = swp.t
            current = swp.i
            voltage = swp.v
            stim_dict['long_squares'][sweep_num] = {'time': time.tolist(), 
                                                    'current': current.tolist(), 
                                                    'voltage': voltage.tolist(),
                                                    'stim_dur': stim_dur,
                                                    'v_baseline': v_baseline,
                                                    'stim_amp': stim_amp}

        file_name = os.path.join(save_dir, 'ephys_ID_' + str(ephys_ID) + '.json')
        # save file
        with open(file_name, 'w') as f:
            json.dump(stim_dict, f)





# last_slash = data_files[0].rfind('/')
# # load in the json file as a dictionary
# json_file = os.path.join(output_dir, data_files[0][last_slash+1:], 'output.json')
# with open(json_file, 'r') as f:
#     data = json.load(f)