import os
import sys
import pickle
import json

import numpy as np
from load_data import get_prediction_path, load_predictions, init_setup
from load_data import get_binary_filename, read_data_from_binary
from common import Logger

usage_msg = "Usage: python clean_CM.py <MODEL> <TESTED> <config_path>"

def edit_one_cm(tsv_filename, features, scores, logger):
    """
    Append model scores to CM, reformat for Py-CMeditor
    Format: ID, Lon, Lat, Depth, SIG H, SIG D, SID, pred, score
    """
    logger.log("Cleaning CM file, {}".format(tsv_filename))
    cm_dir = os.path.dirname(tsv_filename).replace("tsv_all","cm_data/public")
    cm_name = os.path.basename(tsv_filename).replace(".tsv",".cm").strip()
    cm_edit_filename = cm_dir + '/' + cm_name + '.edit.' + model_source

    logger.log("Writing new file, {}".format(cm_edit_filename))
    fwrite = open(cm_edit_filename,'w+')

    with open(tsv_filename, 'r', newline='\n') as fread:
        count = 1
        for line in fread:
            fields = line.strip().split()
            fields = fields[0:7]
            # check that the cm lat/lon and features lat/lon are similar
            if not check_fields(fields[0:2],features[0,0:2]):
                print("Lat and Lon don't match up")
                break

            fields.append(str(scores[0]))
            fields.insert(0,str(count))
            count += 1
            scores = scores[1:]
            features = features[1:,:]
            fwrite.write("{}\n".format(" ".join(fields)))

    fwrite.close()
    return features, scores

def check_fields(cm_fields, pkl_fields):
    """
    Check the lat and lon of the pickled features against the cm file vals
    """

    if float(cm_fields[0]) != pkl_fields[0]:
        return 0
    elif float(cm_fields[1]) != pkl_fields[1]:
        return 0
    else:
        return 1
    

if __name__ == '__main__':
    # args: Model-source Test-source config.json
    # e.g., python clean_CM.py NGDC US_multi2 config.json
    if len(sys.argv) != 4:
        print(usage_msg)
        sys.exit(1)
    model_source = sys.argv[1]
    test_source = sys.argv[2]

    with open(sys.argv[3]) as f:
        config = json.load(f)
    config["base_dir"] = os.path.expanduser(config["base_dir"])
    init_setup(config["base_dir"])

    logger = Logger()
    logfile = os.path.join(config["base_dir"], "CM-cleaning.log")
    logger.set_file_handle(logfile)
    logger.log("begin CM cleaning for {} data trained on {} model".format(test_source, model_source))

    prediction_file = get_prediction_path(config["base_dir"], model_source, test_source)
    features, _, scores, _ = load_predictions(prediction_file,logger)

    #CM_filelist = get_cm_filename(config["testing_files"])
    with open(config["testing_files"]) as f:
        filelist = f.readlines()

    for file in filelist:
        if test_source in os.path.dirname(file.strip()):
            features, scores = edit_one_cm(file.strip(), features, scores, logger)
