import multiprocessing
import lightgbm as lgb
import ray
import random
import os

from .common import Logger
from .booster import train
from .load_data import get_region_data, persist_model
from .tools.split_by_instances import load_examples_from_pickle
TRAIN_PREFIX = "train"
VALID_PREFIX = "valid"
LIMIT = None


def run_training_per_region(
        config, regions, region_str, all_training_files, all_valid_files, is_read_text, logger):
    logger.log("Now training {}".format(region_str))

    logger.log("start constructing datasets")
    (t_features, t_labels, t_weights) = get_region_data(
        config["base_dir"], all_training_files, regions, is_read_text, TRAIN_PREFIX, logger)
    train_dataset = lgb.Dataset(
        t_features, label=t_labels, weight=t_weights, params={'max_bin': config["max_bin"]})
    (v_features, v_labels, v_weights) = get_region_data(
        config["base_dir"], all_valid_files, regions, is_read_text, VALID_PREFIX, logger)
    if len(v_features) == 0:
        logger.log("No validation data provided.")
        valid_dataset = None
    else:
        valid_dataset = lgb.Dataset(
            v_features, label=v_labels, weight=v_weights, params={'max_bin': config["max_bin"]})

    gbm = train(config, train_dataset, valid_dataset, region_str, logger)
    persist_model(config["base_dir"], region_str, gbm)


def run_training(config, regions, is_read_text, logger):
    with open(config["training_files"]) as f:
        all_training_files = f.readlines()
    with open(config["validation_files"]) as f:
        all_valid_files = f.readlines()
    for region in regions:
        run_training_per_region(
            config, [region], region, all_training_files, all_valid_files, is_read_text, logger)


def run_training_all(config, regions, is_read_text, logger):
    with open(config["training_files"]) as f:
        all_training_files = f.readlines()
    with open(config["validation_files"]) as f:
        all_valid_files = f.readlines()
    run_training_per_region(
        config, regions, "all", all_training_files, all_valid_files, is_read_text, logger)

def train_one_random(config,v_features,v_labels,v_weights,str_id,logger,is_read_text,regions):
    print("training one random...")
    filelist_dir = "/Users/huharper/02_multibeam/02_boosting/bathymetry/train-test-split/"
    train_file = "all-train-{}.txt".format(str_id)
    config["training_files"] = os.path.join(filelist_dir,train_file)
    with open(config["training_files"]) as f:
        all_training_files = f.readlines()
    logger.log("start constructing datasets")
    (t_features, t_labels, t_weights) = get_region_data(
        config["base_dir"], all_training_files, regions, is_read_text, TRAIN_PREFIX, logger)
    train_dataset = lgb.Dataset(
        t_features, label=t_labels, weight=t_weights, params={'max_bin': config["max_bin"]})
    if len(v_features) == 0:
        logger.log("No validation data provided.")
        valid_dataset = None
    else:
        valid_dataset = train_dataset.create_valid(v_features,label=v_labels,
            weight=v_weights,params={'max_bin': config["max_bin"]})
    rseed = random.randint(0,100)
    gbm = train(config, train_dataset, valid_dataset, "all", logger, rseed)
    persist_model(config["base_dir"], "all{}".format(str_id), gbm)
    return

def run_training_n_times(config, regions, is_read_text, logger, is_random=True):
    with open(config["validation_files"]) as f:
        all_valid_files = f.readlines()
    (v_features, v_labels, v_weights) = get_region_data(
        config["base_dir"], all_valid_files, regions, is_read_text, VALID_PREFIX, logger)
    #va_id = ray.put(valid_dataset)
    random.seed()
    #result_ids = []
    print("begins loop")
    for n in range(10):
        train_one_random(config,v_features,v_labels,v_weights,str(n),logger,is_read_text,regions)
    return

# Specify a data file
def run_training_specific_file(filenames, region_name, config, logger):
    features, labels, weights = load_examples_from_pickle(filenames)
    train_dataset = lgb.Dataset(
        features, label=labels, weight=weights, params={'max_bin': config["max_bin"]})

    train(config, train_dataset, None, region_name, logger)
