import multiprocessing
import lightgbm as lgb
import ray
import random

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

    train(config, train_dataset, valid_dataset, region_str, logger)


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

def run_training_n_times(config, regions, is_read_text, logger, is_random=True):
    random.seed()
    with open(config["training_files"]) as f:
        all_training_files = f.readlines()
    with open(config["validation_files"]) as f:
        all_valid_files = f.readlines()
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
    result_ids = []
    tr_id = ray.put(train_dataset)
    va_id = ray.put(valid_dataset)
    for n in range(10):
        rseed = random.randint(0,100)
        result_ids.append(train.remote(config, tr_id, va_id,rseed))
        #train(config, train_dataset, valid_dataset, "all", logger, n)
    i = 0
    while len(result_ids):
        done_id, result_ids = ray.wait(result_ids)
        print(ray.get(done_id[0]))
        gbm = ray.get(done_id[0])
        persist_model(config["base_dir"], "all" + str(i), gbm)
        i+=1


# Specify a data file
def run_training_specific_file(filenames, region_name, config, logger):
    features, labels, weights = load_examples_from_pickle(filenames)
    train_dataset = lgb.Dataset(
        features, label=labels, weight=weights, params={'max_bin': config["max_bin"]})

    train(config, train_dataset, None, region_name, logger)
