import os
import random
import numpy as np


data_dir = "/cryosat2/ML_DATA/runtime_data"
regions = ['AGSO', 'JAMSTEC', 'JAMSTEC2', 'NGA', 'NGA2', 'NGDC', 'NOAA_geodas', 'SIO', 'US_multi']

def per_region():
    for region in regions:
        #dirname = os.path.join(data_dir, region)
        dirname = data_dir
        ext = ".tsv.pkl"
        filenames = [filename for filename in os.listdir(dirname) if
                region in filename]
        random.shuffle(filenames)
        filenames = [os.path.join(dirname, filename) for filename in filenames]

        #s0, s1 = int(len(filenames) * 0.15), int(len(filenames) * 0.30)
        s0, s1 = int(len(filenames) * 0.25), int(len(filenames) * 0.50)
        tests, validates, trains = filenames[:s0], filenames[s0:s1],\
            filenames[s1:]
        for name, dataset in [("test", tests), ("validate", validates), ("train", trains)]:
            with open("{}-{}.txt".format(region, name), "w") as f:
                f.write("\n".join(dataset))
                f.write("\n")

dirname = data_dir
ext = ".tsv.pkl"
filenames = [filename for filename in os.listdir(dirname) if "US_multi2" not in filename]
random.shuffle(filenames)
filenames = [os.path.join(dirname, filename) for filename in filenames]

train_keep = []
for i in range(10):
    train_idx = [random.randint(0,len(filenames)) for j in range(1000)]
    trains = [filenames[id] for id in train_idx]
    train_keep.append(trains)
    with open("all-train-{}.txt".format(str(i)),"w") as f:
        f.write("\n".join(trains))
        f.write("\n")

test_filenames = np.setdiff1d(filenames,train_keep)
s0 = int(len(test_filenames)*0.5)
tests, validates = test_filenames[:s0], test_filenames[s0:]
for name, dataset in [("test", tests), ("validate", validates)]:
    with open("{}-{}.txt".format("all", name), "w") as f:
        f.write("\n".join(dataset))
        f.write("\n")


