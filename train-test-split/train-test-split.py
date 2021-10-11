import os
import random


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

#s0, s1 = int(len(filenames) * 0.25), int(len(filenames) * 0.50)
s0 = 100
s1 = 200
tests, validates = filenames[:s0], filenames[s0:s1]
for name, dataset in [("test", tests), ("validate", validates)]:
    with open("{}-{}.txt".format("all", name), "w") as f:
        f.write("\n".join(dataset))
        f.write("\n")

train_filenames = filenames[s1:]
for i in range(10):
    train_idx = [random.randint(0,len(train_filenames)) for j in range(100)]
    trains = [train_filenames[id] for id in train_idx]
    with open("all-train-{}.txt".format(str(i)),"w") as f:
        f.write("\n".join(trains))
        f.write("\n")
