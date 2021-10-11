import os
import random
import pickle


data_dir = "/cryosat2/ML_DATA/runtime_data"
write_dir = "/cryosat2/ML_DATA/runtime_data_chunks"
regions = ['AGSO', 'JAMSTEC', 'JAMSTEC2', 'NGA', 'NGA2', 'NGDC', 'NOAA_geodas', 'SIO', 'US_multi']

os.mkdir(write_dir)
#for region in regions:
#os.mkdir(os.path.join(write_dir, region))

#dirname = os.path.join(data_dir, region)
#ext = ".tsv"
dirname = data_dir
ext = ".tsv.pkl"
filenames = [filename for filename in os.listdir(dirname) if filename.endswith(ext)]

chunk_size = 100000
for filename in filenames:
    basename = filename[:-8]
    path = os.path.join(dirname, filename)
    with open(path,'rb') as f:
        data, labels, weights = pickle.load(f)

    num_line = len(data)
    cursor = 0
    part = 0
    while cursor < num_line:
        start, end = cursor, cursor + chunk_size
        if (num_line - end) * 2 < chunk_size:
            end = num_line
        cursor = end

        part_filename = basename + ".part{}.tsv.pkl".format(part)
        part += 1
        write_loc = os.path.join(write_dir, part_filename)
        with open(write_loc, "wb") as f:
            pickle.dump((data[start:end],labels[start:end],weights[start:end]),f,protocol=4)
            #f.write("".join(data[start:end]))
