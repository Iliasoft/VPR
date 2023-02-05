import pickle
from similarity_multi_dir import Dict2Class
from embeddings_multi_dir import MultiDirDataset
from configs import config1
from files import join

if __name__ == '__main__':
    args = Dict2Class(config1.args)

    data_set = MultiDirDataset(args.small_ds_dir)
    print(f"Generating duplicates list")
    args_ext_number = 1

    for h in range(0, args_ext_number + 1):
        with open(join(args.small_ds_dir, f"similarity_{h}_{0}_{args_ext_number}.pkl"), 'rb') as f:
            pickle.load(f)
