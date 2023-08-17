import numpy as np


a = np.load("lmdb/meld_spec/dev/meta_info.pkl", allow_pickle=True)
print(a['key'])
# print(a)