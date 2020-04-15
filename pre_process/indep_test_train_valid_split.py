import h5py
import os
import numpy as np

mean_frame_num = 0
index_list = []
cropped_keys=[]

#1001-1091

np.random.seed(2)

file_root_dir = "../data/audio"
file_name = "audio_features_6class_long.hdf5"
file_dir = os.path.join(file_root_dir,file_name)

with h5py.File(file_dir, 'r') as f:
    a_group_key = list(f.keys())


cropped_keys = np.array(a_group_key)


total_participant = 1091-1001+1

num_test_size = total_participant//10

random_index = np.random.choice(total_participant, num_test_size*2, replace=False)
random_index = random_index+1001
print(random_index)
test_particip = random_index[:len(random_index)//2].astype(str)
val_particip = random_index[len(random_index)//2:].astype(str)
print(test_particip)
print(val_particip)


test_index = []
val_index = []

for i in range(len(cropped_keys)):
    for j in test_particip:
        if j in cropped_keys[i]:
            test_index.append(i)
    for j in val_particip:
        if j in cropped_keys[i]:
            val_index.append(i)
print(len(test_index))
print(len(val_index))




test_keys = cropped_keys[test_index]
val_keys = cropped_keys[val_index]


print("Test len",len(test_keys))
print("Val len",len(val_keys))
print("Before deleting len",len(cropped_keys))

cropped_keys = np.delete(cropped_keys,test_index+val_index)

print("After deleting len",len(cropped_keys))

file_save_dir = "Ang_hap_sad"
if not os.path.exists(os.path.join(file_root_dir,file_save_dir)):
    os.mkdir(os.path.join(file_root_dir,file_save_dir))


test_file_name = "audio_features_3class_test_long.hdf5"
val_file_name = "audio_features_3class_val_long.hdf5"
train_file_name = "audio_features_3class_train_long.hdf5"



with h5py.File(file_dir, 'r') as f:
    with h5py.File(os.path.join(file_root_dir,file_save_dir,test_file_name),"w") as hdf:
        for i in test_keys:
            if "ANG" in i or "HAP" in i or "SAD" in i:# or "NEU" in i:
                features = np.array(f[i][:])
                hdf.create_dataset(i,data=features)
    print("Test data created")
    with h5py.File(os.path.join(file_root_dir,file_save_dir,val_file_name),"w") as hdf:
        for i in val_keys:
            if "ANG" in i or "HAP" in i or "SAD" in i:# or "NEU" in i:
                features = np.array(f[i][:])
                hdf.create_dataset(i,data=features)
    print("Val data created")
    with h5py.File(os.path.join(file_root_dir,file_save_dir,train_file_name),"w") as hdf:
        for i in cropped_keys:
            if "ANG" in i or "HAP" in i or "SAD" in i:# or "NEU" in i:
                features = np.array(f[i][:])
                hdf.create_dataset(i,data=features)
    print("Train data created")
