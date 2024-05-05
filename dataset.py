import os
from torch.utils.data import Dataset
import numpy as np
import config
import nibabel as nib

def read_list(file):
    file=open(file,"r")
    S=file.read().split()
    p=list(str(i) for i in S)
    return p

def nifti_to_numpy(file):
    data = nib.load(file).get_fdata()
    data = data.astype(np.float32)
    return data

def minmaxnorm(data):
    return (data - np.min(data))/(np.max(data)-np.min(data))

def random_translation(data):
    data1 = np.zeros((128, 128, 128),dtype=np.float32)
    i=np.random.randint(-2,3)
    j=np.random.randint(-2,3)
    z=np.random.randint(-2,3)
    data1[2:125,2:125,2:125] = data[2+i:125+i,2+j:125+j,2+z:125+z]
    return data1

class OneDataset(Dataset):
    def __init__(self, root_MRI = config.whole_MRI, task = config.train, name = "train"):
        self.root_MRI = root_MRI
        self.name = name
        self.MRI = read_list(task)
        self.length_dataset = len(self.MRI)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        MRI_name = self.MRI[index % self.length_dataset] + ".nii"
        path_MRI = os.path.join(self.root_MRI, MRI_name)
        MRI = nifti_to_numpy(path_MRI)
        if self.stage == "train":
            MRI = random_translation(MRI)
        return MRI, MRI_name