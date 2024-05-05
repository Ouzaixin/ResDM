import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 1e-4
batch_size = 1
numworker = 0
epochs = 1000

whole_MRI = "../data/whole_MRI"
whole_Abeta = "../data/whole_Abeta"
whole_Tau = "../data/whole_Tau"
path = "../data/whole_MRI/002S2010.nii"

exp = "exp_1/"
CHECKPOINT_Unet = "result/"+exp+"Unet.pth.tar"
CHECKPOINT_encoder = "result/"+exp+"Text_encoder.pth.tar"

train = "./data_info/train.txt"
validation = "./data_info/validation.txt"
test = "./data_info/test.txt"