import torch
from loss import DiceLoss,DiceScore,MulticlassDiceLoss
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm,trange
from models import isensee2017_model
from brats_dataloader import *
import math
from batchgenerators.dataloading import MultiThreadedAugmenter
import pandas as pd



def get_whole_tumor_mask(data):
    return data > 0


def get_tumor_core_mask(data):
    tmp=np.logical_or(data == 1, data == 3)
    return np.logical_or(tmp ,data==4)


def get_enhancing_tumor_mask(data):
    return data == 4



def dice_coefficient(truth, prediction):
    data=np.sum(truth)+np.sum(prediction)
    if data==0:
        return 0.0
    return 2 * np.sum(truth * prediction)/data




model_path='Isensee2017.pth'

brats_preprocessed_folder='BRATS2015_precessed'
train_ids_path='train_ids.pkl'
valid_ids_path='valid_ids.pkl'

batch_size = 2
n_labels=5

result_path='result.csv'

if __name__ == "__main__":

    header = ("WholeTumor", "TumorCore", "EnhancingTumor")
    masking_functions = (get_whole_tumor_mask, get_tumor_core_mask, get_enhancing_tumor_mask)
    rows = list()
    subject_ids = list()


    
    model=isensee2017_model(labels=n_labels)
    model.load_state_dict(torch.load(model_path))
    model=model.cuda()
    model.eval()

    train=get_file_list(brats_preprocessed_folder,train_ids_path)
    val=get_file_list(brats_preprocessed_folder,valid_ids_path)
    dataloader_validation = brats_dataloader(val,batch_size, None,1,infinite=False,shuffle=False,return_incomplete=True)

    for batch in dataloader_validation:
        
        inputs=torch.from_numpy(batch['data']).cuda()
        with torch.no_grad():
            outputs=model(inputs)
            outputs=torch.argmax(outputs,dim=1).cpu().numpy()
        
        for i in range(len(inputs)):
            rows.append([dice_coefficient(func(outputs[i]),func(batch['seg'][i].astype(int))) for func in masking_functions])
            subject_ids.append(batch['names'][i])
            print(batch['names'][i])
    df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
    df.to_csv(result_path)
