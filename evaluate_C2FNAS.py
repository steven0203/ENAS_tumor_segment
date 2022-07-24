import torch
from loss import DiceLoss,DiceScore,MulticlassDiceLoss
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm,trange
from models import isensee2017_model,Unet3D,C2FNAS
from brats_dataloader import *
import math
from batchgenerators.dataloading import MultiThreadedAugmenter
import pandas as pd
from time import time
import utils
from argparse import ArgumentParser
from preprocess2018 import save_segmentation_as_nifti,get_list_of_patients

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

model_dir='C2FNAS'
model_path=os.path.join(model_dir,'C2FNAS.pth')
result_path=os.path.join(model_dir,'result.csv')
brats_preprocessed_folder='BRATS2018_precessed'
valid_ids_path='valid_ids.pkl'

batch_size = 2
n_labels=4


if __name__ == "__main__":
    parser=ArgumentParser(description='Evaluate C2FNAS')
    parser.add_argument('--data_path',type=str,default=brats_preprocessed_folder)
    parser.add_argument('--model_dir',type=str,default=model_dir)
    parser.add_argument('--n_labels',type=int,default=n_labels)
    args = parser.parse_args()
    model_dir=args.model_dir
    n_labels=args.n_labels
    brats_preprocessed_folder=args.data_path
    model_path=os.path.join(model_dir,'C2FNAS.pth')
    result_path=os.path.join(model_dir,'result.csv')
    seg_path=os.path.join(model_dir,'result_seg')
    maybe_mkdir_p(seg_path)

    header = ("WholeTumor", "TumorCore", "EnhancingTumor")
    masking_functions = (get_whole_tumor_mask, get_tumor_core_mask, get_enhancing_tumor_mask)
    rows = list()
    subject_ids = list()

    model=C2FNAS(labels=n_labels)
    model.load_state_dict(torch.load(model_path))
    model=model.cuda()
    model.eval()

    val=get_file_list(brats_preprocessed_folder,valid_ids_path)
    dataloader_validation = brats_dataloader(val,batch_size, None,1,infinite=False,shuffle=False,return_incomplete=True)
    start_time=time()
    for batch in dataloader_validation:
        inputs=torch.from_numpy(batch['data']).cuda()
        with torch.no_grad():
            outputs=model(inputs)
            outputs=torch.argmax(outputs,dim=1).cpu().numpy()
            if n_labels==4:
                outputs[outputs==3]=4
            for i in range(len(inputs)):
                if n_labels==4:
                    batch['seg'][i]=batch['seg'][i].astype(int)
                    batch['seg'][i][batch['seg'][i]==3]=4
                rows.append([dice_coefficient(func(outputs[i]),func(batch['seg'][i].astype(int))) for func in masking_functions])
                subject_ids.append(batch['names'][i])

                seg_result_path=join(seg_path,os.path.basename(batch['names'][i])+'.nii.gz')
                save_segmentation_as_nifti(outputs[i],batch['metadata'][i],seg_result_path)

                print(batch['names'][i])
    print('inference time : ',time()-start_time)
    print('model size :', utils.count_parameters_in_MB(model))

    df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
    df.to_csv(result_path)
