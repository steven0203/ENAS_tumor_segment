import torch
from loss import DiceLoss,DiceScore,MulticlassDiceLoss
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm,trange
from models import isensee2017_model,Unet3D,C2FNAS,Unet
from brats_dataloader import *
import math
from batchgenerators.dataloading import MultiThreadedAugmenter
import pandas as pd
from time import time
import utils
from argparse import ArgumentParser
from preprocess2018 import save_segmentation_as_nifti,get_list_of_patients
import config
import os
import re
import ast 

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


def get_searched_model(model_dir):
    dag_file=os.path.join(model_dir,'derive_dag.log')
    dag_file=open(dag_file,'r')
    lines=[]
    for line in dag_file:
        lines.append(line)
    result_str=lines[-1].replace('best_dag :','')
    index=[m.start() for m in re.finditer(']',result_str)][-1]
    return ast.literal_eval(result_str[:index+1])


batch_size = 2
n_labels=4
valid_ids_path='valid_ids.pkl'


if __name__ == "__main__":
    args, unparsed = config.get_args()
    model_dir=args.load_path
    model_path=os.path.join(model_dir,'model.pth')
    result_path=os.path.join(model_dir,'result.csv')
    n_labels=args.n_classes
    brats_preprocessed_folder=args.data_path
    dag=get_searched_model(model_dir)
    seg_path=os.path.join(model_dir,'result_seg')
    maybe_mkdir_p(seg_path)

    header = ("WholeTumor", "TumorCore", "EnhancingTumor")
    masking_functions = (get_whole_tumor_mask, get_tumor_core_mask, get_enhancing_tumor_mask)
    rows = list()
    subject_ids = list()

    dag=get_searched_model(model_dir)
    model=Unet(args,dag=dag)
    model.set_arc()
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
