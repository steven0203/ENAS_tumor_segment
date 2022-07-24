import torch
from loss import DiceLoss,DiceScore,MulticlassDiceLoss
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm,trange
from models import isensee2017_model,Unet
from brats_dataloader import *
import math
from batchgenerators.dataloading import MultiThreadedAugmenter
import config
from argparse import ArgumentParser
import os
import re
import ast 
brats_preprocessed_folder='BRATS2018_precessed'
train_ids_path='train_ids.pkl'
valid_ids_path='valid_ids.pkl'


patch_size = (128, 128, 128)
batch_size = 2
num_threads=8
max_epoch=300
n_labels=4

lr=0.0005
weight_decay=5e-5
lr_schedule=0.985


def adjust_lr(optimizer,current_lr,schedule):
    current_lr=current_lr*schedule
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    return current_lr

def get_searched_model(model_dir):
    dag_file=os.path.join(model_dir,'derive_dag.log')
    dag_file=open(dag_file,'r')
    lines=[]
    for line in dag_file:
        lines.append(line)
    result_str=lines[-1].replace('best_dag :','')
    index=[m.start() for m in re.finditer(']',result_str)][-1]
    return ast.literal_eval(result_str[:index+1])
    
if __name__ == "__main__":
    args, unparsed = config.get_args()
    model_dir=args.load_path
    model_path=os.path.join(args.load_path,'model.pth')
    log_path=os.path.join(args.load_path,'training.log')
    model_path_300=os.path.join(args.load_path,'model_300.pth')
    model_path_150=os.path.join(args.load_path,'model_150.pth')
    n_labels=args.n_classes
    brats_preprocessed_folder=args.data_path
    dag=get_searched_model(model_dir)
    model=Unet(args,dag)
    model.set_arc()
    model=model.cuda()
    model.train()
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    loss=MulticlassDiceLoss()

    train=get_file_list(brats_preprocessed_folder,train_ids_path)
    val=get_file_list(brats_preprocessed_folder,valid_ids_path)

    shapes = [brats_dataloader.load_patient(i)[0].shape[1:] for i in train]
    max_shape = np.max(shapes, 0)
    max_shape = list(np.max((max_shape, patch_size), 0))

    dataloader_train = brats_dataloader(train, batch_size, max_shape, num_threads,return_incomplete=True,infinite=False,seed_for_shuffle=np.random.randint(1234))
    dataloader_validation = brats_dataloader(val,batch_size, None,1,infinite=False,shuffle=False,return_incomplete=True)

    tr_transforms = get_train_transform(patch_size)

    tr_gen = MultiThreadedAugmenter(dataloader_train, tr_transforms, num_processes=num_threads,
                                    num_cached_per_queue=3,
                                    seeds=None, pin_memory=False)

    tr_gen.restart()



    log=open(log_path,'w')
    log.write('epoch,loss,valid loss\n')
    min_loss=1000

    num_batches_per_epoch=len(dataloader_train)
    num_validation_batches_per_epoch=len(dataloader_validation)

    current_lr=lr

    for epoch in range(max_epoch):
        raw_loss=0
        with tqdm(total=num_batches_per_epoch) as t:
            for batch in tr_gen:
                inputs=torch.from_numpy(batch['data']).cuda()
                target=torch.from_numpy(batch['seg'].astype(int)).cuda()
                target=get_multi_class_labels(target,n_labels=n_labels)
                outputs=model(inputs)
                losses=loss(outputs,target)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                t.set_description('Training epoch %i' % epoch)
                t.set_postfix(loss=losses.item())
                t.update()

                raw_loss+=losses.item()
        print('epoch : ',epoch,'loss:',raw_loss/num_batches_per_epoch)

        valid_loss=0
        with tqdm(total=num_validation_batches_per_epoch) as t:
            for batch in dataloader_validation:
                inputs=torch.from_numpy(batch['data']).cuda()
                target=torch.from_numpy(batch['seg'].astype(int)).cuda()
                target=get_multi_class_labels(target,n_labels=n_labels)
                with torch.no_grad():
                    outputs=model(inputs)
                    losses=loss(outputs,target)
                t.set_description('Valid epoch %i' % epoch)
                t.set_postfix(loss=losses.item())
                t.update()
                valid_loss+=losses.item()
        print('epoch : ',epoch,'valid loss:',valid_loss/num_validation_batches_per_epoch)
        log.write('%i,%f,%f\n' % (epoch,raw_loss/num_batches_per_epoch,valid_loss/num_validation_batches_per_epoch))
        log.flush()

        if valid_loss/num_validation_batches_per_epoch<min_loss:
            torch.save(model.state_dict(),model_path)
            min_loss=valid_loss/num_validation_batches_per_epoch
        current_lr=adjust_lr(optimizer,current_lr,lr_schedule)
        if epoch==149:
            torch.save(model.state_dict(),model_path_150)
        if epoch==299:
            torch.save(model.state_dict(),model_path_300)

    log.close()
