load_path=$1
data_path=BRATS2018_precessed
n_classes=4

python train_final.py \
       --load_path $load_path \
       --data_path $data_path \
       --n_classes $n_classes
