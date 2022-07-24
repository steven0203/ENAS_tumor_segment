load_path=$1
data_path=BRATS2015_precessed
n_classes=5

python evaluate_final.py \
       --load_path $load_path \
       --data_path $data_path \
       --n_classes $n_classes
