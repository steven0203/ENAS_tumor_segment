mode=derive
load_path=$1
data_path=BRATS2018_precessed
derive_num_sample=100
n_classes=4

python main.py \
       --mode $mode \
       --load_path $load_path \
       --derive_num_sample $derive_num_sample \
       --data_path $data_path \
       --n_classes $n_classes
