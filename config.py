import argparse
from utils import get_logger

logger = get_logger()


arg_lists = []
parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--network_type', type=str, choices=['rnn', 'cnn','unet'], default='unet')

# Controller
net_arg.add_argument('--num_blocks', type=int, default=2)
net_arg.add_argument('--tie_weights', type=str2bool, default=True)
net_arg.add_argument('--controller_hid', type=int, default=100)
net_arg.add_argument('--multi_layer', type=str2bool, default=True)
net_arg.add_argument('--lstm_layer',type=int,default=2)

net_arg.add_argument('--shared_cnn_types', type=eval,
                     default="['3x3x3', '3x3x3 dilation 2', '3x3x3 dilation 3', 'avg pool', 'identity']")
net_arg.add_argument('--filters',type=int,default=8)
net_arg.add_argument('--layers',type=int,default=4)
net_arg.add_argument('--patch_size', type=tuple, default=(128,128,128))
net_arg.add_argument('--n_classes',type=int,default=5)
net_arg.add_argument('--in_channels',type=int,default=4)

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='tumor')
data_arg.add_argument('--data_path', type=str, default='BRATS2015_precessed')
data_arg.add_argument('--train_ids_path', type=str, default='train_ids.pkl')
data_arg.add_argument('--valid_ids_path', type=str, default='valid_ids.pkl')
data_arg.add_argument('--num_threads', type=int, default=8)


# Training / test parameters
learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--mode', type=str, default='train',
                       choices=['train', 'derive', 'test', 'single'],
                       help='train: Training ENAS, derive: Deriving Architectures,\
                       single: training one dag')
learn_arg.add_argument('--batch_size', type=int, default=2)
learn_arg.add_argument('--test_batch_size', type=int, default=2)
learn_arg.add_argument('--max_epoch', type=int, default=300)
learn_arg.add_argument('--entropy_mode', type=str, default='reward', choices=['reward', 'regularizer'])


# Controller
learn_arg.add_argument('--ppl_square', type=str2bool, default=False)
# NOTE(brendan): (Zoph and Le, 2017) page 8 states that c is a constant,
# usually set at 80.
learn_arg.add_argument('--reward_c', type=int, default=80,
                       help="WE DON'T KNOW WHAT THIS VALUE SHOULD BE") # TODO
# NOTE(brendan): irrelevant for actor critic.
learn_arg.add_argument('--ema_baseline_decay', type=float, default=0.95) # TODO: very important
learn_arg.add_argument('--discount', type=float, default=1.0) # TODO
learn_arg.add_argument('--controller_max_step', type=int, default=30,
                       help='step for controller parameters')
learn_arg.add_argument('--controller_optim', type=str, default='adam')
learn_arg.add_argument('--controller_lr', type=float, default=3.5e-4,
                       help="will be ignored if --controller_lr_cosine=True")
learn_arg.add_argument('--controller_lr_cosine', type=str2bool, default=False)
learn_arg.add_argument('--controller_lr_max', type=float, default=0.05,
                       help="lr max for cosine schedule")
learn_arg.add_argument('--controller_lr_min', type=float, default=0.001,
                       help="lr min for cosine schedule")
learn_arg.add_argument('--controller_grad_clip', type=float, default=0)
learn_arg.add_argument('--tanh_c', type=float, default=2.5)
learn_arg.add_argument('--softmax_temperature', type=float, default=5.0)
learn_arg.add_argument('--entropy_coeff', type=float, default=0.1)

# Shared parameters
learn_arg.add_argument('--shared_initial_step', type=int, default=0)
learn_arg.add_argument('--shared_max_step', type=int, default=400,
                       help='step for shared parameters')
# NOTE(brendan): Should be 10 for CNN architectures.
learn_arg.add_argument('--shared_num_sample', type=int, default=1,
                       help='# of Monte Carlo samples')
learn_arg.add_argument('--shared_optim', type=str, default='adam')
learn_arg.add_argument('--shared_lr', type=float, default=0.001)
learn_arg.add_argument('--shared_decay', type=float, default=0.985)
learn_arg.add_argument('--shared_decay_after', type=float, default=15)
learn_arg.add_argument('--shared_l2_reg', type=float, default=5e-5)

# Deriving Architectures
learn_arg.add_argument('--derive_num_sample', type=int, default=20)


# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=50)
misc_arg.add_argument('--save_epoch', type=int, default=10)
misc_arg.add_argument('--max_save_num', type=int, default=4)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--num_gpu', type=int, default=1)
misc_arg.add_argument('--random_seed', type=int, default=12345)
misc_arg.add_argument('--use_tensorboard', type=str2bool, default=False)
misc_arg.add_argument('--dag_path', type=str, default='')
misc_arg.add_argument('--dag_log', type=str, default='dags.log')
misc_arg.add_argument('--loss', type=str, default='MulticlassDiceLoss')

#Reference Network
ref_net_arg=add_argument_group('Reference Network')
ref_net_arg.add_argument('--use_ref',type=str2bool,default=False)
ref_net_arg.add_argument('--ref_arch',type=eval,default="[[0, '3x3x3'], [0, 'identity'], [1, '3x3x3'], [0, 'identity'], \
                                                         [0, '3x3x3'], [0, 'identity'], [1, '3x3x3'], [0, 'identity'],\
                                                         [0, '3x3x3'], [0, 'identity'], [1, '3x3x3'], [0, 'identity'],\
                                                         [0, '3x3x3'], [0, 'identity'], [1, '3x3x3'], [0, 'identity'],\
                                                         [0, '3x3x3'], [0, 'identity'], [1, '3x3x3'], [0, 'identity']]")
ref_net_arg.add_argument('--ref_model_num',type=int,default=5)
ref_net_arg.add_argument('--ref_controller_num',type=int,default=2)

def get_args():
    """Parses all of the arguments above, which mostly correspond to the
    hyperparameters mentioned in the paper.
    """
    args, unparsed = parser.parse_known_args()
    if args.num_gpu > 0:
        setattr(args, 'cuda', True)
    else:
        setattr(args, 'cuda', False)
    if len(unparsed) > 1:
        logger.info(f"Unparsed args: {unparsed}")
    return args, unparsed
