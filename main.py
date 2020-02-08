"""Entry point."""
import os

import torch

import config
import utils
import trainer
import pickle

logger = utils.get_logger()


def main(args):  # pylint:disable=redefined-outer-name
    """main: Entry point."""
    utils.prepare_dirs(args)

    torch.manual_seed(args.random_seed)

    if args.num_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)

    if args.dataset != 'tumor':
        raise NotImplementedError(f"{args.dataset} is not supported")

    trnr = trainer.Trainer(args)

    if args.mode == 'train':
        utils.save_args(args)
        trnr.train()
    elif args.mode == 'derive':
        assert args.load_path != "", ("`--load_path` should be given in "
                                      "`derive` mode")
        trnr.derive_final()
    elif args.mode == 'test':
        if not args.load_path:
            raise Exception("[!] You should specify `load_path` to load a "
                            "pretrained model")
        trnr.test()
    elif args.mode == 'single':
        if not args.dag_path:
            raise Exception("[!] You should specify `dag_path` to load a dag")
        utils.save_args(args)
        trnr.train(single=True)
    else:
        raise Exception(f"[!] Mode not found: {args.mode}")

if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)
