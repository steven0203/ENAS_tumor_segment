"""The module for training ENAS."""
import contextlib
import glob
import math
import os

import numpy as np
import scipy.signal
import torch
from torch import nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from loss import DiceLoss,MulticlassDiceLoss
from brats_dataloader import *
from batchgenerators.dataloading import MultiThreadedAugmenter

import models
import utils
import pickle
import random
import time




def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]


def _get_optimizer(name):
    if name.lower() == 'sgd':
        optim = torch.optim.SGD
    elif name.lower() == 'adam':
        optim = torch.optim.Adam

    return optim


def _get_no_grad_ctx_mgr():
    """Returns a the `torch.no_grad` context manager for PyTorch version >=
    0.4, or a no-op context manager otherwise.
    """
    if float(torch.__version__[0:3]) >= 0.4:
        return torch.no_grad()

    return contextlib.suppress()



class Trainer(object):
    """A class to wrap training code."""
    def __init__(self, args,logger):
        """Constructor for training algorithm.

        Args:
            args: From command line, picked up by `argparse`.

        Initializes:
            - Data: train, val and test.
            - Model: shared and controller.
            - Inference: optimizers for shared and controller parameters.
            - Criticism: cross-entropy loss for training the shared model.
        """
        self.args = args
        self.controller_step = 0
        self.cuda = args.cuda
        self.epoch = 0
        self.shared_step = 0
        self.start_epoch = 0
        self.logger=logger
        """Load dataset""" 
        self.load_dataset()
        if args.mode=='train':
            self.train_data_loader.restart()


        if args.use_tensorboard:
            self.tb = TensorBoard(args.model_dir)
        else:
            self.tb = None
        self.build_model()

        if self.args.load_path:
            self.load_model()

        shared_optimizer = _get_optimizer(self.args.shared_optim)
        controller_optimizer = _get_optimizer(self.args.controller_optim)

        self.shared_optim = shared_optimizer(
            self.shared.parameters(),
            lr=self.shared_lr,)

        self.controller_optim = controller_optimizer(
            self.controller.parameters(),
            lr=self.args.controller_lr)

        self.ce = nn.CrossEntropyLoss()
        if self.args.loss=='MulticlassDiceLoss':
            self.model_loss=MulticlassDiceLoss()
        else:
            self.model_loss=DiceLoss()
        self.time=time.time()
        self.dag_file=open(self.args.model_dir+'/'+self.args.mode+'_dag.log','a')

    def load_dataset(self):
        train = get_file_list(self.args.data_path,self.args.train_ids_path)
        val = get_file_list(self.args.data_path,self.args.valid_ids_path)
        
        shapes = [brats_dataloader.load_patient(i)[0].shape[1:] for i in train]
        max_shape = np.max(shapes, 0)
        max_shape = list(np.max((max_shape, self.args.patch_size), 0))

        dataloader_train = brats_dataloader(train, self.args.batch_size, max_shape,self.args.num_threads)
        dataloader_validation = brats_dataloader(val,self.args.batch_size, None,1)

        tr_transforms = get_train_transform(self.args.patch_size)

        tr_gen = MultiThreadedAugmenter(dataloader_train, tr_transforms, num_processes=self.args.num_threads,
                                        num_cached_per_queue=3,
                                        seeds=None, pin_memory=False)

        self.num_batches_per_epoch=int(math.ceil(len(train)/self.args.batch_size))
        self.train_data_loader=tr_gen
        self.val=val
    
    def build_model(self):
        """Creates and initializes the shared and controller models."""
        if self.args.network_type == 'unet':
            self.shared = models.Unet(self.args)
        else:
            raise NotImplementedError(f'Network type '
                                      f'`{self.args.network_type}` is not '
                                      f'defined')
        self.controller = models.Controller(self.args)

        if self.args.num_gpu == 1:
            self.shared.cuda()
            self.controller.cuda()
        elif self.args.num_gpu > 1:
            raise NotImplementedError('`num_gpu > 1` is in progress')

    def train(self, single=False):
        """Cycles through alternately training the shared parameters and the
        controller, as described in Section 2.2, Training ENAS and Deriving
        Architectures, of the paper.

        From the paper (for Penn Treebank):

        - In the first phase, shared parameters omega are trained for 400
          steps, each on a minibatch of 64 examples.

        - In the second phase, the controller's parameters are trained for 2000
          steps.
          
        Args:
            single (bool): If True it won't train the controller and use the
                           same dag instead of derive().
        """
        dag = utils.load_dag(self.args) if single else None
        
        if self.args.shared_initial_step > 0:
            self.train_shared(self.args.shared_initial_step)
            self.train_controller()

        for self.epoch in range(self.start_epoch, self.args.max_epoch):
            # 1. Training the shared parameters omega of the child models
            self.train_shared(dag=dag)

            # 2. Training the controller parameters theta
            if not single:
                self.train_controller()

            if self.epoch % self.args.save_epoch == 0 and self.epoch!=0:
                with _get_no_grad_ctx_mgr():
                    best_dag = dag if dag else self.derive()
                    self.evaluate(best_dag,batch_size=self.args.batch_size)
                self.save_model()
            """
            if self.epoch >= self.args.shared_decay_after:
                utils.update_lr(self.shared_optim, self.shared_lr)
            """
        self.dag_file.close()
        
    def get_loss(self, inputs, targets, dags):
        """Computes the loss for the same batch for M models.

        This amounts to an estimate of the loss, which is turned into an
        estimate for the gradients of the shared model.
        """
        if not isinstance(dags, list):
            dags = [dags]

        loss = 0
        for dag in dags:
            output = self.shared(inputs, dag)
            sample_loss = (self.model_loss(output, targets) /
                           self.args.shared_num_sample)
            loss += sample_loss

        loss =loss/len(dags)

        return loss

    def get_score(self,inputs,targets,dags):
        if not isinstance(dags, list):
            dags = [dags]

        score=0
        for dag in dags:
            outputs = self.shared(inputs, dag)
            outputs = torch.argmax(outputs,dim=1)
            outputs = get_multi_class_labels(outputs,self.args.n_classes)
            score += self.model_loss(outputs,targets)
        
        return score/len(dags)


    def train_shared(self, dag=None):
        """Train the language model for 400 steps of minibatches of 64
        examples.

        Args:
            max_step: Used to run extra training steps as a warm-up.
            dag: If not None, is used instead of calling sample().

        BPTT is truncated at 35 timesteps.

        For each weight update, gradients are estimated by sampling M models
        from the fixed controller policy, and averaging their gradients
        computed on a batch of training data.
        """
        model = self.shared
        model.train()
        self.controller.eval()
        raw_total_loss = 0
        total_loss = 0

        for step in range(self.num_batches_per_epoch):
            dags = dag if dag else self.controller.sample(
                self.args.shared_num_sample)
            
            batch=next(self.train_data_loader)
            inputs=torch.from_numpy(batch['data']).cuda()
            targets=torch.from_numpy(batch['seg'].astype(int))
            targets=get_multi_class_labels(targets,n_labels=self.args.n_classes).cuda()
            
            print('epoch :',self.epoch,'step :', step, 'time:' ,time.time()-self.time)
            print(dags[0])
            #print('momery',torch.cuda.memory_allocated(device=None))

            loss = self.get_loss(inputs,targets,dags)
            raw_total_loss += loss.data
            
            #print('after model momery',torch.cuda.memory_allocated(device=None))
            print('loss :', loss.item())


            # update
            self.shared_optim.zero_grad()
            loss.backward()
            self.shared_optim.step()

            total_loss += loss.data

            if ((step % self.args.log_step) == 0) and (step > 0):
                self._summarize_shared_train(total_loss, raw_total_loss)
                raw_total_loss = 0
                total_loss = 0


    def get_reward(self, dags, entropies,inputs,targets):
        """Computes the dicescore of a single sampled model on a minibatch of
        validation data.
        """
        if not isinstance(entropies, np.ndarray):
            entropies = entropies.data.cpu().numpy()

        score=self.get_score(inputs,targets,dags)
        R = utils.to_item(score.data)

        if self.args.entropy_mode == 'reward':
            rewards = R + self.args.entropy_coeff * entropies
        elif self.args.entropy_mode == 'regularizer':
            rewards = R * np.ones_like(entropies)
        else:
            raise NotImplementedError(f'Unkown entropy mode: {self.args.entropy_mode}')

        return rewards

    def train_controller(self):
        """Fixes the shared parameters and updates the controller parameters.

        The controller is updated with a score function gradient estimator
        (i.e., REINFORCE), with the reward being c/valid_ppl, where valid_ppl
        is computed on a minibatch of validation data.

        A moving average baseline is used.

        The controller is trained for 2000 steps per epoch (i.e.,
        first (Train Shared) phase -> second (Train Controller) phase).
        """
        model = self.controller
        model.train()
        # TODO(brendan): Why can't we call shared.eval() here? Leads to loss
        # being uniformly zero for the controller.
        # self.shared.eval()

        avg_reward_base = None
        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []

        total_loss = 0
        valid_idx = 0

        valid_dataloader=brats_dataloader(self.val,self.args.batch_size, None,1)
        num_validation_batches_per_epoch=int(math.ceil(len(self.val)/self.args.batch_size))
        for step in range(num_validation_batches_per_epoch):
            if step>self.args.controller_max_step:
                break
            # sample models
            dags, log_probs, entropies = self.controller.sample(
                with_details=True)

            # calculate reward
            np_entropies = entropies.data.cpu().numpy()
            # NOTE(brendan): No gradients should be backpropagated to the
            # shared model during controller training, obviously.
            with _get_no_grad_ctx_mgr():
                batch=next(valid_dataloader)
                inputs=torch.from_numpy(batch['data']).cuda()
                targets=torch.from_numpy(batch['seg'].astype(int))
                targets=get_multi_class_labels(targets,n_labels=self.args.n_classes).cuda()
                #print('momery',torch.cuda.memory_allocated(device=None))
                rewards = self.get_reward(dags,np_entropies,inputs,targets)
                #print('after model momery',torch.cuda.memory_allocated(device=None))

            # discount
            if 1 > self.args.discount > 0:
                rewards = discount(rewards, self.args.discount)

            reward_history.extend(rewards)
            entropy_history.extend(np_entropies)

            # moving average baseline
            if baseline is None:
                baseline = rewards
            else:
                decay = self.args.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * rewards

            adv = rewards - baseline
            adv_history.extend(adv)

            # policy loss
            loss = -log_probs*utils.get_variable(adv,
                                                 self.cuda,
                                                 requires_grad=False)
            if self.args.entropy_mode == 'regularizer':
                loss -= self.args.entropy_coeff * entropies

            loss = loss.sum()  # or loss.mean()

            # update
            self.controller_optim.zero_grad()
            loss.backward()

            if self.args.controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(),
                                              self.args.controller_grad_clip)
            self.controller_optim.step()

            total_loss += utils.to_item(loss.data)
            self.controller_step += 1

        self._summarize_controller_train(total_loss,
                                        adv_history,
                                        entropy_history,
                                        reward_history,
                                        avg_reward_base)





    def evaluate(self, dag, batch_size=1, ):
        """Evaluate on the validation set.

        NOTE(brendan): We should not be using the test set to develop the
        algorithm (basic machine learning good practices).
        """
        self.shared.eval()
        self.controller.eval()

        val_loss = 0
        dice_score=0
        valid_dataloader=brats_dataloader(self.val,self.args.batch_size, None,1,infinite=False,return_incomplete=True)
        for batch in valid_dataloader:
            inputs=torch.from_numpy(batch['data']).cuda()
            targets=torch.from_numpy(batch['seg'].astype(int))
            targets=get_multi_class_labels(targets,n_labels=self.args.n_classes).cuda()
            
            loss = self.get_loss(inputs,targets,dag)
            val_loss += uitls.to_item(loss)
            dice_score +=uitls.to_item(self.get_score(inputs,targets,dag))

        val_loss =val_loss/len(valid_dataloader)
        dice_score=dice_score/len(valid_dataloader)

        """
        self.tb.scalar_summary(f'eval/{name}_loss', val_loss, self.epoch)
        self.tb.scalar_summary(f'eval/{name}_dice_score', dice_score, self.epoch)
        """
        self.logger.info(f'eval | loss: {val_loss:8.2f} | dice_score: {dice_score:8.2f}')
        return dice_score

    def derive(self, sample_num=None, valid_idx=0):
        """TODO(brendan): We are always deriving based on the very first batch
        of validation data? This seems wrong...
        """
        if sample_num is None:
            sample_num = self.args.derive_num_sample

        dags, _, entropies = self.controller.sample(sample_num,
                                                    with_details=True)

        max_score = 0
        best_dag = None

        valid_dataloader=brats_dataloader(self.val,self.args.batch_size, None,1)
        batch=next(valid_dataloader)
        inputs=torch.from_numpy(batch['data']).cuda()
        targets=torch.from_numpy(batch['seg'].astype(int))
        targets=get_multi_class_labels(targets,n_labels=self.args.n_classes).cuda()

        self.dag_file.write('Epoch : %i \n' %self.epoch)
        for dag in dags:
            dag=[dag]
            score = self.get_score(inputs,targets,dag)
            self.dag_file.write(str(dag)+'\n'+str(score.item()))
            if score > max_score:
                max_score = score
                best_dag = dag
        self.dag_file.write('best_dag :'+str(best_dag)+'\n'+str(max_score))
        self.dag_file.flush()
        self.logger.info(f'derive | max_score: {max_R:8.6f}')
        #fname = (f'{self.epoch:03d}-{self.controller_step:06d}-'
        #         f'{max_R:6.4f}-best.png')
        #path = os.path.join(self.args.model_dir, 'networks', fname)
        #utils.draw_network(best_dag, path)
        #self.tb.image_summary('derive/best', [path], self.epoch)

        return best_dag

    
    def derive_final(self, sample_num=None):
        """TODO(brendan): We are always deriving based on the very first batch
        of validation data? This seems wrong...
        """
        if sample_num is None:
            sample_num = self.args.derive_num_sample

        dags, _, entropies = self.controller.sample(sample_num,
                                                    with_details=True)
        max_score = 0
        for dag in dags:
            with _get_no_grad_ctx_mgr():
                dice_score=self.evaluate([dag],batch_size=self.args.test_batch_size)
            print(dag)
            self.dag_file.write(str(dag)+' '+str(dice_score)+'\n')
            if dice_score > max_score:
                max_score=dice_score
                best_dag = dag
            
        self.dag_file.write('best_dag :'+str(best_dag)+' '+str(max_score)+'\n')
        self.dag_file.flush()
        self.dag_file.close()



    @property
    def shared_lr(self):
        degree = max(self.epoch - self.args.shared_decay_after + 1, 0)
        return self.args.shared_lr * (self.args.shared_decay ** degree)

    @property
    def controller_lr(self):
        return self.args.controller_lr

    @property
    def shared_path(self):
        return f'{self.args.model_dir}/shared_epoch{self.epoch}_step{self.shared_step}.pth'

    @property
    def controller_path(self):
        return f'{self.args.model_dir}/controller_epoch{self.epoch}_step{self.controller_step}.pth'

    def get_saved_models_info(self):
        paths = glob.glob(os.path.join(self.args.model_dir, '*.pth'))
        paths.sort()

        def get_numbers(items, delimiter, idx, replace_word, must_contain=''):
            return list(set([int(
                    name.split(delimiter)[idx].replace(replace_word, ''))
                    for name in basenames if must_contain in name]))

        basenames = [os.path.basename(path.rsplit('.', 1)[0]) for path in paths]
        epochs = get_numbers(basenames, '_', 1, 'epoch')
        shared_steps = get_numbers(basenames, '_', 2, 'step', 'shared')
        controller_steps = get_numbers(basenames, '_', 2, 'step', 'controller')

        epochs.sort()
        shared_steps.sort()
        controller_steps.sort()

        return epochs, shared_steps, controller_steps

    def save_model(self):
        torch.save(self.shared.state_dict(), self.shared_path)
        self.logger.info(f'[*] SAVED: {self.shared_path}')

        torch.save(self.controller.state_dict(), self.controller_path)
        self.logger.info(f'[*] SAVED: {self.controller_path}')

        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        for epoch in epochs[:-self.args.max_save_num]:
            paths = glob.glob(
                os.path.join(self.args.model_dir, f'*_epoch{epoch}_*.pth'))

            for path in paths:
                utils.remove_file(path)

    def load_model(self):
        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        if len(epochs) == 0:
            self.logger.info(f'[!] No checkpoint found in {self.args.model_dir}...')
            return

        self.epoch = self.start_epoch = max(epochs)
        self.shared_step = max(shared_steps)
        self.controller_step = max(controller_steps)

        if self.args.num_gpu == 0:
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        self.shared.load_state_dict(
            torch.load(self.shared_path, map_location=map_location))
        self.logger.info(f'[*] LOADED: {self.shared_path}')

        self.controller.load_state_dict(
            torch.load(self.controller_path, map_location=map_location))
        self.logger.info(f'[*] LOADED: {self.controller_path}')

    def _summarize_controller_train(self,
                                    total_loss,
                                    adv_history,
                                    entropy_history,
                                    reward_history,
                                    avg_reward_base):
        """Logs the controller's progress for this training epoch."""
        cur_loss = total_loss / self.args.log_step

        avg_adv = np.mean(adv_history)
        avg_entropy = np.mean(entropy_history)
        avg_reward = np.mean(reward_history)

        if avg_reward_base is None:
            avg_reward_base = avg_reward

        self.logger.info(
            f'| epoch {self.epoch:3d} | lr {self.controller_lr:.5f} '
            f'| R {avg_reward:.5f} | entropy {avg_entropy:.4f} '
            f'| loss {cur_loss:.5f}')

        # Tensorboard
        if self.tb is not None:
            self.tb.scalar_summary('controller/loss',
                                   cur_loss,
                                   self.controller_step)
            self.tb.scalar_summary('controller/reward',
                                   avg_reward,
                                   self.controller_step)
            self.tb.scalar_summary('controller/reward-B_per_epoch',
                                   avg_reward - avg_reward_base,
                                   self.controller_step)
            self.tb.scalar_summary('controller/entropy',
                                   avg_entropy,
                                   self.controller_step)
            self.tb.scalar_summary('controller/adv',
                                   avg_adv,
                                   self.controller_step)
            """
            paths = []
            for dag in dags:
                fname = (f'{self.epoch:03d}-{self.controller_step:06d}-'
                         f'{avg_reward:6.4f}.png')
                path = os.path.join(self.args.model_dir, 'networks', fname)
                #utils.draw_network(dag, path)
                paths.append(path)
            self.tb.image_summary('controller/sample',
                                  paths,
                                  self.controller_step)
            """
    def _summarize_shared_train(self, total_loss, raw_total_loss):
        """Logs a set of training steps."""
        cur_loss = utils.to_item(total_loss) / self.args.log_step
        # NOTE(brendan): The raw loss, without adding in the activation
        # regularization terms, should be used to compute ppl.
        cur_raw_loss = utils.to_item(raw_total_loss) / self.args.log_step

        self.logger.info(f'| epoch {self.epoch:3d} '
                    f'| lr {self.shared_lr:4.2f} '
                    f'| raw loss {cur_raw_loss:.2f} '
                    f'| loss {cur_loss:.2f} ')
        """
        # Tensorboard
        if self.tb is not None:
            self.tb.scalar_summary('shared/loss',
                                   cur_loss,
                                   self.shared_step)
        """
