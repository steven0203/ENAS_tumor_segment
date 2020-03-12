"""A module with NAS controller-related code."""
import collections
import os

import torch
import torch.nn.functional as F

import utils
from utils import Node


def _construct_dags(prev_nodes, activations, func_names):
    dags = []
    for nodes, func_ids in zip(prev_nodes, activations):
        dag=[]
        for idx,func_id in zip(nodes,func_ids):
            dag.append([idx.item(),func_names[func_id]])
        dags.append(dag)
    return dags

class Controller(torch.nn.Module):
    """Based on
    https://github.com/pytorch/examples/blob/master/word_language_model/model.py

    TODO(brendan): RL controllers do not necessarily have much to do with
    language models.

    Base the controller RNN on the GRU from:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
    """
    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.args = args
        self.num_tokens=[]
        for idx in range(self.args.num_blocks):
            self.num_tokens+=[idx+1,idx+1,len(args.shared_cnn_types)
                              ,len(args.shared_cnn_types)]
        self.func_names=args.shared_cnn_types

        num_total_tokens = sum(self.num_tokens)

        self.encoder = torch.nn.Embedding(num_total_tokens,
                                          args.controller_hid)

        
        self.lstm = []
        for i in range(self.args.lstm_layer):
            self.lstm.append(torch.nn.LSTMCell(args.controller_hid, args.controller_hid))

        self._lstm=torch.nn.ModuleList(self.lstm)

        #self.lstm = torch.nn.LSTMCell(args.controller_hid, args.controller_hid)

        # TODO(brendan): Perhaps these weights in the decoder should be
        # shared? At least for the activation functions, which all have the
        # same size.
        self.decoders = []
        for idx, size in enumerate(self.num_tokens):
            decoder = torch.nn.Linear(args.controller_hid, size)
            self.decoders.append(decoder)

        self._decoders = torch.nn.ModuleList(self.decoders)

        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)

        def _get_default_hidden(key):
            return utils.get_variable(
                torch.zeros(key, self.args.controller_hid),
                self.args.cuda,
                requires_grad=False)

        self.static_inputs = utils.keydefaultdict(_get_default_hidden)
        self.multi_layer = args.multi_layer
        self.arch_layer= args.layers+1

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)

    def forward(self,  # pylint:disable=arguments-differ
                inputs,
                hidden,
                block_idx,
                is_embed):
        if not is_embed:
            embed = self.encoder(inputs)
        else:
            embed = inputs
        
        return_hidden=[]
        hx=embed
        for i in range(self.args.lstm_layer):
            hx, cx = self.lstm[i](hx, hidden[i])
            return_hidden.append((hx,cx))
        logits = self.decoders[block_idx](hx)

        logits /= self.args.softmax_temperature
        
        # exploration
        if self.args.mode == 'train':
            logits = (self.args.tanh_c*F.tanh(logits))

        return logits, return_hidden

    def sample(self, batch_size=1, with_details=False, save_dir=None):
        """Samples a set of `args.num_blocks` many computational nodes from the
        controller, where each node is made up of an activation function, and
        each node except the last also includes a previous node.
        """
        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')

        # [B, L, H]
        inputs = self.static_inputs[batch_size]
        hidden = [self.static_init_hidden[batch_size] for i in range(self.args.lstm_layer)]

        activations = []
        entropies = []
        log_probs = []
        prev_nodes = []
        # NOTE(brendan): The RNN controller alternately outputs an activation,
        # followed by a previous node, for each block except the last one,
        # which only gets an activation function. The last node is the output
        # node, and its previous node is the average of all leaf nodes.
        if self.multi_layer:
            layers=self.arch_layer
        else :
            layers=1
        for layer in range(layers):
            for block_idx in range(4*self.args.num_blocks):
                logits, hidden = self.forward(inputs,
                                              hidden,
                                              block_idx,
                                              is_embed=(block_idx == 0 and layer==0))

                probs = F.softmax(logits, dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)
                # TODO(brendan): .mean() for entropy?
                entropy = -(log_prob * probs).sum(1, keepdim=False)

                action = probs.multinomial(num_samples=1).data
                selected_log_prob = log_prob.gather(
                    1, utils.get_variable(action, requires_grad=False))

                # TODO(brendan): why the [:, 0] here? Should it be .squeeze(), or
                # .view()? Same below with `action`.
                entropies.append(entropy)
                log_probs.append(selected_log_prob[:, 0])

                # 0,1,:previous node 2,3: function name
                mode = block_idx % 4
                inputs = utils.get_variable(
                    action[:, 0] + sum(self.num_tokens[:block_idx]),
                    requires_grad=False)

                if mode == 2 or mode ==3:
                    activations.append(action[:, 0])
                elif mode == 0 or mode ==1:
                    prev_nodes.append(action[:, 0])

        prev_nodes = torch.stack(prev_nodes).transpose(0, 1)
        activations = torch.stack(activations).transpose(0, 1)

        dags = _construct_dags(prev_nodes,
                               activations,
                               self.func_names)

        if save_dir is not None:
            for idx, dag in enumerate(dags):
                """
                utils.draw_network(dag,
                                   os.path.join(save_dir, f'graph{idx}.png'))
                """
                pass
        if with_details:
            return dags, torch.cat(log_probs), torch.cat(entropies)

        return dags

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.args.controller_hid)
        return (utils.get_variable(zeros, self.args.cuda, requires_grad=False),
                utils.get_variable(zeros.clone(), self.args.cuda, requires_grad=False))

    def forward_with_ref(self,ref_net,batch_size=1):
        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')

        # [B, L, H]
        inputs = self.static_inputs[batch_size]
        hidden = [self.static_init_hidden[batch_size] for i in range(self.args.lstm_layer)]

        entropies = []
        log_probs = []

        if self.multi_layer:
            layers=self.arch_layer
        else :
            layers=1
        for layer in range(layers):
            for block_idx in range(4*self.args.num_blocks):
                logits, hidden = self.forward(inputs,
                                              hidden,
                                              block_idx,
                                              is_embed=(block_idx == 0 and layer==0))
                probs = F.softmax(logits, dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)
                entropy = -(log_prob * probs).sum(1, keepdim=False)
                index = layer*(4*self.args.num_blocks)+block_idx
                action = ref_net[:,index]
                selected_log_prob = log_prob.gather(1, utils.get_variable(action,self.args.cuda, requires_grad=False))
                entropies.append(entropy)
                log_probs.append(selected_log_prob[:, 0])

                inputs = utils.get_variable(
                    action[:, 0] + sum(self.num_tokens[:block_idx]),
                    requires_grad=False)


        return torch.cat(log_probs), torch.cat(entropies)
