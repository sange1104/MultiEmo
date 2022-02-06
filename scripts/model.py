''' 
Implements deepmoji and other modules from torchmoji repo
https://github.com/huggingface/torchMoji/tree/master/torchmoji
'''
 
import torch
import math
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
 
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
                            nn.Linear(input_dim, hidden_dim), 
                            nn.Linear(hidden_dim, num_classes)
                       )
        
    def forward(self, x):
        y = self.decoder(x)
        return y.unsqueeze(0)

class MultiEmo(nn.Module):
    '''
    # Arguments
      - aux_task: a list of auxiliary task ex. ['emo', 'sent']
      - torchmoji: pre-trained deepmoji for backbone feature extraction
      - input_dim: the dimension of tensor after passing torchmoji
    '''
    def __init__(self, aux_task, torchmoji, input_dim):
        super(MultiEmo, self).__init__()
        self.backbone = torchmoji 
        self.hidden_dim = 256
        self.aux_task = aux_task
        aux_num = len(self.aux_task)
        self.num_emoji = 64
        self.main_idx = 0
        self.task_to_numclasses = {'emo':27, 'Ekman':6, 'sent':3} 
        self.task_to_idx = {'emo':1, 'Ekman':2, 'sent':3}
        
        # build a decoder dictionary ex. {1:decoder_block_1, 2:decoder_block_2, ...}
        task_to_decoder = {}
        for task in aux_task:
            # get the number of classes for each auxiliary task
            num_classes = self.task_to_numclasses[task]
            # get the task index
            idx = self.task_to_idx[task]
            # construct a decoder
            task_to_decoder[str(idx)] = Decoder(input_dim, self.hidden_dim, num_classes) 
        self.aux_decoders = nn.ModuleDict(task_to_decoder)
        
        self.main_decoder = Decoder(input_dim, self.hidden_dim, self.num_emoji) 

    def forward(self, tokens, tasks): 
        x = self.backbone(tokens) 
        
        # construct output dictionary
        output = {0 : torch.empty(0, self.num_emoji).cuda()}
        for task in self.aux_task:
            idx = self.task_to_idx[task]
            output[idx] = torch.empty(0, self.task_to_numclasses[task]).cuda()
        
        # feed tokens to decoder
        for i in range(len(x)): 
            task = tasks[i].item() 
            # main task
            if task == self.main_idx : 
                y = self.main_decoder(x[i, :])
            # auxiliary task
            else:
                assert task in [1, 2, 3]                
                y = self.aux_decoders[str(task)](x[i, :])
            # cumulate outputs
            output[task] = torch.cat([output[task], y], dim=0) 
 
        return output

def AutogradRNN(input_size, hidden_size, num_layers=1, batch_first=False,
                dropout=0, train=True, bidirectional=False, batch_sizes=None,
                dropout_state=None, flat_weight=None):

    cell = LSTMCell

    if batch_sizes is None:
        rec_factory = Recurrent
    else:
        rec_factory = variable_recurrent_factory(batch_sizes)

    if bidirectional:
        layer = (rec_factory(cell), rec_factory(cell, reverse=True))
    else:
        layer = (rec_factory(cell),)

    func = StackedRNN(layer,
                      num_layers,
                      True,
                      dropout=dropout,
                      train=train)

    def forward(input, weight, hidden):
        if batch_first and batch_sizes is None:
            input = input.transpose(0, 1)

        nexth, output = func(input, hidden, weight)

        if batch_first and batch_sizes is None:
            output = output.transpose(0, 1)

        return output, nexth

    return forward

def Recurrent(inner, reverse=False):
    def forward(input, hidden, weight):
        output = []
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
        for i in steps:
            hidden = inner(input[i], hidden, *weight)
            # hack to handle LSTM
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        if reverse:
            output.reverse()
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return hidden, output

    return forward


def variable_recurrent_factory(batch_sizes):
    def fac(inner, reverse=False):
        if reverse:
            return VariableRecurrentReverse(batch_sizes, inner)
        else:
            return VariableRecurrent(batch_sizes, inner)
    return fac

def VariableRecurrent(batch_sizes, inner):
    def forward(input, hidden, weight):
        output = []
        input_offset = 0
        last_batch_size = batch_sizes[0]
        hiddens = []
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
        for batch_size in batch_sizes:
            step_input = input[input_offset:input_offset + batch_size]
            input_offset += batch_size

            dec = last_batch_size - batch_size
            if dec > 0:
                hiddens.append(tuple(h[-dec:] for h in hidden))
                hidden = tuple(h[:-dec] for h in hidden)
            last_batch_size = batch_size

            if flat_hidden:
                hidden = (inner(step_input, hidden[0], *weight),)
            else:
                hidden = inner(step_input, hidden, *weight)

            output.append(hidden[0])
        hiddens.append(hidden)
        hiddens.reverse()

        hidden = tuple(torch.cat(h, 0) for h in zip(*hiddens))
        assert hidden[0].size(0) == batch_sizes[0]
        if flat_hidden:
            hidden = hidden[0]
        output = torch.cat(output, 0)

        return hidden, output

    return forward


def VariableRecurrentReverse(batch_sizes, inner):
    def forward(input, hidden, weight):
        output = []
        input_offset = input.size(0)
        last_batch_size = batch_sizes[-1]
        initial_hidden = hidden
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
            initial_hidden = (initial_hidden,)
        hidden = tuple(h[:batch_sizes[-1]] for h in hidden)
        for batch_size in reversed(batch_sizes):
            inc = batch_size - last_batch_size
            if inc > 0:
                hidden = tuple(torch.cat((h, ih[last_batch_size:batch_size]), 0)
                               for h, ih in zip(hidden, initial_hidden))
            last_batch_size = batch_size
            step_input = input[input_offset - batch_size:input_offset]
            input_offset -= batch_size

            if flat_hidden:
                hidden = (inner(step_input, hidden[0], *weight),)
            else:
                hidden = inner(step_input, hidden, *weight)
            output.append(hidden[0])

        output.reverse()
        output = torch.cat(output, 0)
        if flat_hidden:
            hidden = hidden[0]
        return hidden, output

    return forward

def StackedRNN(inners, num_layers, lstm=False, dropout=0, train=True):

    num_directions = len(inners)
    total_layers = num_layers * num_directions

    def forward(input, hidden, weight):
        assert(len(weight) == total_layers)
        next_hidden = []

        if lstm:
            hidden = list(zip(*hidden))

        for i in range(num_layers):
            all_output = []
            for j, inner in enumerate(inners):
                l = i * num_directions + j

                hy, output = inner(input, hidden[l], weight[l])
                next_hidden.append(hy)
                all_output.append(output)

            input = torch.cat(all_output, input.dim() - 1)

            if dropout != 0 and i < num_layers - 1:
                input = F.dropout(input, p=dropout, training=train, inplace=False)

        if lstm:
            next_h, next_c = zip(*next_hidden)
            next_hidden = (
                torch.cat(next_h, 0).view(total_layers, *next_h[0].size()),
                torch.cat(next_c, 0).view(total_layers, *next_c[0].size())
            )
        else:
            next_hidden = torch.cat(next_hidden, 0).view(
                total_layers, *next_hidden[0].size())

        return next_hidden, input

    return forward

def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    """
    A modified LSTM cell with hard sigmoid activation on the input, forget and output gates.
    """
    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = hard_sigmoid(ingate)
    forgetgate = hard_sigmoid(forgetgate)
    cellgate = nn.Tanh()(cellgate)
    outgate = hard_sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * nn.Tanh()(cy)

    return hy, cy

def hard_sigmoid(x):
    """
    Computes element-wise hard sigmoid of x.
    See e.g. https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L279
    """
    x = (0.2 * x) + 0.5
    x = F.threshold(-x, -1, -1)
    x = F.threshold(-x, 0, 0)
    return x

class Attention(nn.Module):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, attention_size, return_attention=False):
        """ Initialize the attention layer
        # Arguments:
            attention_size: Size of the attention vector.
            return_attention: If true, output will include the weight for each input token
                              used for the prediction
        """
        super(Attention, self).__init__()
        self.return_attention = return_attention
        self.attention_size = attention_size
        self.attention_vector = Parameter(torch.FloatTensor(attention_size))
        self.attention_vector.data.normal_(std=0.05) # Initialize attention vector

    def __repr__(self):
        s = '{name}({attention_size}, return attention={return_attention})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, input_lengths):
        """ Forward pass.
        # Arguments:
            inputs (Torch.Variable): Tensor of input sequences
            input_lengths (torch.LongTensor): Lengths of the sequences
        # Return:
            Tuple with (representations and attentions if self.return_attention else None).
        """
        logits = inputs.matmul(self.attention_vector)
        unnorm_ai = (logits - logits.max()).exp()

        # Compute a mask for the attention on the padded sequences
        # See e.g. https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
        max_len = unnorm_ai.size(1)
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        mask = Variable((idxes < input_lengths.unsqueeze(1)).float()).cuda()

        # apply mask and renormalize attention scores (weights)
        masked_weights = unnorm_ai * mask
        att_sums = masked_weights.sum(dim=1, keepdim=True)  # sums per sequence
        attentions = masked_weights.div(att_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(dim=1)

        return (representations, attentions if self.return_attention else None)

class LSTMHardSigmoid(nn.Module): 
    def __init__(self, input_size, hidden_size,
                num_layers=1, bias=True, batch_first=False,
                dropout=0, bidirectional=False):
        super(LSTMHardSigmoid, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        gate_size = 4 * hidden_size

        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
                b_ih = Parameter(torch.Tensor(gate_size))
                b_hh = Parameter(torch.Tensor(gate_size))
                layer_params = (w_ih, w_hh, b_ih, b_hh)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)

        self.flatten_parameters()
        self.reset_parameters()

    def flatten_parameters(self):
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this is a no-op wince we don't use CUDA acceleration.
        """
        self._data_ptrs = []

    def _apply(self, fn):
        ret = super(LSTMHardSigmoid, self)._apply(fn)
        self.flatten_parameters()
        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_2, hx=None):
        is_packed = isinstance(input_2, PackedSequence) 
        if is_packed:
            input, batch_sizes = input_2[0], input_2[1]
            max_batch_size = batch_sizes[0]
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.autograd.Variable(input.data.new(self.num_layers *
                                                        num_directions,
                                                        max_batch_size,
                                                        self.hidden_size).zero_(), requires_grad=False)
            hx = (hx, hx)

        has_flat_weights = list(p.data.data_ptr() for p in self.parameters()) == self._data_ptrs
        if has_flat_weights:
            first_data = next(self.parameters()).data
            assert first_data.storage().size() == self._param_buf_size
            flat_weight = first_data.new().set_(first_data.storage(), 0, torch.Size([self._param_buf_size]))
        else:
            flat_weight = None
        func = AutogradRNN(
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            train=self.training,
            bidirectional=self.bidirectional,
            batch_sizes=batch_sizes,
            dropout_state=self.dropout_state,
            flat_weight=flat_weight
        )
        output, hidden = func(input, self.all_weights, hx)
        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, hidden

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def __setstate__(self, d):
        super(LSTMHardSigmoid, self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                if self.bias:
                    self._all_weights += [weights]
                else:
                    self._all_weights += [weights[:2]]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]
    
class TorchMoji(nn.Module):
    def __init__(self, nb_classes=None, nb_tokens=50000, feature_output=True, output_logits=False,
                 embed_dropout_rate=0, final_dropout_rate=0, return_attention=False):
        """
        torchMoji model.
        IMPORTANT: The model is loaded in evaluation mode by default (self.eval())

        # Arguments:
            nb_classes: Number of classes in the dataset.
            nb_tokens: Number of tokens in the dataset (i.e. vocabulary size).
            feature_output: If True the model returns the penultimate
                            feature vector rather than Softmax probabilities
                            (defaults to False).
            output_logits:  If True the model returns logits rather than probabilities
                            (defaults to False).
            embed_dropout_rate: Dropout rate for the embedding layer.
            final_dropout_rate: Dropout rate for the final Softmax layer.
            return_attention: If True the model also returns attention weights over the sentence
                              (defaults to False).
        """
        super(TorchMoji, self).__init__()

        embedding_dim = 256
        hidden_size = 512
        attention_size = 4 * hidden_size + embedding_dim

        self.feature_output = feature_output
        self.embed_dropout_rate = embed_dropout_rate
        self.final_dropout_rate = final_dropout_rate
        self.return_attention = return_attention
        self.hidden_size = hidden_size
        self.output_logits = output_logits
        self.nb_classes = nb_classes

        self.add_module('embed', nn.Embedding(nb_tokens, embedding_dim))
        # dropout2D: embedding channels are dropped out instead of words
        # many exampels in the datasets contain few words that losing one or more words can alter the emotions completely
        self.add_module('embed_dropout', nn.Dropout2d(embed_dropout_rate))
        self.add_module('lstm_0', LSTMHardSigmoid(embedding_dim, hidden_size, batch_first=True, bidirectional=True))
        self.add_module('lstm_1', LSTMHardSigmoid(hidden_size*2, hidden_size, batch_first=True, bidirectional=True))
        self.add_module('attention_layer', Attention(attention_size=attention_size, return_attention=return_attention))
        if not feature_output:
            self.add_module('final_dropout', nn.Dropout(final_dropout_rate))
            if output_logits:
                self.add_module('output_layer', nn.Sequential(nn.Linear(attention_size, nb_classes if self.nb_classes > 2 else 1)))
            else:
                self.add_module('output_layer', nn.Sequential(nn.Linear(attention_size, nb_classes if self.nb_classes > 2 else 1),
                                                              nn.Softmax() if self.nb_classes > 2 else nn.Sigmoid()))
        self.init_weights()
        # Put model in evaluation mode by default
        self.eval()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        nn.init.uniform_(self.embed.weight.data, a=-0.5, b=0.5)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)
        if not self.feature_output:
            nn.init.xavier_uniform(self.output_layer[0].weight.data)

    def forward(self, input_seqs):
        """ Forward pass.

        # Arguments:
            input_seqs: Can be one of Numpy array, Torch.LongTensor, Torch.Variable, Torch.PackedSequence.

        # Return:
            Same format as input format (except for PackedSequence returned as Variable).
        """
        # Check if we have Torch.LongTensor inputs or not Torch.Variable (assume Numpy array in this case), take note to return same format
        return_numpy = False
        return_tensor = False
        if isinstance(input_seqs, (torch.LongTensor, torch.cuda.LongTensor)):
            input_seqs = Variable(input_seqs)
            return_tensor = True
        elif not isinstance(input_seqs, Variable):
            input_seqs = Variable(torch.from_numpy(input_seqs.astype('int64')).long())
            return_numpy = True

        # If we don't have a packed inputs, let's pack it
        reorder_output = False
        if not isinstance(input_seqs, PackedSequence):
            ho = self.lstm_0.weight_hh_l0.data.new(2, input_seqs.size()[0], self.hidden_size).zero_()
            co = self.lstm_0.weight_hh_l0.data.new(2, input_seqs.size()[0], self.hidden_size).zero_()

            # Reorder batch by sequence length
            input_lengths = torch.LongTensor([torch.max(input_seqs[i, :].data.nonzero()) + 1 for i in range(input_seqs.size()[0])])
            input_lengths, perm_idx = input_lengths.sort(0, descending=True)
            input_seqs = input_seqs[perm_idx][:, :input_lengths.max()]

            # Pack sequence and work on data tensor to reduce embeddings/dropout computations
            packed_input = pack_padded_sequence(input_seqs, input_lengths.cpu().numpy(), batch_first=True)
            reorder_output = True
        else:
            ho = self.lstm_0.weight_hh_l0.data.data.new(2, input_seqs.size()[0], self.hidden_size).zero_()
            co = self.lstm_0.weight_hh_l0.data.data.new(2, input_seqs.size()[0], self.hidden_size).zero_()
            input_lengths = input_seqs.batch_sizes
            packed_input = input_seqs

        hidden = (Variable(ho, requires_grad=False), Variable(co, requires_grad=False))

        # Embed with an activation function to bound the values of the embeddings
        x = self.embed(packed_input.data)
        x = nn.Tanh()(x)

        # pyTorch 2D dropout2d operate on axis 1 which is fine for us
        x = self.embed_dropout(x)

        # Update packed sequence data for RNN
        packed_input = PackedSequence(x, packed_input.batch_sizes)

        # skip-connection from embedding to output eases gradient-flow and allows access to lower-level features
        # ordering of the way the merge is done is important for consistency with the pretrained model
        lstm_0_output, _ = self.lstm_0(packed_input, hidden)
        lstm_1_output, _ = self.lstm_1(lstm_0_output, hidden)

        # Update packed sequence data for attention layer
        packed_input = PackedSequence(torch.cat((lstm_1_output.data,
                                                 lstm_0_output.data,
                                                 packed_input.data), dim=1),
                                      packed_input.batch_sizes)

        input_seqs, _ = pad_packed_sequence(packed_input, batch_first=True)

        x, att_weights = self.attention_layer(input_seqs, input_lengths)

        # output class probabilities or penultimate feature vector
        if not self.feature_output:
            x = self.final_dropout(x)
            outputs = self.output_layer(x)
        else:
            outputs = x

        # Reorder output if needed
        if reorder_output:
            reorered = Variable(outputs.data.new(outputs.size()))
            reorered[perm_idx] = outputs
            outputs = reorered

        # Adapt return format if needed
        if return_tensor:
            outputs = outputs.data
        if return_numpy:
            outputs = outputs.data.numpy()

        if self.return_attention:
            return outputs, att_weights
        else:
            return outputs

def load_specific_weights(model, weight_path, exclude_names=[], extend_embedding=0, verbose=False):
    """ Loads model weights from the given file path, excluding any
        given layers.

    # Arguments:
        model: Model whose weights should be loaded.
        weight_path: Path to file containing model weights.
        exclude_names: List of layer names whose weights should not be loaded.
        extend_embedding: Number of new words being added to vocabulary.
        verbose: Verbosity flag.

    # Raises:
        ValueError if the file at weight_path does not exist.
    """
    if not os.path.exists(weight_path):
        raise ValueError('ERROR (load_weights): The weights file at {} does '
                         'not exist. Refer to the README for instructions.'
                         .format(weight_path))

    if extend_embedding and 'embed' in exclude_names:
        raise ValueError('ERROR (load_weights): Cannot extend a vocabulary '
                         'without loading the embedding weights.')

    # Copy only weights from the temporary model that are wanted
    # for the specific task (e.g. the Softmax is often ignored)
    weights = torch.load(weight_path)
    for key, weight in weights.items():
        if any(excluded in key for excluded in exclude_names):
            if verbose:
                print('Ignoring weights for {}'.format(key))
            continue

        try:
            model_w = model.state_dict()[key]
        except KeyError:
            raise KeyError("Weights had parameters {},".format(key)
                           + " but could not find this parameters in model.")

        if verbose:
            print('Loading weights for {}'.format(key))

        # extend embedding layer to allow new randomly initialized words
        # if requested. Otherwise, just load the weights for the layer.
        if 'embed' in key and extend_embedding > 0:
            weight = torch.cat((weight, model_w[NB_TOKENS:, :]), dim=0)
            if verbose:
                print('Extended vocabulary for embedding layer ' +
                      'from {} to {} tokens.'.format(
                        NB_TOKENS, NB_TOKENS + extend_embedding))
        try:
            model_w.copy_(weight)
        except:
            print('While copying the weigths named {}, whose dimensions in the model are'
                  ' {} and whose dimensions in the saved file are {}, ...'.format(
                        key, model_w.size(), weight.size()))
            raise
