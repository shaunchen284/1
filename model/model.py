import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import shutil
from pathlib import Path
'''
这是一个名为RNNPredictor的类，它是一个容器模块，包含一个编码器、一个循环模块和一个解码器。该类的构造函数接受以下参数：
rnn_type: 循环神经网络的类型，可以是LSTM、GRU、SRU、RNN_TANH或RNN_RELU
enc_inp_size: 编码器输入的大小
rnn_inp_size: 循环神经网络输入的大小
rnn_hid_size: 循环神经网络隐藏层的大小
dec_out_size:  解码器输出的大小
nlayers: 循环神经网络中的层数
dropout: dropout概率，默认为0.5
tie_weights: 是否将解码器权重与编码器权重绑定，默认为False
res_connection: 是否使用残差连接，默认为False
该类包含以下成员变量：
enc_input_size: 编码器输入的大小
drop: dropout层
encoder: 编码器
rnn: 循环神经网络
decoder: 解码器
res_connection: 是否使用残差连接
rnn_type: 循环神经网络的类型
rnn_hid_size: 循环神经网络隐藏层的大小
nlayers: 循环神经网络中的层数
其中，编码器和解码器都是由线性层组成，循环神经网络可以是LSTM、GRU、SRU或RNN。
如果循环神经网络是LSTM或GRU，则使用PyTorch中对应的类；如果循环神经网络是SRU，则使用cuda_functional中的SRU类；
如果循环神经网络是RNN_TANH或RNN_RELU，则使用PyTorch中对应的类。如果循环神经网络不是LSTM、GRU、SRU、RNN_TANH或RNN_RELU，
则会抛出异常。如果tie_weights为True，则解码器权重与编码器权重绑定。
'''
class RNNPredictor(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    '''
    编码器和解码器都是由线性层组成，循环神经网络可以是LSTM、GRU、SRU或RNN。
    如果循环神经网络是LSTM或GRU，则使用PyTorch中对应的类；如果循环神经网络是SRU，则使用cuda_functional中的SRU类；
    如果循环神经网络是RNN_TANH或RNN_RELU，则使用PyTorch中对应的类。
    如果循环神经网络不是LSTM、GRU、SRU、RNN_TANH或RNN_RELU，则会抛出异常。
    如果tie_weights为True，则解码器权重与编码器权重绑定。
    '''
    def __init__(self, rnn_type, enc_inp_size, rnn_inp_size, rnn_hid_size, dec_out_size, nlayers, dropout=0.5,
                 tie_weights=False,res_connection=False):
        super(RNNPredictor, self).__init__()
        self.enc_input_size = enc_inp_size

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Linear(enc_inp_size, rnn_inp_size)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(rnn_inp_size, rnn_hid_size, nlayers, dropout=dropout)
            '''
            getattr(nn, rnn_type)返回的是一个类，后面的括号是用来创建这个类的实例的。
            例如，如果rnn_type是LSTM，则getattr(nn, rnn_type)返回的是PyTorch中的LSTM类，
            后面的括号则用来创建一个LSTM类的实例。
            '''
        elif rnn_type == 'SRU':
            from cuda_functional import SRU, SRUCell
            self.rnn = SRU(input_size=rnn_inp_size,hidden_size=rnn_hid_size,num_layers=nlayers,dropout=dropout,
                           use_tanh=False,use_selu=True,layer_norm=True)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
                '''如果rnn_type不在字典{‘RNN_TANH’: ‘tanh’, ‘RNN_RELU’: ‘relu’}中，则会抛出KeyError异常'''
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'SRU', 'RNN_TANH' or 'RNN_RELU']""")
            '''try语句块中包含可能会抛出异常的代码，如果try语句块中的代码执行成功，则跳过except语句块；
            否则，执行except语句块中的代码。'''
            self.rnn = nn.RNN(rnn_inp_size, rnn_hid_size, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(rnn_hid_size, dec_out_size)


        if tie_weights:
            if rnn_hid_size != rnn_inp_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            '''nhid是循环神经网络隐藏层的大小，emsize是rnn_inp_size，即：输入的大小，且emsize是embedding size的缩写'''
            self.decoder.weight = self.encoder.weight
        self.res_connection=res_connection
        self.init_weights()
        self.rnn_type = rnn_type
        self.rnn_hid_size = rnn_hid_size
        self.nlayers = nlayers
        #self.layerNorm1=nn.LayerNorm(normalized_shape=rnn_inp_size)
        #self.layerNorm2=nn.LayerNorm(normalized_shape=rnn_hid_size)
    '''
    这些方法都是用于模型训练和保存的。其中，
    repackage_hidden方法将隐藏状态包装在新的变量中，以将其与历史记录分离；
    save_checkpoint方法用于保存检查点；
    extract_hidden方法用于提取隐藏状态；
    initialize方法用于初始化模型；
    load_checkpoint方法用于加载检查点。
    '''
    def init_weights(self):
        initrange = 0.1
        '''
        uniform_(-initrange, initrange)是一个PyTorch中的函数，用于初始化权重。
        其中，initrange是初始化权重的范围。这个函数会将权重初始化为在[-initrange, initrange]之间的均匀分布。
        '''
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
    '''init_weights()方法用于初始化权重'''
    def forward(self, input, hidden, return_hiddens=False, noise=False):
        emb = self.drop(self.encoder(input.contiguous().view(-1,self.enc_input_size))) # [(seq_len x batch_size) * feature_size]
        '''contiguous()是PyTorch中的一个方法，用于将一个Tensor变成在内存中连续分布的形式'''
        '''与reshape()方法不同，view()方法只能用于具有连续内存布局的Tensor'''
    
        emb = emb.view(-1, input.size(1), self.rnn_hid_size) # [ seq_len * batch_size * feature_size]
        '''
        这里也许是错的，即：self.rnn_hid_size 变成 self.rnn_input_size
        '''
        if noise:
            # emb_noise = Variable(torch.randn(emb.size()))
            # hidden_noise = Variable(torch.randn(hidden[0].size()))
            # if next(self.parameters()).is_cuda:
            #     emb_noise=emb_noise.cuda()
            #     hidden_noise=hidden_noise.cuda()
            # emb = emb+emb_noise
            hidden = (F.dropout(hidden[0],training=True,p=0.9),F.dropout(hidden[1],training=True,p=0.9))
        '''如果需要添加噪声，则对隐藏层进行dropout处理。'''
        #emb = self.layerNorm1(emb)
        output, hidden = self.rnn(emb, hidden)
        #output = self.layerNorm2(output)

        output = self.drop(output)
        '''RNN模型的输出进行dropout处理'''
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2))) # [(seq_len x batch_size) * feature_size]
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1)) # [ seq_len * batch_size * feature_size]
        if self.res_connection:
            decoded = decoded + input
        if return_hiddens:
            return decoded,hidden,output

        return decoded, hidden




    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.rnn_hid_size).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.rnn_hid_size).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.rnn_hid_size).zero_())
    '''初始化隐藏状态'''


    def repackage_hidden(self,h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == tuple:
            return tuple(self.repackage_hidden(v) for v in h)
        else:
            return h.detach()
    '''将隐藏状态包装在新的变量中，以将其与历史记录分离。'''
    def save_checkpoint(self,state, is_best):
        print("=> saving checkpoint ..")
        args = state['args']
        checkpoint_dir = Path('save',args.data,'checkpoint')
        checkpoint_dir.mkdir(parents=True,exist_ok=True)
        checkpoint = checkpoint_dir.joinpath(args.filename).with_suffix('.pth')

        torch.save(state, checkpoint)
        if is_best:
            model_best_dir = Path('save',args.data,'model_best')
            model_best_dir.mkdir(parents=True,exist_ok=True)

            shutil.copyfile(checkpoint, model_best_dir.joinpath(args.filename).with_suffix('.pth'))

        print('=> checkpoint saved.')
    ''' 保存检查点 '''
    def extract_hidden(self, hidden):
        if self.rnn_type == 'LSTM':
            return hidden[0][-1].data.cpu()  # hidden state last layer (hidden[1] is cell state)
        else:
            return hidden[-1].data.cpu()  # last layer
    ''' 提取隐藏状态。'''
    def initialize(self,args,feature_dim):
        self.__init__(rnn_type = args.model,
                           enc_inp_size=feature_dim,
                           rnn_inp_size = args.emsize,
                           rnn_hid_size = args.nhid,
                           dec_out_size=feature_dim,
                           nlayers = args.nlayers,
                           dropout = args.dropout,
                           tie_weights= args.tied,
                           res_connection=args.res_connection)
        self.to(args.device)
        '''将该类转换为指定设备上的张量。'''
    '''初始化模型'''
    def load_checkpoint(self, args, checkpoint, feature_dim):
        start_epoch = checkpoint['epoch'] +1
        best_val_loss = checkpoint['best_loss']
        args_ = checkpoint['args']
        args_.resume = args.resume
        args_.pretrained = args.pretrained
        args_.epochs = args.epochs
        args_.save_interval = args.save_interval
        args_.prediction_window_size=args.prediction_window_size
        self.initialize(args_, feature_dim=feature_dim)
        self.load_state_dict(checkpoint['state_dict'])

        return args_, start_epoch, best_val_loss
    ''' 加载检查点'''