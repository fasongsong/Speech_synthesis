from math import sqrt, ceil
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, ConvNorm2d, LinearNorm, LstmNorm
from utils import to_gpu, get_mask_from_lengths
from fp16_optimizer import fp32_to_fp16, fp16_to_fp32
from module import GradientReversal
from module import grad_reverse

# class BiRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super(BiRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection
#
#     def forward(self, x):
#         # Set initial states
#         h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)  # 2 for bidirection
#         c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).
#
#         # Forward propagate LSTM
#         out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
#
#         # Decode the hidden state of the last time step
#         out = self.fc(out[:, -1, :])
#         return out


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=False)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Speaker_Encoder(nn.Module):
    def __init__(self, hparams):
        super(Speaker_Encoder, self).__init__()
        self.z_s = LinearNorm(hparams.reference_dim, hparams.z_speaker_dim)
        self.n_mels = hparams.n_mel_channels
        convolutions = []
        for i in range(hparams.n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm2d(hparams.conv_dim_in[i],
                           hparams.conv_dim_out[i],
                           kernel_size=3, stride=2,
                           padding=1,
                           dilation=1, w_init_gain='relu'),
                nn.BatchNorm2d(hparams.conv_dim_out[i]))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        out_channels = self.calculate_channels(hparams.n_mel_channels, 3, 2, 1, 2)
        self.ceil_n_mel_64 = int(ceil(hparams.n_mel_channels / 64))
        self.lstm = nn.LSTM(hparams.reference_dim*self.ceil_n_mel_64, hparams.reference_dim//2, 1, batch_first=True, bidirectional=True)

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            # elif 'bias' in name:
            #     nn.init.constant(param, 0.0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L

    def forward(self, x, input_lengths):

        N = x.size(0)
        x = x.contiguous().view(N, 1, -1, self.n_mels)  # [N, 1, Ty, n_mels]

        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)        ### 이거쓸지 아래쓸지.
        #
        # for conv in self.convolutions:
        #     x = F.relu(conv(x))

        x = x.transpose(1, 2)

        #input_lengths = input_lengths.cpu().numpy()
        # input_lengths = input_lengths.sort(0, descending=True)[0].cpu().numpy()
        # x = nn.utils.rnn.pack_padded_sequence(
        #     x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        T = x.size(1)
        N = x.size(0)
        x = x.contiguous().view(N, T, -1)
        outputs, _ = self.lstm(x)
        outputs = outputs.mean(1)
        #
        # outputs, _ = nn.utils.rnn.pad_packed_sequence(
        #     outputs, batch_first=True)
        z_s = self.z_s(outputs)
        return z_s

    def inference(self, x):
        N = 1
        x = x.contiguous().view(N, 1, -1, self.n_mels)  # [N, 1, Ty, n_mels]

        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)  ### 이거쓸지 아래쓸지.

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        T = x.size(1)
        N = x.size(0)
        x = x.contiguous().view(N, T, -1)
        outputs, _ = self.lstm(x)
        outputs = outputs.mean(1)

        z_s = self.z_s(outputs)

        return z_s

class Residual_Encoder(nn.Module):
    def __init__(self, hparams):
        super(Residual_Encoder, self).__init__()

        self.n_mels = hparams.n_mel_channels
        convolutions = []
        self.z_r = LinearNorm(hparams.reference_dim, hparams.z_residual_dim)

        self.n_mels = hparams.n_mel_channels
        self.batchs = hparams.batch_size
        self.z_r_dim = hparams.z_residual_dim
        for i in range(hparams.n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm2d(hparams.conv_dim_in[i],
                           hparams.conv_dim_out[i],
                           kernel_size=3, stride=2,
                           padding=1,
                           dilation=1, w_init_gain='relu'),
                nn.BatchNorm2d(hparams.conv_dim_out[i]))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        out_channels = self.calculate_channels(hparams.n_mel_channels, 3, 2, 1, 2)
        self.ceil_n_mel_64 = int(ceil(hparams.n_mel_channels / 64))
        self.lstm = nn.LSTM(hparams.reference_dim*self.ceil_n_mel_64, hparams.reference_dim//2, 1, batch_first=True, bidirectional=True)

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            # elif 'bias' in name:
            #     nn.init.constant(param, 0.0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L

    def forward(self, x, warp_label, input_lengths):

        # x = x.squeeze(1).transpose(2, 1)             ### 있어야하나?
        N = x.size(0)
        x = x.contiguous().view(N, 1, -1, self.n_mels)  # [N, 1, Ty, n_mels]

        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)  ### 이거쓸지 아래쓸지.
        #
        # for conv in self.convolutions:
        #     x = F.relu(conv(x))

        x = x.transpose(1, 2)
        #
        # input_lengths = input_lengths.cpu().numpy()
        # x = nn.utils.rnn.pack_padded_sequence(
        #     x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        T = x.size(1)
        N = x.size(0)
        x = x.contiguous().view(N, T, -1)
        outputs, _ = self.lstm(x)
        outputs = outputs.mean(1)

        outputs = torch.cat([outputs, warp_label], 1)
        return outputs

    def inference(self, x, warp_label):
        N = 1
        x = x.contiguous().view(N, 1, -1, self.n_mels)  # [N, 1, Ty, n_mels]

        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)  ### 이거쓸지 아래쓸지.

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        T = x.size(1)
        N = x.size(0)
        x = x.contiguous().view(N, T, -1)
        outputs, _ = self.lstm(x)
        outputs = outputs.mean(1)

        outputs = torch.cat([outputs, warp_label], 1)
        return outputs


class speaker_classifier(nn.Module):
    def __init__(self,hparams):
        super(speaker_classifier, self).__init__()

        self.hidden1 = LinearNorm(hparams.z_speaker_dim,256)
        self.output = LinearNorm(256, hparams.speaker_num)

    def forward(self, z_s):

        x = F.dropout(F.relu(self.hidden1(z_s)))
        x = self.output(x)

        return x


class augmentation_classifier(nn.Module):
    def __init__(self, hparams):
        super(augmentation_classifier, self).__init__()

        self.hidden2 = LinearNorm(hparams.z_speaker_dim, 256)
        self.output = LinearNorm(256, 2)
        self.lambd = 1

    def forward(self, z_s):
        x = grad_reverse(z_s, self.lambd)
        x = F.dropout(F.relu(self.hidden2(z_s)))
        x = self.output(x)

        return x


class VAE(nn.Module):
    def __init__(self, hparams):
        super(VAE, self).__init__()

        self.fc_r_mu = LinearNorm(hparams.reference_dim+3, hparams.z_residual_dim)
        self.fc_r_lv = LinearNorm(hparams.reference_dim+3, hparams.z_residual_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, inputs):

        mu = self.fc_r_mu(inputs)
        logvar = self.fc_r_lv(inputs)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar


    def inference(self, inputs, ref_logvar=1, condition_on_ref=True):

        if condition_on_ref==True:
            mu = self.fc_r_mu(inputs)
            logvar = self.fc_r_lv(inputs)
            z = self.reparameterize(mu, logvar * ref_logvar)
        else:
            mu = self.fc_r_mu(inputs)
            logvar = self.fc_r_lv(inputs)
            z = self.reparameterize(mu, logvar)

        return z


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            # elif 'bias' in name:
            #     nn.init.constant(param, 0.0)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim + hparams.z_speaker_dim + hparams.z_residual_dim + 3
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim +  self.encoder_embedding_dim,
            hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim,  self.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim +  self.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim +  self.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim +  self.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)
        self.attention_cell = F.dropout(
            self.attention_cell, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)
        self.decoder_cell = F.dropout(
            self.decoder_cell, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        # self.speaker_embedding = LinearNorm(
        #     hparams.n_speakers, hparams.speaker_embedding_dim, bias=True, w_init_gain='tanh')
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
        self.speaker_encoder = Speaker_Encoder(hparams)
        self.residual_encoder = Residual_Encoder(hparams)
        self.speaker_class = speaker_classifier(hparams)
        #self.augmenation_class = nn.Sequential(GradientReversal(),
         #                                      augmentation_classifier())
        self.augmenation_class = augmentation_classifier(hparams)
        self.vae = VAE(hparams)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, speaker_id, aug_labels, warp_labels = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        speaker_id = to_gpu(speaker_id).float()
        aug_labels = to_gpu(aug_labels).float()
        warp_labels = to_gpu(warp_labels).float()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths, warp_labels),
            (mel_padded, gate_padded, speaker_id, aug_labels))

    def parse_input(self, inputs):
        inputs = fp32_to_fp16(inputs) if self.fp16_run else inputs
        return inputs

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        outputs = fp16_to_fp32(outputs) if self.fp16_run else outputs
        return outputs

    def forward(self, inputs):
        inputs, input_lengths, targets, max_len, \
            output_lengths, warp_labels = self.parse_input(inputs)
        input_lengths, output_lengths = input_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(inputs).transpose(1, 2)


        ############################### VAE ##############################

        speaker_encoding = self.speaker_encoder(targets, output_lengths)
        residual_encoding, r_mu, r_lv = self.vae(self.residual_encoder(targets, warp_labels, output_lengths))
        residual_encoding = torch.cat((residual_encoding, warp_labels), dim=-1)
        concat_z = torch.cat((speaker_encoding, residual_encoding), dim=-1)

        #####################################################################

        transcript_outputs = self.encoder(embedded_inputs, input_lengths)
        reference_output = concat_z.unsqueeze(1).expand(-1,input_lengths[0],-1)

        encoder_outputs = torch.cat((transcript_outputs, reference_output), dim=-1)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=input_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        speaker_output = self.speaker_class(speaker_encoding)
        augmentation_output = self.augmenation_class(speaker_encoding)


        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments, r_mu, r_lv, speaker_output, augmentation_output],
            output_lengths)

    ### condition_on_ref : Transfer Or control
    ### condition :   (0) : time,  (1) : freq
    ### logvar : how many

    def inference(self, inputs, refence_mel, ref_length, condition_on_ref=True, condition=0, logvar=0):
        inputs = self.parse_input(inputs)
        embedded_inputs = self.embedding(inputs).transpose(1, 2)

        speaker_encoding = self.speaker_encoder.inference(refence_mel)

        if condition_on_ref:
            if condition == 0:
                warp_label = torch.LongTensor([0, 1, 0])
                residual_encoding = self.vae.inference(self.residual_encoder.inference(refence_mel, warp_label), logvar, condition_on_ref=True)
                residual_encoding = torch.cat((residual_encoding, warp_label), dim=1)
            elif condition == 1:
                warp_label = torch.LongTensor([0, 0, 1])
                residual_encoding = self.vae.inference(self.residual_encoder.inference(refence_mel, warp_label), logvar, condition_on_ref=True)
                residual_encoding = torch.cat((residual_encoding, warp_label), dim=1)
        else:
            warp_label = torch.LongTensor([1, 0, 0])
            residual_encoding = self.vae.inference(self.residual_encoder.inference(refence_mel, warp_label), logvar=1, condition_on_ref=False)
            residual_encoding = torch.cat((residual_encoding, warp_label), dim=1)

        concat_z = torch.cat((speaker_encoding.cuda(), residual_encoding.cuda()), dim=-1)

        transcript_outputs = self.encoder.inference(embedded_inputs)
        reference_output = concat_z.unsqueeze(1).expand(-1, transcript_outputs.size(1), -1)

        encoder_outputs = torch.cat((transcript_outputs, reference_output), dim=-1)

        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs
