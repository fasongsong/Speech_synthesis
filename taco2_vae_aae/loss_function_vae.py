from torch import nn
import torch
import numpy as np


class Tacotron2Loss(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2Loss, self).__init__()
        self.anneal_function = hparams.anneal_function
        self.lag = hparams.anneal_lag
        self.k = hparams.anneal_k
        self.x0 = hparams.anneal_x0
        self.upper = hparams.anneal_upper

    def kl_anneal_function(self, anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            # if step > 10000:
            #     step = step - (step // 10000) * 10000
            return float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            return min(1, step/x0)
        elif anneal_function == 'constant':
            return 0.001
    # def kl_anneal_function(self, anneal_function, lag, step, k, x0, upper):
    #     if anneal_function == 'logistic':
    #         return float(upper/(upper+np.exp(-k*(step-x0))))
    #     elif anneal_function == 'linear':
    #         if step > lag:
    #             return min(upper, step/x0)
    #         else:
    #             return 0
    #     elif anneal_function == 'constant':
    #         return 0.001

    def forward(self, model_output, targets, step):         ########## target에 og / aug 구분 라벨추가 코드필요
        mel_target, gate_target, speaker_target, aug_target = targets[0], targets[1], targets[2], targets[3]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        speaker_target.requires_grad = False
        aug_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, alignment, r_mu, r_lv, speaker_out, aug_out = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        _, speaker_label = torch.max(speaker_target, 1)
        _, aug_label = torch.max(aug_target, 1)

        speaker_loss = nn.CrossEntropyLoss()(speaker_out, speaker_label)
        augment_loss = nn.CrossEntropyLoss()(aug_out, aug_label)

        R_kl_loss = -0.5 * torch.sum(1 + r_lv - r_mu.pow(2) - r_lv.exp())
        R_kl_weight = self.kl_anneal_function(self.anneal_function, step, self.k, self.x0)

        tmp_loss = mel_loss+gate_loss

        N = alignment.size(1)
        T = alignment.size(2)
        k = 0.4
        lamb = 0.0005
        alignment_loss = torch.FloatTensor(0)
        g = torch.zeros([N, T], dtype=torch.float32).cuda()
        for n in range(0, N):
            for t in range(0, T):
                g[n][t] = 1 - np.exp(-(n / N - t / T) ** 2 / k)
        g.expand_as(alignment)
        alignment_loss = torch.abs(torch.sum(torch.mul(alignment, g)))

        total_loss = 10 * (mel_loss + gate_loss) + 0.01 * (R_kl_loss*R_kl_weight) + 0.1 * speaker_loss + 0.1 * augment_loss + lamb * alignment_loss

        return total_loss, tmp_loss, R_kl_loss, R_kl_weight, speaker_loss, augment_loss, alignment_loss
