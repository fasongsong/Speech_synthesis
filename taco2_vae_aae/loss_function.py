from torch import nn
import torch
import numpy as np

class AAE_D_loss(nn.Module):
    def __init__(self):
        super(AAE_D_loss, self).__init__()

    def forward(self, model_output):
        mel_out, mel_out_postnet, gate_out, alignment, speaker_out, augment_out, D_real_logits, D_fake_logits = model_output
        #################################################
        ############### Residual encoder AAE ############
        #################################################
        D_loss_fake = nn.BCEWithLogitsLoss()(D_fake_logits, torch.zeros_like(D_fake_logits))
        D_loss_true = nn.BCEWithLogitsLoss()(D_real_logits, torch.ones_like(D_real_logits))
        D_loss =  D_loss_fake +  D_loss_true

        return D_loss


class AAE_G_loss(nn.Module):
    def __init__(self):
        super(AAE_G_loss, self).__init__()

    def forward(self, model_output):
        mel_out, mel_out_postnet, gate_out, alignment, speaker_out, augment_out, D_real_logits, D_fake_logits = model_output
        ###############torch.mean(torch.stack(my_list), dim=0)
        G_loss = nn.BCEWithLogitsLoss()(D_fake_logits, torch.ones_like(D_fake_logits))

        return G_loss


class Tacotron2Loss(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2Loss, self).__init__()
        self.anneal_function = hparams.anneal_function
        self.lag = hparams.anneal_lag
        self.k = hparams.anneal_k
        self.x0 = hparams.anneal_x0
        self.upper = hparams.anneal_upper
        self.distribute = hparams.distributed_run


    def forward(self, model_output, targets, step):         ########## target에 og / aug 구분 라벨추가 코드필요
        mel_target, gate_target, speaker_target, aug_target = targets[0], targets[1], targets[2], targets[3]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        speaker_target.requires_grad = False
        aug_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, alignment, speaker_out, augment_out, D_real_logits, D_fake_logits = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        #################################################
        ################   Speaker encoder ##############
        #################################################
        _, speaker_label = torch.max(speaker_target, 1)
        _, aug_label = torch.max(aug_target, 1)
        speaker_loss = nn.CrossEntropyLoss()(speaker_out, speaker_label)
        augment_loss = nn.CrossEntropyLoss()(augment_out, aug_label)

        tmp_loss = mel_loss+gate_loss

        #################################################
        ################## Alignment loss ###############
        #################################################
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
        alignment_loss = alignment_loss * lamb

        #total_loss = 10*(mel_loss + gate_loss) + 0.01*(S_kl_loss*S_kl_weight + R_kl_loss*R_kl_weight) + 0.01*speaker_loss + 0.0001*augment_loss + alignment_loss*lamb

        #total_loss = 10 * (mel_loss + gate_loss) + 0.01 * (S_kl_loss*S_kl_weight + R_kl_loss*R_kl_weight) + 0.1 * speaker_loss + 0.1 * augment_loss + alignment_loss*lamb
        #total_loss = mel_loss + gate_loss + S_kl_loss*S_kl_weight + R_kl_loss*R_kl_weight + speaker_loss + augment_loss + alignment_loss*lamb
        if self.distribute:
            total_loss = 10 * tmp_loss + 0.1 * (speaker_loss + augment_loss) + alignment_loss
        else:
            total_loss = 10 * tmp_loss + 0.1 * (speaker_loss + augment_loss) + alignment_loss
        #total_loss = mel_loss + gate_loss + S_kl_loss*S_kl_weight + R_kl_loss*R_kl_weight + speaker_loss + augment_loss

        return total_loss, tmp_loss, speaker_loss, augment_loss, alignment_loss
