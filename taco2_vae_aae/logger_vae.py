import random
import torch.nn.functional as F
import torch
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration, recon_loss, R_kl_loss, R_kl_weight, speaker_loss, augment_loss, alignment_loss, iteration):
            self.add_scalar("training.loss", reduced_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)
            self.add_scalar("duration", duration, iteration)
            self.add_scalar("R_KL_loss", R_kl_loss, iteration)
            self.add_scalar("R_KL_weight", R_kl_weight, iteration)
            self.add_scalar("R_KL_weight * R_KL_loss", R_kl_loss*R_kl_weight, iteration)
            self.add_scalar("recon_loss", recon_loss, iteration)
            self.add_scalar("speaker_class_loss", speaker_loss, iteration)
            self.add_scalar("augment_class_loss", augment_loss, iteration)
            self.add_scalar("alingmnet_loss", alignment_loss, iteration)


    def log_validation(self, reduced_loss, model, y, y_pred, iteration, speaker_acc=0, augment_acc=0):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        self.add_scalar("Speaker_classifier_ACC", speaker_acc, iteration)
        self.add_scalar("Augment_classifier_ACC", augment_acc, iteration)
        _, mel_outputs, gate_outputs, alignments, _, _, speaker_output, augmentation_output = y_pred
        mel_targets, gate_targets, speaker_id, labels = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_image(
            "alignment",
            torch.from_numpy(plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T)).permute(2,0,1),
            iteration)
        self.add_image(
            "mel_target",
            torch.from_numpy(plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy())).permute(2,0,1), iteration)
        self.add_image(
            "mel_predicted",
            torch.from_numpy(plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy())).permute(2,0,1),
            iteration)
        self.add_image(
            "gate",
            torch.from_numpy(plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                F.sigmoid(gate_outputs[idx]).data.cpu().numpy())).permute(2,0,1),
            iteration)



        # self.add_image(
        #     "alignment",
        #     plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
        #     iteration)
        # self.add_image(
        #     "mel_target",
        #     plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
        #     iteration)
        # self.add_image(
        #     "mel_predicted",
        #     plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
        #     iteration)
        # self.add_image(
        #     "gate",
        #     plot_gate_outputs_to_numpy(
        #         gate_targets[idx].data.cpu().numpy(),
        #         F.sigmoid(gate_outputs[idx]).data.cpu().numpy()), iteration)
