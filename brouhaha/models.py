from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pyannote.audio import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.models.segmentation.debug import SimpleSegmentationModel

SNR_MIN = -15
SNR_MAX = 80
C50_MIN = -10
C50_MAX = 60


class ParametricSigmoid(nn.Module):
    def __init__(self, alpha: float, beta: float) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor):
        return (self.beta - self.alpha) * F.sigmoid(x) + self.alpha


class CustomClassifier(nn.Module):
    def __init__(self, in_features, out_features: int) -> None:
        super().__init__()
        self.linears = nn.ModuleDict({
            'vad': nn.Linear(in_features, out_features),
            'snr': nn.Linear(in_features, 1),
            'c50': nn.Linear(in_features, 1),
        })

    def forward(self, x: torch.Tensor):
        out = dict()
        for mode, linear in self.linears.items():
            _output = linear(x)
            out[mode] = _output

        return out


class CustomActivation(nn.Module):
    # Try something else for snr and c50
    # TODO : print and save alpha and beta values of ParametricSigmoid
    def __init__(self) -> None:
        super().__init__()
        self.activations = nn.ModuleDict({
            'vad': nn.Sigmoid(),
            'snr': ParametricSigmoid(SNR_MAX, SNR_MIN),
            'c50': ParametricSigmoid(C50_MAX, C50_MIN),
        })

    def forward(self, x: torch.Tensor):
        out = list()
        for mode, activation in self.activations.items():
            _output = activation(x[mode])
            out.append(_output)

        out = torch.stack(out)
        out = rearrange(out, "n b t o -> b t (n o)")
        return out


class RegressiveSegmentationModelMixin(Model):
    def build(self):
        """
        Debug architecture
        """
        nb_classif = len(set(self.specifications.classes) - set(['snr', 'c50']))
        self.classifier = CustomClassifier(32 * 2, nb_classif)
        self.activation = CustomActivation()


class CustomSimpleSegmentationModel(RegressiveSegmentationModelMixin, SimpleSegmentationModel):
    pass


class CustomPyanNetModel(RegressiveSegmentationModelMixin, PyanNet):
    SINCNET_WINDOW_SIZE_SAMPLES = 991
    SINCNET_STEP_SAMPLES = 270

    def __init__(
        self,
        sincnet: dict = None,
        lstm: dict = None,
        linear: dict = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
        num_new_frames: int = 6,  # Number of frames which are new to the lstm after we
                                  # apply sincnet
    ):
        super().__init__(
            sincnet=sincnet,
            lstm=lstm,
            linear=linear,
            sample_rate=sample_rate,
            num_channels=num_channels,
            task=task,
        )
        self.num_new_frames = num_new_frames
        self.hx = None

    def reset_backward_pass(self, hx):
        hn = hx[0].clone()
        cn = hx[1].clone()
        selection = range(1, hn.shape[0], 2)
        hn[selection, :, :] = 0
        cn[selection, :, :] = 0
        return (hn, cn)

    def build(self):
        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self.hparams.lstm["hidden_size"] * (
                2 if self.hparams.lstm["bidirectional"] else 1
            )
        nb_classif = len(set(self.specifications.classes) - set(['snr', 'c50']))
        self.classifier = CustomClassifier(in_features, nb_classif)
        self.activation = CustomActivation()

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """

        outputs = self.sincnet(waveforms)

        # Only keep new frames
        outputs = outputs[:, :, -self.num_new_frames:]
        if self.hparams.lstm["monolithic"]:
            outputs, self.hx = self.lstm(
                rearrange(outputs, "batch feature frame -> batch frame feature"),
                self.hx,
            )
        else:
            outputs = rearrange(outputs, "batch feature frame -> batch frame feature")
            for i, lstm in enumerate(self.lstm):
                outputs, self.hx = lstm(outputs, self.hx)
                if i + 1 < self.hparams.lstm["num_layers"]:
                    outputs = self.dropout(outputs)
        self.hx = self.reset_backward_pass(self.hx)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))
