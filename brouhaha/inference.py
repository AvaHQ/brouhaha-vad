# MIT License
#
# Copyright (c) 2020-2021 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import warnings
from pathlib import Path
from typing import Any, Callable, List, Optional, Text, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature
from pytorch_lightning.utilities.memory import is_oom_error

from pyannote.audio import Inference
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Resolution
from pyannote.audio.utils.permutation import mae_cost_func, permutate
from pyannote.audio.utils.progress import InferenceProgressHook

TaskName = Union[Text, None]


class CustomInference(Inference):
    """CustomInference"""

    def slide(self, waveform: torch.Tensor, sample_rate: int) -> SlidingWindowFeature:
        """Slide model on a waveform

        Parameters
        ----------
        waveform: (num_channels, num_samples) torch.Tensor
            Waveform.
        sample_rate : int
            Sample rate.

        Returns
        -------
        output : SlidingWindowFeature
            Model output. Shape is (num_chunks, dimension) for chunk-level tasks,
            and (num_frames, dimension) for frame-level tasks.
        """

        window_size: int = round(self.duration * sample_rate)
        num_channels, num_samples = waveform.shape

        specifications = self.model.specifications
        resolution = specifications.resolution
        introspection = self.model.introspection
        if resolution == Resolution.CHUNK:
            frames = SlidingWindow(start=0.0, duration=self.duration, step=self.step)
        elif resolution == Resolution.FRAME:
            frames = introspection.frames
        frame_step_size = int(frames.step * sample_rate)
        step_size: int = frame_step_size * self.model.num_new_frames
        sincnet_real_window_size_samples = 991
        window_size_output = sincnet_real_window_size_samples + frame_step_size * (self.model.num_new_frames - 1)

        # prepare first (incomplete) chunks
        if num_samples >= window_size_output:
            # compute maximum number of incomplete chunks we might have to handle
            # given the window size
            max_num_chunks = math.floor((window_size - window_size_output) / step_size + 1)
            # compute number of chunks available
            num_chunks_available = math.floor((num_samples - window_size_output) / step_size + 1)
            # compute the end of the last chunk we want to consider
            end_last = window_size_output + (min(max_num_chunks, num_chunks_available) - 1) * step_size
            ends = np.array(
                range(window_size_output, end_last + 1, step_size)
            )
            first_chunks = [waveform[:, 0:e].unsqueeze(0) for e in ends]
            num_first_chunks = len(first_chunks)
        else:
            num_first_chunks = 0

        # we can realign the waveform to have the complete chunks start at index 0
        # that makes the following computations easier
        start_complete_chunks = ends[-1] + step_size - window_size
        waveform = waveform[:, start_complete_chunks:]
        num_samples = waveform.shape[1]
        # prepare complete chunks
        if num_samples >= window_size:
            chunks: torch.Tensor = rearrange(
                waveform.unfold(1, window_size, step_size),
                "channel chunk frame -> chunk channel frame",
            )
            num_chunks, _, _ = chunks.shape
        else:
            num_chunks = 0

        # prepare last incomplete chunk
        if num_samples > window_size:
            num_last_frames = ((num_samples - window_size) % step_size) // frame_step_size
            has_last_chunk = num_last_frames > 0
        if has_last_chunk:
            last_chunk: torch.Tensor = waveform[:, num_chunks * step_size :]

        outputs: Union[List[np.ndarray], np.ndarray] = list()

        if self.progress_hook is not None:
            self.progress_hook(0, num_first_chunks + num_chunks + has_last_chunk)

        # process first chunks
        for c in np.arange(0, num_first_chunks):
            outputs.append(self.infer(first_chunks[c]))
            if self.progress_hook is not None:
                self.progress_hook(c, num_first_chunks + num_chunks + has_last_chunk)

        # slide over audio chunks in batch
        for c in np.arange(0, num_chunks):
            batch: torch.Tensor = chunks[c]
            outputs.append(self.infer(batch))
            if self.progress_hook is not None:
                self.progress_hook(
                    num_first_chunks + c + 1,
                    num_first_chunks + num_chunks + has_last_chunk
                )

        # process orphan last chunk
        if has_last_chunk:

            last_output = self.infer(last_chunk[None])[:, -num_last_frames:, :]

            if specifications.resolution == Resolution.FRAME:
                pad = self.model.num_new_frames - last_output.shape[1]
                last_output = np.pad(last_output, ((0, 0), (0, pad), (0, 0)))

            outputs.append(last_output)
            if self.progress_hook is not None:
                self.progress_hook(
                    num_first_chunks + num_chunks + has_last_chunk,
                    num_first_chunks + num_chunks + has_last_chunk
                )

        outputs = np.vstack(outputs)

        # skip aggregation when requested,
        # or when model outputs just one vector per chunk
        # or when model is permutation-invariant (and not post-processed)
        if (
            self.skip_aggregation
            or specifications.resolution == Resolution.CHUNK
            or (
                specifications.permutation_invariant
                and self.pre_aggregation_hook is None
            )
        ):
            step = step_size / sample_rate
            frames = SlidingWindow(start=0.0, duration=step, step=step)
            shape = outputs.shape
            outputs = outputs.reshape(shape[0] * shape[1], shape[2])
            return SlidingWindowFeature(outputs, frames)

        if self.pre_aggregation_hook is not None:
            outputs = self.pre_aggregation_hook(outputs)

        aggregated = self.aggregate(
            SlidingWindowFeature(
                outputs,
                SlidingWindow(start=0.0, duration=self.duration, step=self.step),
            ),
            frames=frames,
            warm_up=self.warm_up,
            hamming=True,
            missing=0.0,
        )

        if has_last_chunk:
            num_frames = aggregated.data.shape[0]
            aggregated.data = aggregated.data[: num_frames - pad, :]

        return aggregated
