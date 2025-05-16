import math
import torch
from torch import Tensor
import numpy as np

from sound_model.data import is_silent


class SI_SDRi:
    """"""

    def __init__(
        self,
        chunk_duration: float = 4.0,
        overlap_duration: float = 2.0,
        sample_rate: int = 22050,
        eps: float = 1e-8,
        filter_single_source: bool = True,
    ):

        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.sample_rate = sample_rate
        self.eps = eps
        self.filter_single_source = filter_single_source

        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(overlap_duration * sample_rate)

        # Calculate the step size between consecutive sub-chunks
        self.step_size = chunk_samples - overlap_samples
        self.chunk_samples = chunk_samples
        self.overlap_samples = overlap_samples

    def score(self, reconstruction: Tensor, reference: Tensor):
        """
        reconstruction : Tensor of shape (n_instruments, len_chunk)

        reference : Tensor of shape (n_instruments, len_chunk)
        """
        mixture = reference.sum(dim=0, keepdim=True)
        n_instruments = reference.shape[0]

        # Determine the number of evaluation chunks based on step_size
        num_eval_chunks = math.ceil(
            (mixture.shape[-1] - self.overlap_samples) / self.step_size
        )

        sub_scores = [[] for _ in range(n_instruments)]
        for i in range(num_eval_chunks):
            start_sample = i * self.step_size
            end_sample = start_sample + self.chunk_samples

            # Determine number of active signals in sub-chunk
            num_active_signals = 0
            for instrument in reference:
                o = instrument[None, start_sample:end_sample]
                if not is_silent(o):
                    num_active_signals += 1

            # Skip sub-chunk if necessary
            if self.filter_single_source and num_active_signals <= 1:
                continue

            # Compute SI-SNRi for each stem
            m = mixture[:, start_sample:end_sample]
            for j, (ref_instrument, rec_instrument) in enumerate(
                zip(reference, reconstruction)
            ):
                o = ref_instrument[None, start_sample:end_sample]
                s = rec_instrument[None, start_sample:end_sample]

                sub_scores[j].append(
                    (sisnr(s, o, self.eps) - sisnr(m, o, self.eps)).item()
                )

        return [np.mean(score_i) for score_i in sub_scores]


# NOTE: copy/paste of MSDM code
def sdr(preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    s_target = torch.norm(target, dim=-1) ** 2 + eps
    s_error = torch.norm(target - preds, dim=-1) ** 2 + eps
    return 10 * torch.log10(s_target / s_error)


def sisnr(preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + eps) / (
        torch.sum(target**2, dim=-1, keepdim=True) + eps
    )
    target_scaled = alpha * target
    noise = target_scaled - preds
    s_target = torch.sum(target_scaled**2, dim=-1) + eps
    s_error = torch.sum(noise**2, dim=-1) + eps
    return 10 * torch.log10(s_target / s_error)
