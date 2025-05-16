import warnings
import itertools
from pathlib import Path

import torch, torchaudio
from ddrm.functions.svd_replacement import H_functions
from sound_model.data import is_multi_source, is_silent


def generate_invp(
    x_orig: torch.Tensor,
    task: str,
    obs_std: float,
    device: float,
):
    """Generate inverse problem.

    Supported tasks are
        - separation
    """
    n_instruments, len_track = x_orig.shape
    operator = OPERATORS[task](n_instruments, len_track, device)

    # generate obseration
    x_orig = x_orig.unsqueeze(0)
    obs = operator.H(x_orig)
    obs = obs  # + obs_std * torch.randn_like(obs)

    def log_pot(x):
        diff = obs.reshape(1, -1) - operator.H(x)
        return -0.5 * torch.norm(diff) ** 2 / obs_std**2

    return obs, operator, log_pot


def load_chunck(
    track_name,
    chunk_idx,
    *,
    # NOTE refer to config file to set these params
    sample_rate,
    chunk_size,
    silence_threshold,
    keep_only_multisource,
    sample_eps_in_sec,
    stem_names,
    data_dir,
):
    """"""
    sample_eps = sample_eps_in_sec * sample_rate
    data_dir = Path(data_dir)

    # paths of each instrument
    stem_paths = {stem: data_dir / track_name / f"{stem}.wav" for stem in stem_names}
    stem_paths = {
        stem: stem_path for stem, stem_path in stem_paths.items() if stem_path.exists()
    }
    assert len(stem_paths) >= 1, track_name

    # load track
    stems_tracks = {}
    for stem, stem_path in stem_paths.items():
        audio_track, sr = torchaudio.load(stem_path)

        assert (
            sr == sample_rate
        ), f"sample rate {sr} is different from target sample rate {sample_rate}"
        stems_tracks[stem] = audio_track

    # check all intruments have the same length
    channels, samples = zip(*[t.shape for t in stems_tracks.values()])
    for s1, s2 in itertools.product(samples, samples):
        assert abs(s1 - s2) <= sample_eps, f"{track_name}: {abs(s1 - s2)}"
        if s1 != s2:
            warnings.warn(
                f"The tracks with name {track_name} have a different number of samples ({s1}, {s2})"
            )

    # truncate the track
    n_samples = min(samples)
    n_channels = channels[0]
    stems_tracks = {s: t[:, :n_samples] for s, t in stems_tracks.items()}

    # set absent instrument to zero
    for stem in stem_names:
        if not stem in stems_tracks:
            stems_tracks[stem] = torch.zeros(
                (n_channels, n_samples), device=audio_track.device
            )

    separated_track = tuple([stems_tracks[stem] for stem in stem_names])
    separated_track = torch.cat(separated_track)

    # extract chunk
    chunk = separated_track[:, chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
    _, chunk_samples = chunk.shape

    # Keep checks performed by authors
    # Remove if silent
    if silence_threshold is not None and is_silent(
        chunk.sum(0, keepdims=True), silence_threshold
    ):
        raise ValueError(f"Silent chunk, {track_name}--{chunk_idx}")

    # Remove if it contains only one source
    if keep_only_multisource and not is_multi_source(chunk):
        raise ValueError(f"Single source chunk {track_name}--{chunk_idx}")

    # Remove if it contains less than the minimum chunk size
    if chunk_samples < chunk_size:
        raise ValueError(
            f"Small size {track_name}--{chunk_idx}, expect {chunk_size} got {chunk_samples}"
        )

    return chunk


# --- operators
class MixtureOperator(H_functions):
    """"""

    def __init__(self, n_instruments, duration, device):
        self.n_instruments = n_instruments
        self.duration = duration

        H = torch.ones((1, n_instruments)).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    def V(self, vec):
        # get the needles
        needles = (
            vec.clone().reshape(vec.shape[0], self.n_instruments, -1).permute(0, 2, 1)
        )  # shape: B, WH, C'
        # multiply each needle by the small V
        needles = torch.matmul(
            self.V_small, needles.reshape(-1, self.n_instruments, 1)
        ).reshape(
            vec.shape[0], -1, self.n_instruments
        )  # shape: B, WH, C
        # permute back to vector representation
        recon = needles.permute(0, 2, 1)  # shape: B, C, WH
        return recon.reshape(vec.shape[0], -1)

    def Vt(self, vec):
        # get the needles
        needles = (
            vec.clone().reshape(vec.shape[0], self.n_instruments, -1).permute(0, 2, 1)
        )  # shape: B, WH, C
        # multiply each needle by the small V transposed
        needles = torch.matmul(
            self.Vt_small, needles.reshape(-1, self.n_instruments, 1)
        ).reshape(
            vec.shape[0], -1, self.n_instruments
        )  # shape: B, WH, C'
        # reorder the vector so that the first entry of each needle is at the top
        recon = needles.permute(0, 2, 1).reshape(vec.shape[0], -1)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):  # U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.duration)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros(
            (vec.shape[0], self.n_instruments * self.duration), device=vec.device
        )
        temp[:, : self.duration] = reshaped
        return temp


OPERATORS = {
    "separation": MixtureOperator,
}
