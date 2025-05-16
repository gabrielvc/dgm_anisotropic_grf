
from torch.utils.data import Dataset
from typing import Tuple, Dict, Iterator, SupportsIndex
import h5py
import hashlib


class H5Dataset(Dataset):
    """Interface the H5 dataset"""

    def __init__(
        self,
        path: str,
        data_axis: int = 1,
        data_name: str = "data"
    ):
        super().__init__()
        self._path = path
        self._file = None
        self.data_axis = data_axis
        self.data_name = data_name
        with h5py.File(self._path, "r") as file:
            self._num_records = int(file[self.data_name].shape[self.data_axis])

    def __enter__(self):
        return self

    def create_reader(self):
        self._file = h5py.File(self._path, "r")

    def __exit__(self, exc_type, exc_value, traceback):
        if self._file is not None:
            self._dataset = None
            self._file.close()

    def __len__(self) -> int:
        return self._num_records

    def format_element(self, index) -> Dict:
        raise NotImplementedError("Format element not implemented")

    def __iter__(self) -> Iterator[Dict]:
        self._ensure_reader_exists()
        for index in range(self._num_records):
            yield self.format_element(index)

    def _ensure_reader_exists(self) -> None:
        if self._file is None:
            self.create_reader()

    def __getitem__(self, record_key: SupportsIndex) -> Dict:
        record_key = record_key.__index__()
        self._ensure_reader_exists()
        if record_key < 0 or record_key >= self._num_records:
            raise ValueError(f"Record key should be in [0, {self._num_records})")
        return self.format_element(record_key)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_file"]
        del state["_dataset"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # We open readers lazily when we need to read from them. Thus, we don't
        # need to re-open the same files as before pickling.
        self._file = None
        self._dataset = None

    def __repr__(self) -> str:
        """Storing a hash of paths since paths can be a very long list."""
        h = hashlib.sha1()
        h.update(self._path.encode())
        return f"GaussH5(path={h.hexdigest()})"
