from torch.utils.data import Dataset
from pyioopt.core import reader

class PyiooptDataset(Dataset) :

    def __init__(self, reader, transform = None, start_fraction = 0., use_fraction = 1.0) :

        self._transform = transform

        # Check input fractions makes sense (stolen from watchmal dataloader)
        assert start_fraction >= 0. and start_fraction < 1.
        assert use_fraction > 0. and use_fraction <= 1.
        assert (start_fraction + use_fraction) <= 1.

        self._start_event = int(len(self._reader)*start_fraction)
        self._end_event = int(len(self._reader)*(start_fraction+use_fraction))
        self._len = self._end_event - self._start_event
        
    def __len__(self) :
        return self._len

    def __getitem__(self, i) :
        return self._reader[i+self._start_event][0]

    def worker_init(self) :
        self._reader = reader
        
