from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import iotools.hkdataset
from torch.utils.data import DataLoader

def loader_factory(name, batch_size,
                   shuffle=True,
                   num_workers=1,
                   worker_init_fn=None,
                   collate_fn=iotools.hkdataset.Collate,
                   pin_memory=True,
                   **args):
    ds = getattr(iotools.hkdataset,name)(**args)
    loader = DataLoader(ds,
                        batch_size  = batch_size,
                        shuffle     = shuffle,
                        num_workers = num_workers,
                        worker_init_fn = worker_init_fn,
                        collate_fn  = collate_fn,
                        pin_memory = pin_memory)
    return loader
