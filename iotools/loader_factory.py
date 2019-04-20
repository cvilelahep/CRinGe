from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import iotools.hkdataset
from torch.utils.data import DataLoader

def loader_factory(name, batch_size,
                   shuffle=True,
                   num_workers=1,
                   collate_fn=iotools.hkdataset.Collate,
                   **args):
    ds = getattr(iotools.hkdataset,name)(**args)
    loader = DataLoader(ds,
                        batch_size  = batch_size,
                        shuffle     = shuffle,
                        num_workers = num_workers,
                        collate_fn  = collate_fn)
    return loader

