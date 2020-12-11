import torch
from torch.utils.data import Dataset, TensorDataset, Sampler
from pyioopt.core import reader

import os

import numpy as np

class PyiooptDataset(Dataset) :
    
    def __init__(self, reader, transform = None, start_fraction = 0., use_fraction = 1.0) :

        self._transform = transform

        # Check input fractions makes sense (stolen from watchmal dataloader)
        assert start_fraction >= 0. and start_fraction < 1.
        assert use_fraction > 0. and use_fraction <= 1.
        assert (start_fraction + use_fraction) <= 1.

        self._reader = reader
        
        self._start_event = int(len(self._reader)*start_fraction)
        self._end_event = int(len(self._reader)*(start_fraction+use_fraction))
        self._len = self._end_event - self._start_event

        self._onehot = {22 : 0,
                        11 : 1,
                        13 : 2}

        self._masses = {22 : 0,
                        11 : 0.511,
                        13 : 105.7}

        print("DATASET! START {0} END {1} LENGTH{2}".format(self._start_event,self._end_event,self._len))
        
    def __len__(self) :
        return self._len
    
    def __getitem__(self, i) :

#        print(">>>GETTINGITEM", os.getpid(), i)
        
        # Get zeroth sub-event
        thisEvent = self._reader[i+self._start_event][0] 

        # 0: bottom, 1: barrel, 2: top
        t_np = [np.zeros(self._reader.mask[i].shape) for i in range(3)]
        q_np = [np.zeros(self._reader.mask[i].shape) for i in range(3)]

        for i_region in range(3) :

            thisHits = thisEvent["hits"][self._reader.pmts()[thisEvent["hits"]["pmtNumber"]-1]["location"] == i_region]
            
            t_np[i_region][self._reader.pmts()[thisHits["pmtNumber"]-1]["column"],
                           self._reader.pmts()[thisHits["pmtNumber"]-1]["row"]] = thisHits["t"]
            q_np[i_region][self._reader.pmts()[thisHits["pmtNumber"]-1]["column"],
                           self._reader.pmts()[thisHits["pmtNumber"]-1]["row"]] = thisHits["q"]


        t_top = torch.from_numpy(t_np[0])
        q_top = torch.from_numpy(q_np[0])

        t_barrel = torch.from_numpy(t_np[1])
        q_barrel = torch.from_numpy(q_np[1])

        t_bottom = torch.from_numpy(t_np[2])
        q_bottom = torch.from_numpy(q_np[2])
            
        particles = thisEvent["trueTracks"][0]

        pid = torch.zeros(len(self._onehot))
        pid[self._onehot[particles["PDG_code"]]] = 1.

        Ekin = torch.subtract(torch.tensor([particles['E']]), self._masses[particles["PDG_code"]])
        
        direction = torch.tensor([particles['dirx'], particles['diry'], particles['dirz']])

        pos = torch.tensor([thisEvent["vertex"][0]["vtx_x"], thisEvent["vertex"][0]["vtx_y"], thisEvent["vertex"][0]["vtx_z"]])

        return { "q_top" : q_top,
                 "t_top" : t_top,
                 "q_barrel" : q_barrel,
                 "t_barrel" : t_barrel,
                 "q_bottom" : q_bottom,
                 "t_bottom" : t_bottom,
                 "pid" : pid,
                 "pos" : pos,
                 "dir" : direction,
                 "Ekin" : Ekin }
                 

class SemiRandomSampler(Sampler) :
    def __init__ (self, data_source, sequence_length) :
        self.sequence_length = sequence_length

        self.i_sequence = self.sequence_length

        self.seq_start = 0
        
        self.data_source = data_source

    def __len__(self) :
        return len(self.data_source)
        
    def __iter__(self) :
        return self
        
    def __next__(self) :
        if self.i_sequence < self.sequence_length :
            self.i_sequence += 1
        else :
            self.i_sequence = 0
            self.seq_start = np.random.randint(low = 0, high = len(self.data_source)-self.sequence_length)

        return self.seq_start+self.i_sequence
