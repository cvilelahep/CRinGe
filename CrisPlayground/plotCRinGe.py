import matplotlib
matplotlib.use('Agg')

import torch
import matplotlib.pyplot as plt
import sys


# CRinGeNet
class CRinGeNet(torch.nn.Module) :
    def __init__(self) :
        super(CRinGeNet, self).__init__()

        self._mlp_pid = torch.nn.Sequential(
            torch.nn.Linear(2,512), torch.nn.ReLU(),
            torch.nn.Linear(512,512), torch.nn.ReLU()
        )

        self._mlp_pos = torch.nn.Sequential(
            torch.nn.Linear(3,512), torch.nn.ReLU(),
            torch.nn.Linear(512,512), torch.nn.ReLU()
        )

        self._mlp_dir = torch.nn.Sequential(
            torch.nn.Linear(3,512), torch.nn.ReLU(),
            torch.nn.Linear(512,512), torch.nn.ReLU()
        )

        self._mlp_E = torch.nn.Sequential(
            torch.nn.Linear(1,512), torch.nn.ReLU(),
            torch.nn.Linear(512,512), torch.nn.ReLU()
        )

        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 14784), torch.nn.ReLU()
        )


        self._upconvs = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 4, 2), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3), torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(64, 32, 4, 2), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3), torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(32, 32, 4, 2), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 1, 3)
        )

    def forward(self, x) :
        # Concatenate MLPs that treat PID, pos, dir and energy inputs separately
        net = torch.cat( (self._mlp_pid(x[:,0:2]),self._mlp_pos(x[:,2:5]),self._mlp_dir(x[:,5:8]),self._mlp_E(x[:,8].reshape(len(x[:,8]),1))), 1)

        # MegaMLP 
        net = self._mlp(net)
        
        # Reshape into 11 x 21 figure in 64 channels. Enough?!
        net = net.view(-1, 64, 11, 21)

        # Need to flatten? Maybe...
        return self._upconvs(net).view(-1, 88*168)

net = CRinGeNet()
net.load_state_dict(torch.load("testCRinGe_50epochs.cnn"))

torch.set_grad_enabled(False)
net.eval()

if len(sys.argv) == 10 :
    data = [[float(x) for x in sys.argv[1:]]]
else :
    data = [[1, # is electron
             0, # is muon
             0., 100., 0., # position
             0., 0., 1., # direction
             200.]] # Energy

data = torch.as_tensor(data).cpu()

event = net(data).cpu().detach().numpy()[0]

event = event.reshape((88,168))

print event

plt.imshow(event)
plt.savefig("CRinGeEvent.pdf")
plt.show()
