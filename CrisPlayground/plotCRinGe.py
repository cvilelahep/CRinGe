import matplotlib
matplotlib.use('Agg')

from math import atan2

import torch
import matplotlib.pyplot as plt
import sys

class CRinGeNet(torch.nn.Module) :
    def __init__(self) :
        super(CRinGeNet, self).__init__()

        self._mlp_pid = torch.nn.Sequential(
            torch.nn.Linear(3,512), torch.nn.ReLU(),
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
        net = torch.cat( (self._mlp_pid(x[:,0:3]),self._mlp_pos(x[:,3:6]),self._mlp_dir(x[:,6:9]),self._mlp_E(x[:,9].reshape(len(x[:,9]),1))), 1)

        # MegaMLP 
        net = self._mlp(net)
        
        # Reshape into 11 x 21 figure in 64 channels. Enough?!
        net = net.view(-1, 64, 11, 21)

        # Need to flatten? Maybe...
        return self._upconvs(net).view(-1, 88*168)

net = CRinGeNet().cpu()
net.load_state_dict(torch.load("testCRinGe_10epochs_emugamma.cnn", map_location=lambda storage, loc: storage))

torch.set_grad_enabled(False)
net.eval()


plt.figure(figsize=(24,12))


index = 1


for data in [ [[1, 0, 0, 0., 0., 0., 0., 0., 1., 200]],
              [[0, 1, 0, 0., 0., 0., 0., 0., 1., 200]],
              [[0, 0, 1, 0., 0., 0., 0., 0., 1., 200]],
              
              [[1, 0, 0, 0., 0., 0., -0.6, 0., 0.8, 500]],
              [[0, 1, 0, 0., 0., 0., -0.6, 0., 0.8, 500]],
              [[0, 0, 1, 0., 0., 0., -0.6, 0., 0.8, 500]],
              
              [[1, 0, 0, 0., 0., 0., -0.9797958971132712, 0., 0.2, 700]],
              [[0, 1, 0, 0., 0., 0., -0.9797958971132712, 0., 0.2, 700]],
              [[0, 0, 1, 0., 0., 0., -0.9797958971132712, 0., 0.2, 700]] ] :
    
    plt.subplot(3, 3, index)

    data = torch.as_tensor(data).cpu()

    event = net(data).cpu().detach().numpy()[0]

    event = event.reshape((88,168))

    data = data.numpy()
    
    plt.imshow(event)

    if data[0][0] == 1 :
        plt.title(r'$\gamma$ E='+str(data[0][-1])+' MeV $\phi$='+"{:.2f}".format(atan2(data[0][8], data[0][6])))
    elif data[0][1] == 1 :
        plt.title(r'$e$ E='+str(data[0][-1])+' MeV $\phi$='+"{:.2f}".format(atan2(data[0][8], data[0][6])))
    elif data[0][2] == 1 :
        plt.title(r'$\mu$ E='+str(data[0][-1])+' MeV $\phi$='+"{:.2f}".format(atan2(data[0][8], data[0][6])))
    else :
        print("INVALID PID", data)
        
    plt.colorbar()
    index +=1


plt.tight_layout()
plt.savefig("CRinGeEvent.png")
plt.show()
