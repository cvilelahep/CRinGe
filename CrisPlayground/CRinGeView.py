from bokeh.io import curdoc
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import row, column
from bokeh.models import  Slider
from bokeh.models import ColumnDataSource, RadioButtonGroup

import numpy as np
import torch
import sys

# CRinGeNet
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
#net.load_state_dict(torch.load("testCRinGe_50epochs.cnn", map_location=lambda storage, loc: storage))
#net.load_state_dict(torch.load("testCRinGe.cnn", map_location=lambda storage, loc: storage))
net.load_state_dict(torch.load("testCRinGe_10epochs_emugamma.cnn", map_location=lambda storage, loc: storage))

torch.set_grad_enabled(False)
net.eval()

data = [[0, # is gamma
         1, # is electron
         0, # is muon
         0., 100., 0., # position
         0., 0., 1., # direction
         200.]] # Energy

data = torch.as_tensor(data).cpu()

event = net(data).cpu().detach().numpy()[0]
event = event.reshape((88,168))
source = ColumnDataSource(data=dict(image=[event]))

data = data.cpu().detach().numpy()

p = figure(plot_height=int(880*0.8), plot_width=int(1680*0.8))

p.image(image='image', x=0, y=0, dw=168, dh=88, palette="Viridis256", source = source)

energy_slider = Slider(title="Energy", value = data[0][8], start = 0., end = 2000., step = 10.)
phi_slider = Slider(title="Azimuthal angle", value = np.arctan2(data[0][7], data[0][5]), start = -np.pi, end = np.pi, step = np.pi/10.)
coseta_slider = Slider(title="Cosine of zenith angle", value = data[0][6], start = -0.999, end = 0.999, step = 1/20.)

posx_slider = Slider(title="Position x", value = data[0][2], start = -250., end = 250., step = 500./20)
posy_slider = Slider(title="Position y", value = data[0][3], start = -250., end = 250., step = 500./20)
posz_slider = Slider(title="Position z", value = data[0][4], start = -250., end = 250., step = 500./20)

PID_button = RadioButtonGroup(labels=["gamma", "electron", "muon"], active=0)

inputs = row(column(PID_button,energy_slider), column(phi_slider,coseta_slider), column(posx_slider, posy_slider, posz_slider))

def update_data(attrname, old, new) :

    energy = energy_slider.value
    phi = phi_slider.value
    coseta = coseta_slider.value

    dx = np.cos(phi)
    dy = coseta
    dz = np.sin(phi)

    r = (1-dy**2)**0.5
    if r :
        dx /= r
        dz /= r

    # dx, dy, dz, doesn't seem to be working well ...
        
    x = posx_slider.value
    y = posy_slider.value
    z = posz_slider.value

    pid = PID_button.active
    
    thisData = []
    if pid == 0 :
        thisData.append(1.)
        thisData.append(0.)
        thisData.append(0.)
    elif pid == 1 :
        thisData.append(0.)
        thisData.append(1.)
        thisData.append(0.)
    elif pid == 2 :
        thisData.append(0.)
        thisData.append(0.)
        thisData.append(1.)
        
    thisData.append(x)
    thisData.append(y)
    thisData.append(z)
    thisData.append(dx)
    thisData.append(dy)
    thisData.append(dz)
    thisData.append(energy)

    thisData = [thisData]
    
    data = torch.as_tensor(thisData).cpu()
    event = net(data).cpu().detach().numpy()[0]
    event = event.reshape((88,168))
    source.data = {'image': [event]}

for w in [energy_slider, phi_slider, coseta_slider, posx_slider, posy_slider, posz_slider, PID_button] :
    if hasattr(w, 'value') :
        w.on_change('value', update_data)
    elif hasattr(w, 'active') :
        w.on_change('active', update_data)

curdoc().add_root(column(p, inputs, width=1680))

curdoc().title = "CRinGeView"
