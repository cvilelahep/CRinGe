import matplotlib
#import tkinter
#matplotlib.use('TkAgg')
matplotlib.use('Qt5Agg')
import random
import math
from math import atan2
import timeit

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import LinearLocator
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm

from copy import copy
import sys
import getopt
import os

import numpy as np
import h5py
import scipy.stats as stats
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp2d
from scipy import optimize

from iotools import loader_factory

torch.set_printoptions(precision=10)

start = timeit.default_timer()

n_parameters_per_gaus = 3
n_event_to_scan = 2000
seed=0
N_GAUS = 3
clipQ = False
chargeclip = 100000
chargescale = 2500.
timescale = 1000.

energyscale = 5000.
rscale = 1690.
zscale = 1810.

loss_scale = 1.e+7

try:
    opts, args = getopt.getopt(sys.argv[1:], "mhn:", ["npeak="])
except getopt.GetoptError:
    print('LLScanEvent_SK_MultiGausTimeCorr_G.py -n <number of peaks>')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('LLScanEvent_SK_MultiGausTimeCorr_G.py -n <number of peaks, default=1>')
        sys.exit()
    elif opt in ("-n", "--npeak") :
        N_GAUS = int(arg)

print("RUNNING WITH "+str(N_GAUS)+" GAUSSIANS")
print("Applying scaling to charge "+str(chargescale))
print('Applying scaling to timing '+str(timescale))
print("Applying scaling to event energy "+str(energyscale))
print("Applying scaling to vertex r "+str(rscale))
print("Applying scaling to vertex z "+str(zscale))

nbatch=1

DATA_DIRS=['SKML/v1']
mu_data=loader_factory('H5Dataset', batch_size=nbatch, shuffle=False, num_workers=4, data_dirs=DATA_DIRS, flavour='mu-', start_fraction=0., use_fraction=1, read_keys= ["positions","directions", "energies", "event_data_top", "event_data_bottom"])
e_data=loader_factory('H5Dataset', batch_size=nbatch, shuffle=False, num_workers=4, data_dirs=DATA_DIRS, flavour='e-', start_fraction=0., use_fraction=1, read_keys= ["positions","directions", "energies", "event_data_top", "event_data_bottom"])


def computeDwall_(vertex):
    x = vertex[0]
    y = vertex[1]
    z = vertex[2]
    
    Rmax = 1690.
    Zmax = 1810.
    rr   = (x*x + y*y)**0.5
    absz = np.abs(z)
    #check if vertex is outside tank
    signflg = 1.
    if absz>Zmax or rr > Rmax:
        signflg = -1.
    #find min distance to wall      
    distz = np.abs(Zmax-absz)
    distr = np.abs(Rmax-rr)
    wall = signflg*np.minimum(distz,distr)
    return wall

def computeTowall_(vertex, direction):
    x = vertex[0]
    y = vertex[1]
    z = vertex[2]
    dx = direction[0]
    dy = direction[1]
    dz = direction[2]
    
    R=1690.
    l_b=100000.0
    l_t=100000.0
    H = 0.0
    if dx!=0 or dy!=0:
        A = dx*dx+dy*dy
        B = 2*(x*dx+y*dy)
        C = x*x+y*y-R*R
        RAD = (B*B) - (4*A*C)
        l_b = ((-1*B) + RAD**0.5)/(2*A)
    if dz==0:
        return l_b
    elif dz > 0:
        H=1810
    elif dz < 0:
        H=-1810
    l_t=(H - z)/dz;
    return np.minimum(l_t,l_b)

def _get_mask(filename):
    file_handle = h5py.File(filename, mode='r')
    dim_cap = file_handle['event_data_top'][0].shape
    mask = [torch.as_tensor(file_handle['mask'][0],dtype=torch.bool).reshape(-1, dim_cap[0]*dim_cap[1]), torch.as_tensor(file_handle['mask'][1], dtype=torch.bool).reshape(-1, dim_cap[0]*dim_cap[1])]
    file_handle.close()
    return mask

def event_loader(filename, n_event, energyCut_l, energyCut_h, wallCut, towallCut):
    file_handle = h5py.File(filename, mode='r')
    eventNum = file_handle['event_data'].shape[0]
    dim_barrel = file_handle['event_data'][0].shape
    dim_cap = file_handle['event_data_top'][0].shape
    
    dwall_list = []
    towall_list = []
    mask = [torch.as_tensor(file_handle['mask'][0],dtype=torch.bool).reshape(-1, dim_cap[0]*dim_cap[1]), torch.as_tensor(file_handle['mask'][1], dtype=torch.bool).reshape(-1, dim_cap[0]*dim_cap[1])]

    ids=[]
    i = 0
    while i < n_event:
        i += 1
        ii = random.randint(0,eventNum) 
        if ii in ids:
            i -= 1
            continue        
        if file_handle['energies'][ii] < energyCut_l:
            i -= 1
            continue
        if file_handle['energies'][ii] > energyCut_h:
            i -= 1
            continue

        wall = computeDwall_(file_handle['positions'][i][0])
        towall = computeTowall_(file_handle['positions'][i][0], file_handle['directions'][i][0])
        if wall < wallCut or towall < towallCut:
            i -= 1
            continue        
        ids.append(ii)
        
    datas=[]
    labels=[]
    scan_size = len(ids)
    j=0
    for ID in ids:
        if j >= scan_size:
            break
 
        p_position = file_handle['positions'][ID][0] 
        p_direction = file_handle['directions'][ID][0]
        p_energy = file_handle['energies'][ID]        
        p_pid = file_handle['pids'][ID]

        p_position[0] /= rscale
        p_position[1] /= rscale
        p_position[2] /= zscale
        p_energy /= energyscale
        
        if p_pid == 13:
            pid_vec = [0,0,1]
        else:
            pid_vec = [0,1,0]
        
        data_info = np.expand_dims(np.hstack((pid_vec,p_position,p_direction,p_energy)), axis=0)
        datas.append(data_info)

        barrel_label = torch.as_tensor(file_handle['event_data'][ID][:,:,0].reshape(-1,dim_barrel[0]*dim_barrel[1]), dtype=torch.float) / chargescale
        top_label = torch.as_tensor(file_handle['event_data_top'][ID][:,:,0].reshape(-1,dim_cap[0]*dim_cap[1]), dtype=torch.float) / chargescale
        bot_label = torch.as_tensor(file_handle['event_data_bottom'][ID][:,:,0].reshape(-1,dim_cap[0]*dim_cap[1]), dtype=torch.float) / chargescale
        label_info = [barrel_label,bot_label,top_label]
        labels.append(label_info)
        j = j+1
        
    file_handle.close() 
    return ids, datas, labels, mask, dwall_list, towall_list

def _fillData(data) :
    oneHotGamma = np.array(data[1] == 0)
    oneHotE = np.array(data[1] == 1)
    oneHotMu = np.array(data[1] == 2)
    
    out_data =  np.hstack((oneHotGamma.reshape(len(oneHotGamma),1), oneHotE.reshape(len(oneHotE),1), oneHotMu.reshape(len(oneHotMu),1), # One-hot PID
                           data[2][:,0,:], # Positions                                                        
                           data[3][:,0,:], # Directions                                                       
                           data[4][:,0].reshape(len(data[4][:,0]),1) ) ) # Energy                             
    out_data[:,3] /= rscale
    out_data[:,4] /= rscale
    out_data[:,5] /= zscale
    out_data[:,9] /= energyscale
    
    return out_data

def _fillLabel (data) :
    # Label is vector of charges. Mind = Blown
    dim = data[0].shape
    dim_cap = data[5].shape
    #label = torch.as_tensor(data[0][:,:,:,0].reshape(-1,dim[1]*dim[2]), dtype=torch.float) / chargescale
    #label_top = torch.as_tensor(data[5][:,:,:,0].reshape(-1, dim_cap[1]*dim_cap[2]), dtype=torch.float) / chargescale
    #label_bottom = torch.as_tensor(data[6][:,:,:,0].reshape(-1, dim_cap[1]*dim_cap[2]), dtype=torch.float) / chargescale
    
    label = torch.from_numpy(data[0][:,:,:,0].reshape(-1,dim[1]*dim[2])) / chargescale
    label_top = torch.from_numpy(data[5][:,:,:,0].reshape(-1, dim_cap[1]*dim_cap[2])) / chargescale
    label_bottom = torch.from_numpy(data[6][:,:,:,0].reshape(-1, dim_cap[1]*dim_cap[2])) / chargescale

    return [label, label_bottom, label_top]


def _multigaus_loss(prediction, charge, mask = None):

    charge_n = torch.stack( [ charge for i in range(N_GAUS) ], dim = 1)
    
    if mask is None:
        mask = torch.full_like(prediction[:,0], True, dtype = torch.bool, device = torch.device('cpu'))
    
    punhit = prediction[:,0]*mask
        
    logvar = torch.stack( [ prediction[:, i*(n_parameters_per_gaus-1) + 1]*mask for i in range(N_GAUS) ], dim = 1)
    var = torch.exp(logvar)
    logmu = torch.stack( [ prediction[:, i*(n_parameters_per_gaus-1) + 2]*mask for i in range(N_GAUS) ], dim = 1)
    mu = torch.exp(logmu)
    coeff = torch.nn.functional.softmax(prediction[:, -N_GAUS:], dim = 1)
    coefficients = torch.stack( [ coeff[:, i]*mask for i in range(N_GAUS) ], dim = 1)

    
    hitMask = torch.squeeze(charge > 0,0)
    
    hit_loss_tensor = torch.nn.BCEWithLogitsLoss(reduction="none")(punhit, (charge == 0).float())
    hit_loss = hit_loss_tensor[mask].sum()

    charge_loss = hitMask.sum()*(1/2.)*np.log(2*np.pi) # Constant term
    nll_charge = torch.log(coefficients) - 1/2.*logvar - 1/2.*(charge_n - mu)**2/var
    charge_loss += torch.squeeze(-torch.logsumexp(nll_charge, dim = 1), 0)[hitMask].sum()

    loss = (hit_loss + charge_loss)/loss_scale
    charge_diff = (charge_n-mu).sum().item()
    return loss, charge_diff


def _get_loss(prediction, label, cap_mask):
    loss = 0
    
    prediction_top = prediction[2].detach().cpu()
    prediction_bottom = prediction[1].detach().cpu()
    prediction = prediction[0].detach().cpu()

    label_top = label[2]
    label_bottom = label[1]
    label = label[0]
    
    ######barrel#######
    barrel_loss, barrel_qdiff = _multigaus_loss(prediction, label)
    #######top########
    top_loss, top_qdiff = _multigaus_loss(prediction_top, label_top, cap_mask[1])
    #######bottom#######
    bottom_loss, bottom_qdiff = _multigaus_loss(prediction_bottom, label_bottom, cap_mask[0])

    loss = barrel_loss + top_loss + bottom_loss
    #qdiff = barrel_qdiff + top_qdiff + bottom_qdiff
    
    return loss

def _stack_hit_event_display(label, data, cap_mask):
    label_top = label[2].reshape(48,48)*chargescale
    label_bottom = label[1].reshape(48,48)*chargescale
    label_barrel = label[0].reshape(51,150)*chargescale
    
    data = torch.as_tensor(data, dtype=torch.float).detach().cpu()
    pred = net(data)
    pred_top = pred[2].detach().cpu()
    pred_bottom = pred[1].detach().cpu()
    pred_barrel = pred[0].detach().cpu().numpy()
    
    unhit_top = (1/(1+torch.exp(pred_top[:, 0]))*cap_mask[1]).numpy().reshape(48,48)
    unhit_bottom = (1/(1+torch.exp(pred_bottom[:, 0]))*cap_mask[0]).numpy().reshape(48,48)
    unhit_barrel = (1/(1+np.exp(pred_barrel[:, 0]))).reshape(51,150)    

    label_barrel=np.flipud(label_barrel) # the 1d array starts from bottom?
    label_bottom=np.flipud(label_bottom) 
    
    unhit_barrel = np.flipud(unhit_barrel)
    unhit_bottom = np.flipud(unhit_bottom)
    
    dim_barrel = label_barrel.shape
    dim_cap = label_top.shape #(row, column)
    #make a new  array including all 3 regions in a rectangular
    
    new_combined_event_disp = np.zeros((2*dim_cap[0]+dim_barrel[0], dim_barrel[1]))
    new_combined_hitprob = np.zeros((2*dim_cap[0]+dim_barrel[0], dim_barrel[1]))
    #put cap in the center
    cap_start = int(0.5*(dim_barrel[1]-dim_cap[1]))
    new_combined_event_disp[0:dim_cap[0],cap_start:(cap_start+dim_cap[1])] = np.log(label_top+1e-10)
    new_combined_event_disp[dim_cap[0]:(dim_cap[0]+dim_barrel[0]),0:dim_barrel[1]] = np.log(label_barrel+1e-10)
    new_combined_event_disp[(dim_cap[0]+dim_barrel[0]):new_combined_event_disp.shape[0], cap_start:(cap_start+dim_cap[1])] = np.log(label_bottom+1e-10)
    
    new_combined_hitprob[0:dim_cap[0],cap_start:(cap_start+dim_cap[1])] = unhit_top
    new_combined_hitprob[dim_cap[0]:(dim_cap[0]+dim_barrel[0]),0:dim_barrel[1]] = unhit_barrel
    new_combined_hitprob[(dim_cap[0]+dim_barrel[0]):new_combined_event_disp.shape[0], cap_start:(cap_start+dim_cap[1])] = unhit_bottom

    nhit = np.count_nonzero(new_combined_event_disp>0)
    
    return new_combined_event_disp, new_combined_hitprob, nhit

def _scan_lossvE(data, label, cap_mask):
    Loss = []
    energy = []
    dcharge_total = []
    data = torch.as_tensor(data, dtype=torch.float).detach().cpu()
    origE = data[0][9].item()*energyscale
    orig_pred = net(data)
    orig_loss = _get_loss(orig_pred, label, cap_mask).cpu().item()

    eSpace = np.linspace(0.2*origE, 1.8*origE, 100).tolist()
    for iE in eSpace:
        data[0][9] = float(iE/energyscale)
        energy.append(iE)
        new_pred = net(data)
        Loss.append(_get_loss(new_pred, label, cap_mask).item())
        dcharge_barrel = [(new_pred[0][:,i*n_parameters_per_gaus+2] - orig_pred[0][:,i*n_parameters_per_gaus+2]).sum() for i in range(N_GAUS)][0].item()
        dcharge_bottom = [(new_pred[1][:,i*n_parameters_per_gaus+2] - orig_pred[1][:,i*n_parameters_per_gaus+2]).sum() for i in range(N_GAUS)][0].item()
        dcharge_top = [(new_pred[2][:,i*n_parameters_per_gaus+2] - orig_pred[2][:,i*n_parameters_per_gaus+2]).sum() for i in range(N_GAUS)][0].item()

        dcharge_total.append(dcharge_barrel+dcharge_bottom+dcharge_top)
        
    return energy, Loss, origE, orig_loss, dcharge_total

def _scan_lossvPID(data, label, energy, cap_mask):
    data = torch.as_tensor(data, dtype=torch.float).detach().cpu()
    data[0][9] = energy/energyscale
    isMu = int(data[0][2]==1)
    data[0][1] = 1
    data[0][2] = 0
    elike_pred = net(data)
    LossE =_get_loss(elike_pred,label,cap_mask).item()
    data[0][1] = 0
    data[0][2] = 1
    mulike_pred = net(data)
    LossMu =_get_loss(mulike_pred,label,cap_mask).item()

    LossPID = LossE - LossMu
    return LossPID, energy

class CRinGeNet(torch.nn.Module) :
    def __init__(self) :
        super(CRinGeNet, self).__init__()

        self._mlp_pid = torch.nn.Sequential(
#           torch.nn.BatchNorm1d(3),
            torch.nn.Linear(3,512), torch.nn.ReLU(),
            torch.nn.Linear(512,512), torch.nn.ReLU()
        )

        self._mlp_pos = torch.nn.Sequential(
            torch.nn.BatchNorm1d(3),
            torch.nn.Linear(3,512), torch.nn.ReLU(),
            torch.nn.Linear(512,512), torch.nn.ReLU()
        )

        self._mlp_dir = torch.nn.Sequential(
            torch.nn.BatchNorm1d(3),
            torch.nn.Linear(3,512), torch.nn.ReLU(),
            torch.nn.Linear(512,512), torch.nn.ReLU()
        )

        self._mlp_E = torch.nn.Sequential(
            torch.nn.BatchNorm1d(1),
            torch.nn.Linear(1,512), torch.nn.ReLU(),
            torch.nn.Linear(512,512), torch.nn.ReLU()
        )

        self._mlp_barrel = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 8512), torch.nn.ReLU()   # 64 * 7 * 19
        )
        self._mlp_top = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 2304), torch.nn.ReLU()   # 64 * 6 * 6 
            #torch.nn.Linear(2048, 256), torch.nn.ReLU(),
            #torch.nn.Linear(256, 256), torch.nn.ReLU(),
            #torch.nn.Linear(256, 2304), torch.nn.ReLU()   # 64 * 6 * 6 
        )
        self._mlp_bottom = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 2304), torch.nn.ReLU()
            #torch.nn.Linear(2048, 256), torch.nn.ReLU(),
            #torch.nn.Linear(256, 256), torch.nn.ReLU(),
            #torch.nn.Linear(256, 2304), torch.nn.ReLU()
        )

        self._upconvs_barrel = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 4, 2),  torch.nn.ReLU(),  # 16 x 40
            torch.nn.Conv2d(64, 64, 3), torch.nn.ReLU(),               # 14 x 38
            torch.nn.ConvTranspose2d(64, 32, 4, 2), torch.nn.ReLU(),   # 30 x 78
            torch.nn.Conv2d(32, 32, 3),  torch.nn.ReLU(),              # 28 x 76
            torch.nn.ConvTranspose2d(32, 32, 4, 2), torch.nn.ReLU(),   # 58 x 154
            #unhit, (tvar, tmean, qvar, qmean, varcorr)*N_GAUS, coeff*N_GAUS     
            torch.nn.Conv2d(32, 1+N_GAUS*n_parameters_per_gaus, 3)     # 56 x 152
        )

        self._upconvs_top = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 4, 2),  torch.nn.ReLU(),  # 14 x 14
            torch.nn.Conv2d(64, 64, 3), torch.nn.ReLU(),               # 12 x 12
            torch.nn.ConvTranspose2d(64, 32, 4, 2), torch.nn.ReLU(),   # 26 x 26
            torch.nn.Conv2d(32, 32, 3),  torch.nn.ReLU(),              # 24 x 24
            torch.nn.ConvTranspose2d(32, 32, 4, 2), torch.nn.ReLU(),   # 50 x 50
            torch.nn.Conv2d(32, 1+N_GAUS*n_parameters_per_gaus, 3) # 48 x 48
        )
        self._upconvs_bottom = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 4, 2),  torch.nn.ReLU(),  # 14 x 14
            torch.nn.Conv2d(64, 64, 3), torch.nn.ReLU(),               # 12 x 12
            torch.nn.ConvTranspose2d(64, 32, 4, 2), torch.nn.ReLU(),   # 26 x 26
            torch.nn.Conv2d(32, 32, 3),  torch.nn.ReLU(),              # 24 x 24
            torch.nn.ConvTranspose2d(32, 32, 4, 2), torch.nn.ReLU(),   # 50 x 50
            torch.nn.Conv2d(32, 1+N_GAUS*n_parameters_per_gaus, 3) # 48 x 48
        )
        self._sigmoid = torch.nn.Sigmoid()
        self._tanh = torch.nn.Tanh()

    def forward(self, x) :
        # Concatenate MLPs that treat PID, pos, dir and energy inputs separately
        net = torch.cat( (self._mlp_pid(x[:,0:3]),self._mlp_pos(x[:,3:6]),self._mlp_dir(x[:,6:9]),self._mlp_E(x[:,9].reshape(len(x[:,9]),1))), 1)

        net_barrel = self._mlp_barrel(net)
        net_top = self._mlp_top(net)
        net_bottom = self._mlp_bottom(net)
        # Reshape into 7 x 19 figure in 64 channels. Enough?!
        net_barrel = net_barrel.view(-1, 64, 7, 19)
        net_top = net_top.view(-1, 64, 6, 6)
        net_bottom = net_bottom.view(-1, 64, 6, 6)
        # Upconvs layers                                   
        net_barrel = self._upconvs_barrel(net_barrel)[:,:,2:-3,1:-1]
        net_barrel = net_barrel.reshape(-1, 1+N_GAUS*n_parameters_per_gaus, 51*150)
        net_top = self._upconvs_top(net_top).view(-1, 1+N_GAUS*n_parameters_per_gaus, 48*48)
        net_bottom = self._upconvs_bottom(net_bottom).view(-1, 1+N_GAUS*n_parameters_per_gaus, 48*48)

        return [net_barrel, net_bottom, net_top]

def quadratic_spline_roots(spl):
    root = []
    knots = spl.get_knots()
    for a, b in zip(knots[:-1], knots[1:]):
        u, v, w = spl(a), spl((a+b)/2), spl(b)
        t = np.roots([u+w-2*v, w-u, 2*v])
        t = t[np.isreal(t) & (np.abs(t) <= 1)]
        root.extend(t*(b-a)/2 + (b+a)/2)
    return np.array(root)

def find_cubicspline_min(spl, root):
    cr_vals = spl(root)
    min_index = np.argmin(cr_vals)
    min_pt = root[min_index]
    return min_pt


net = CRinGeNet().cpu()
net.load_state_dict(torch.load("/home/junjiex/pro-junjiex/CRinGe/framework_training_output/CRinGe_SK_MultiGaus_"+str(N_GAUS)+"/CRinGe_SK_MultiGaus.cnn", map_location=lambda storage, loc: storage))
torch.set_grad_enabled(False)
net.eval()

h5file_mu="/home/junjiex/pro-junjiex/CRinGe/SKML/v1/WCSim_mu-_npztoh5_test.h5"

mask_mu = _get_mask(h5file_mu)
event_used = 0

# rect = patches.Rectangle((0, 48),150,51,linewidth=1, edgecolor='w', facecolor='none')
# circ_top = patches.Circle((75, 24),24, linewidth=1, edgecolor='w', facecolor='none')
# circ_bottom = patches.Circle((75, 123),24, linewidth=1, edgecolor='w', facecolor='none')

fscan_mu = open("/home/junjiex/pro-junjiex/CRinGe/scan_output/MultiGaus_SK_"+str(N_GAUS)+"_mu_LLH_Curves_"+str(n_event_to_scan)+".txt","wb")
fscan_e = open("/home/junjiex/pro-junjiex/CRinGe/scan_output/MultiGaus_SK_"+str(N_GAUS)+"_e_LLH_Curves_"+str(n_event_to_scan)+".txt","wb")
fout = open("/home/junjiex/pro-junjiex/CRinGe/scan_output/MultiGaus_SK_"+str(N_GAUS)+"_LLScan_test_"+str(n_event_to_scan)+".txt","wb")
#pdfout = PdfPages("/home/junjiex/pro-junjiex/CRinGe/scan_output/MultiGaus_SK_"+str(N_GAUS)+"_LLScan_test_curious_scans_"+str(n_event_to_scan)+".pdf")

for (data_mu, data_e) in zip(mu_data, e_data):
    try:
        label_mu = _fillLabel(data_mu)
        label_e = _fillLabel(data_e)
        data_mu = _fillData(data_mu)
        data_e = _fillData(data_e)
        
        xrand = random.random()
        
        mu_onbound = 0
        e_onbound = 0
        mu_local_min = 0
        e_local_min = 0
        
        #print(label_mu[0].shape, label_mu[1].shape, label_mu[2].shape, data_mu)
        #print("////////////")
        #print(label_e[0].shape, label_e[1].shape, label_e[2].shape, data_e)
        event_used += 1
        
        ids = np.column_stack([event_used, event_used])
        
        position_mu = [data_mu[0][3]*rscale, data_mu[0][4]*rscale, data_mu[0][5]*zscale]
        position_e = [data_e[0][3]*rscale, data_e[0][4]*rscale, data_e[0][5]*zscale]
        wall_mu = computeDwall_(position_mu)
        wall_e = computeDwall_(position_e)
        towall_mu = computeTowall_(position_mu, data_mu[0][6:9])
        towall_e = computeTowall_(position_e, data_e[0][6:9])
        
        energy_scanlist_mu, loss_scanlist_mu, orig_Emu, orig_Lossmu, dcharge_mu = _scan_lossvE(data_mu, label_mu, mask_mu)
        energy_scanlist_e, loss_scanlist_e, orig_Ee, orig_Losse, dcharge_e = _scan_lossvE(data_e, label_e, mask_mu)

        np.savetxt(fscan_mu, [["Muon scan energy"] + energy_scanlist_mu], fmt='%s')
        np.savetxt(fscan_mu, [["Muon scan loss"] + loss_scanlist_mu], fmt='%s')
        np.savetxt(fscan_mu, [["Muon true energy and loss"] +[orig_Emu, orig_Lossmu]], fmt='%s') 
        fscan_mu.flush()
        # typically the above line would do. however this is used to ensure that the file is written
        os.fsync(fscan_mu.fileno())
        
        np.savetxt(fscan_e, [["Electron scan energy"] + energy_scanlist_e], fmt='%s')
        np.savetxt(fscan_e, [["Electron scan loss"] + loss_scanlist_e], fmt='%s')
        np.savetxt(fscan_e, [["Electron true energy and loss"] +[orig_Ee, orig_Losse]], fmt='%s')
        fscan_e.flush()
        # typically the above line would do. however this is used to ensure that the file is written
        os.fsync(fscan_e.fileno())

        label_stack, pred_stack, nhit_mu = _stack_hit_event_display(label_mu, data_mu, mask_mu)
        
        '''
        figScan_mu = plt.figure(figsize=(17,5))
        axscan_mu = figScan_mu.add_subplot(131)
        axevent_mu = figScan_mu.add_subplot(132)
        axhit_mu = figScan_mu.add_subplot(133)
            
        mudisp = axevent_mu.imshow(label_stack, vmin=0, vmax=np.max(label_stack)+1)
        axevent_mu.set_axis_off()
        rect_mu = copy(rect)
        circt_mu = copy(circ_top)
        circb_mu = copy(circ_bottom)
        axevent_mu.add_patch(rect_mu)
        axevent_mu.add_patch(circt_mu) 
        axevent_mu.add_patch(circb_mu) 
        plt.colorbar(mudisp, ax=axevent_mu)
    
        muhit = axhit_mu.imshow(pred_stack, vmin=0, vmax=1)
        axhit_mu.set_axis_off()
        rect_muhit = copy(rect)
        circt_muhit = copy(circ_top)
        circb_muhit = copy(circ_bottom)
        axhit_mu.add_patch(rect_muhit)
        axhit_mu.add_patch(circt_muhit) 
        axhit_mu.add_patch(circb_muhit) 
        plt.colorbar(muhit, ax=axhit_mu)
        
        axscan_mu.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
        axscan_mu.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
        '''
        ##################################
        ## mu like ##
        ##################################
        
        splELoss_mu = InterpolatedUnivariateSpline(energy_scanlist_mu, loss_scanlist_mu, k=4)       
        crptsELoss_mu = splELoss_mu.derivative().roots()

        loss_mu_range = np.array(loss_scanlist_mu).max() - np.array(loss_scanlist_mu).min()
        if len(crptsELoss_mu) > 0:
            minLoss_mu = find_cubicspline_min(splELoss_mu, crptsELoss_mu)
            
        else :
            minLoss_mu = energy_scanlist_mu[np.argmin(loss_scanlist_mu)]
            mu_onbound = 1

        if splELoss_mu(minLoss_mu) > np.min(loss_scanlist_mu): # this is indeed a min loss
            mu_local_min = 1
        '''    
        if xrand < 0.01 or mu_local_min == 0 or mu_onbound == 1:
            axscan_mu.plot(energy_scanlist_mu, loss_scanlist_mu, color="blue", alpha=0.75)
            axscan_mu.scatter(orig_Emu, orig_Lossmu, color="red", label="Truth")
            axscan_mu.scatter(crptsELoss_mu, splELoss_mu(crptsELoss_mu), color = "orange", s=10, label="Local min/max")
            axscan_mu.scatter(minLoss_mu, splELoss_mu(minLoss_mu), color="violet", label="Reco", marker="^", s=30)
            axscan_mu.text(0.1, 0.95, "Muon: E=%.2f MeV \nWall=%.2f cm | Towall=%.2f cm" % (orig_Emu, wall_mu, towall_mu), verticalalignment = 'top', horizontalalignment='left', transform=axscan_mu.transAxes, color='black', fontsize=7, bbox={'facecolor': 'white', 'alpha': 1., 'pad': 10})
            axscan_mu.set_ylim([np.array(loss_scanlist_mu).min()-0.05*loss_mu_range, np.array(loss_scanlist_mu).max()+0.05*loss_mu_range])
            axscan_mu.set_ylabel("Loss (1.e7)")
            axscan_mu.set_xlabel("Energy [MeV]")
            axscan_mu.legend(loc='upper right', framealpha=0)
        
            pdfout.savefig(figScan_mu)
            #plt.show()
    
        axevent_mu.cla()
        axhit_mu.cla()
        axscan_mu.cla()
        plt.close(figScan_mu)
        '''
        ##################################
        ## e like ##
        ##################################
        
        label_stack, pred_stack, nhit_e = _stack_hit_event_display(label_e, data_e, mask_mu)
        '''       
        figScan_e = plt.figure(figsize=(17,5))
        axscan_e = figScan_e.add_subplot(131)
        axevent_e = figScan_e.add_subplot(132)
        axhit_e = figScan_e.add_subplot(133)
        axscan_e.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
        axscan_e.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
                
        edisp = axevent_e.imshow(label_stack, vmin=0.01, vmax=np.max(label_stack[label_stack<1e10])+1)
        axevent_e.set_axis_off()
        rect_e = copy(rect)
        circt_e = copy(circ_top)
        circb_e = copy(circ_bottom)
        axevent_e.add_patch(rect_e)
        axevent_e.add_patch(circt_e) 
        axevent_e.add_patch(circb_e)
        plt.colorbar(edisp, ax=axevent_e)
        
        ehit = axhit_e.imshow(pred_stack, vmin=0, vmax=1)
        axhit_e.set_axis_off()
        rect_ehit = copy(rect)
        circt_ehit = copy(circ_top)
        circb_ehit = copy(circ_bottom)
        axhit_e.add_patch(rect_ehit)
        axhit_e.add_patch(circt_ehit) 
        axhit_e.add_patch(circb_ehit) 
        plt.colorbar(ehit, ax=axhit_e)
        '''       
        splELoss_e = InterpolatedUnivariateSpline(energy_scanlist_e, loss_scanlist_e, k=4) 
        crptsELoss_e = splELoss_e.derivative().roots()
    
        loss_e_range = np.array(loss_scanlist_e).max() - np.array(loss_scanlist_e).min()
        
        if len(crptsELoss_e) > 0:
            minLoss_e = find_cubicspline_min(splELoss_e, crptsELoss_e)
        else:
            minLoss_e = energy_scanlist_e[np.argmin(loss_scanlist_e)]
            e_onbound = 1

        if splELoss_e(minLoss_e) > np.min(loss_scanlist_e):
            e_local_min = 1
        '''
        if xrand < 0.01 or e_local_min == 0 or e_onbound == 1:            
            axscan_e.plot(energy_scanlist_e, loss_scanlist_e, color="blue", alpha=0.75)
            axscan_e.scatter(orig_Ee, orig_Losse, color="red", label="Truth")
            axscan_e.scatter(crptsELoss_e, splELoss_e(crptsELoss_e), color = "orange", s=10, label="Local min/max")
            axscan_e.scatter(minLoss_e, splELoss_e(minLoss_e), color="violet", label="Reco", marker="^", s=30)
            axscan_e.text(0.1, 0.95, "Electron: E=%.2f MeV \nWall=%.2f cm | Towall=%.2f cm" % (orig_Ee, wall_e, towall_e), verticalalignment = 'top', horizontalalignment='left', transform=axscan_e.transAxes, color='black', fontsize=7, bbox={'facecolor': 'white', 'alpha': 1., 'pad': 10})
            axscan_e.set_ylim([np.array(loss_scanlist_e).min()-0.05*loss_e_range, np.array(loss_scanlist_e).max()+0.05*loss_e_range])
            axscan_e.set_ylabel("Loss (1.e7)")
            axscan_e.set_xlabel("Energy [MeV]")
            axscan_e.legend(loc='upper right', framealpha=0)
            pdfout.savefig(figScan_e)
            #plt.show()
            
        axscan_e.cla()
        axevent_e.cla()
        axhit_e.cla()
        plt.close(figScan_e)
        ##end of 2 events##
        '''        
        on_boundary = [mu_onbound,e_onbound]
        orig_energy = [orig_Emu, orig_Ee]
        minloss_energy = [minLoss_mu, minLoss_e]
        n_zero_deriv = [len(crptsELoss_mu), len(crptsELoss_e)]
        
        ##pid scan at reco energy##
        pid_mu, energy_mu = _scan_lossvPID(data_mu, label_mu, minLoss_mu, mask_mu)
        pid_e, energy_e = _scan_lossvPID(data_e, label_e, minLoss_e, mask_mu)
        pid_diffs = [pid_mu, pid_e]
        
        line_output = [event_used] + orig_energy + minloss_energy + on_boundary  + [wall_mu, wall_e] + [towall_mu, towall_e] + pid_diffs + n_zero_deriv + [mu_local_min, e_local_min] + [nhit_mu, nhit_e]
        print(event_used)
        np.savetxt(fout, [line_output], fmt='%s')
        fout.flush()
        # typically the above line would do. however this is used to ensure that the file is written
        os.fsync(fout.fileno())
        
        if event_used > n_event_to_scan:
            break

    except KeyboardInterrupt:
        break

#pdfout.close()
fscan_mu.close()
fscan_e.close()
fout.close()

stop = timeit.default_timer()
print('Time: ', stop - start) 
sys.exit()
