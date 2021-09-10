import matplotlib
import tkinter
#matplotlib.use('TkAgg')
import random
import math
from math import atan2

import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import cm

import sys

import numpy as np
import h5py
import scipy.stats as stats
from scipy.interpolate import CubicSpline


seed=0
N_GAUS = 2
iPMT = 7787
probThr = 0.5

if len(sys.argv) > 1 :
    N_GAUS = int(sys.argv[1])
    if len(sys.argv) == 3 :
        seed = int(sys.argv[2])
    if len(sys.argv) == 4 :
        iPMT = int(sys.argv[3])
        
print("Random Seed set to "+str(seed))
print("RUNNING WITH "+str(N_GAUS)+" GAUSSIANS and Timing Peak")
print("Scanning for PMT No."+str(iPMT))

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True        

StepSizeQ = 0.5
StepSizeT = 0.05
nStep = 40
#Qpoint=[-1, 2.5, 7]
#TpointIndep=[1.5, 0.8, -0.5]
#TpointCorr=[0.45, 0.26, 0.]

class CRinGeNet(torch.nn.Module) :
    def __init__(self) :
        super(CRinGeNet, self).__init__()

        self._mlp_pid = torch.nn.Sequential(
#            torch.nn.BatchNorm1d(3),
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

        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 14784), torch.nn.ReLU()
        )


        self._upconvs = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 4, 2),  torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3), torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(64, 32, 4, 2), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3),  torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(32, 32, 4, 2), torch.nn.ReLU(),
            #                    phit, gaussian, logvar, logmu, coefficient
            torch.nn.Conv2d(32, 1+N_GAUS*6, 3)
        )

        self._sigmoid = torch.nn.Sigmoid()
        self._tanh = torch.nn.Tanh()

    def forward(self, x) :
        # Concatenate MLPs that treat PID, pos, dir and energy inputs separately
        net = torch.cat( (self._mlp_pid(x[:,0:3]),self._mlp_pos(x[:,3:6]),self._mlp_dir(x[:,6:9]),self._mlp_E(x[:,9].reshape(len(x[:,9]),1))), 1)

        # MegaMLP 
        net = self._mlp(net)
        
        # Reshape into 11 x 21 figure in 64 channels. Enough?!
        net = net.view(-1, 64, 11, 21)

        # Need to flatten? Maybe...
        net = self._upconvs(net).view(-1, 1+N_GAUS*6, 88*168)
        
        for i in range(N_GAUS):
            a11 = torch.exp(net[:, 1+i*5, :])
            a22 = torch.exp(net[:, 1+i*5+2, :]) 
            a12 = net[:,1+i*5+4,:]
            
            #net[:,1+i*5+4,:] = (a11+a22)*self._sigmoid(a12) - 0.5*(a11+a22)
            net[:,1+i*5+4,:] = 0.5*(a11+a22)*self._tanh(a12)

        #corr has to be less than 1
        #corr = self._sigmoid(net[:, -N_GAUS:].view(-1, N_GAUS, 88*168))
        #net = torch.cat( (net[:, 0:-N_GAUS], corr), dim=1)
        # return net
        # 0th channel is probability, pass through Sigmoid
        #hitprob = self._sigmoid(net[:,0]).view(-1, 1, 88*168)
        #hitprob = net[:,0].view(-1, 1, 88*168)

        # Last N_GAUS channels are coefficients, pass through softmax
        #coeffs = self._softmax(net[:,-N_GAUS:])

        #net = torch.cat( (hitprob, net[:,1:-N_GAUS], coeffs), dim=1)
        #net = torch.cat( (hitprob, net[:,1:]), dim=1)

        return net

#def gaussian2D(x, t, coeff, mu, var, tmu, tvar, varshare):
def gaussian2D(x, t, coeff, mu, a22, tmu, a11, a12):
    
    gauss2d = coeff*a11*a22/(2*np.pi)*np.exp(-1./2*(((t-tmu)*a11)**2+(x-mu)**2*(a22**2+a12**2)+2*a11*a12*(t-tmu)*(x-mu)))
    return gauss2d

def gaussian2x1D(x, t, coeff, mu, var, tmu, tvar, varshare):
    VAR = var + varshare
    tVAR = tvar + varshare
    gauss2x1d = coeff/(2*np.pi*(VAR*tVAR)**0.5)*np.exp(-1./2*(x - mu)**2/VAR-1./2*(t - tmu)**2/tVAR)
    return gauss2x1d

def user_softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

net = CRinGeNet().cpu()
net.load_state_dict(torch.load("testCRinGe_MultiGausTimeCorr_"+str(N_GAUS)+".cnn", map_location=lambda storage, loc: storage))
#net.load_state_dict(torch.load("testCRinGe_MultiGausTimeCorr_"+str(N_GAUS)+"_i_22088.0.cnn", map_location=lambda storage, loc: storage))

torch.set_grad_enabled(False)
net.eval()

def _init_fn(worker_id):
    np.random.seed(int(seed)+worker_id)

class BLOB :
    pass
blob = BLOB()
blob.data = None
blob.label = None


nbatch = 200
# Data loaders
from iotools import loader_factory
DATA_DIRS=['/home/junjiex/projects/def-pdeperio/junjiex/HKML/varyAll']
data_loader=loader_factory('H5Dataset', batch_size=nbatch, shuffle=True, num_workers=1, worker_init_fn=_init_fn, pin_memory=True, data_dirs=DATA_DIRS, flavour='1M.h5', start_fraction=0.99, use_fraction=0.01, read_keys= ["positions","directions", "energies"])

nloop = 10
xloop = 0

nHitTrue = 0
hithitProb = [] # hit prob when true hit
hitunhitProb = [] # unhit prob when true hit
nUnHitTrue = 0
unhithitProb = [] # hit prob when true unhit
unhitunhitProb = [] # unhit prob when true unhit


#gammaArr = []
#muonArr = []
#eArr = []
#energyArr = []
#sinArr = []
#cosArr = []

dcorrLLQArr = []
dindepLLQArr = []
dcorrLLTArr = []
dindepLLTArr = []

figQLLScan = plt.figure(figsize=(5,5))
figTLLScan = plt.figure(figsize=(5,5))

thisAxQ = figQLLScan.add_subplot(111)
#thisAxQindep = figQLLScan.add_subplot(111)
thisAxT = figTLLScan.add_subplot(111)
#thisAxTindep = figTLLScan.add_subplot(111)

pdfLLQ = PdfPages("/project/6008045/junjiex/CRinGe/PNGplots/LLNScanQ_MultiGausTCorr_"+str(N_GAUS)+"_PMT_"+str(iPMT)+".pdf")
pdfLLT = PdfPages("/project/6008045/junjiex/CRinGe/PNGplots/LLNScanT_MultiGausTCorr_"+str(N_GAUS)+"_PMT_"+str(iPMT)+".pdf")


#label structure: [nbatch, nPMT, charge&time]
#data structure: [nbatch, gamma, e, mu, x, y, z, dx, dy, dz, E] 
for i,data in enumerate(data_loader):
    dim = data[0].shape
    labelQ = data[0][:,:,:,0].reshape(-1,dim[1]*dim[2])
    labelT = 0.1*(data[0][:,:,:,1]).reshape(-1, dim[1]*dim[2])

    oneHotGamma = np.array(data[1] == 0)
    oneHotE = np.array(data[1] == 1)
    oneHotMu = np.array(data[1] == 2)
    
    data =  np.hstack((oneHotGamma.reshape(len(oneHotGamma),1), oneHotE.reshape(len(oneHotE),1), oneHotMu.reshape(len(oneHotMu),1), # One-hot PID
                       data[2][:,0,:], # Positions
                       data[3][:,0,:], # Directions
                       data[4][:,0].reshape(len(data[4][:,0]),1) ) ) # Energy
    
#    if xloop >= nloop:
#        break
    for ibatch in range(nbatch):

        if ibatch % 10 == 0:
            print("Iterating at loop", xloop, "batch", ibatch)
        tmuArr = []
        a11Arr = []
        tvarArr = []
        muArr = []
        a22Arr = []
        varArr = []
        muHitProbArr = []
        coeffArr = []
        corrArr = []
        a12Arr = []
        
        Dtrack=[data[ibatch]]
        Dtrack=torch.as_tensor(Dtrack).cpu()
        #Dtrack=torch.as_tensor(blob.data[ibatch]).cpu()
        
        eventHitProb = 1./(1+np.exp(net(Dtrack).cpu().detach().numpy()[0,0,iPMT])).item()
        charge=labelQ[ibatch][iPMT]
        time=labelT[ibatch][iPMT]
        
        if charge > 0 :
            nHitTrue += 1
            hithitProb.append(eventHitProb)
            hitunhitProb.append(1-eventHitProb)
        else:
            nUnHitTrue += 1
            unhithitProb.append(eventHitProb)
            unhitunhitProb.append(eventHitProb)
            
        if eventHitProb < probThr:
            continue

        x = np.arange(charge-0.5*StepSizeQ*nStep, charge+0.5*StepSizeQ*nStep+StepSizeQ, StepSizeQ)
        t = np.arange(time-0.5*StepSizeT*nStep, time+0.5*StepSizeT*nStep+StepSizeT, StepSizeT)

        #gammaArr.append(data[ibatch][0])
        #eArr.append(data[ibatch][1])
        #muonArr.append(data[ibatch][2])
        #energyArr.append(data[ibatch][-1])
        #sinArr.append(data[ibatch][8])
        #cosArr.append(data[ibatch][6])


        for iax in range(N_GAUS):
            eventa11 = np.exp(net(Dtrack).cpu().detach().numpy()[0,1+iax*5,iPMT]).item()
            a11Arr.append(eventa11)
            
            eventTime = net(Dtrack).cpu().detach().numpy()[0, 1+iax*5+1,iPMT].item()
            tmuArr.append(eventTime)
        
            eventa22 = np.exp(net(Dtrack).cpu().detach().numpy()[0,1+iax*5+2,iPMT]).item()
            a22Arr.append(eventa22)
            
            eventMu = np.exp(net(Dtrack).cpu().detach().numpy()[0,1+iax*5+3,iPMT]).item()
            muArr.append(eventMu)
            
            eventa12 = np.exp(net(Dtrack).cpu().detach().numpy()[0, 1+iax*5+4,iPMT]).item()
            a12Arr.append(eventa12)
            
            eventCoeff = net(Dtrack).cpu().detach().numpy()[0, -N_GAUS+iax,iPMT].item()            
            coeffArr.append(eventCoeff)
            
        corrQLL = []
        indepQLL = []
        corrTLL = []
        indepTLL = []
        
        coeffArr = user_softmax(coeffArr)
        for istep in range(nStep+1):
#            print("Correlated charge scan", charge-0.5*nStep*StepSizeQ+istep*StepSizeQ, time, sum(gaussian2D(charge-0.5*nStep*StepSizeQ+istep*StepSizeQ, time, coeffArr[ig], muArr[ig], a22Arr[ig], tmuArr[ig], a11Arr[ig], a12Arr[ig]) for ig in range(N_GAUS)))
#            print("Correlated time scan", charge, time-0.5*nStep*StepSizeT+istep*StepSizeT, sum(gaussian2x1D(charge-0.5*nStep*StepSizeQ+istep*StepSizeQ,time,coeffArr[ig],muArr[ig], 1/(a22Arr[ig]**2), tmuArr[ig],(a12Arr[ig]**2+a22Arr[ig]**2)/((a11Arr[ig]*a12Arr[ig])**0.5),0.0) for ig in range(N_GAUS)))
            
            corrQLL.append(-2.*np.log(1.E-10+sum(gaussian2D(charge-0.5*nStep*StepSizeQ+istep*StepSizeQ, 
                                                            time, 
                                                            coeffArr[ig], 
                                                            muArr[ig], 
                                                            a22Arr[ig], 
                                                            tmuArr[ig],
                                                            a11Arr[ig],
                                                            a12Arr[ig]) for ig in range(N_GAUS))))
            indepQLL.append(-2.*np.log(1.E-10+sum(gaussian2x1D(charge-0.5*nStep*StepSizeQ+istep*StepSizeQ, 
                                                               time, 
                                                               coeffArr[ig], 
                                                               muArr[ig], 
                                                               1/(a22Arr[ig]**2), 
                                                               tmuArr[ig],
                                                               (a12Arr[ig]**2+a22Arr[ig]**2)/((a11Arr[ig]*a12Arr[ig])**0.5),
                                                               0.0) for ig in range(N_GAUS))))
            corrTLL.append(-2.*np.log(1.E-10+sum(gaussian2D(charge, 
                                                            time-0.5*nStep*StepSizeT+istep*StepSizeT, 
                                                            coeffArr[ig], 
                                                            muArr[ig], 
                                                            a22Arr[ig], 
                                                            tmuArr[ig],
                                                            a11Arr[ig],
                                                            a12Arr[ig]) for ig in range(N_GAUS))))
            indepTLL.append(-2.*np.log(1.E-10+sum(gaussian2x1D(charge, 
                                                               time-0.5*nStep*StepSizeT+istep*StepSizeT,  
                                                               coeffArr[ig], 
                                                               muArr[ig], 
                                                               1/(a22Arr[ig]**2), 
                                                               tmuArr[ig],
                                                               (a12Arr[ig]**2+a22Arr[ig]**2)/((a11Arr[ig]*a12Arr[ig])**0.5),
                                                               0.0) for ig in range(N_GAUS))))
        
        #if ibatch%100 == 0:
        print("Batch",ibatch,"has a hit in PMT", iPMT)
            
        thisAxQ.plot(x, corrQLL, color = 'b', linestyle = 'solid', alpha = 0.5, label = "Corr")    
        thisAxQ.scatter(charge, corrQLL[int(0.5*nStep)], s = 10, c = 'green')
        thisAxQ.plot(x, indepQLL, color = 'r', linestyle = 'solid', alpha = 0.5, label = "Indep" % (time*10))
        thisAxQ.scatter(charge, indepQLL[int(0.5*nStep)], s = 10, c = 'green')
        thisAxT.plot(t, corrTLL, color = 'b', linestyle = 'solid', alpha = 0.5, label = "Corr" % (charge))
        thisAxT.scatter(time, corrTLL[int(0.5*nStep)], s = 10, c = 'green')
        thisAxT.plot(t, indepTLL, color = 'r', linestyle = 'solid', alpha = 0.5, label = "Indep")
        thisAxT.scatter(time, indepTLL[int(0.5*nStep)], s = 10, c = 'green')

        thisAxQ.text(0.5, 0.95, "P-hit=%.2f | T = %.1f ns | Q = %.1f PE" % (eventHitProb,time*10,charge), verticalalignment = 'top', horizontalalignment='center', transform=thisAxQcorr.transAxes, color='black', fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
        thisAxT.text(0.5, 0.95, "P-hit=%.2f | T = %.1f ns | Q = %.1f PE" % (eventHitProb,time*10,charge), verticalalignment = 'top', horizontalalignment='center', transform=thisAxTcorr.transAxes, color='black', fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        thisAxQ.set_xlim(0, x[-1])
        #thisAxQindep.set_xlim(0, x[-1])
        thisAxQ.set_ylim(0, indepQLL[-1]*2.)
        #thisAxQindep.set_ylim(0, indepQLL[-1]*2.)
        
        
        thisAxQ.legend(loc='lower right')
        #thisAxQindep.legend(loc='lower right')
        thisAxT.legend(loc='lower right')
        #thisAxTindep.legend(loc='lower right')
        
        thisAxQ.set_xlabel('Charge [P.E.]')
        thisAxQ.set_ylabel('-2log(ln)')
        #thisAxQindep.set_xlabel('Charge [P.E.]')
        #thisAxQindep.set_ylabel('-2log(ln)')
        thisAxT.set_xlabel('Time [10ns]')
        thisAxT.set_ylabel('-2log(ln)')
        #thisAxTindep.set_xlabel('Time [10ns]')
        #thisAxTindep.set_ylabel('-2log(ln)')
        
        
        #figQLLScan.tight_layout()
        #figQLLScan.savefig("/project/6008045/junjiex/CRinGe/PNGplots/LLNScanQ_MultiGausTCorr_"+str(N_GAUS)+"_PMT_"+str(iPMT)+"_iloop_"+str(xloop)+"_batch_"+str(ibatch)+".png")
        #figTLLScan.tight_layout()
        #figTLLScan.savefig("/project/6008045/junjiex/CRinGe/PNGplots/LLNScanT_MultiGausTCorr_"+str(N_GAUS)+"_PMT_"+str(iPMT)+"_iloop_"+str(xloop)+"_batch_"+str(ibatch)+".png")

        figQLLScan.savefig(pdfLLQ, format='pdf')
        figTLLScan.savefig(pdfLLT, format='pdf')
        
        thisAxQ.cla()
        #thisAxQindep.cla()
        thisAxT.cla()
        #thisAxTindep.cla()
        
        #plt.close(figQLLScan)
        #plt.close(figTLLScan)
        #del Dtrack
    xloop += 1
#false positive ratio FPR = FP/(FP+TN)

pdfLLQ.close()
pdfLLT.close()

FP = sum(np.array(unhithitProb)[np.array(unhithitProb) > probThr])
FPR = FP/(FP+nUnHitTrue)
print("FPR is", FPR)

