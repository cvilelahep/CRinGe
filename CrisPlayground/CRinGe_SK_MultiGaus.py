import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import sys, getopt
import random

seed=0
N_GAUS=10
learnR=0.0002
gradClip=1

clipQ = False
chargeclip = 2500
chargescale = 2500

energyscale = 5000
rscale = 1690
zscale = 1810

q_Epsilon = 1.E-4

'''
try:
    opts, args = getopt.getopt(sys.argv[1:], "hn:s:l:g:c:q:e:r:z:", ["npeak=","seed=","learning-rate=","gradclipper="])
except getopt.GetoptError:
    print('CRinGe_SK_MultiGaus.py -n <number of peaks> -s <random seed> -l <optimizer learning rate> -g <gradclipper norm>')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('CRinGe_SK_MultiGaus.py -n <number of peaks, default=1> -s <random seed, default=0> -l <optimizer learning rate, default=0.0002> -g <gradclipper norm, default=1>')
        sys.exit()
    elif opt in ("-n", "--npeak") :
        N_GAUS = int(arg)
        #print("RUNNING WITH "+str(N_GAUS)+" GAUSSIANS")
    elif opt in ("-s", "--seed") :
        seed = int(arg)
        #print("Random Seed set to "+str(seed))
    elif opt in ("-l", "--learing-rate") :
        learnR = float(arg)
        #print("Optimizer learning rate "+str(learnR))
    elif opt in ("-g", "--gradclipper") :
        gradClip = float(arg)
        #print("Gradient clipper "+str(gradClip))
    elif opt in ("-q", "--chargescale"):
        chargescale = float(arg)
        #print('Applying scaling to charge '+str(chargescale))
    elif opt in ("-e", "--energyscale"):
        energyscale = float(arg)
        #print('Applying scaling to event energy '+str(energyscale))
    elif opt in ("-r", "--rscale"):
        rscale = float(arg)
        #print('Applying scaling to vertex r '+str(rscale))
    elif opt in ("-z", "--zscale"):
        zscale = float(arg)
        #print('Applying scaling to vertex z '+str(zscale))
    elif opt in ("-c", "--chargeclipper"):
        chargeclip = float(arg)
        clipQ = True
        #print('Applying clipping to charge at '+str(chargeclip))
'''
print("RUNNING WITH "+str(N_GAUS)+" GAUSSIANS")
print("Random Seed set to "+str(seed))
print("Optimizer learning rate "+str(learnR))
print("Gradient clipper "+str(gradClip))
print("Applying scaling to charge "+str(chargescale))
print("Applying scaling to event energy "+str(energyscale))
print("Applying scaling to vertex r "+str(rscale))
print("Applying scaling to vertex z "+str(zscale))
print("Applying clipping to charge at "+str(chargeclip))

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# CRinGeNet
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
        self._mlp_barrel = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 8512), torch.nn.ReLU()   # 64 * 7 * 19
        )
        self._mlp_top = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, 256), torch.nn.ReLU(),            
            torch.nn.Linear(256, 2304), torch.nn.ReLU()   # 64 * 6 * 6
	)
        self._mlp_bottom = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024), torch.nn.ReLU(),        
            torch.nn.Linear(1024, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 2304), torch.nn.ReLU()
        )

        self._upconvs = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 4, 2),  torch.nn.ReLU(),  # 16 x 40
            torch.nn.Conv2d(64, 64, 3), torch.nn.ReLU(),               # 14 x 38 
            torch.nn.ConvTranspose2d(64, 32, 4, 2), torch.nn.ReLU(),   # 30 x 78
            torch.nn.Conv2d(32, 32, 3),  torch.nn.ReLU(),              # 28 x 76 
            torch.nn.ConvTranspose2d(32, 32, 4, 2), torch.nn.ReLU(),   # 58 x 154
            #unhit, (tvar, tmean, qvar, qmean, varcorr)*N_GAUS, coeff*N_GAUS     
            torch.nn.Conv2d(32, 1+N_GAUS*3, 3)                         # 56 x 152
        )
        self._upconvs_top = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 4, 2),  torch.nn.ReLU(),  # 14 x 14
            torch.nn.Conv2d(64, 64, 3), torch.nn.ReLU(),               # 12 x 12
            torch.nn.ConvTranspose2d(64, 32, 4, 2), torch.nn.ReLU(),   # 26 x 26 
            torch.nn.Conv2d(32, 32, 3),  torch.nn.ReLU(),              # 24 x 24
            torch.nn.ConvTranspose2d(32, 32, 4, 2), torch.nn.ReLU(),   # 50 x 50
            torch.nn.Conv2d(32, 1+N_GAUS*3, 3)                         # 48 x 48
        )
        self._upconvs_bottom = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 4, 2),  torch.nn.ReLU(),  # 14 x 14
            torch.nn.Conv2d(64, 64, 3), torch.nn.ReLU(),               # 12 x 12
            torch.nn.ConvTranspose2d(64, 32, 4, 2), torch.nn.ReLU(),   # 26 x 26
            torch.nn.Conv2d(32, 32, 3),  torch.nn.ReLU(),              # 24 x 24
            torch.nn.ConvTranspose2d(32, 32, 4, 2), torch.nn.ReLU(),   # 50 x 50
            torch.nn.Conv2d(32, 1+N_GAUS*3, 3)                         # 48 x 48
        )

#        self._sigmoid = torch.nn.Sigmoid()
#        self._softmax = torch.nn.Softmax(dim=1)

    def forward(self, x) :
        # Concatenate MLPs that treat PID, pos, dir and energy inputs separately
        net = torch.cat( (self._mlp_pid(x[:,0:3]),self._mlp_pos(x[:,3:6]),self._mlp_dir(x[:,6:9]),self._mlp_E(x[:,9].reshape(len(x[:,9]),1))), 1)

        # MegaMLP 
        #net = self._mlp(net)
        net_barrel = self._mlp_barrel(net)
        net_top = self._mlp_top(net)
        net_bottom = self._mlp_bottom(net)

        # Reshape into 7 x 19 figure in 64 channels. Enough?!
        net_barrel = net_barrel.view(-1, 64, 7, 19)
        net_top = net_top.view(-1, 64, 6, 6)
        net_bottom = net_bottom.view(-1, 64, 6, 6)
        # Upconvs layers
        net_barrel = self._upconvs(net_barrel)[:,:,2:-3,1:-1]
        net_barrel = net_barrel.reshape(-1, 1+N_GAUS*3, 51*150)
        net_top = self._upconvs_top(net_top).view(-1, 1+N_GAUS*3, 48*48)
        net_bottom = self._upconvs_bottom(net_bottom).view(-1, 1+N_GAUS*3, 48*48)

        return [net_barrel, net_bottom, net_top]

def gradient_clipper(model: torch.nn.Module, val: float) -> torch.nn.Module:
    for parameter in model.parameters():
        parameter.register_hook(lambda grad: grad.clamp_(-val, val))
    
    return model

# blob blob
class BLOB :
    pass
blob = BLOB()

blob.net = CRinGeNet().cuda()
#blob.bceloss = torch.nn.BCELoss(reduction = 'mean')
blob.bceloss = torch.nn.BCEWithLogitsLoss(reduction = 'sum')
#problem - having torch 1.8
#blob.criterion = torch.nn.SmoothL1Loss()
blob.optimizer = torch.optim.Adam(blob.net.parameters(), lr = learnR)
blob.data = None
blob.label = None
blob.label_top = None
blob.label_bottom = None
blob.top_mask = None
blob.bottom_mask = None
blob.last_good_state = None
blob.last_good_opt_state = None
N_LAST_GOOD_STATE = 5

# Forward path
def forward(blob, train=True) :
    with torch.set_grad_enabled(train) :
        data = torch.as_tensor(blob.data).cuda()
        prediction = blob.net(data)
        prediction_top = prediction[2]
        prediction_bottom = prediction[1]
        prediction = prediction[0]
 
        # Training
        loss, acc = 0, 0
        chargeloss = 0
        nhitTotal = 0
        if blob.label is not None :
            label = torch.as_tensor(blob.label).type(torch.FloatTensor).cuda()
            label_n = torch.stack( [ label for i in range(N_GAUS) ] ) # better way of doing this?

            punhit = prediction[:,0]
            logvar = torch.stack( [ prediction[:,i*2+1] for i in range(N_GAUS) ] )
            var = torch.exp(logvar)+q_Epsilon
            logmu = torch.stack( [ prediction[:,i*2+2] for i in range(N_GAUS) ] )
            mu = torch.exp(logmu)
            coeff = torch.nn.functional.softmax(prediction[:, -N_GAUS:], dim=1)
            coefficients = torch.stack( [ coeff[:,i]  for i in range(N_GAUS) ] )

            unhitMask = (label == 0)
            unhitTarget = torch.as_tensor(unhitMask).type(torch.FloatTensor).cuda()
            hitTarget = torch.as_tensor(~unhitMask).type(torch.FloatTensor).cuda()
            nhitTotal += hitTarget.sum()
            fracUnhit = unhitTarget.sum()/unhitTarget.numel()
            #loss += blob.bceloss(1-punhit[~unhitMask], hitTarget[~unhitMask]) + blob.bceloss(punhit[unhitMask], unhitTarget[unhitMask])
            loss += blob.bceloss(punhit, unhitTarget)
            
            if fracUnhit >= (1 - 1/unhitTarget.numel()) :
                chargeloss += 0
            else:
                chargeloss += (unhitMask==0).sum()*(1/2.)*np.log(2*np.pi)
                chargeloss += -torch.logsumexp(torch.log(coefficients[:,~unhitMask]) - 1/2.*logvar[:,~unhitMask] -1/2.*(label_n[:,~unhitMask]-mu[:,~unhitMask])**2/var[:,~unhitMask], dim = 0).sum()

        if blob.label_top is not None :
            label_top = torch.as_tensor(blob.label_top).type(torch.FloatTensor).cuda()
            label_n_top = torch.stack( [ label_top for i in range(N_GAUS) ] ) # better way of doing this?
            mask_top = torch.as_tensor(blob.top_mask).type(torch.BoolTensor).cuda()

            stack_mask_top = torch.stack( [ mask_top for i in range(label_top.shape[0])], dim = 0)
            n_mask_top = torch.squeeze(stack_mask_top, 1)
            
            punhit_top = prediction_top[:,0]*mask_top
            logvar_top = torch.stack( [ prediction_top[:,i*2+1]*mask_top for i in range(N_GAUS) ] )
            var_top = torch.exp(logvar_top)+q_Epsilon
            logmu_top = torch.stack( [ prediction_top[:,i*2+2]*mask_top for i in range(N_GAUS) ] )
            mu_top = torch.exp(logmu_top)
            coeff_top = torch.nn.functional.softmax(prediction_top[:, -N_GAUS:], dim=1)
            coefficients_top = torch.stack( [ coeff_top[:,i]*mask_top  for i in range(N_GAUS) ] )
            
            unhitMask_top = (label_top == 0) & (n_mask_top != 0)
            hitMask_top = (label_top > 0) 
            unhitTarget_top = torch.as_tensor(unhitMask_top).type(torch.FloatTensor).cuda()            
            hitTarget_top = torch.as_tensor(hitMask_top).type(torch.FloatTensor).cuda()
            nhitTotal += hitTarget_top.sum()
            fracUnhit_top = unhitTarget_top.sum()/torch.count_nonzero(n_mask_top).item()

            #loss += blob.bceloss(punhit_top[unhitMask_top], unhitTarget_top[unhitMask_top]) + blob.bceloss(1-punhit_top[hitMask_top], hitTarget_top[hitMask_top])
            loss += blob.bceloss(punhit_top[n_mask_top], unhitTarget_top[n_mask_top])
            
            if fracUnhit_top >= (1 - 1./torch.count_nonzero(n_mask_top).item()) :
                chargeloss += 0
            else:
                chargeloss += hitMask_top.sum()*(1/2.)*np.log(2*np.pi)
                chargeloss += -torch.logsumexp(torch.log(coefficients_top[:,hitMask_top]) - 1/2.*logvar_top[:,hitMask_top] -1/2.*(label_n_top[:,hitMask_top]-mu_top[:,hitMask_top])**2/var_top[:,hitMask_top], dim = 0).sum()
                #chargeloss += -(1-fracUnhit_top)*(torch.sum(torch.log(coefficients_top[:,~unhitMask_bottom] - blob.gausNLLloss(mu_top[:, ~unhitMask_bottom], label_n_top[:, ~unhitMask_bottom], var_top[:, ~unhitMask_bottom])), dim=0)).mean()

        else:
            print("No label on top cap!")
            raise ValueError
                
        if blob.label_bottom is not None :
            label_bottom = torch.as_tensor(blob.label_bottom).type(torch.FloatTensor).cuda()
            label_n_bottom = torch.stack( [ label_bottom for i in range(N_GAUS) ] ) # better way of doing this?
            mask_bottom = torch.as_tensor(blob.bottom_mask).type(torch.BoolTensor).cuda()

            stack_mask_bottom = torch.stack( [ mask_bottom for i in range(label_bottom.shape[0])], dim = 0)
            n_mask_bottom = torch.squeeze(stack_mask_bottom, 1)

            punhit_bottom = prediction_bottom[:,0]*mask_bottom
            logvar_bottom = torch.stack( [ prediction_bottom[:,i*2+1]*mask_bottom for i in range(N_GAUS) ] )
            var_bottom = torch.exp(logvar_bottom)+q_Epsilon
            logmu_bottom = torch.stack( [ prediction_bottom[:,i*2+2]*mask_bottom for i in range(N_GAUS) ] )
            mu_bottom = torch.exp(logmu_bottom)
            coeff_bottom = torch.nn.functional.softmax(prediction_bottom[:, -N_GAUS:], dim=1)
            coefficients_bottom = torch.stack( [ coeff_bottom[:,i]*mask_bottom  for i in range(N_GAUS) ] )
                
            unhitMask_bottom = (label_bottom == 0) & (mask_bottom != 0)
            hitMask_bottom = (label_bottom > 0)
            unhitTarget_bottom = torch.as_tensor(unhitMask_bottom).type(torch.FloatTensor).cuda()
            hitTarget_bottom = torch.as_tensor(hitMask_bottom).type(torch.FloatTensor).cuda()
            nhitTotal += hitTarget_bottom.sum()
            fracUnhit_bottom = unhitTarget_bottom.sum()/torch.count_nonzero(n_mask_bottom).item()

            #loss += blob.bceloss(punhit_bottom[unhitMask_bottom], unhitTarget_bottom[unhitMask_bottom]) + blob.bceloss(1-punhit_bottom[hitMask_bottom], hitTarget_bottom[hitMask_bottom])
            loss += blob.bceloss(punhit_bottom[n_mask_bottom], unhitTarget_bottom[n_mask_bottom])
            
            if fracUnhit_bottom >= (1 - 1./torch.count_nonzero(n_mask_bottom).item()) :
                chargeloss += 0
            else:
                chargeloss += hitMask_bottom.sum()*(1/2.)*np.log(2*np.pi)                
                chargeloss += -torch.logsumexp(torch.log(coefficients_bottom[:,hitMask_bottom]) - 1/2.*logvar_bottom[:,hitMask_bottom] -1/2.*(label_n_bottom[:,hitMask_bottom]-mu_bottom[:,hitMask_bottom])**2/var_bottom[:,hitMask_bottom], dim = 0).sum()

        else:
            print("No label on bottom cap!")
            raise ValueError

        #if nhitTotal > 0 : 
        #    chargeloss = chargeloss / nhitTotal
        #    loss = loss / nhitTotal
        
        loss += chargeloss        

        
        
        if chargeloss != chargeloss:
            print("################################################################################")
            print("NAN Charge LOSS")
            print("################################################################################")
            #exit()

        elif loss != loss :
            with open("fDebug.p", "wb") as f:
                print("################################################################################")
                print("NAN LOSS")
                print("################################################################################")
                print("LOSS")
                print(loss.cpu().detach().numpy())
                pickle.dump(loss.cpu().detach().numpy(), f)
                print("################################################################################")
                print("DATA")
                print(data.cpu().numpy())
                pickle.dump(data.cpu().numpy(), f)
                print("--------------------------------------------------------------------------------")
                print("LABEL")
                print(label.cpu().numpy())
                pickle.dump(label.cpu().numpy(), f)
                print("--------------------------------------------------------------------------------")
                print("punhit")
                print(punhit.cpu().detach().numpy())
                pickle.dump(punhit.cpu().detach().numpy(), f)
                print("--------------------------------------------------------------------------------")
                print("MU")
                print(mu.cpu().detach().numpy())
                pickle.dump(mu.cpu().detach().numpy(), f)
                print("--------------------------------------------------------------------------------")
                print("VAR")
                pickle.dump(var.cpu().detach().numpy(), f)
                print(var.cpu().detach().numpy())
                print("--------------------------------------------------------------------------------")
                print("LOGMU")
                print(logmu.cpu().detach().numpy())
                pickle.dump(logmu.cpu().detach().numpy(), f)
                print("--------------------------------------------------------------------------------")
                print("LOGVAR")
                pickle.dump(logvar.cpu().detach().numpy(), f)
                print(logvar.cpu().detach().numpy())
                sys.stdout.flush()
            #exit()
        else :
            if blob.last_good_state == None :
                blob.last_good_state = [None]*N_LAST_GOOD_STATE
                blob.last_good_opt_state = [None]*N_LAST_GOOD_STATE
                for i_state in range(N_LAST_GOOD_STATE) :
                    blob.last_good_state[i_state] = blob.net.state_dict().copy()
                    blob.last_good_opt_state[i_state] = blob.optimizer.state_dict().copy()
                
            for i_state in range(N_LAST_GOOD_STATE-1) :
                blob.last_good_state[i_state] = blob.last_good_state[i_state+1].copy()
                blob.last_good_opt_state[i_state] = blob.last_good_opt_state[i_state+1].copy()
                
            blob.last_good_state[-1] = blob.net.state_dict().copy()
            blob.last_good_opt_state[-1] = blob.optimizer.state_dict().copy()
        blob.loss = loss
        return {'prediction' : prediction.cpu().detach().numpy(),
                'loss' : loss.cpu().detach().item(),
                'chargeloss' : chargeloss.cpu().detach().item()}
    # Backward path
def backward(blob) :
    blob.optimizer.zero_grad()
    blob.loss.backward()
    # Clip gradient norm to avoid exploding gradients:
    #torch.nn.utils.clip_grad.clip_grad_norm_(blob.net.parameters(), gradClip)
    blob.optimizer.step()

def _init_fn(worker_id):
    np.random.seed(int(seed)+worker_id)

# Data loaders
from iotools import loader_factory
DATA_DIRS=['/home/junjiex/pro-junjiex/CRinGe/SKML/v1']
import h5py
file_handle = h5py.File(DATA_DIRS[0]+"/WCSim_mu-_npztoh5_test.h5", mode='r')
dim_cap = file_handle['mask'][0].shape
blob.top_mask = file_handle['mask'][0].reshape(-1, dim_cap[0]*dim_cap[1])
blob.bottom_mask = file_handle['mask'][1].reshape(-1, dim_cap[0]*dim_cap[1])
file_handle.close()


#DATA_DIRS=['/home/cvilela/HKML/varyAll/']
#DATA_DIRS=['/storage/shared/cvilela/HKML/varyAll']
#DATA_DIRS=['/home/junjiex/projects/def-pdeperio/junjiex/HKML/varyAll']
#reading in mask array
train_loader=loader_factory('H5Dataset', batch_size=200, shuffle=True, num_workers=4, worker_init_fn=_init_fn, pin_memory=True,  data_dirs=DATA_DIRS, flavour='test.h5', start_fraction=0, use_fraction=0.75, read_keys= ["positions","directions", "energies", "event_data_top", "event_data_bottom"])
test_loader=loader_factory('H5Dataset', batch_size=200, shuffle=True, num_workers=2, worker_init_fn=_init_fn, pin_memory=True, data_dirs=DATA_DIRS, flavour='test.h5', start_fraction=0.75, use_fraction=0.24, read_keys= ["positions","directions", "energies", "event_data_top", "event_data_bottom"])

# Useful function
def fillLabel (blob,data) :
    # Label is vector of charges. Mind = Blown
    dim = data[0].shape
    blob.label = data[0][:,:,:,0].reshape(-1,dim[1]*dim[2])/chargescale
    dim_cap = data[5].shape
    blob.label_top = data[5][:,:,:,0].reshape(-1, dim_cap[1]*dim_cap[2])/chargescale
    blob.label_bottom = data[6][:,:,:,0].reshape(-1, dim_cap[1]*dim_cap[2])/chargescale

    if clipQ:
        blob.label.clip(max = chargeclip, out = blob.label)
        blob.label_top.clip(max = chargeclip, out = blob.label_top)
        blob.label_bottom.clip(max = chargeclip, out = blob.label_bottom)

def fillData (blob,data) :
    # Data is particle state
    oneHotGamma = np.array(data[1] == 0)
    oneHotE = np.array(data[1] == 1)
    oneHotMu = np.array(data[1] == 2)
    
    blob.data =  np.hstack((oneHotGamma.reshape(len(oneHotGamma),1), oneHotE.reshape(len(oneHotE),1), oneHotMu.reshape(len(oneHotMu),1), # One-hot PID
                            data[2][:,0,:], # Positions
                            data[3][:,0,:], # Directions
                            data[4][:,0].reshape(len(data[4][:,0]),1) ) ) # Energy

    blob.data[:,3] /= rscale
    blob.data[:,4] /= rscale
    blob.data[:,5] /= zscale
    blob.data[:,9] /= energyscale
    
# Training loop
TRAIN_EPOCH = 10.
blob.net.train()
epoch = 0.
iteration = 0.
#sys.stdout = open("N2GausTrain.log","w")

while epoch < TRAIN_EPOCH :
    print('Epoch', epoch, int(epoch+0.5), 'Starting @',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for i,data in enumerate(train_loader) :
        fillLabel(blob,data)
        fillData(blob,data)

        res = forward(blob, True)
        
        if blob.loss != blob.loss :
            print("BAD LOSS, loading last good state")
            del blob.net
            del blob.optimizer

            blob.net = CRinGeNet().cuda()
            blob.optimizer = torch.optim.Adam(blob.net.parameters(), lr = 2e-4)

            blob.net.load_state_dict(blob.last_good_state[0])
            blob.optimizer.load_state_dict(blob.last_good_opt_state[0])
            continue
        
        backward(blob)
        epoch += 1./len(train_loader)
        iteration += 1

        # Report progress
        if i == 0 or (i+1)%10 == 0 :
            print('TRAINING', 'Iteration', iteration, 'Epoch', epoch, 'HitProb Loss', res['loss']-res['chargeloss'], 'Charge Loss', res['chargeloss'])
            
        if (i+1)%100 == 0 :
            with torch.no_grad() :
                blob.net.eval()
                test_data = next(iter(test_loader))
                fillLabel(blob,test_data)
                fillData(blob,test_data)
                res = forward(blob, False)
                print('VALIDATION', 'Iteration', iteration, 'Epoch', epoch, 'HitProb Loss', res['loss']-res['chargeloss'], 'Charge Loss', res['chargeloss'])

        if (iteration+1)%7363 == 0 :
            torch.save(blob.net.state_dict(), "testCRinGe_SK_MultiGaus_"+str(N_GAUS)+"_GradClip_"+str(gradClip)+"_LR_"+str(learnR)+"_i_"+str(iteration)+"_Qclip_"+str(chargeclip)+"_Qscale_"+str(chargescale)+"_rscale_"+str(rscale)+"_zscale_"+str(zscale)+"_Escale_"+str(energyscale)+".cnn")
        if epoch >= TRAIN_EPOCH :
            break

torch.save(blob.net.state_dict(), "testCRinGe_SK_MultiGaus_"+str(N_GAUS)+"_GradClip_"+str(gradClip)+"_LR_"+str(learnR)+"_Qclip_"+str(chargeclip)+"_Qscale_"+str(chargescale)+"_rscale_"+str(rscale)+"_zscale_"+str(zscale)+"_Escale_"+str(energyscale)+".cnn")
#sys.stdout.close
