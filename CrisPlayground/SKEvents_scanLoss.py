import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import math

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

        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024), torch.nn.ReLU(),
        )
        self._mlp_barrel = torch.nn.Sequential(
            torch.nn.Linear(1024, 8512), torch.nn.ReLU()   # 64 * 7 * 19
        )
        self._mlp_top = torch.nn.Sequential(
            torch.nn.Linear(1024, 2304), torch.nn.ReLU()   # 64 * 6 * 6
        )
        self._mlp_bottom = torch.nn.Sequential(
            torch.nn.Linear(1024, 2304), torch.nn.ReLU()
        )

        self._upconvs = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 4, 2),  torch.nn.ReLU(),  # 16 x 40 
            torch.nn.Conv2d(64, 64, 3), torch.nn.ReLU(),               # 14 x 38 
                                                                                 
            torch.nn.ConvTranspose2d(64, 32, 4, 2), torch.nn.ReLU(),   # 30 x 78 
            torch.nn.Conv2d(32, 32, 3),  torch.nn.ReLU(),              # 28 x 76 
                                                                                 
            torch.nn.ConvTranspose2d(32, 32, 4, 2), torch.nn.ReLU(),   # 58 x 154
            torch.nn.Conv2d(32, 3, 3)                                  # 56 x 152
        )
        self._upconvs_top = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 4, 2),  torch.nn.ReLU(),  # 14 x 14 
            torch.nn.Conv2d(64, 64, 3), torch.nn.ReLU(),               # 12 x 12 
                                                                                 
            torch.nn.ConvTranspose2d(64, 32, 4, 2), torch.nn.ReLU(),   # 26 x 26 
            torch.nn.Conv2d(32, 32, 3),  torch.nn.ReLU(),              # 24 x 24 
                                                                                 
            torch.nn.ConvTranspose2d(32, 32, 4, 2), torch.nn.ReLU(),   # 50 x 50
            torch.nn.Conv2d(32, 3, 3)                                  # 48 x 48
        )
        self._upconvs_bottom = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 4, 2),  torch.nn.ReLU(),  # 14 x 14 
            torch.nn.Conv2d(64, 64, 3), torch.nn.ReLU(),               # 12 x 12 
                                                                                 
            torch.nn.ConvTranspose2d(64, 32, 4, 2), torch.nn.ReLU(),   # 26 x 26 
            torch.nn.Conv2d(32, 32, 3),  torch.nn.ReLU(),              # 24 x 24 
                                                                                 
            torch.nn.ConvTranspose2d(32, 32, 4, 2), torch.nn.ReLU(),   # 50 x 50
            torch.nn.Conv2d(32, 3, 3)                                  # 48 x 48
        )

        self._sigmoid = torch.nn.Sigmoid()

    def forward(self, x) :
        # Concatenate MLPs that treat PID, pos, dir and energy inputs separately
        net = torch.cat( (self._mlp_pid(x[:,0:3]),self._mlp_pos(x[:,3:6]),self._mlp_dir(x[:,6:9]),self._mlp_E(x[:,9].reshape(len(x[:,9]),1))), 1)

        # MegaMLP 
        net = self._mlp(net)
        net_barrel = self._mlp_barrel(net)
        net_top = self._mlp_top(net)
        net_bottom = self._mlp_bottom(net)
 
        # Reshape into 7 x 19 figure in 64 channels. Enough?!
        net_barrel = net_barrel.view(-1, 64, 7, 19)
        net_top = net_top.view(-1, 64, 6, 6)
        net_bottom = net_bottom.view(-1, 64, 6, 6)
        # Upconvs layers
        net_barrel = self._upconvs(net_barrel)[:,:,2:-3,1:-1]
        net_barrel = net_barrel.reshape(-1, 3, 51*150)
        net_top = self._upconvs_top(net_top).view(-1, 3, 48*48)
        net_bottom = self._upconvs_bottom(net_bottom).view(-1, 3, 48*48)
        
        return [net_barrel, net_bottom, net_top]

torch.set_printoptions(precision=10)

net = CRinGeNet().cpu()
net.load_state_dict(torch.load("testAllCRinGe_Gaus_SK_CapBar.cnn", map_location=lambda storage, loc: storage))

torch.set_grad_enabled(False)
net.eval()

def lossFunction_Gaus_Charge(pred, label):
    '''
       One-Gaussian Charge Only loss
    '''
    label = torch.as_tensor(label).type(torch.FloatTensor).cuda()

    logvar = torch.as_tensor(pred[0].cpu().detach().numpy()).type(torch.FloatTensor).cuda()
    logmu = torch.as_tensor(pred[1].cpu().detach().numpy()).type(torch.FloatTensor).cuda()
    punhit = torch.as_tensor(pred[2].cpu().detach().numpy()).type(torch.FloatTensor).cuda()

    var = torch.exp(logvar)
    mu = torch.exp(logmu)
    unhitMask = (label == 0)

    unhitTarget = torch.as_tensor(unhitMask).type(torch.FloatTensor).cuda()
    fracUnhit = unhitTarget.sum()/unhitTarget.numel()

    loss = torch.nn.BCEWithLogitsLoss()(punhit, unhitTarget)
    loss += (1-fracUnhit)*(1/2.)*(logvar[~unhitMask] + (label[~unhitMask]-mu[~unhitMask])**2/var[~unhitMask]).mean()
    loss += (1-fracUnhit)*(1/2.)*np.log(2*np.pi)
  
    return loss

def _loss(prediction, label, mask):
    '''
       Calculation of loss function for a given event
    '''
    dim_bar = label[0].shape 
    dim_top = label[2].shape
    dim_bot = label[1].shape
    label_bar = label[0].reshape( dim_bar[1]*dim_bar[0])
    label_top = label[2].reshape( dim_top[1]*dim_top[0])
    label_bot = label[1].reshape( dim_bot[1]*dim_bot[0])
 
    dim_top = mask[1].shape
    dim_bot = mask[0].shape
    mask_top = torch.as_tensor(mask[1].reshape(-1,dim_top[0]*dim_top[1])).type(torch.FloatTensor).cpu()
    mask_bot = torch.as_tensor(mask[0].reshape(-1,dim_bot[0]*dim_bot[1])).type(torch.FloatTensor).cpu()

    prediction_top = prediction[2]
    prediction_bot = prediction[1]
    prediction_bar = prediction[0]

    pred_top = prediction_top[0, :] * mask_top
    pred_bot = prediction_bot[0, :] * mask_bot
    pred_bar = prediction_bar[0]                # pred_bar should be a 2-d array  

    loss_bar = lossFunction_Gaus_Charge(pred_bar,label_bar)
    loss_top = lossFunction_Gaus_Charge(pred_top,label_top)
    loss_bot = lossFunction_Gaus_Charge(pred_bot,label_bot)

    return (loss_bar + loss_top + loss_bot)

def min_scan_loss(data, label, mask):
    '''
      energy scan for a given event.
      Return the list of the scanned energies along with the one of the corresopnding losses, true mc energy and reconstructed energy
    '''
    Loss = []

    data = np.expand_dims(data, axis=0).tolist()
    origE = data[0][9]
    lowerbound = origE * 0.2
    uperbound = origE * 1.8    
    energyList = np.linspace(lowerbound, uperbound, 300).tolist()

    data = torch.as_tensor(data).cpu()
    orig_pred = net(data)
    orig_loss = _loss(orig_pred, label, mask).cpu().detach().numpy() 

    for i in energyList:
        data[0][9] = i
        data = torch.as_tensor(data).cpu()
        prediction = net(data)
        loss = _loss(prediction, label, mask).cpu().detach().numpy()

        Loss.append(loss)

    min_loss = np.amin(Loss)
    min_loss_idx = np.where(Loss == min_loss)[0][0]
    min_loss_energy = energyList[min_loss_idx]
    return Loss, energyList, origE, min_loss_energy

#function to get the z position of the projected track of the incoming particle
def getZProj(vertex,direction):
    '''
       the z-compoent of the position at which the event vertex is projected on the tank 
       along  particle dirction
    '''
    a = direction[0]*direction[0] + direction[1]*direction[1]
    b = vertex[0]*direction[0] + vertex[1]*direction[1]
    c = vertex[0]*vertex[0] + vertex[1]*vertex[1] - (33.6815*100/2)**2

    t = (math.sqrt(b*b - a*c) - b)/a
    if t < 0:
        t = (-math.sqrt(b*b - a*c) - b)/a
        if t < 0:
            raise ValueError
    z =  vertex[2] + t*direction[2]
    return z

def getDWall(vertex):
    '''
       shortest distance between event vertex and the tank wall
    ''' 
    x = vertex[0]
    y = vertex[1]
    z = vertex[2]
    
    R = 1690.
    H_half = 1810.
 
    r_vtx =  math.sqrt(x*x + y*y)
    z_abs = abs(z)

    sign = 1.
    if r_vtx > R:
        sign = -1.
    if z_abs > H_half:
        sign = -1.

    disz = abs(H_half - z_abs)
    disr = abs(R - r_vtx)
   
    dWall = sign * min(disz, disr)
    return dWall

# function to provide data and label lists 
import h5py
def event_loader_nocuts(filename):
    file_handle = h5py.File(filename, mode='r')
    eventNum = file_handle['event_data'].shape[0]
  
    mask = [file_handle['mask'][0], file_handle['mask'][1]]
    datas=[]
    labels=[]
    scan_size = 10
    for i in range(eventNum):
        if i >= scan_size:
            break
 
        p_position = file_handle['positions'][i][0] 
        p_direction = file_handle['directions'][i][0]
        p_energy = file_handle['energies'][i]        
        p_pid = file_handle['pids'][i]
        
        if p_pid == 13:
            pid_vec = [0,0,1]
        else:
            pid_vec = [0,1,0]
        
        data_info = np.hstack((pid_vec,p_position,p_direction,p_energy))
        datas.append(data_info)

        barrel_label = file_handle['event_data'][i][:,:,0]
        top_label = file_handle['event_data_top'][i][:,:,0]
        bot_label = file_handle['event_data_bottom'][i][:,:,0]
        label_info = [barrel_label,bot_label,top_label]
        labels.append(label_info)
        
    file_handle.close() 
    return datas, labels, mask   

def event_loader(filename):
    '''
       Prepare the event data list for the calucation of loss funtion 
    '''
    energyCut_h = 2000.
    energyCut_l = 1500.
    wallCut = 500.

    file_handle = h5py.File(filename, mode='r')
    eventNum = file_handle['event_data'].shape[0]
  
    mask = [file_handle['mask'][0], file_handle['mask'][1]]

    ids=[]
    for i in range(eventNum):
        if file_handle['energies'][i] < energyCut_l:
            continue
        if file_handle['energies'][i] > energyCut_h:
            continue

        wall = getDWall(file_handle['positions'][i][0])
        if wall < wallCut:
            continue

        ids.append(i)
#for a full scan comment the following two lines out             
#        if len(ids) > 100: 
#           break
  
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
        
        if p_pid == 13:
            pid_vec = [0,0,1]
        else:
            pid_vec = [0,1,0]
        
        data_info = np.hstack((pid_vec,p_position,p_direction,p_energy))
        datas.append(data_info)

        barrel_label = file_handle['event_data'][ID][:,:,0]
        top_label = file_handle['event_data_top'][ID][:,:,0]
        bot_label = file_handle['event_data_bottom'][ID][:,:,0]
        label_info = [barrel_label,bot_label,top_label]
        labels.append(label_info)
        j = j+1
        
    file_handle.close() 
    return ids, datas, labels, mask   

############################################################################
def plot_reconstr(filename, ID):
    file_handle = h5py.File(filename, mode='r')

    mask = [file_handle['mask'][0], file_handle['mask'][1]]

    p_position = file_handle['positions'][ID][0] 
    p_direction = file_handle['directions'][ID][0]
    p_energy = file_handle['energies'][ID]        
    p_pid = file_handle['pids'][ID]        
    if p_pid == 13:
        pid_vec = [0,0,1]
        particle = 'Mu'
    else:
        pid_vec = [0,1,0]
        particle = 'E'

    data_info = np.hstack((pid_vec,p_position,p_direction,p_energy))

    barrel_label = file_handle['event_data'][ID][:,:,0]
    top_label = file_handle['event_data_top'][ID][:,:,0]
    bot_label = file_handle['event_data_bottom'][ID][:,:,0]
    label_info = [barrel_label,bot_label,top_label]

    losses, energies, E_org, E_minLoss = min_scan_loss(data_info,label_info,mask)

    data_info = np.expand_dims(data_info, axis=0).tolist()
    data_info[0][9] = E_minLoss
    data_info = torch.as_tensor(data_info).cpu()
    prediction = net(data_info)

    prediction_top = prediction[2]
    prediction_bot = prediction[1]
    prediction_bar = prediction[0]

    dim_top = mask[1].shape
    dim_bot = mask[0].shape
    mask_top = mask[1].reshape(-1,dim_top[0]*dim_top[1])
    mask_bot = mask[0].reshape(-1,dim_bot[0]*dim_bot[1])

    fig = plt.figure(figsize = (34, 20))  
    fig.suptitle('event ID: {}'.format(ID))
    a = fig.add_subplot(3,4,1)
    var_top = np.exp(prediction_top.cpu().detach().numpy()[0,0] * mask_top).reshape((48,48))
    imgplot = plt.imshow(var_top,origin='lower',vmin=0,vmax=25)
    a.set_title('min_var_top')

    a = fig.add_subplot(3,4,2)
    mu_top = np.exp(prediction_top.cpu().detach().numpy()[0,1] * mask_top).reshape((48,48))
    imgplot = plt.imshow(mu_top,origin='lower',vmin=0,vmax=5)
    a.set_title('min_mu_top')

    a = fig.add_subplot(3,4,3)
    hitProb_top = (1./(1 + np.exp(prediction_top.cpu().detach().numpy()[0,2])) * mask_top).reshape((48,48))
    imgplot = plt.imshow(hitProb_top,origin='lower',vmin=0,vmax=1)
    a.set_title('min_hitProb_top')

    a = fig.add_subplot(3,4,4)
    imgplot = plt.imshow(top_label,origin='lower',vmin=0,vmax=5)
    a.set_title('original')    

    a = fig.add_subplot(3,4,5)
    var_bar = np.exp(prediction_bar.cpu().detach().numpy()[0,0]).reshape(51,150)
    imgplot = plt.imshow(var_bar,origin='lower',vmin=0,vmax=25)
    a.set_title('min_var_bar')

    a = fig.add_subplot(3,4,6)
    mu_bar = np.exp(prediction_bar.cpu().detach().numpy()[0,1]).reshape(51,150)
    imgplot = plt.imshow(mu_bar,origin='lower',vmin=0,vmax=5)
    a.set_title('min_mu_bar')
 
    a = fig.add_subplot(3,4,7)
    hitProb_bar = (1./(1+np.exp(prediction_bar.cpu().detach().numpy()[0,2]))).reshape(51,150)
    imgplot = plt.imshow(hitProb_bar,origin='lower',vmin=0,vmax=1)
    a.set_title('min_hitProb_bar')
 
    a = fig.add_subplot(3,4,8)
    imgplot = plt.imshow(barrel_label,origin='lower',vmin=0,vmax=5)
    a.set_title('original')

    a = fig.add_subplot(3,4,9)
    var_bot = np.exp(prediction_bot.cpu().detach().numpy()[0,0] * mask_bot).reshape((48,48))
    imgplot = plt.imshow(var_bot,origin='lower',vmin=0,vmax=25)
    a.set_title('min_var_bot')

    a = fig.add_subplot(3,4,10)
    mu_bot = np.exp(prediction_bot.cpu().detach().numpy()[0,1] * mask_bot).reshape((48,48))
    imgplot = plt.imshow(mu_bot,origin='lower',vmin=0,vmax=5)
    a.set_title('min_mu_bot')

    a = fig.add_subplot(3,4,11)
    hitProb_bot = (1./(1 + np.exp(prediction_bot.cpu().detach().numpy()[0,2])) * mask_bot).reshape((48,48))
    imgplot = plt.imshow(hitProb_bot,origin='lower',vmin=0,vmax=1)
    a.set_title('min_hitProb_bot')

    a = fig.add_subplot(3,4,12)
    imgplot = plt.imshow(bot_label,origin='lower',vmin=0,vmax=5)
    a.set_title('original')    
    
    fig.savefig("Figure_{}_evt{}.png".format(particle, ID))
############################################################################
# Functions for analysis
def makeScanningPlots(h5file):
    import random 

    #h5file="/storage/shared/mojia/trainingSamples/WCSim_e-_npztoh5_test.h5"

    eventIDs, datas, labels, mask = event_loader(h5file)

    print(len(datas))
 
    fig = plt.figure(figsize=(24,12))

    indexs = []
    indexs.append(random.randint(0,len(datas)))
    while len(indexs) < 10:
        ii = random.randint(0,len(datas))
        if ii in indexs:
            continue
        indexs.append(ii)

    print("Randomly chosen {} events".format(len(indexs)))

    j=0
    for i in indexs:
        print(eventIDs[i])
    
        plot_reconstr(h5file, eventIDs[i])
    
        losses, energies, E_org, E_minLoss = min_scan_loss(datas[i],labels[i],mask)
        print("eventID:", eventIDs[i], "--Ture Energy: ", E_org, "--Min Loss Energy: ", E_minLoss)
        a = fig.add_subplot(2,5,j+1)
        j = j+1
        a.plot(energies, losses)
        a.axvline(x = E_org, label = 'true E: {:.2f}'.format(E_org), color='r',linestyle='--')
        a.axvline(x = E_minLoss, label ='min loss E: {:.2f}'.format(E_minLoss),color='b',linestyle='--')
        a.set_xlabel("energy")
        a.set_ylabel("Loss")
        a.set_title('Event ID:{}'.format(eventIDs[i]))
        a.legend()

    fig.tight_layout()
    fig.savefig("E_scanningSamples_Dwall_E1500-2000.png")

def makeScatterPlot(h5file):
    eventIDs, datas, labels, mask = event_loader(h5file)

    print(len(datas))
    trueEs = []
    recoEs = []
    f_e = open("/storage/shared/mojia/reconE/trueE_reconE_varied.txt", "wb")

    for i in range(len(datas)):
        print(eventIDs[i])

        losses, energies, E_org, E_minLoss = min_scan_loss(datas[i],labels[i],mask)
        print("eventID:", eventIDs[i], "--Ture Energy: ", E_org, "--Min Loss Energy: ", E_minLoss)
        trueEs.append(E_org)
        recoEs.append(E_minLoss)
        E_data = np.column_stack([eventIDs[i], E_org, E_minLoss, (E_minLoss-E_org)/E_org])
        np.savetxt(f_e, E_data, fmt='%s')

    f_e.close()

    fig = plt.figure(figsize=(12, 12))
    ax=plt.axes()
    ax.scatter(trueEs, recoEs)
    ax.plot([0,500,1000,1500,2000],[0,500,1000,1500,2000],linestyle='dashed',color='red')
    ax.set_xlabel('True E')
    ax.set_ylabel('Reconstructed E')

    fig.savefig("sk_true_recon_E_mu-.png")

def makeHisto(h5file):
    eventIDs, datas, labels, mask = event_loader(h5file)

    print(len(datas))
    recoEs = []
    f_e = open("/storage/shared/mojia/reconE/trueE_reconE_fixed.txt","wb")
    for i in range(len(datas)):
        print(eventIDs[i])

        losses, energies, E_org, E_minLoss = min_scan_loss(datas[i],labels[i],mask)
        print("eventID:", eventIDs[i], "--Ture Energy: ", E_org, "--Min Loss Energy: ", E_minLoss)
        recoEs.append(E_minLoss)
        E_data = np.column_stack([eventIDs[i], E_org, E_minLoss, (E_minLoss-E_org)/E_org])
        np.savetxt(f_e, E_data, fmt='%s')

    f_e.close()
    fig = plt.figure(figsize=(12, 12))
    ax=plt.axes()
    ax.hist(recoEs, bins= 50)

    ax.set_xlabel('Reconstructed E')
    ax.set_ylabel('event numbers')

    fig.savefig("sk_recon_EHist_mu-.png")

#################################################################################
# main function:
h5file="/storage/shared/mojia/trainingSamples/WCSim_mu-_npztoh5_test.h5"

#makeScanningPlots(h5file)
makeScatterPlot(h5file)
#makeHisto(h5file)
