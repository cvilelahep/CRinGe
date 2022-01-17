import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import LinearLocator
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm

from copy import copy

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

def _scan_lossvE(net, flip_tb):
    data = torch.as_tensor(net.data, dtype=torch.float, device=net.device)
    if flip_tb:
        net.charge_top, net.charge_bottom = net.charge_bottom, net.charge_top
        if net.use_time :
            net.time_top, net.time_bottom = net.time_bottom, net.time_top

    origE = data[0][9].item()*net.energy_scale
    orig_loss = net.evaluate(False)['loss']
    
    loss = []
    energy = []
    eSpace = np.linspace(0.2*origE, 1.8*origE, 300).tolist()
    for iE in eSpace:
        net.data[0][9] = float(iE/net.energy_scale)
        energy.append(iE)
        loss.append(net.evaluate(False)['loss'])

    return energy, loss, origE, orig_loss

def _scan_lossvPID(net, energy, flip_tb):
    data = torch.as_tensor(net.data, dtype=torch.float, device=net.device)
    net.data[0][9] = energy/net.energy_scale
    if flip_tb:
        net.charge_top, net.charge_bottom = net.charge_bottom, net.charge_top
        if net.use_time :
            net.time_top, net.time_bottom = net.time_bottom, net.time_top

    net.data[0][1] = 1
    net.data[0][2] = 0
    LossE =net.evaluate(False)['loss']
    net.data[0][1] = 0
    net.data[0][2] = 1
    LossMu =net.evaluate(False)['loss']

    LossPID = LossE - LossMu
    return LossPID, energy

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

def _stack_hit_event_display(net, flip_tb):
    label_top = net.charge_top.detach().cpu().numpy().reshape(48,48)*net.charge_scale
    label_bottom = net.charge_bottom.detach().cpu().numpy().reshape(48,48)*net.charge_scale
    label_barrel = net.charge_barrel.detach().cpu().numpy().reshape(51,150)*net.charge_scale
    
    data = torch.as_tensor(net.data, dtype=torch.float, device=net.device)
    pred_barrel, pred_bottom, pred_top = net(data)
    
    unhit_top = (1/(1+torch.exp(pred_top[:, 0]).detach().cpu().numpy())*net.top_mask).reshape(48,48)
    unhit_bottom = (1/(1+torch.exp(pred_bottom[:, 0]).detach().cpu().numpy())*net.bottom_mask).reshape(48,48)
    unhit_barrel = (1/(1+torch.exp(pred_barrel[:, 0]).detach().cpu().numpy())).reshape(51,150)

    if flip_tb:
        unhit_top, unhit_bottom = unhit_bottom, unhit_top

    label_barrel=np.flipud(label_barrel) # the 1d array starts from bottom?
    label_bottom=np.flipud(label_bottom) 
    
    unhit_barrel = np.flipud(unhit_barrel)
    unhit_bottom = np.flipud(unhit_bottom)
    
    dim_barrel = label_barrel.shape
    dim_cap = label_top.shape #(row, column)
    #make a new array including all 3 regions in a rectangular
    
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


def _save_scan_curve(flavor, plot_dict):
    
    loss_range = np.array(plot_dict['loss_scanlist']).max() - np.array(plot_dict['loss_scanlist']).min()    
    rect = copy(plot_dict['rect'])
    circ_top = copy(plot_dict['circ_top'])
    circ_bottom = copy(plot_dict['circ_bottom'])
    rect_cp = copy(rect)
    circt = copy(circ_top)
    circb = copy(circ_bottom)
    
    axscan = plot_dict['figure'].add_subplot(131)
    axevent = plot_dict['figure'].add_subplot(132)
    axhit = plot_dict['figure'].add_subplot(133)

    disp = axevent.imshow(plot_dict['label_stack'], vmin=0, vmax=np.max(plot_dict['label_stack'])+1)
    axevent.set_axis_off()
    axevent.add_patch(rect)
    axevent.add_patch(circ_top)
    axevent.add_patch(circ_bottom)
    plt.colorbar(disp, ax=axevent)

    hit = axhit.imshow(plot['pred_stack'], vmin=0, vmax=1)
    axhit.set_axis_off()
    axhit.add_patch(rect_cp)
    axhit.add_patch(circt)
    axhit.add_patch(circb)
    plt.colorbar(hit, ax=axhit)

    axscan.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    axscan.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

    axscan.plot(plot_dict['energy_scanlist'], plot_dict['loss_scanlist'], color="blue", alpha=0.75)
    axscan.scatter(plot_dict['orig_E'], plot_dict['orig_Loss'], color="red", label="Truth")
    axscan.scatter(plot_dict['crptsELoss'], plot_dict['splELoss_mu'](plot_dict['crptsELoss']), color = "orange", s=10, label="Local min/max")
    axscan.scatter(plot_dict['minLoss'], plot_dict['splELoss'](plot_dict['minLoss']), color="violet", label="Reco", marker="^", s=30)
    axscan.text(0.1, 0.95, "%s: E=%.2f MeV \nWall=%.2f cm | Towall=%.2f cm" % (flavor, plot_dict['orig_E'], plot_dict['wall'], plot_dict['towall']), verticalalignment = 'top', horizontalalignment='left', transform=axscan.transAxes, color='black', fontsize=7, bbox={'facecolor': 'white', 'alpha': 1., 'pad': 10})
    axscan.set_ylim([np.array(plot_dict['loss_scanlist']).min()-0.05*loss_range, np.array(plot_dict['loss_scanlist']).max()+0.05*loss_range])
    axscan.set_ylabel("Loss")
    axscan.set_xlabel("Energy [MeV]")
    axscan.legend(loc='upper right', framealpha=0)

    plot_dict['pdfout'].savefig(plot_dict['figure'])

    axevent.cla()
    axhit.cla()
    axscan.cla()
    
