import torch
import numpy as np
import string

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import LinearLocator
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm

from scipy.interpolate import griddata
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp2d
import scipy.stats as stats
from scipy import optimize

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
    eSpace = np.linspace(0.2*origE, 1.8*origE, 100).tolist()
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

    hit = axhit.imshow(plot_dict['pred_stack'], vmin=0, vmax=1)
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
    

def _plot_npeak_comparison(outdir, npeak_total, info_dict, n_scan, tflag, cflag) :
    #info_dict = {npeak{flavor{keys}}}
    fig, (axmu, axe) = plt.subplots(2,2)
    fig.set_size_inches(8,7)

    wallcut = [0, 200, 500, 1700]
    colorid = ['red','green','blue']
    markers = ['o', '+', 's']

    flavor = ['muon', 'electron']
    residual = [[],[]]
    pid = [[],[]]
    wall = [[],[]]
    towall = [[],[]]
    onbound = [[],[]]
    not_min = [[],[]]

    ngaus = [ i+1 for i in range(npeak_total)]
    
    for ig in range(npeak_total):
        for j,f in enumerate(flavor):
            residual[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['energy_res'][0:n_scan])
            pid[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['pid'][0:n_scan])
            wall[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['dwall'][0:n_scan])
            towall[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['towall'][0:n_scan]) 
            onbound[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['onbound'][0:n_scan])
            not_min[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['not_local_min'][0:n_scan])
            
    mask = np.array((np.array(onbound)==0)&(np.array(not_min)==0), dtype=bool)

    #############E resolution###############
    for ip in range(1,4):
        axmu[0].errorbar(ngaus, np.nanmean(np.array(np.where((np.array(wall)[0] < wallcut[ip]) & (np.array(wall)[0] >= wallcut[ip-1]) & (mask[0]), np.array(residual)[0], np.nan)), axis = 1), yerr = 0.5*np.nanstd(np.array(np.where((np.array(wall)[0] < wallcut[ip]) & (np.array(wall)[0] >= wallcut[ip-1]) & (mask[0]), np.array(residual)[0], np.nan)), axis = 1), marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':', linewidth = 1, capsize=1, elinewidth=1, markeredgewidth=1,label="{:.1f} $\leq$ Dwall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))
        axmu[1].errorbar(ngaus, np.nanmean(np.array(np.where((np.array(towall)[0] < wallcut[ip]) & (np.array(towall)[0] >= wallcut[ip-1]) & (mask[0]), np.array(residual)[0], np.nan)), axis = 1), yerr = 0.5*np.nanstd(np.array(np.where((np.array(towall)[0] < wallcut[ip]) & (np.array(towall)[0] >= wallcut[ip-1]) & (mask[0]), np.array(residual)[0], np.nan)), axis = 1), marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':', linewidth = 1, capsize=1, elinewidth=1, markeredgewidth=1,label="{:.1f} $\leq$ Towall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))
        axe[0].errorbar(ngaus, np.nanmean(np.array(np.where((np.array(wall)[1] < wallcut[ip]) & (np.array(wall)[1] >= wallcut[ip-1]) & (mask[1]), np.array(residual)[1], np.nan)), axis = 1), yerr = 0.5*np.nanstd(np.array(np.where((np.array(wall)[1] < wallcut[ip]) & (np.array(wall)[1] >= wallcut[ip-1]) & (mask[1]), np.array(residual)[1], np.nan)), axis = 1), marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':', linewidth = 1, capsize=1, elinewidth=1, markeredgewidth=1,label="{:.1f} $\leq$ Dwall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))
        axe[1].errorbar(ngaus, np.nanmean(np.array(np.where((np.array(towall)[1] < wallcut[ip]) & (np.array(towall)[1] >= wallcut[ip-1]) & (mask[1]), np.array(residual)[1], np.nan)), axis = 1), yerr = 0.5*np.nanstd(np.array(np.where((np.array(towall)[1] < wallcut[ip]) & (np.array(towall)[1] >= wallcut[ip-1]) & (mask[1]), np.array(residual)[1], np.nan)), axis = 1), marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':', linewidth = 1, capsize=1, elinewidth=1, markeredgewidth=1,label="{:.1f} $\leq$ Towall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))        

    for iax in range(2):
        axmu[iax].set_ylim(-0.8, 0.6)
        axmu[iax].set_xlim(0, 10)
        axmu[iax].set_xlabel(r"$N_{Gaussian}$", fontsize=8, loc='right')
        axmu[iax].set_ylabel("Fractional Energy Residual", fontsize=8, loc='top')
        axmu[iax].tick_params(axis='x', labelsize=8)
        axmu[iax].tick_params(axis='y', labelsize=8)
        axmu[iax].axhline(y=0., color='grey', linewidth = 1, linestyle=':', alpha=0.5)
        axe[iax].set_ylim(-0.8, 0.6)
        axe[iax].set_xlim(0, 10)
        axe[iax].set_xlabel(r"$N_{Gaussian}$", fontsize=8, loc='right')
        axe[iax].set_ylabel("Fractional Energy Residual", fontsize=8, loc='top')
        axe[iax].tick_params(axis='x', labelsize=8)
        axe[iax].tick_params(axis='y', labelsize=8)        
        axe[iax].axhline(y=0., color='grey', linewidth = 1, linestyle=':', alpha=0.5)
        
        axmu[iax].legend(loc='upper left', prop={'size': 5})
        axe[iax].legend(loc='upper left', prop={'size': 5})

    axmu[0].set_title(r'(a) $\mu^-$ divided by Dwall', fontsize=10, y=1)
    axmu[1].set_title(r'(b) $\mu^-$ divided by Towall', fontsize=10, y=1)
    axe[0].set_title(r'(c) $e^-$ divided by Dwall', fontsize=10, y=1)
    axe[1].set_title(r'(d) $e^-$ divided by Towall', fontsize=10, y=1)
    
    fig.tight_layout(pad=0.8)
    pp = PdfPages(outdir+'/SK_MultiGaus_Ereso_vs_Walls_NGaus_1_to_'+str(npeak_total)+'_time_'+str(tflag)+'_corr_'+str(cflag)+'_'+str(n_scan)+'_events.pdf')
    pp.savefig(fig)
    pp.close()

    ################PID####################
    for iax in range(2):
        axmu[iax].cla()
        axe[iax].cla()
        axmu[iax].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        axe[iax].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    for ip in range(1,4):        
        axmu[0].plot(ngaus, np.sum(np.array(np.where((np.array(wall)[0] < wallcut[ip]) & (np.array(wall)[0] >= wallcut[ip-1]) & (mask[0]) & (np.array(pid)[0]<=0), 1, 0)), axis = 1)/(1e-10+np.sum(np.array(np.where((np.array(wall)[0] < wallcut[ip]) & (np.array(wall)[0] >= wallcut[ip-1]) & (mask[0]), 1, 0)), axis = 1)),  marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':', linewidth = 1, label="{:.1f} $\leq$ Dwall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))
        axmu[1].plot(ngaus, np.sum(np.array(np.where((np.array(towall)[0] < wallcut[ip]) & (np.array(towall)[0] >= wallcut[ip-1]) & (mask[0]) & (np.array(pid)[0]<=0), 1, 0)), axis = 1)/(1e-10+np.sum(np.array(np.where((np.array(towall)[0] < wallcut[ip]) & (np.array(towall)[0] >= wallcut[ip-1]) & (mask[0]), 1, 0)), axis = 1)),  marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':', linewidth = 1, label="{:.1f} $\leq$ Towall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))
        axe[0].plot(ngaus, np.sum(np.array(np.where((np.array(wall)[1] < wallcut[ip]) & (np.array(wall)[1] >= wallcut[ip-1]) & (mask[1]) & (np.array(pid)[1]>=0), 1, 0)), axis = 1)/(1e-10+np.sum(np.array(np.where((np.array(wall)[1] < wallcut[ip]) & (np.array(wall)[1] >= wallcut[ip-1]) & (mask[1]), 1, 0)), axis = 1)),  marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':', linewidth = 1, label="{:.1f} $\leq$ Dwall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))
        axe[1].plot(ngaus, np.sum(np.array(np.where((np.array(towall)[1] < wallcut[ip]) & (np.array(towall)[1] >= wallcut[ip-1]) & (mask[1]) & (np.array(pid)[1]>=0), 1, 0)), axis = 1)/(1e-10+np.sum(np.array(np.where((np.array(towall)[1] < wallcut[ip]) & (np.array(towall)[1] >= wallcut[ip-1]) & (mask[1]), 1, 0)), axis = 1)),  marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':', linewidth = 1, label="{:.1f} $\leq$ Towall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))

    for iax in range(2):
        axmu[iax].set_ylim(-0.01, 0.01)
        axmu[iax].set_xlim(0, 10)        
        axmu[iax].set_xlabel(r"$N_{Gaussian}$", fontsize=8, loc='right')
        axmu[iax].set_ylabel("Mis-PID Rate", fontsize=8, loc='top')
        axmu[iax].tick_params(axis='x', labelsize=8)
        axmu[iax].tick_params(axis='y', labelsize=8)
        axmu[iax].axhline(y=0., color='grey', linewidth = 1, linestyle=':', alpha=0.5)
        axe[iax].set_ylim(-0.01, 0.01)
        axe[iax].set_xlim(0, 10)
        axe[iax].set_xlabel(r"$N_{Gaussian}$", fontsize=8, loc='right')
        axe[iax].set_ylabel("Mis-PID Rate", fontsize=8, loc='top')
        axe[iax].tick_params(axis='x', labelsize=8)
        axe[iax].tick_params(axis='y', labelsize=8)
        axe[iax].axhline(y=0., color='grey', linewidth = 1, linestyle=':', alpha=0.5)
        
        axmu[iax].legend(loc='upper left', prop={'size': 5})
        axe[iax].legend(loc='upper left', prop={'size': 5})

    axmu[0].set_title(r'(a) $\mu^-$ divided by Dwall', y=1, fontsize=10)
    axmu[1].set_title(r'(b) $\mu^-$ divided by Towall', y=1, fontsize=10)
    axe[0].set_title(r'(c) $e^-$ divided by Dwall', y=1, fontsize=10)
    axe[1].set_title(r'(d) $e^-$ divided by Towall', y=1, fontsize=10)

    #fig.tight_layout(pad=0.8)
    pp = PdfPages(outdir+'/SK_MultiGaus_misPID_vs_Walls_NGaus_1_to_'+str(npeak_total)+'_time_'+str(tflag)+'_corr_'+str(cflag)+'_'+str(n_scan)+'_events.pdf')
    pp.savefig(fig)
    pp.close()
    fig.clf()

    #print useful info
    print('$N_{Gaussian}$ & \multicolumn{2}{c}{Failed reconstruction} & \multicolumn{2}{c}{No local min/max} & \multicolumn{2}{c}{Not a local min}\\\\ ')
    print(' & $\mu^-$ & $e^-$ & $\mu^-$ & $e^-$ & $\mu^-$ & $e^-$\\\\ ')
    print('\hline')
    for ig in range(npeak_total):
        print('{:d} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\ '.format(ig+1, 1 - np.sum(mask[0][ig])/n_scan, 1 - np.sum(mask[1][ig])/n_scan, np.sum(onbound[0][ig])/n_scan, np.sum(onbound[1][ig])/n_scan, np.sum(not_min[0][ig])/n_scan, np.sum(not_min[1][ig])/n_scan))
    print('\hline \hline')

    '''
    ##############Nmin################
    figNminAll, (axmu, axe) = plt.subplots(1,2)
    figNminAll.set_size_inches(8,3.5)
    axmu.set_prop_cycle(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    axe.set_prop_cycle(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    nminbins = [1,2,3,4,5,6,7,8,9,10,11]
    for ig in range(npeak_total):
        axmu.hist(np.clip(mu_nmin_all[ig][~np.isnan(mu_nmin_all[ig])], nminbins[0], nminbins[-1]), bins=nminbins, histtype='step', density = True, label=r'$\mu$ {:d} Gaussian'.format(ig+1))
        axe.hist(np.clip(e_nmin_all[ig][~np.isnan(e_nmin_all[ig])], nminbins[0], nminbins[-1]), bins=nminbins, histtype='step', density = True, label=r'e {:d} Gaussian'.format(ig+1))  
        axmu.set_ylabel("Event Rate")
        axe.set_ylabel("Event Rate")
        axmu.set_xlabel("Number of local min/max")
        axe.set_xlabel("Number of local min/max")
        axmu.set_ylim(0, 0.5)
        axmu.set_xlim(1, 11)
        axe.set_ylim(0, 0.5)
        axe.set_xlim(1, 11)
        axmu.legend(loc='upper right', ncol = 2, prop={'size': 5})
        axe.legend(loc='upper right', ncol = 2, prop={'size': 5})
        figNminAll.tight_layout(pad=0.3)
    pp = PdfPages(outdir+'/SK_MultiGaus_Nmin_NGaus_1_to_'+str(npeak_total)+'_time_'+str(tflag)+'_corr_'+str(cflag)+'_'+str(n_scan)+'_events.pdf')
    pp.savefig(figNminAll)
    plt.close(figNminAll)
    pp.close()   
    '''

def _plot_2D_heatmap(outdir, npeak_total, info_dict, n_scan, tflag, cflag) :

    fig =  plt.figure(figsize=(6*3,5*int(round(npeak_total/3))))
    ax = []
    npmt = 11146
    nlevel = 20

    #some constants and axis range
    nhitax = np.linspace(0, 1, 100)
    wallax = np.linspace(0, 2000, 200)
    towallax = np.linspace(0, 3900, 200)
    dwmesh, twmesh = np.meshgrid(wallax, towallax)

    flavor = ['muon', 'electron']
    residual = [[],[]]
    etrue=[[],[]]
    nhit=[[],[]]
    wall = [[],[]]
    towall = [[],[]]
    onbound = [[],[]]
    not_min = [[],[]]

    ngaus = [ i+1 for i in range(npeak_total)]

    for ig in range(npeak_total):
        for j,f in enumerate(flavor):
            residual[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['energy_res'][0:n_scan])
            etrue[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['orig_energy'][0:n_scan])
            nhit[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['nhit'][0:n_scan])
            wall[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['dwall'][0:n_scan])
            towall[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['towall'][0:n_scan]) 
            onbound[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['onbound'][0:n_scan])
            not_min[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['not_local_min'][0:n_scan])
            
    mask = np.array((np.array(onbound)==0)&(np.array(not_min)==0), dtype=bool)

    ####plot muon######
    for ig in range(npeak_total):
        ax.append(fig.add_subplot(3, int(round(npeak_total/3)), ig+1))
        ax[ig].set_xlabel('True Energy (MeV)', fontsize=8, loc='right')
        ax[ig].set_ylabel('True Nhit Fraction', fontsize=8, loc='top')
        ax[ig].tick_params(axis='x', labelsize=8)
        ax[ig].tick_params(axis='y', labelsize=8)
        #ax[ig].set_xlim(0, np.max(np.array(np.where(mask, np.array(etrue), np.nan)[0][ig])))
        ax[ig].set_xlim(0, np.max(np.array(etrue)[0][ig]))
        ax[ig].set_ylim(0, 1)
        energyax = np.linspace(0, np.max(np.array(etrue)[0][ig]), 100)
        Emesh, hitmesh = np.meshgrid(energyax, nhitax)
        Z = griddata((np.array(etrue)[0][ig], np.array(nhit)[0][ig]/npmt), np.abs(np.array(residual)[0][ig]), (Emesh, hitmesh), method='linear')
        heatmap = ax[ig].contourf(Emesh, hitmesh, Z, nlevel, vmin = -0.8, vmax = 0.8, cmap= 'magma')
        cbar=plt.colorbar(heatmap, ax=ax[ig])
        cbar.ax.set_ylabel(r'|$\Delta_{E}$|', rotation=270)
        ax[ig].set_title(r'({:s}) $\mu^-$ {:d} Gaussians'.format(string.ascii_lowercase[ig:ig+1], ig+1), y=1, fontsize=10)

    fig.tight_layout(pad=0.9)
    pp = PdfPages(outdir+'/SK_MultiGaus_mu_Etrue_v_nhit_Heatmap_NGaus_1_to_'+str(npeak_total)+'_time_'+str(tflag)+'_corr_'+str(cflag)+'_'+str(n_scan)+'_events.pdf')
    pp.savefig(fig)
    pp.close()

    for ig in range(npeak_total):
        ax[ig].cla()
        ax[ig].set_xlabel('Dwall (cm)', fontsize=8, loc='right')
        ax[ig].set_ylabel('Towall (cm)', fontsize=8, loc='top')
        ax[ig].tick_params(axis='x', labelsize=8)
        ax[ig].tick_params(axis='y', labelsize=8)
        #ax[ig].set_xlim(0, np.max(np.array(np.where(mask, np.array(etrue), np.nan)[0][ig])))
        ax[ig].set_xlim(0, 2000)
        ax[ig].set_ylim(0, 3900)
        Z = griddata((np.array(wall)[0][ig], np.array(towall)[0][ig]), np.abs(np.array(residual)[0][ig]), (dwmesh, twmesh), method='linear')
        heatmap = ax[ig].contourf(dwmesh, twmesh, Z, nlevel, vmin = -0.8, vmax = 0.8, cmap= 'magma')
        ax[ig].set_title(r'({:s}) $\mu^-$ {:d} Gaussians'.format(string.ascii_lowercase[ig:ig+1], ig+1), y=1, fontsize=10)

    pp = PdfPages(outdir+'/SK_MultiGaus_mu_wall_v_towall_Heatmap_NGaus_1_to_'+str(npeak_total)+'_time_'+str(tflag)+'_corr_'+str(cflag)+'_'+str(n_scan)+'_events.pdf')
    pp.savefig(fig)
    pp.close()

    ####plot electron######
    for ig in range(npeak_total):
        ax[ig].set_xlabel('True Energy (MeV)', fontsize=8, loc='right')
        ax[ig].set_ylabel('True Nhit Fraction', fontsize=8, loc='top')
        ax[ig].tick_params(axis='x', labelsize=8)
        ax[ig].tick_params(axis='y', labelsize=8)
        #ax[ig].set_xlim(0, np.max(np.array(np.where(mask, np.array(etrue), np.nan)[0][ig])))
        ax[ig].set_xlim(0, np.max(np.array(etrue)[1][ig]))
        ax[ig].set_ylim(0, 1)
        energyax = np.linspace(0, np.max(np.array(etrue)[1][ig]), 100)
        Emesh, hitmesh = np.meshgrid(energyax, nhitax)
        Z = griddata((np.array(etrue)[1][ig], np.array(nhit)[1][ig]/npmt), np.abs(np.array(residual)[1][ig]), (Emesh, hitmesh), method='linear')
        heatmap = ax[ig].contourf(Emesh, hitmesh, Z, nlevel, vmin = -0.8, vmax = 0.8, cmap= 'magma')
        ax[ig].set_title(r'({:s}) $e^-$ {:d} Gaussians'.format(string.ascii_lowercase[ig:ig+1], ig+1), y=1, fontsize=10)

    pp = PdfPages(outdir+'/SK_MultiGaus_e_Etrue_v_nhit_Heatmap_NGaus_1_to_'+str(npeak_total)+'_time_'+str(tflag)+'_corr_'+str(cflag)+'_'+str(n_scan)+'_events.pdf')
    pp.savefig(fig)
    pp.close()

    for ig in range(npeak_total):
        ax[ig].cla()
        ax[ig].set_xlabel('Dwall (cm)', fontsize=8, loc='right')
        ax[ig].set_ylabel('Towall (cm)', fontsize=8, loc='top')
        ax[ig].tick_params(axis='x', labelsize=8)
        ax[ig].tick_params(axis='y', labelsize=8)
        #ax[ig].set_xlim(0, np.max(np.array(np.where(mask, np.array(etrue), np.nan)[0][ig])))
        ax[ig].set_xlim(0, 2000)
        ax[ig].set_ylim(0, 3900)
        Z = griddata((np.array(wall)[1][ig], np.array(towall)[1][ig]), np.abs(np.array(residual)[1][ig]), (dwmesh, twmesh), method='linear')
        heatmap = ax[ig].contourf(dwmesh, twmesh, Z, nlevel, vmin = -0.8, vmax = 0.8, cmap= 'magma')
        #cbar=plt.colorbar(heatmap, ax=ax[ig])
        #cbar.ax.set_ylabel(r'|$\Delta_{E}$|', rotation=270)
        ax[ig].set_title(r'({:s}) $e^-$ {:d} Gaussians'.format(string.ascii_lowercase[ig:ig+1], ig+1), y=1, fontsize=10)

    pp = PdfPages(outdir+'/SK_MultiGaus_e_wall_v_towall_Heatmap_NGaus_1_to_'+str(npeak_total)+'_time_'+str(tflag)+'_corr_'+str(cflag)+'_'+str(n_scan)+'_events.pdf')
    pp.savefig(fig)
    pp.close()
    fig.clf()
