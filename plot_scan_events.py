import numpy as np
import argparse
import importlib
import os
import random
import sys

from cycler import cycler
import timeit
import scantools.scanlib as sclib
import iotools
import h5py
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import LinearLocator
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm

from copy import copy

def plot_scan_events(args):

    if args.npeak_max < 1:
        print("Need at least 1 peak for PMT PDFs, please modify your inputs.")
        sys.exit()

    if args.n_scan_to_use > args.n_scan:
        print("Using more scans than what is available, please modify your inputs.")
        sys.exit()
    
    try:
        os.makedirs(args.output_dir)
    except FileExistsError :
        pass

    if args.event_id is None:
        args.event_id = int(random.random()*args.n_scan_to_use)


    # Initialize data loaders                                                                                      
    print("Data directory: "+args.data_dirs)
    
    mu_data=iotools.loader_factory('H5Dataset', batch_size=1, shuffle=False, num_workers=1, pin_memory = True, data_dirs=args.data_dirs.split(","), flavour='mu-', start_fraction=args.begin_fraction, use_fraction=1.-args.begin_fraction, read_keys= ["positions","directions", "energies", "event_data_top", "event_data_bottom"])
    e_data =iotools.loader_factory('H5Dataset', batch_size=1, shuffle=False, num_workers=1, pin_memory = True, data_dirs=args.data_dirs.split(","), flavour='e-', start_fraction=args.begin_fraction, use_fraction=1.-args.begin_fraction, read_keys= ["positions","directions", "energies", "event_data_top", "event_data_bottom"])

    event_load = 0
    for (data_mu, data_e) in zip(mu_data, e_data) :
        event_load += 1
        if args.event_id != event_load :
            continue
        else:
            dim_barrel = data_mu[0].shape
            dim_cap = data_mu[5].shape

            charge_mu_barrel = data_mu[0][:,:,:,0].reshape(-1,dim_barrel[1]*dim_barrel[2])
            charge_mu_top = data_mu[5][:,:,:,0].reshape(-1, dim_cap[1]*dim_cap[2])
            charge_mu_bottom = data_mu[6][:,:,:,0].reshape(-1, dim_cap[1]*dim_cap[2])

            charge_e_barrel = data_e[0][:,:,:,0].reshape(-1,dim_barrel[1]*dim_barrel[2])
            charge_e_top = data_e[5][:,:,:,0].reshape(-1, dim_cap[1]*dim_cap[2])
            charge_e_bottom = data_e[6][:,:,:,0].reshape(-1, dim_cap[1]*dim_cap[2])

            if args.use_time:
                time_mu_barrel = data_mu[0][:,:,:,1].reshape(-1,dim_barrel[1]*dim_barrel[2])
                time_mu_top = data_mu[5][:,:,:,1].reshape(-1, dim_cap[1]*dim_cap[2])
                time_mu_bottom = data_mu[6][:,:,:,1].reshape(-1, dim_cap[1]*dim_cap[2])
            
                time_e_barrel = data_e[0][:,:,:,1].reshape(-1,dim_barrel[1]*dim_barrel[2])
                time_e_top = data_e[5][:,:,:,1].reshape(-1, dim_cap[1]*dim_cap[2])
                time_e_bottom = data_e[6][:,:,:,1].reshape(-1, dim_cap[1]*dim_cap[2])

            break
    
    # Grab end-cap masks from one of the input files                                                               
    with h5py.File(glob(args.data_dirs+"/*mu-*")[0], mode = "r") as f :
        top_mask = f['mask'][0]
        top_mask = top_mask.reshape(-1, top_mask.shape[0]*top_mask.shape[1])
        bottom_mask = f['mask'][1]
        bottom_mask = bottom_mask.reshape(-1, bottom_mask.shape[0]*bottom_mask.shape[1])


    fig, (axmu, axe) = plt.subplots(2,2)
    fig.set_size_inches(8, 5.5)

    mu_charge_stack = sclib._stack_event_display([charge_mu_barrel, charge_mu_bottom, charge_mu_top], [], [top_mask, bottom_mask], False)
    e_charge_stack = sclib._stack_event_display([charge_e_barrel, charge_e_bottom, charge_e_top], [], [top_mask, bottom_mask], False)

    rect = patches.Rectangle((0, 48),150,51,linewidth=1, edgecolor='b', facecolor='none')
    circ_top = patches.Circle((75, 24),24, linewidth=1, edgecolor='b', facecolor='none')
    circ_bottom = patches.Circle((75, 123),24, linewidth=1, edgecolor='b', facecolor='none')
    rect_cp = copy(rect)
    circt = copy(circ_top)
    circb = copy(circ_bottom)
    
    disp_mu = axmu[0].imshow(mu_charge_stack, vmin=0, vmax=np.nanmax(mu_charge_stack))
    axmu[0].set_axis_off()
    #axmu[0].add_patch(rect)
    #axmu[0].add_patch(circ_top)
    #axmu[0].add_patch(circ_bottom)
    cbar_mu = plt.colorbar(disp_mu, ax=axmu[0])
    cbar_mu.ax.set_ylabel('Log(Q [p.e.])', rotation=270, labelpad=12, fontsize=10)
    
    
    disp_e = axe[0].imshow(e_charge_stack, vmin=0, vmax=np.nanmax(e_charge_stack))
    axe[0].set_axis_off()
    #axe[0].add_patch(rect_cp)
    #axe[0].add_patch(circt)
    #axe[0].add_patch(circb)
    cbar_e = plt.colorbar(disp_e, ax=axe[0])
    cbar_e.ax.set_ylabel('Log(Q [p.e.])', rotation=270, labelpad=12, fontsize=10)
        
    #color cycle for the stack
    axmu[1].set_prop_cycle(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    axe[1].set_prop_cycle(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    axmu[1].set_ylim(-1.e-10, 1.e-10)
    axe[1].set_ylim(-1.e-10, 1.e-10)
    axmu[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axe[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axmu[1].tick_params(axis="x", labelsize=8)
    axmu[1].tick_params(axis="y", labelsize=8)
    axe[1].tick_params(axis="x", labelsize=8)
    axe[1].tick_params(axis="y", labelsize=8)    
    axmu[1].set_xlabel("Energy (MeV)")
    axmu[1].set_ylabel("Loss")
    axe[1].set_xlabel("Energy (MeV)")
    axe[1].set_ylabel("Loss")
    
    #create a big nested dictionary
    for ig in range(args.npeak_max):
        
        all_in_one = {}
        keys = ['ID', 'orig_energy', 'orig_loss', 'scan_energy', 'scan_loss']
        all_in_one['NPeak{:d}'.format(ig+1)] = {}
        all_in_one['NPeak{:d}'.format(ig+1)]['muon'] = {key: [] for key in keys}
        all_in_one['NPeak{:d}'.format(ig+1)]['electron'] = {key: [] for key in keys}

        with open(args.input_dir+'_'+str(ig+1)+'/'+args.model+'_mu_LLH_curves_'+str(args.n_scan)+'_events.txt') as file:
            curves = filter(None, (curve.rstrip('\n').split() for curve in file))
            c_id = 0
            for curve in curves:
                #old syntax
                if curve[0:3] == ['Muon', 'scan', 'energy']:
                    c_id += 1

                if c_id == args.event_id:
                    all_in_one['NPeak{:d}'.format(ig+1)]['muon']['ID'].append(c_id)
                    all_in_one['NPeak{:d}'.format(ig+1)]['muon']['scan_energy'].append(curve[3:])
                    curve = next(curves)
                    if curve[0:3] == ['Muon', 'scan', 'loss']:
                        all_in_one['NPeak{:d}'.format(ig+1)]['muon']['scan_loss'].append(curve[3:])
                    curve = next(curves)
                    if curve[0:5] == ['Muon', 'true', 'energy', 'and', 'loss']:
                        all_in_one['NPeak{:d}'.format(ig+1)]['muon']['orig_energy'].append(float(curve[-2]))
                        all_in_one['NPeak{:d}'.format(ig+1)]['muon']['orig_loss'].append(float(curve[-1]))

                    break
        
                '''
                if curve[0:4] == [str(c_id), 'Muon', 'scan', 'energy']:                
                    all_in_one['NPeak{:d}'.format(ig+1)]['muon']['scan_energy'].append(float(curve[4:]))
                elif curve[0:4] == [str(c_id), 'Muon', 'scan', 'loss']:
                    all_in_one['NPeak{:d}'.format(ig+1)]['muon']['scan_loss'].append(float(curve[4:]))
                elif curve[0:6] == [str(c_id), 'Muon', 'true', 'energy', 'and', 'loss']:
                    all_in_one['NPeak{:d}'.format(ig+1)]['muon']['orig_energy'].append(float(curve[-2]))
                    all_in_one['NPeak{:d}'.format(ig+1)]['muon']['orig_loss'].append(float(curve[-1]))                    
                c_id = len(all_in_one['NPeak{:d}'.format(ig+1)]['muon']['scan_loss'])+1
                '''
        with open(args.input_dir+'_'+str(ig+1)+'/'+args.model+'_e_LLH_curves_'+str(args.n_scan)+'_events.txt') as file:
            curves = filter(None, (curve.rstrip('\n').split() for curve in file))
            c_id = 0
            for curve in curves:
                #old syntax
                if curve[0:3] == ['Electron', 'scan', 'energy']:
                    c_id += 1

                if c_id == args.event_id:
                    all_in_one['NPeak{:d}'.format(ig+1)]['electron']['ID'].append(c_id)
                    all_in_one['NPeak{:d}'.format(ig+1)]['electron']['scan_energy'].append(curve[3:])
                    curve = next(curves)
                    if curve[0:3] == ['Electron', 'scan', 'loss']:
                        all_in_one['NPeak{:d}'.format(ig+1)]['electron']['scan_loss'].append(curve[3:])
                    curve = next(curves)
                    if curve[0:5] == ['Electron', 'true', 'energy', 'and', 'loss']:
                        all_in_one['NPeak{:d}'.format(ig+1)]['electron']['orig_energy'].append(float(curve[-2]))
                        all_in_one['NPeak{:d}'.format(ig+1)]['electron']['orig_loss'].append(float(curve[-1]))

                    break
        
                '''
                if curve[0:4] == [str(c_id), 'Muon', 'scan', 'energy']:                
                    all_in_one['NPeak{:d}'.format(ig+1)]['muon']['scan_energy'].append(float(curve[4:]))
                elif curve[0:4] == [str(c_id), 'Muon', 'scan', 'loss']:
                    all_in_one['NPeak{:d}'.format(ig+1)]['muon']['scan_loss'].append(float(curve[4:]))
                elif curve[0:6] == [str(c_id), 'Muon', 'true', 'energy', 'and', 'loss']:
                    all_in_one['NPeak{:d}'.format(ig+1)]['muon']['orig_energy'].append(float(curve[-2]))
                    all_in_one['NPeak{:d}'.format(ig+1)]['muon']['orig_loss'].append(float(curve[-1]))                    
                c_id = len(all_in_one['NPeak{:d}'.format(ig+1)]['muon']['scan_loss'])+1
                '''
        
        #print(all_in_one)
        #print('Size of the big dictionary {:d} peak is '.format(ig+1), sys.getsizeof(all_in_one['NPeak{:d}'.format(ig+1)]))
        sclib._stack_scan_curves(fig, axmu, all_in_one['NPeak{:d}'.format(ig+1)]['muon'], ig)
        sclib._stack_scan_curves(fig, axe, all_in_one['NPeak{:d}'.format(ig+1)]['electron'], ig)

    plt.figtext(0.5,0.98, r"(a) {:.2f} MeV $\mu^-$ event".format(all_in_one['NPeak{:d}'.format(ig+1)]['muon']['orig_energy'][0]), va="center", ha="center", size=10)
    plt.figtext(0.5,0.5, r"(b) {:.2f} MeV $e^-$ event".format(all_in_one['NPeak{:d}'.format(ig+1)]['electron']['orig_energy'][0]), va="center", ha="center", size=10)

    axmu[1].axvline(x=all_in_one['NPeak{:d}'.format(ig+1)]['muon']['orig_energy'][0], color='grey', linewidth = 1, linestyle='-', alpha=0.5, label = "True Energy")
    axmu[1].legend(loc='upper right', framealpha=1, facecolor='w', prop={'size': 5})
    axe[1].axvline(x=all_in_one['NPeak{:d}'.format(ig+1)]['electron']['orig_energy'][0], color='grey', linewidth = 1, linestyle='-', alpha=0.5, label = "True Energy")
    axe[1].legend(loc='upper right', framealpha=1, facecolor='w', prop={'size': 5})

    fig.tight_layout()
    pp = PdfPages(args.output_dir+'/SK_MultiGaus_scancurves_eventdisp_eventID_'+str(args.event_id)+'_NGaus_1_to_'+str(args.npeak_max)+'_time_'+str(args.use_time)+'_corr_'+str(args.use_corr)+'_'+str(args.n_scan_to_use)+'_events.pdf')
    pp.savefig(fig)
    pp.close()
    fig.clf()

#    sclib._event_display()
    

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Application to scan Water Cherenkov generative neural network loss function.')
    parser.add_argument('-b', '--begin_fraction', type = float, help = "Starting fraction of data loading", default = 0.0, required = False)
    parser.add_argument('-c', '--use_corr', type = bool, help = "Correlate PMT charge and timing", default = False, required = False)
    parser.add_argument('-e', '--event_id', type = int, help = "ID of event to plot", default = None, required = False)
    parser.add_argument('-i', '--input_dir', type = str, help = "Input directory", default = "./", required = False)
    parser.add_argument('-l', '--loss_scale', type = float, help = "Scaling factor of loss to regularize PID plot range", default = 1.e7, required = False)
    parser.add_argument('-m', '--model', type = str, help = "Name of model", default = 'CRinGe_SK_MultiGaus', required = False)
    parser.add_argument('-n', '--n_scan', type = int, help = "Number of scans to include", default = 10000, required = False)
    parser.add_argument('-o', '--output_dir', type = str, help = "Output directory", default = "./", required = False)
    parser.add_argument('-t', '--use_time', type = bool, help = "Using PMT timing", default = False, required = False)
    #parser.add_argument('-w', '--weight_file', type = str, help = "Input cnn weights", default = "./", required = False)
    parser.add_argument('-x', '--npeak_max', type = int, help = "Max subcomponent number to include in the plots", default = 10, required = False)
    parser.add_argument('-y', '--n_scan_to_use', type = int, help = "Number of scan to use, must be less than n_scan", default = 10000, required = False)
    parser.add_argument('data_dirs', type = str, help = "Directory with training data")

    args = parser.parse_args()
    print(args)

    plot_scan_events(args)
