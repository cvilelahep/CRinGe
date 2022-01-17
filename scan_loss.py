import argparse
import importlib
import os
from glob import glob
import h5py
import pickle
import random
import os

import torch
import numpy as np
import iotools
import scantools.scanlib as sclib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import LinearLocator
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm

from copy import copy

import scipy.stats as stats
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp2d
from scipy import optimize

def scan_loss(args) :

    # Set random seed
    if args.random_seed is not None :
        print("Setting random seed to {0}".args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)  # if you are using multi-GPU.
        torch.manual_seed(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        np.random.seed(args.random_seed)

    # Get and initialize model
  
    print("Loading model: "+args.model)
    
    # Collect model options in a dictionary
    model_args_dict = {}
    if len(args.model_arguments) :
        for model_arg in args.model_arguments :
            arg_split = model_arg.split(":")
            try :
                arg_value = float(arg_split[1])
                if arg_value.is_integer() :
                    arg_value = int(arg_value)
            except :
                arg_value = arg_split[1]
            model_args_dict[arg_split[0]] = arg_value

        print("With options: ", model_args_dict)
    
    # import model
    model_module = importlib.import_module("models."+args.model)
    
    # Initialize model
    network = model_module.model(**model_args_dict)

    # read cnn weight
    network.load_state_dict(torch.load(args.input_dir+"/"+args.model+".cnn", map_location=lambda storage, loc: storage))
    torch.set_grad_enabled(False)
    network.eval()
    
    # Initialize data loaders
    print("Data directory: "+args.data_dirs)

    mu_data=iotools.loader_factory('H5Dataset', batch_size=1, shuffle=args.shuffle_loader, num_workers=args.num_workers, pin_memory = True, data_dirs=args.data_dirs.split(","), flavour='mu-', start_fraction=args.begin_fraction, use_fraction=1.-args.begin_fraction, read_keys= ["positions","directions", "energies", "event_data_top", "event_data_bottom"])
    e_data =iotools.loader_factory('H5Dataset', batch_size=1, shuffle=args.shuffle_loader, num_workers=args.num_workers, pin_memory = True, data_dirs=args.data_dirs.split(","), flavour='e-', start_fraction=args.begin_fraction, use_fraction=1.-args.begin_fraction, read_keys= ["positions","directions", "energies", "event_data_top", "event_data_bottom"])

    # Grab end-cap masks from one of the input files
    with h5py.File(glob(args.data_dirs+"/*mu-*")[0], mode = "r") as f :
        network.top_mask = f['mask'][0]
        network.top_mask = network.top_mask.reshape(-1, network.top_mask.shape[0]*network.top_mask.shape[1])
        network.bottom_mask = f['mask'][1]
        network.bottom_mask = network.bottom_mask.reshape(-1, network.bottom_mask.shape[0]*network.bottom_mask.shape[1])
    
    # Create output directory
    try :
        os.makedirs(args.output_dir)
    except FileExistsError :
        pass

    # Save scan outputs and curves to file
    if args.plot_pdf:
        pdfout = PdfPages(args.output_dir+"/"+args.model+"_LLScan_test_curious_scans_"+str(args.num_scan)+"_events.pdf")
        rect = patches.Rectangle((0, 48),150,51,linewidth=1, edgecolor='w', facecolor='none')
        circ_top = patches.Circle((75, 24),24, linewidth=1, edgecolor='w', facecolor='none')
        circ_bottom = patches.Circle((75, 123),24, linewidth=1, edgecolor='w', facecolor='none')
        
        xrand = random.random()
    
    fout = open(args.output_dir+"/"+args.model+"_LLScan_test_"+str(args.num_scan)+"_events.txt", "wb")
    fscan_mu = open(args.output_dir+"/"+args.model+"_mu_LLH_curves_"+str(args.num_scan)+"_events.txt","wb")
    fscan_e = open(args.output_dir+"/"+args.model+"_e_LLH_curves_"+str(args.num_scan)+"_events.txt","wb")

    # Scan loop
    event_used = 0
    for (data_mu, data_e) in zip(mu_data, e_data) :
        try:
            ######### muon event ############
            network.fillData(data_mu)
            network.fillLabel(data_mu)

            # flags of interpolation result: edge reco or local max instead of min
            mu_onbound = 0
            mu_local_max = 0

            position_mu = [network.data[0][3]*network.xy_scale,network.data[0][4]*network.xy_scale,network.data[0][5]*network.z_scale]
            wall_mu = sclib.computeDwall_(position_mu)
            towall_mu = sclib.computeTowall_(position_mu, network.data[0][6:9])
            label_stack_mu, pred_stack_mu, nhit_mu = sclib._stack_hit_event_display(network)
            energy_scanlist_mu, loss_scanlist_mu, orig_Emu, orig_Lossmu = sclib._scan_lossvE(network, args.flip_top_bottom)

            splELoss_mu = InterpolatedUnivariateSpline(energy_scanlist_mu, loss_scanlist_mu, k=4)       
            crptsELoss_mu = splELoss_mu.derivative().roots()
            
            if len(crptsELoss_mu) > 0:
                minLoss_mu = sclib.find_cubicspline_min(splELoss_mu, crptsELoss_mu) # energy of min loss for the muon event             
            else : # no local min or max
                minLoss_mu = energy_scanlist_mu[np.argmin(loss_scanlist_mu)]
                mu_onbound = 1
                    
            if splELoss_mu(minLoss_mu) > np.min(loss_scanlist_mu): # reco at local max
                mu_local_max = 1


            ############ save muon scan curve to pdf ################
            if args.plot_pdf and (xrand < 0.01 or mu_local_max == 1 or mu_onbound == 1):
                figScan_mu = plt.figure(figsize=(17,5))                
                mu_plot_dict = {"energy_scanlist": energy_scanlist_mu, "loss_scanlist": loss_scanlist_mu, "orig_E": orig_Emu, "orig_Loss": orig_Lossmu, "crptsELoss": crptsELoss_mu, "splELoss": splELoss_mu, "minLoss": minLoss_mu, "wall": wall_mu, "towall": towall_mu, "rect": rect, "circ_top": circ_top, "circ_bottom": circ_bottom, "figure": figScan_mu, "label_stack": label_stack_mu, "pred_stack": pred_stack_mu, "pdfout": pdfout}
                sclib._save_scan_curve("Muon", mu_plot_dict)
                plt.close(figScan_mu)
            
            pid_mu, energy_mu = sclib._scan_lossvPID(network, minLoss_mu, args.flip_top_bottom)

            ######### electron event ############
            network.fillData(data_e)
            network.fillLabel(data_e)

            e_onbound = 0
            e_local_max = 0
            label_stack_e, pred_stack_e, nhit_e = sclib._stack_hit_event_display(network)
            position_e = [network.data[0][3]*network.xy_scale,network.data[0][4]*network.xy_scale,network.data[0][5]*network.z_scale]
            wall_e = sclib.computeDwall_(position_e)
            towall_e = sclib.computeTowall_(position_e, network.data[0][6:9])

            energy_scanlist_e, loss_scanlist_e, orig_Ee, orig_Losse = sclib._scan_lossvE(network, args.flip_top_bottom)

            splELoss_e = InterpolatedUnivariateSpline(energy_scanlist_e, loss_scanlist_e, k=4) 
            crptsELoss_e = splELoss_e.derivative().roots()
                
            if len(crptsELoss_e) > 0:
                minLoss_e = sclib.find_cubicspline_min(splELoss_e, crptsELoss_e)
            else:
                minLoss_e = energy_scanlist_e[np.argmin(loss_scanlist_e)]
                e_onbound = 1
                
            if splELoss_e(minLoss_e) > np.min(loss_scanlist_e):
                e_local_max = 1

            ############ save electron scan curve to pdf ################
            if args.plot_pdf and (xrand < 0.01 or e_local_max == 1 or e_onbound == 1):
                figScan_e = plt.figure(figsize=(17,5))
                e_plot_dict = {"energy_scanlist": energy_scanlist_e, "loss_scanlist": loss_scanlist_e, "orig_E": orig_Ee, "orig_Loss": orig_Losse, "crptsELoss": crptsELoss_e, "splELoss": splELoss_e, "minLoss": minLoss_e, "wall": wall_e, "towall": towall_e, "rect": rect, "circ_top": circ_top, "circ_bottom": circ_bottom, "figure": figScan_e, "label_stack": label_stack_e, "pred_stack": pred_stack_e, "pdfout": pdfout}
                sclib._save_scan_curve("Electron", e_plot_dict)
                plt.close(figScan_e)


            pid_e, energy_e = sclib._scan_lossvPID(network, minLoss_e, args.flip_top_bottom)

            ######## write to output ##########            
            event_used += 1

            line_output = [event_used] + [orig_Emu, orig_Ee] + [minLoss_mu, minLoss_e] + [mu_onbound, e_onbound] + [wall_mu, wall_e] + [towall_mu, towall_e] + [pid_mu, pid_e] + [len(crptsELoss_mu), len(crptsELoss_e)] + [mu_local_max, e_local_max] + [nhit_mu, nhit_e]
            np.savetxt(fout, [line_output], fmt='%s')
            fout.flush()
            # typically the above line would do. however this is used to ensure that the file is written
            os.fsync(fout.fileno())

            np.savetxt(fscan_mu, [[event_used] + ["Muon scan energy"] + energy_scanlist_mu], fmt='%s')
            np.savetxt(fscan_mu, [[event_used] + ["Muon scan loss"] + loss_scanlist_mu], fmt='%s')
            np.savetxt(fscan_mu, [[event_used] + ["Muon true energy and loss"] +[orig_Emu, orig_Lossmu]], fmt='%s') 
            fscan_mu.flush()
            # typically the above line would do. however this is used to ensure that the file is written
            os.fsync(fscan_mu.fileno())

            np.savetxt(fscan_e, [[event_used] + ["Electron scan energy"] + energy_scanlist_e], fmt='%s')
            np.savetxt(fscan_e, [[event_used] + ["Electron scan loss"] + loss_scanlist_e], fmt='%s')
            np.savetxt(fscan_e, [[event_used] + ["Electron true energy and loss"] +[orig_Ee, orig_Losse]], fmt='%s') 
            fscan_e.flush()
            # typically the above line would do. however this is used to ensure that the file is written
            os.fsync(fscan_e.fileno())

        ##end of 2 events##
            if event_used >= args.num_scan :
                break

        # save output even keyboard interrupted    
        except KeyboardInterrupt:
            break
        
    print("Scan done")

    if args.plot_pdf is True:
        pdfout.close()

    fscan_mu.close()
    fscan_e.close()
    fout.close()

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Application to scan Water Cherenkov generative neural network loss function.')
    parser.add_argument('-b', '--begin_fraction', type = float, help = "Starting fraction of data loading", default = 0.0, required = False)
    parser.add_argument('-i', '--input_dir', type = str, help = "Input directory of cnn weights", default = "./", required = False)
    parser.add_argument('-f', '--flip_top_bottom', type = bool, help = "Flip top and bottom caps (to deal with bug in training)", default = False, required = False)
    parser.add_argument('-j', '--num_workers', type = int, help = "Number of CPUs for loading data", default = 8, required = False)
    parser.add_argument('-n', '--num_scan', type = int, help = "Number of events to scan", default = 2000, required = False)
    parser.add_argument('-o', '--output_dir', type = str, help = "Output directory", default = "./", required = False)
    parser.add_argument('-p', '--plot_pdf', type = bool, help = "Save scan curve to pdf", default = False, required = False)
    parser.add_argument('-r', '--random_seed', type = int, help = "Random Seed", default = None, required = False)
    parser.add_argument('-s', '--shuffle_loader', type = bool, help = "Shuffle data to load", default = False, required = False)
    parser.add_argument('data_dirs', type = str, help = "Directory with training data")
    parser.add_argument('model', type = str, help = "Name of model to train")
    parser.add_argument('model_arguments', type = str, help = "Arguments to pass to model, in format \"name1:value1 name2:value2 ...\"", nargs = "*", default = "")

    args = parser.parse_args()
    
    print(args)
    
    scan_loss(args)

