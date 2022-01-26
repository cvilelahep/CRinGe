import numpy as np
import argparse
import importlib
import os
import random
import sys

from cycler import cycler
import timeit
import scantools.scanlib as sclib

def plot_scan_output(args):

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

    #create a big nested dictionary
    all_in_one = {}
    keys = ['ID', 'orig_energy', 'reco_energy', 'onbound', 'dwall', 'towall', 'pid', 'nmin', 'not_local_min', 'nhit', 'energy_res']
    for ig in range(args.npeak_max):
        all_in_one['NPeak{:d}'.format(ig+1)] = {}
        all_in_one['NPeak{:d}'.format(ig+1)]['muon'] = {key: [] for key in keys}
        all_in_one['NPeak{:d}'.format(ig+1)]['electron'] = {key: [] for key in keys}

        with open(args.input_dir+'_'+str(ig+1)+'/'+args.model+'_LLScan_test_'+str(args.n_scan)+'_events.txt') as file:
            lines = filter(None, (line.rstrip('\n').split() for line in file))
            for line in lines:
                all_in_one['NPeak{:d}'.format(ig+1)]['muon']['ID'].append(float(line[0]))
                all_in_one['NPeak{:d}'.format(ig+1)]['electron']['ID'].append(float(line[0]))
                for ik, key in enumerate(keys[1:-1]):
                    all_in_one['NPeak{:d}'.format(ig+1)]['muon'][key].append(float(line[ik*2+1]))
                    all_in_one['NPeak{:d}'.format(ig+1)]['electron'][key].append(float(line[(ik+1)*2]))
                all_in_one['NPeak{:d}'.format(ig+1)]['muon']['energy_res'].append((float(line[3])-float(line[1]))/float(line[1]))
                all_in_one['NPeak{:d}'.format(ig+1)]['electron']['energy_res'].append((float(line[4])-float(line[2]))/float(line[2]))
        print('Size of the big dictionary {:d} peak is '.format(ig+1), sys.getsizeof(all_in_one['NPeak{:d}'.format(ig+1)]))

    sclib._plot_2D_heatmap(args.output_dir, args.npeak_max, all_in_one, args.n_scan_to_use, args.use_time, args.use_corr)
    sclib._plot_npeak_comparison(args.output_dir, args.npeak_max, all_in_one, args.n_scan_to_use, args.use_time, args.use_corr)
    

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Application to scan Water Cherenkov generative neural network loss function.')
    parser.add_argument('-c', '--use_corr', type = bool, help = "Correlate PMT charge and timing", default = False, required = False)
    parser.add_argument('-i', '--input_dir', type = str, help = "Input directory", default = "./", required = False)
    parser.add_argument('-l', '--loss_scale', type = float, help = "Scaling factor of loss to regularize PID plot range", default = 1.e7, required = False)
    parser.add_argument('-m', '--model', type = str, help = "Name of model", default = 'CRinGe_SK_MultiGaus', required = False)
    parser.add_argument('-n', '--n_scan', type = int, help = "Number of scans to include", default = 10000, required = False)
    parser.add_argument('-o', '--output_dir', type = str, help = "Output directory", default = "./", required = False)
    parser.add_argument('-t', '--use_time', type = bool, help = "Using PMT timing", default = False, required = False)
    parser.add_argument('-x', '--npeak_max', type = int, help = "Max subcomponent number to include in the plots", default = 10, required = False)
    parser.add_argument('-y', '--n_scan_to_use', type = int, help = "Number of scan to use, must be less than n_scan", default = 10000, required = False)

    args = parser.parse_args()
    print(args)

    plot_scan_output(args)
