import argparse
import importlib
from glob import glob

import numpy as np
import scipy.optimize
import h5py

import torch
import iotools

def function(x, model, PID) :

    E = np.linalg.norm(x[3:6])

    model.data = torch.FloatTensor(np.expand_dims(np.concatenate([PID,
                                                                  x[:3],
                                                                  x[3:6]/E,
                                                                  np.array([E])]), axis = 0), device = model.device)
    if model.use_time :
        t0 = x[6]
    else :
        t0 = 0.
    ret = model.evaluate(Train = False, t0 = t0)["loss"]
    return ret
        
def pre_fit(model, event) :
    # For now this is just a placeholder. Using truth information, pick a random point, reasonably close to the truth.
    if model.use_time :
        seed = np.concatenate([model.data[0,3:6], model.data[0,6:9]*model.data[0,9], [0]])
    else :
        seed = np.concatenate([model.data[0,3:6], model.data[0,6:9]*model.data[0,9]])
        
    print("TRUTH")
    print(seed)

    # Randomize truth for seed. To mimick a pre-fit algorithm.
    # For position use 5/sqrt(3) m sigma on each coordinate
    # For direction / energy: mutiply direction cosines by total energy, and then smear each component with 50%/sqrt(3).
    # For time use sigma = 15 ns

    sigma_space = 500. # cm
    sigma_time = 15. # ns
    sigma_E = 0.5 # Fractional

    if model.use_time :
        seed = np.concatenate([np.random.normal(loc = seed[0:3], scale = [sigma_space/model.xy_scale/3**0.5, sigma_space/model.xy_scale/3**0.5, sigma_space/model.z_scale/3**0.5]),
                               np.random.normal(loc = seed[3:6], scale = np.abs(seed[3:6]*sigma_E/3**0.5)),
                               np.expand_dims(np.random.normal(loc = 0, scale = sigma_time/model.time_scale), axis = 0)])
    else :
        seed = np.concatenate([np.random.normal(loc = seed[0:3], scale = [sigma_space/model.xy_scale/3**0.5, sigma_space/model.xy_scale/3**0.5, sigma_space/model.z_scale/3**0.5]),
                               np.random.normal(loc = seed[3:6], scale = np.abs(seed[3:6]*sigma_E/3**0.5))])
    print("SEED")
    print(seed)

    return seed
    
def load_model(args) :
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

    # Load weigths
    network.load_state_dict(torch.load(args.model_weights_path, map_location=network.device))

    return network

if __name__ == "__main__" :
    
    parser = argparse.ArgumentParser(description='Application to fit neural network model to events.')
    parser.add_argument('data_dirs', type = str, help = "Directory with training data")
    parser.add_argument('data_flavour', type = str, help = "Expression that matches training data file ending")
    parser.add_argument('n_events', type = int, help = "Number of events to fit")
    parser.add_argument('-t', '--train_fraction', type = float, help = "Fraction of data used for training", default = 0.75, required = False)
    parser.add_argument('model', type = str, help = "Name of model to train")
    parser.add_argument('model_weights_path', type = str, help = "Path to saved model weights")
    parser.add_argument('model_arguments', type = str, help = "Arguments to pass to model, in format \"name1:value1 name2:value2 ...\"", nargs = "*", default = "")

    
    args = parser.parse_args()

    network = load_model(args)

    output = []

    test_loader = iotools.loader_factory('H5Dataset', batch_size=1, shuffle=False, num_workers=1, pin_memory = True, data_dirs=args.data_dirs.split(","), flavour=args.data_flavour, start_fraction=args.train_fraction, use_fraction=1.-args.train_fraction, read_keys= ["positions","directions", "energies", "event_data_top", "event_data_bottom"])

    # Grab end-cap masks from one of the input files
    with h5py.File(glob(args.data_dirs+"/*"+args.data_flavour)[0], mode = "r") as f :
        network.top_mask = f['mask'][0]
        network.top_mask = network.top_mask.reshape(-1, network.top_mask.shape[0]*network.top_mask.shape[1])
        network.bottom_mask = f['mask'][1]
        network.bottom_mask = network.bottom_mask.reshape(-1, network.bottom_mask.shape[0]*network.bottom_mask.shape[1])

    for i_event, event in enumerate(test_loader) :
        if i_event >= args.n_events :
            break
        print(i_event)

        network.fillLabel(event)
        network.fillData(event)
        
        seed = pre_fit(network, event)
        
        fit_result_e = scipy.optimize.minimize(function, seed, args=(network, [0., 1., 0.]), method = "Nelder-Mead")
        print(fit_result_e)
        fit_result_mu = scipy.optimize.minimize(function, seed, args=(network, [0., 0., 1.]), method = "Nelder-Mead")
        print(fit_result_mu)

