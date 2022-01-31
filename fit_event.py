import argparse
import importlib
from glob import glob

import numpy as np
import scipy.optimize
import h5py

import torch
import iotools

import collections

def function(x, model, PID) :

    E = np.linalg.norm(x[3:6])

    model.data = torch.tensor(np.expand_dims(np.concatenate([PID,
                                                             x[:3],
                                                             x[3:6]/E,
                                                             np.array([E])]), axis = 0), 
                              dtype = torch.float,
                              device = model.device)
    if model.use_time :
        t0 = x[6]
    else :
        t0 = 0.
    ret = model.evaluate(Train = False, t0 = t0)["loss"]
    return ret
        
def pre_fit(model, event) :
    # For now this is just a placeholder. Using truth information, pick a random point, reasonably close to the truth.

    sigma_space = 500. # cm
    sigma_time = 15. # ns
    sigma_E = 0.5 # Fractional
    sigma_angle = 20 # degrees
    
    # Randomize truth for seed. To mimick a pre-fit algorithm.
    # Energy:
    ran_E = model.data[0,9] * np.random.normal(loc = 1., scale = sigma_E*model.data[0,9])
    
    # Random theta
    ran_theta = np.abs(np.random.normal(0., sigma_angle*np.pi/180))
    ran_phi = np.random.uniform(0, 2*np.pi)

    original_dir = model.data[0,6:9]
    # Just to make sure
    original_dir = original_dir/np.linalg.norm(original_dir)

    unit_vects = [np.array([1, 0, 0]),
                  np.array([0, 1, 0]),
                  np.array([0, 0, 1])]

    # Find which coordinate is most orthogonal to original direction
    i_coord = np.argmin(np.dot(original_dir, unit_vects))

    # Get one orthogonal vector:
    v_T_1 = np.cross(original_dir, unit_vects[i_coord])
    # And get another orthogonal vector to complete the basis:
    v_T_2 = np.cross(original_dir, v_T_1)

    # Normalize basis
    v_T_1 = v_T_1/np.linalg.norm(v_T_1)
    v_T_2 = v_T_2/np.linalg.norm(v_T_2)

    # Build rotation matrix
    R = np.transpose(np.array([v_T_1,
                               v_T_2,
                               original_dir]))
    
    # Random direction in new basis
    ran_vec = np.array([np.sin(ran_theta)*np.cos(ran_phi),
                        np.sin(ran_theta)*np.sin(ran_phi),
                        np.cos(ran_theta)])
    
    # Random direction in detector coordinates:
    ran_dir = np.matmul(R, ran_vec)
    
    # Now randomize position:
    ran_pos = np.random.normal(loc = model.data[0,3:6], scale = [sigma_space/model.xy_scale/3**0.5, sigma_space/model.xy_scale/3**0.5, sigma_space/model.z_scale/3**0.5])
    
    # And randomize time
    if model.use_time :
        ran_t = np.random.normal(loc = 0., scale = sigma_time)

    print(ran_dir)
    print(ran_dir*ran_E)

    # Seed:
    if model.use_time :
        seed = np.concatenate([ran_pos, ran_dir*ran_E, [ran_t]])
    else :
        seed = np.concatenate([ran_pos, ran_dir*ran_E])

    print("TRUTH")
    print(np.concatenate([model.data[0,3:6], model.data[0,6:9]*model.data[0,9]]))

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
    parser.add_argument('out_file_name', type = str, help = "Output file name")
    parser.add_argument('model', type = str, help = "Name of model to train")
    parser.add_argument('model_weights_path', type = str, help = "Path to saved model weights")
    parser.add_argument('model_arguments', type = str, help = "Arguments to pass to model, in format \"name1:value1 name2:value2 ...\"", nargs = "*", default = "")
    
    
    args = parser.parse_args()

    network = load_model(args)

    test_loader = iotools.loader_factory('H5Dataset', batch_size=1, shuffle=False, num_workers=0, pin_memory = True, data_dirs=args.data_dirs.split(","), flavour=args.data_flavour, start_fraction=args.train_fraction, use_fraction=1.-args.train_fraction, read_keys= ["positions","directions", "energies", "event_data_top", "event_data_bottom"])

    # Grab end-cap masks from one of the input files
    with h5py.File(glob(args.data_dirs+"/*"+args.data_flavour)[0], mode = "r") as f :
        network.top_mask = f['mask'][0]
        network.top_mask = network.top_mask.reshape(-1, network.top_mask.shape[0]*network.top_mask.shape[1])
        network.bottom_mask = f['mask'][1]
        network.bottom_mask = network.bottom_mask.reshape(-1, network.bottom_mask.shape[0]*network.bottom_mask.shape[1])

    # Output file
    data = {}
    data["PID"] = collections.defaultdict(list)
    data["pos"] = collections.defaultdict(list)
    data["dir"] = collections.defaultdict(list)
    data["t0"] = collections.defaultdict(list)
    data["E"] = collections.defaultdict(list)
    data["nll"] = collections.defaultdict(list)
    data["fit_success"] = collections.defaultdict(list)
    
    for i_event, event in enumerate(test_loader) :
        if i_event >= args.n_events :
            break
        print(i_event)

        network.fillLabel(event)
        network.fillData(event)
        
        data["pos"]["truth"].append([network.data[0,3]*network.xy_scale,
                             network.data[0,4]*network.xy_scale,
                             network.data[0,5]*network.z_scale])
        data["dir"]["truth"].append([network.data[0,6],
                             network.data[0,7],
                             network.data[0,8]])
        data["E"]["truth"].append(network.data[0,9]*network.energy_scale)
        data["PID"]["truth"].append([network.data[0,0],
                             network.data[0,1],
                             network.data[0,2]])
        
        # Run fake prefit
        seed = pre_fit(network, event)

        seed_E = np.linalg.norm(seed[3:6])
        data["pos"]["seed"].append([seed[0]*network.xy_scale,
                            seed[1]*network.xy_scale,
                            seed[2]*network.z_scale])
        data["dir"]["seed"].append([seed[3]/seed_E,
                            seed[4]/seed_E,
                            seed[5]/seed_E])
        data["E"]["seed"].append(seed_E*network.energy_scale)

        if network.use_time :
            data["t0"]["seed"].append(seed[6]*network.time_scale+network.time_offset)
        
        fit_result_e = scipy.optimize.minimize(function, seed, args=(network, [0., 1., 0.]), method = "Nelder-Mead")
        
        e_X = fit_result_e.x

        e_E = np.linalg.norm(e_X[3:6])

        data["pos"]["fit_e"].append([e_X[0]*network.xy_scale,
                             e_X[1]*network.xy_scale,
                             e_X[2]*network.z_scale])
        data["dir"]["fit_e"].append([e_X[3]/e_E,
                             e_X[4]/e_E,
                             e_X[5]/e_E])
        data["E"]["fit_e"].append(e_E*network.energy_scale)

        if network.use_time :
            data["t0"]["fit_e"].append(e_X[6]*network.time_scale+network.time_offset)
        
        data["nll"]["fit_e"].append(fit_result_e.fun)
        data["fit_success"]["fit_e"].append(fit_result_e.success)

        fit_result_mu = scipy.optimize.minimize(function, seed, args=(network, [0., 0., 1.]), method = "Nelder-Mead")

        mu_X = fit_result_mu.x

        mu_E = np.linalg.norm(mu_X[3:6])

        data["pos"]["fit_mu"].append([mu_X[0]*network.xy_scale,
                              mu_X[1]*network.xy_scale,
                              mu_X[2]*network.z_scale])
        data["dir"]["fit_mu"].append([mu_X[3]/mu_E,
                              mu_X[4]/mu_E,
                              mu_X[5]/mu_E])
        data["E"]["fit_mu"].append(mu_E*network.energy_scale)

        if network.use_time :
            data["t0"]["fit_mu"].append(mu_X[6]*network.time_scale+network.time_offset)
        
        data["nll"]["fit_mu"].append(fit_result_mu.fun)
        data["fit_success"]["fit_mu"].append(fit_result_mu.success)
        
    with h5py.File(args.out_file_name, mode = "w") as f :
        for group_name in data["E"].keys() :
            g = f.create_group(group_name)
            for var_name in data.keys() :
                if group_name in data[var_name] :
                    g.create_dataset(var_name, data = data[var_name][group_name])
