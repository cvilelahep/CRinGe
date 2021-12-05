import argparse
import importlib
import os
from glob import glob
import h5py
import pickle

import torch

import iotools

from torch.profiler import profile, record_function, ProfilerActivity

def train_model(args) :

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
    
    # Initialize data loaders
    print("Data directory: "+args.data_dirs)
    print("Data flavour: "+args.data_flavour)

    train_loader=iotools.loader_factory('H5Dataset', batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory = True, data_dirs=args.data_dirs.split(","), flavour=args.data_flavour, start_fraction=0.0, use_fraction=args.train_fraction, read_keys= ["positions","directions", "energies", "event_data_top", "event_data_bottom"])
    test_loader =iotools.loader_factory('H5Dataset', batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory = True, data_dirs=args.data_dirs.split(","), flavour=args.data_flavour, start_fraction=args.train_fraction, use_fraction=1.-args.train_fraction, read_keys= ["positions","directions", "energies", "event_data_top", "event_data_bottom"])

    # Grab end-cap masks from one of the input files
    with h5py.File(glob(args.data_dirs+"/*"+args.data_flavour)[0], mode = "r") as f :
        network.top_mask = f['mask'][0]
        network.top_mask = network.top_mask.reshape(-1, network.top_mask.shape[0]*network.top_mask.shape[1])
        network.bottom_mask = f['mask'][1]
        network.bottom_mask = network.bottom_mask.reshape(-1, network.bottom_mask.shape[0]*network.bottom_mask.shape[1])
    
    # Create output directory
    try :
        os.makedirs(args.output_dir)
    except FileExistsError :
        pass

    # Save training options to file
    with open(args.output_dir+"/"+args.model+"_config.p", "wb") as f_out_conf :
        pickle.dump(args, f_out_conf)

    # Lists to store training progress
    train_record = []
    test_record = []

    # Training loop
    current_epoch = 0.
    network.train()
    while current_epoch < args.epochs :
        print("STARTING EPOCH {0}".format(current_epoch))
        for iteration, data in enumerate(train_loader) :
        
            network.fillData(data)
            network.fillLabel(data)
        
            res = network.evaluate(True)
            network.backward()
        
            current_epoch += 1./len(train_loader)
            
            res.update({'epoch' : current_epoch, 'iteration' : iteration})
            train_record.append(res)
            # Report progress
            if iteration == 0 or (iteration+1)%10 == 0 :
                print('TRAINING', 'Iteration', iteration, 'Epoch', current_epoch, 'Loss', res['loss'], res['loss_breakdown'])
                
            if (iteration+1)%100 == 0 :
                with torch.no_grad() :
                    network.eval()
                    test_data = next(iter(test_loader))
                    network.fillLabel(test_data)
                    network.fillData(test_data)
                    res = network.evaluate(False)

                    res.update({'epoch' : current_epoch, 'iteration' : iteration})
                    test_record.append(res)
                    print('VALIDATION', 'Iteration', iteration, 'Epoch', current_epoch, 'Loss', res['loss'], res['loss_breakdown'])
                network.train()
        
            # Save network periodically
            if (iteration+1)%args.save_interval == 0 :
                print("Saving network state")
                torch.save(network.state_dict(), args.output_dir+"/"+args.model+"_"+str(iteration)+".cnn")
                torch.save(network.optimizer.state_dict(), args.output_dir+"/"+args.model+"_optimizer_"+str(iteration)+".cnn")

            if current_epoch >= args.epochs :
                break

    torch.save(network.state_dict(), args.output_dir+"/"+args.model+".cnn")
    torch.save(network.optimizer.state_dict(), args.output_dir+"/"+args.model+"_optimizer.cnn")
    with open(args.output_dir+"/"+args.model+"_train_record.o", "wb") as f :
        pickle.dump(train_record, f)
    with open(args.output_dir+"/"+args.model+"_test_record.o", "wb") as f :
        pickle.dump(test_record, f)

    print("Training done")

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Application to train Water Cherenkov generative neural networks.')
    parser.add_argument('-e', '--epochs', type = float, help = "Number of epochs to train for", default = 1., required = False)
    parser.add_argument('-b', '--batch_size', type = int, help = "Batch size", default = 200, required = False)
    parser.add_argument('-j', '--num_workers', type = int, help = "Number of CPUs for loading data", default = 8, required = False)
    parser.add_argument('-t', '--train_fraction', type = float, help = "Fraction of data used for training", default = 0.75, required = False)
    parser.add_argument('-s', '--save_interval', type = int, help = "Save network state every <save_interval> iterations", default = 5000, required = False)
    parser.add_argument('-o', '--output_dir', type = str, help = "Output directory", default = "./", required = False)
    parser.add_argument('-r', '--random_seed', type = int, help = "Output directory", default = None, required = False)
    parser.add_argument('data_dirs', type = str, help = "Directory with training data")
    parser.add_argument('data_flavour', type = str, help = "Expression that matches training data file ending")
    parser.add_argument('model', type = str, help = "Name of model to train")
    parser.add_argument('model_arguments', type = str, help = "Arguments to pass to model, in format \"name1:value1 name2:value2 ...\"", nargs = "*", default = "")

    args = parser.parse_args()
    
    print(args)
    
    train_model(args)

