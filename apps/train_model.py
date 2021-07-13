import argparse
import importlib

import torch

from iotools import loader_factory

def train_model(args) :
    
    # Get and initialize model
    print("Loading model: "+args.model)
    model_module = importlib.import_module("models."+args.model)
    network = model_module.model()
    
    # Initialize data loaders
    print(args.data_dirs)
    print(args.data_flavour)

    train_loader=loader_factory('H5Dataset', batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, data_dirs=args.data_dirs.split(","), flavour=args.data_flavour, start_fraction=0.0, use_fraction=args.train_fraction, read_keys= ["positions","directions", "energies"])
    test_loader=loader_factory('H5Dataset', batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, data_dirs=args.data_dirs.split(","), flavour=args.data_flavour, start_fraction=args.train_fraction, use_fraction=1.-args.train_fraction, read_keys= [ "positions","directions", "energies"])
    
    # Training loop
    current_epoch = 0.
    for iteration, data in enumerate(train_loader) :

        network.train()

        network.fillData(data)
        network.fillLabel(data)

        res = network.evaluate(True)
        network.backward()

        current_epoch += 1./len(train_loader)
        
        # Report progress
        if iteration == 0 or (iteration+1)%10 == 0 :
            print('TRAINING', 'Iteration', iteration, 'Epoch', current_epoch, 'Loss', res['loss'])
            
        if (iteration+1)%100 == 0 :
            with torch.no_grad() :
                network.eval()
                test_data = next(iter(test_loader))
                network.fillLabel(test_data)
                network.fillData(test_data)
                res = network.evaluate(False)
                print('VALIDATION', 'Iteration', iteration, 'Epoch', current_epochepoch, 'Loss', res['loss'])

        # Save network periodically
        if (iteration+1)%args.save_interval == 0 :
            torch.save(network.state_dict(), model_name+"_"+str(iteration)+".cnn")

        if current_epoch >= args.epochs :
            break
    torch.save(network.state_dict(), model_name+".cnn")

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Application to train Water Cherenkov generative neural networks.')
    parser.add_argument('-e', '--epochs', type = float, help = "Number of epochs to train for", default = 1., required = False)
    parser.add_argument('-b', '--batch_size', type = int, help = "Batch size", default = 200, required = False)
    parser.add_argument('-j', '--num_workers', type = int, help = "Number of CPUs for loading data", default = 8, required = False)
    parser.add_argument('-t', '--train_fraction', type = float, help = "Fraction of data used for training", default = 0.75, required = False)
    parser.add_argument('-s', '--save_interval', type = int, help = "Save network state every <save_interval> iterations", default = 5000, required = False)
    parser.add_argument('data_dirs', type = str, help = "Directory with training data")
    parser.add_argument('data_flavour', type = str, help = "Expression that matches training data file ending")
    parser.add_argument('model', type = str, help = "Name of model to train")

    args = parser.parse_args()
    
    print(args)
    
    train_model(args)

