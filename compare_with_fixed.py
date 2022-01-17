import argparse
import importlib

import matplotlib.pyplot as plt
import numpy as np
import h5py

import torch

particle_name = ["gamma", "e", "mu"]

def compare_with_fixed(args) :
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

    pmts_to_study_xy = [[22, 75], [22, 90], [22, 110]]
    pmt_names = ["Center", "Edge", "Outside"]

    with h5py.File(args.event_path) as f :
        flavour = [0, 0, 0] 
        i_flavour = int(f['labels'][0])
        flavour[i_flavour] = 1
        print("Flavour", flavour)
        event_p = np.zeros((f['event_data'].shape[1], f['event_data'].shape[2]), dtype = np.float64)
        event_q = np.zeros((f['event_data'].shape[1], f['event_data'].shape[2]), dtype = np.float64)
        event_t = np.zeros((f['event_data'].shape[1], f['event_data'].shape[2]), dtype = np.float64)

        pmts_to_study_data_temp = np.zeros((len(pmts_to_study_xy), f['event_data'].shape[0], 2), dtype = np.float64)

        for i_event, event in enumerate(f['event_data']) :
            event_p += event[:,:,0] > 0
            event_q += event[:,:,0]
            event_t += event[:,:,1]

            for i_pmt, pmt in enumerate(pmts_to_study_xy) :
                pmts_to_study_data_temp[i_pmt][i_event][0] = event[pmt[0],pmt[1],0]
                pmts_to_study_data_temp[i_pmt][i_event][1] = event[pmt[0],pmt[1],1]

        event_p /= f['event_data'].shape[0]
        event_q /= f['event_data'].shape[0]
        event_t /= f['event_data'].shape[0]

        pmts_to_study = []
        for this_data in pmts_to_study_data_temp :
            pmts_to_study.append(this_data[this_data[:,0] > 0])
        
#        self.charge_scale = charge_scale
#        self.time_scale = time_scale
#        self.time_offset = time_offset
#        self.energy_scale = energy_scale
#        self.z_scale = z_scale
#        self.xy_scale = xy_scale

    net_input = torch.as_tensor([[flavour[0], flavour[1], flavour[2],
                                  0./network.xy_scale, 0./network.xy_scale, 0./network.z_scale,
                                  1., 0., 0.,
                                  500/network.energy_scale]], device = network.device)
    network.eval()
    with torch.set_grad_enabled(False) :
        prediction = network(net_input)[0].cpu().detach().numpy()

    hit_prob = (1./(1+np.exp(prediction[0][0]))).reshape((51, 150))

    # Hit probability
    plt.figure()
    plt.figure(figsize = (6.4, 6.4))
    plt.subplot(3, 1, 1)
    plt.imshow(event_p, origin = "lower", vmin = 0, vmax = 1)
    plt.title("Fixed events hit probability")
    plt.colorbar()
    plt.subplot(3, 1, 2)
    plt.imshow(hit_prob, origin = "lower", vmin = 0, vmax = 1)
    plt.title("Hit probability prediction")
    plt.colorbar()
    plt.subplot(3, 1, 3)
    plt.imshow(event_p - hit_prob, origin = "lower", vmin = -0.5, vmax = 0.5)
    plt.title("Hit probability prediction - fixed events hit probability")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("compare_fixed_event_{0}_hit_prob_NGAUS_{1}.png".format(particle_name[i_flavour], network.N_GAUS))

    # PMTs
    if not network.use_time :
        plt.figure(figsize = (6.4, 6.4))
        for i_pmt in range(len(pmts_to_study)) :
            this_pmt_prediction = torch.tensor(prediction.reshape((-1, prediction.shape[1], 51,150))[:, :, pmts_to_study_xy[i_pmt][0], pmts_to_study_xy[i_pmt][1]]).unsqueeze(dim=2)

            plt.subplot(len(pmts_to_study), 1, i_pmt+1)
            contents, bins, _ = plt.hist(pmts_to_study[i_pmt][:,0], bins = 200, density = True)

            xx = np.linspace(min(bins), max(bins), 500)
            pdf_values = []
            for x in xx :
                loss = network.multiGausLoss(this_pmt_prediction, torch.tensor([x/network.charge_scale]).unsqueeze(dim=0))
                pdf_values.append(loss['qt_loss'])

            pdf_values = np.array(pdf_values)
        
            plt.plot(xx, np.exp(-pdf_values)/network.charge_scale)

        plt.tight_layout()
        plt.savefig("compare_fixed_event_{0}_charge_NGAUS_{1}.png".format(particle_name[i_flavour], network.N_GAUS))

    else :
        for i_pmt in range(len(pmts_to_study)) :

#        plt.figure()
#        plt.figure(figsize = (6.4, 6.4))
#        for i_pmt in range(len(pmts_to_study)) :
            this_pmt_prediction = torch.tensor(prediction.reshape((-1, prediction.shape[1], 51,150))[:, :, pmts_to_study_xy[i_pmt][0], pmts_to_study_xy[i_pmt][1]]).unsqueeze(dim=2)

            this_ax = plt.subplot(len(pmts_to_study), 1, i_pmt+1)
            #contents, bins, _ = plt.hist(pmts_to_study[i_pmt][:,0], bins = 200, density = True)
            
            x_1D = np.linspace(min(pmts_to_study[i_pmt][:,0]), max(pmts_to_study[i_pmt][:,0],), 200)
            y_1D = np.linspace(min(pmts_to_study[i_pmt][:,1]), max(pmts_to_study[i_pmt][:,1],), 200)
            
            xx, yy = np.meshgrid(x_1D, y_1D)
            
            pdf_values = []
            for x, y in zip(xx.flatten(), yy.flatten()) :
                loss = network.multiGausLoss(this_pmt_prediction, torch.tensor([x/network.charge_scale]).unsqueeze(dim=0), mask = None, time = torch.tensor([(y - network.time_offset)/network.time_scale]).unsqueeze(dim=0))
                pdf_values.append(loss['qt_loss'])
            
            pdf_values = np.array(pdf_values).reshape(xx.shape)
            
            fig = plt.figure()

            left, width = 0.1, 0.65
            bottom, height = 0.1, 0.65
            spacing = 0.005

            rect_scatter = [left, bottom, width, height]
            rect_histx = [left, bottom + height + spacing, width, 0.2]
            rect_histy = [left + width + spacing, bottom, 0.2, height]

            ax = fig.add_axes(rect_scatter)
            ax_histx = fig.add_axes(rect_histx, sharex=ax)
            ax_histy = fig.add_axes(rect_histy, sharey=ax)
            
            ax.contourf(yy, xx, np.exp(-pdf_values)/(network.charge_scale*network.time_scale))
            
            ax_histx.hist(pmts_to_study[i_pmt][:,1], bins = 200, density = True)
            ax_histx.plot(y_1D, (np.exp(-pdf_values)/(network.charge_scale*network.time_scale)).sum(axis = 1))
            ax_histy.hist(pmts_to_study[i_pmt][:,0], bins = 200, density = True, orientation = 'horizontal')
            ax_histy.plot((np.exp(-pdf_values)/(network.charge_scale*network.time_scale)).sum(axis = 0), x_1D) #, orientation = 'horizontal')

            plt.tight_layout()
            plt.savefig("compare_fixed_event_{0}_charge_NGAUS_{1}_{2}.png".format(particle_name[i_flavour], network.N_GAUS, i_pmt))
    

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Application to compare Water Cherenkov generative neural network output to events.')

    parser.add_argument('event_path', type = str, help = "Path to event file")
    parser.add_argument('model', type = str, help = "Name of model to train")
    parser.add_argument('model_weights_path', type = str, help = "Path to saved model weights")
    parser.add_argument('model_arguments', type = str, help = "Arguments to pass to model, in format \"name1:value1 name2:value2 ...\"", nargs = "*", default = "")
    args = parser.parse_args()
    
    compare_with_fixed(args)
    
    



