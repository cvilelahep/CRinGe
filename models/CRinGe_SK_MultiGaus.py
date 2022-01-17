import torch
import numpy as np
import sys

class model(torch.nn.Module) :
    
    def __init__(self, N_GAUS = 1, use_time = False, use_corr = False, charge_scale = 2500., time_scale = 1000., time_offset = 1000., energy_scale = 5000., z_scale = 1810., xy_scale = 1690.) :
        
        # Model name
        self.name = "CRinGe_SK_MultiGaus"
        
        # Detector regions implemented in the model
        self.detector_regions = ["barrel", "top", "bottom"]
        
        # Model inputs
        self.inputs = ["particle_id", "position", "direction", "energy"]

        # Model outputs
        self.outputs = ["hit_probability", "logmu", "logvar", "coefficients"]
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Network parameters
        self.N_GAUS = N_GAUS
        self.use_time = use_time
        self.use_corr = use_corr
        if not self.use_time :
            self.n_parameters_per_gaus = 3 # Coefficient, mean, sigma
            if self.use_corr:
               print("Correlated loss needs timing! Please turn on use_time for this network!")
               sys.exit()
        else :
            if not self.use_corr:
                self.n_parameters_per_gaus = 5 # Coefficient, q mean, q sigma, t mean, t sigma
            else:
                self.n_parameters_per_gaus = 6 # Coefficient, q mean, q sigma, t mean, t sigma, corr
                
        self.charge_scale = charge_scale
        self.time_scale = time_scale
        self.time_offset = time_offset
        self.energy_scale = energy_scale
        self.z_scale = z_scale
        self.xy_scale = xy_scale

        # Initialize network
        super(model, self).__init__()
        self.initialize_network()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.0002)

        self.bceloss = torch.nn.BCEWithLogitsLoss(reduction = "none")

        self.to(self.device)

    def initialize_network(self):

        # Neural network components
        self._mlp_pid = torch.nn.Sequential(
            torch.nn.Linear(3,512), torch.nn.ReLU(),
            torch.nn.Linear(512,512), torch.nn.ReLU()
        )

        self._mlp_pos = torch.nn.Sequential(
            torch.nn.BatchNorm1d(3),
            torch.nn.Linear(3,512), torch.nn.ReLU(),
            torch.nn.Linear(512,512), torch.nn.ReLU()
        )

        self._mlp_dir = torch.nn.Sequential(
            torch.nn.BatchNorm1d(3),
            torch.nn.Linear(3,512), torch.nn.ReLU(),
            torch.nn.Linear(512,512), torch.nn.ReLU()
        )

        self._mlp_E = torch.nn.Sequential(
            torch.nn.BatchNorm1d(1),
            torch.nn.Linear(1,512), torch.nn.ReLU(),
            torch.nn.Linear(512,512), torch.nn.ReLU()
        )
        
        self._mlp_barrel = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 8512), torch.nn.ReLU() # 64 * 7 * 19
        )
        
        self._mlp_top = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 2304), torch.nn.ReLU() # 64 * 6 * 6
	)
        self._mlp_bottom = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 2304), torch.nn.ReLU() # 64 * 6 * 6
        )

        self._upconvs_barrel = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 4, 2),  torch.nn.ReLU(),  # 16 x 40
            torch.nn.Conv2d(64, 64, 3), torch.nn.ReLU(),               # 14 x 38 
            torch.nn.ConvTranspose2d(64, 32, 4, 2), torch.nn.ReLU(),   # 30 x 78
            torch.nn.Conv2d(32, 32, 3),  torch.nn.ReLU(),              # 28 x 76 
            torch.nn.ConvTranspose2d(32, 32, 4, 2), torch.nn.ReLU(),   # 58 x 154
            torch.nn.Conv2d(32, 1+self.N_GAUS*self.n_parameters_per_gaus, 3)                         # 56 x 152
        )
        
        self._upconvs_top = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 4, 2),  torch.nn.ReLU(),  # 14 x 14
            torch.nn.Conv2d(64, 64, 3), torch.nn.ReLU(),               # 12 x 12
            torch.nn.ConvTranspose2d(64, 32, 4, 2), torch.nn.ReLU(),   # 26 x 26 
            torch.nn.Conv2d(32, 32, 3),  torch.nn.ReLU(),              # 24 x 24
            torch.nn.ConvTranspose2d(32, 32, 4, 2), torch.nn.ReLU(),   # 50 x 50
            torch.nn.Conv2d(32, 1+self.N_GAUS*self.n_parameters_per_gaus, 3) # 48 x 48
        )
        self._upconvs_bottom = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 4, 2),  torch.nn.ReLU(),  # 14 x 14
            torch.nn.Conv2d(64, 64, 3), torch.nn.ReLU(),               # 12 x 12
            torch.nn.ConvTranspose2d(64, 32, 4, 2), torch.nn.ReLU(),   # 26 x 26
            torch.nn.Conv2d(32, 32, 3),  torch.nn.ReLU(),              # 24 x 24
            torch.nn.ConvTranspose2d(32, 32, 4, 2), torch.nn.ReLU(),   # 50 x 50
            torch.nn.Conv2d(32, 1+self.N_GAUS*self.n_parameters_per_gaus, 3) # 48 x 48
        )
        self._tanh = torch.nn.Tanh()
        
    # Forward neural network with input x
    def forward(self, x) :
        # Concatenate MLPs that treat PID, pos, dir and energy inputs separately
        net = torch.cat( (self._mlp_pid(x[:,0:3]),self._mlp_pos(x[:,3:6]),self._mlp_dir(x[:,6:9]),self._mlp_E(x[:,9].reshape(len(x[:,9]),1))), 1)

        # MLPs
        net_barrel = self._mlp_barrel(net)
        net_top = self._mlp_top(net)
        net_bottom = self._mlp_bottom(net)
        
        # Reshape MLP outputs
        net_barrel = net_barrel.view(-1, 64, 7, 19)
        net_top = net_top.view(-1, 64, 6, 6)
        net_bottom = net_bottom.view(-1, 64, 6, 6)

        # Upconv layers
        net_barrel = self._upconvs_barrel(net_barrel)[:,:,2:-3,1:-1]
        net_barrel = net_barrel.reshape(-1, 1+self.N_GAUS*self.n_parameters_per_gaus, 51*150)
        net_top = self._upconvs_top(net_top).view(-1, 1+self.N_GAUS*self.n_parameters_per_gaus, 48*48)
        net_bottom = self._upconvs_bottom(net_bottom).view(-1, 1+self.N_GAUS*self.n_parameters_per_gaus, 48*48)

        if self.use_corr:
            # 5th < (1st + 3rd)/2                                                                                      
            # |a12| < |a11+a22|/2            
            for i in range(self.N_GAUS):
                #0th element is always unhit probability
                a11_barrel = torch.exp(net_barrel[:, i*(self.n_parameters_per_gaus-1)+3, :])
                a22_barrel = torch.exp(net_barrel[:, i*(self.n_parameters_per_gaus-1)+1, :])
                a12_barrel = net_barrel[:,i*(self.n_parameters_per_gaus-1)+5,:]
                net_barrel[:,1+i*5+4,:] = 0.5*(a11_barrel+a22_barrel)*self._tanh(a12_barrel)
                
                a11_top = torch.exp(net_top[:, i*(self.n_parameters_per_gaus-1)+3, :])
                a22_top = torch.exp(net_top[:, i*(self.n_parameters_per_gaus-1)+1, :])
                a12_top = net_top[:,i*(self.n_parameters_per_gaus-1)+5,:]
                net_top[:,1+i*5+4,:] = 0.5*(a11_top+a22_top)*self._tanh(a12_top)
                
                a11_bottom = torch.exp(net_bottom[:, i*(self.n_parameters_per_gaus-1)+3, :])
                a22_bottom = torch.exp(net_bottom[:, i*(self.n_parameters_per_gaus-1)+1, :])
                a12_bottom = net_bottom[:,i*(self.n_parameters_per_gaus-1)+5,:]
                net_bottom[:,1+i*5+4,:] = 0.5*(a11_bottom+a22_bottom)*self._tanh(a12_bottom)
                
        return [net_barrel, net_bottom, net_top]

    # Fill data
    def fillData(self, data) :
        oneHotGamma = np.array(data[1] == 0)
        oneHotE = np.array(data[1] == 1)
        oneHotMu = np.array(data[1] == 2)
    
        self.data =  np.hstack((oneHotGamma.reshape(len(oneHotGamma),1), oneHotE.reshape(len(oneHotE),1), oneHotMu.reshape(len(oneHotMu),1), # One-hot PID
                                data[2][:,0,:], # Positions
                                data[3][:,0,:], # Directions
                                data[4][:,0].reshape(len(data[4][:,0]),1) ) ) # Energy
        self.data[:,3] /= self.xy_scale
        self.data[:,4] /= self.xy_scale
        self.data[:,5] /= self.z_scale
        self.data[:,9] /= self.energy_scale

    # Fill label
    def fillLabel(self, data) :
        dim_barrel = data[0].shape
        self.charge_barrel = torch.tensor(data[0][:,:,:,0].reshape(-1,dim_barrel[1]*dim_barrel[2])/self.charge_scale, device = self.device)

        dim_cap = data[5].shape
        self.charge_top = torch.tensor(data[5][:,:,:,0].reshape(-1, dim_cap[1]*dim_cap[2])/self.charge_scale, device = self.device)
        self.charge_bottom = torch.tensor(data[6][:,:,:,0].reshape(-1, dim_cap[1]*dim_cap[2])/self.charge_scale, device = self.device)

        if self.use_time :
            self.time_barrel = torch.tensor((data[0][:,:,:,1].reshape(-1,dim_barrel[1]*dim_barrel[2]) - self.time_offset)/self.time_scale, device = self.device)
            self.time_top = torch.tensor((data[5][:,:,:,1].reshape(-1, dim_cap[1]*dim_cap[2]) - self.time_offset)/self.time_scale, device = self.device)
            self.time_bottom = torch.tensor((data[6][:,:,:,1].reshape(-1, dim_cap[1]*dim_cap[2]) - self.time_offset)/self.time_scale, device = self.device)

    def multiGausLoss(self, prediction, charge, mask = None, time = None) :
        
        charge_n = torch.stack( [ charge for i in range(self.N_GAUS) ], dim = 1 )
        if time is not None:
            time_n = torch.stack([ time for i in range(self.N_GAUS) ], dim = 1)

        punhit = prediction[:,0]
        hitMask = charge > 0
        hit_loss_tensor = self.bceloss(punhit, (charge == 0).float())
        hit_loss = hit_loss_tensor[:,mask].sum()
        if mask is None :
            mask = torch.full_like(punhit[0], True, dtype = torch.bool, device = self.device)
        else :
            mask = torch.squeeze(torch.tensor(mask, dtype = torch.bool, device = self.device), dim = 0)

        coefficients = torch.nn.functional.softmax(prediction[:, -self.N_GAUS:], dim = 1)

        if not self.use_corr:         

            logvar = torch.stack( [ prediction[:, i*(self.n_parameters_per_gaus - 1) + 1] for i in range(self.N_GAUS) ], dim = 1 )
            var = torch.exp(logvar)
            
            logmu = torch.stack( [ prediction[:, i*(self.n_parameters_per_gaus - 1) + 2] for i in range(self.N_GAUS) ], dim = 1 )
            mu = torch.exp(logmu)
                            
            charge_loss = hitMask.sum()*(1/2.)*np.log(2*np.pi) # Constant term
            
            nll_charge = torch.log(coefficients) - 1/2.*logvar - 1/2.*(charge_n - mu)**2/var
            
            charge_loss += - torch.logsumexp(nll_charge, dim = 1)[hitMask].sum()
        
            ret = {"hit_loss" : hit_loss, "charge_loss" : charge_loss}

            if time is not None:
            
                logvar_t = torch.stack( [ prediction[:, i*(self.n_parameters_per_gaus - 1) + 3] for i in range(self.N_GAUS) ], dim = 1)
                var_t = torch.exp(logvar_t)
                mu_t = torch.stack( [ prediction[:, i*(self.n_parameters_per_gaus - 1) + 4] for i in range(self.N_GAUS) ], dim = 1)
                
                time_loss = hitMask.sum()*(1/2.)*np.log(2*np.pi) # Constant term
                
                nll_time = torch.log(coefficients) - 1/2.*logvar_t - 1/2.*(time_n - mu_t)**2/var_t
                
                time_loss += - torch.logsumexp(nll_time, dim = 1)[hitMask].sum()
                
                ret.update({"time_loss" : time_loss})

        else :
            
            a12 = torch.stack( [ prediction[:, i*(self.n_parameters_per_gaus - 1) + 5] for i in range(self.N_GAUS) ], dim = 1)
            
            loga11 = torch.stack( [ prediction[:, i*(self.n_parameters_per_gaus - 1) + 3] for i in range(self.N_GAUS) ], dim = 1)
            a11 = torch.exp(loga11) # ~ 1/sigma_t
            mu_t = torch.stack( [ prediction[:, i*(self.n_parameters_per_gaus - 1) + 4] for i in range(self.N_GAUS) ], dim = 1)
            mu_t_diff = time_n - mu_t
            
            loga22 = torch.stack( [ prediction[:, i*(self.n_parameters_per_gaus - 1) + 1] for i in range(self.N_GAUS) ], dim = 1)
            a22 = torch.exp(loga22) # ~ 1/sigma_q
            logmu = torch.stack( [ prediction[:, i*(self.n_parameters_per_gaus - 1) + 2] for i in range(self.N_GAUS) ], dim = 1)
            mu = torch.exp(logmu)
            mu_diff = charge_n - mu

            corr_loss = hitMask.sum()*np.log(2*np.pi) # Constant term
            
            nll_corr = torch.log(coefficients) - loga11 - loga22 - 1/2.*((mu_t_diff*a11)**2 + mu_diff**2*(a22**2 + a12**2) + 2*mu_t_diff*mu_diff*a11*a12)
            
            corr_loss += - torch.logsumexp(nll_corr, dim = 1)[hitMask].sum()

            ret = {"hit_loss" : hit_loss, "correlated_loss" : corr_loss}
            
        return ret

    # Evaluate network
    def evaluate(self, Train = True) :
        with torch.set_grad_enabled(Train) :
            data = torch.as_tensor(self.data, device = self.device)
            #wrong order before
            prediction_barrel, prediction_bottom, prediction_top = self(data)
            
            if self.use_time :
                barrel_loss = self.multiGausLoss(prediction_barrel, self.charge_barrel, mask = None, time = self.time_barrel)
                top_loss = self.multiGausLoss(prediction_top, self.charge_top, mask = self.top_mask, time = self.time_top)
                bottom_loss = self.multiGausLoss(prediction_bottom, self.charge_bottom, mask = self.bottom_mask, time = self.time_bottom)
            else :
                barrel_loss = self.multiGausLoss(prediction_barrel, self.charge_barrel, mask = None)
                top_loss = self.multiGausLoss(prediction_top, self.charge_top, mask = self.top_mask)
                bottom_loss = self.multiGausLoss(prediction_bottom, self.charge_bottom, mask = self.bottom_mask)
                
            # Collect all losses separately for later analysis
            loss_breakdown = {}
            for key, item in barrel_loss.items() :
                loss_breakdown["barrel_"+key] = item.item()
            for key, item in top_loss.items() :
                loss_breakdown["top_"+key] = item.item()
            for key, item in bottom_loss.items() :
                loss_breakdown["bottom_"+key] = item.item()

            # The actual loss that is used to update the gradients
            self.loss = torch.stack([ barrel_loss[k] for k in barrel_loss.keys() ]).sum()
            self.loss += torch.stack([ top_loss[k] for k in top_loss.keys() ]).sum()
            self.loss += torch.stack([ bottom_loss[k] for k in bottom_loss.keys() ]).sum()
                
            return { 'loss' : self.loss.item(),
                     'loss_breakdown' : loss_breakdown }
#                     'prediction' : [prediction_barrel.cpu().detach().numpy(),
#                                     prediction_top.cpu().detach().numpy(),
#                                     prediction_bottom.cpu().detach().numpy()] }

    def backward(self) :
        #self.optimizer.zero_grad()
        for param in self.parameters() :
            param.grad = None
        self.loss.backward()
        self.optimizer.step()
