import torch
import numpy as np

class model(torch.nn.Module) :
    
    def __init__(self) :
        
        # Model name
        self.name = "CRinGe_Gaus"
        
        # Detector regions implemented in the model
        self.detector_regions = ["barrel"]
        
        # Model inputs
        self.inputs = ["particle_id", "position", "direction", "energy"]

        # Model outputs
        self.outputs = ["hit_probability", "logmu", "logvar"]
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize network
        super(model, self).__init__()
        self.initialize_network()
        
        self.bceloss = torch.nn.BCEWithLogitsLoss()

        self.to(self.device)

    def initialize_network(self):

        # Neural network components
        self._mlp_pid = torch.nn.Sequential(
#            torch.nn.BatchNorm1d(3),
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

        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 14784), torch.nn.ReLU()
        )


        self._upconvs = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 4, 2),  torch.nn.ReLU(),  # 24 x 44 
            torch.nn.Conv2d(64, 64, 3), torch.nn.ReLU(),               # 22 x 42 
                                                                                 
            torch.nn.ConvTranspose2d(64, 32, 4, 2), torch.nn.ReLU(),   # 46 x 86 
            torch.nn.Conv2d(32, 32, 3),  torch.nn.ReLU(),              # 44 x 84 
                                                                                 
            torch.nn.ConvTranspose2d(32, 32, 4, 2), torch.nn.ReLU(),   # 90 x 170
            torch.nn.Conv2d(32, 3, 3)                                  # 88 x 168
        )

        self._sigmoid = torch.nn.Sigmoid()

        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.0002)
        torch.nn.utils.clip_grad.clip_grad_norm_(self.parameters(), 1.0)
        
    # Forward neural network with input x
    def forward(self, x) :
        # Concatenate MLPs that treat PID, pos, dir and energy inputs separately
        net = torch.cat( (self._mlp_pid(x[:,0:3]),self._mlp_pos(x[:,3:6]),self._mlp_dir(x[:,6:9]),self._mlp_E(x[:,9].reshape(len(x[:,9]),1))), 1)

        # MegaMLP 
        net = self._mlp(net)
        
        # Reshape into 11 x 21 figure in 64 channels. Enough?!
        net = net.view(-1, 64, 11, 21)

        # Need to flatten? Maybe...
        net = self._upconvs(net).view(-1, 3, 88*168)
        return net

    # Fill data
    def fillData(self, data) :
        oneHotGamma = np.array(data[1] == 0)
        oneHotE = np.array(data[1] == 1)
        oneHotMu = np.array(data[1] == 2)
    
        self.data =  np.hstack((oneHotGamma.reshape(len(oneHotGamma),1), oneHotE.reshape(len(oneHotE),1), oneHotMu.reshape(len(oneHotMu),1), # One-hot PID
                                data[2][:,0,:], # Positions
                                data[3][:,0,:], # Directions
                                data[4][:,0].reshape(len(data[4][:,0]),1) ) ) # Energy
    # Fill label
    def fillLabel(self, data) :
        dim = data[0].shape
        self.label = data[0][:,:,:,0].reshape(-1,dim[1]*dim[2])

    # Evaluate network
    def evaluate(self, Train = True) :
        with torch.set_grad_enabled(Train) :
            data = torch.as_tensor(self.data).to(self.device)
            prediction = self(data)

            logvar = prediction[:,0]
            logmu = prediction[:,1]
            punhit = prediction[:,2]
            
            var = torch.exp(logvar)
            mu = torch.exp(logmu)

            if self.label is not None :
                label = torch.as_tensor(self.label).to(self.device)
                
                unhitMask = (label == 0)
                            
                unhitTarget = torch.as_tensor(unhitMask).type(torch.FloatTensor).to(self.device)
                fracUnhit = unhitTarget.sum()/unhitTarget.numel()
            
                loss = fracUnhit*self.bceloss(punhit, unhitTarget)
                
                loss += (1-fracUnhit)*(1/2.)*(logvar[~unhitMask] + (label[~unhitMask]-mu[~unhitMask])**2/var[~unhitMask]).mean()
                loss += (1-fracUnhit)*(1/2.)*np.log(2*np.pi)
                
                self.loss = loss

                return {'prediction' : prediction.cpu().detach().numpy(),
                        'loss' : loss.cpu().detach().item()}

            return {'prediction' : prediction.cpu().detach().numpy()}

    def backward(self) :
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
