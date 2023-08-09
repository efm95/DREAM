import torch

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
#from sklearn.preprocessing import StandardScaler



class DREAM_gp:
    def __init__(self) -> None:
        pass


    def train_repetitions(self,
                          model, #NeuREM model
                          x_event:torch.tensor,
                          x_non_event:torch.tensor,
                          stat_test:torch.tensor,
                          repetitions:int=5,
                          batch_size:int= 2**14,
                          gradient_clipping:float = 1.0):
        
        
        self.stat_test = stat_test
        self.repetitions = repetitions
        self.stat_test_sample_size = self.stat_test.shape[1]
        num_stat = x_event.shape[1]
        self.stat_test_sample_size = self.stat_test.shape[0]
        self.outputs = torch.zeros(num_stat,self.repetitions,self.stat_test_sample_size)
        
        for rep in range(self.repetitions):
            print(f'Repetition: {rep+1}')
            
            #Shuffling
            torch.manual_seed(rep)
            event_set=x_event[torch.randperm(x_event.size()[0])]
            non_event_set=x_non_event[torch.randperm(x_non_event.size()[0])]
            
            #Training
            self.model = model
            self.model.fit(events=event_set,
                           non_events=non_event_set,
                           batch_size=batch_size,
                           verbose=False,
                           gradient_clipping=gradient_clipping)
            
            for stat in range(num_stat):
                self.outputs[stat,rep,:] = self.model.feature_out(stat,self.stat_test[:,stat])
        
        
    def gp_feature(self,
                   fetaure_id:int,
                   scale_y:bool=True,
                   scale_x:bool=False):
        
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
        
        X = self.stat_test[:,fetaure_id].repeat(self.repetitions).reshape(-1,1)
        y = self.outputs[fetaure_id].flatten().reshape(-1,1)
        
        if scale_y:
            m_y = y[:self.stat_test_sample_size].mean()
            s_y = y[:self.stat_test_sample_size].std()
            y = (y-m_y)/s_y
        
        if scale_x:
            m_X = X[:self.stat_test_sample_size].mean()
            s_X = X[:self.stat_test_sample_size].std()
            X = (X-m_X)/s_X
            
        #scaler_X = StandardScaler().fit(self.stat_test[:,fetaure_id].repeat(self.repetitions).reshape(-1,1))
        #scaler_y = StandardScaler().fit(self.outputs[fetaure_id].flatten().reshape(-1,1))
        
        #X_scaled = scaler_X.transform(self.stat_test[:,fetaure_id].repeat(self.repetitions).reshape(-1,1))
        #y_scaled = scaler_y.transform(self.outputs[fetaure_id].flatten().reshape(-1,1)).ravel()
        
        #X = self.stat_test[:,fetaure_id].repeat(self.repetitions).reshape(-1,1)

        self.gp = gp = GaussianProcessRegressor(kernel=self.kernel)
        self.gp.fit(X, y)
        
        y_mean, y_std = gp.predict(X[:self.stat_test_sample_size], return_std=True)
        y_mean = torch.tensor(y_mean)
        y_std = torch.tensor(y_std)
        
        if scale_y:
            y_mean = (y_mean*s_y)+m_y
            y_std = y_std*s_y
            
        #y_mean_original_scale = scaler_y.inverse_transform(y_mean.reshape(-1, 1)).ravel()
        #y_std_original_scale = y_std * scaler_y.scale_
        
        return y_mean,y_std
    
    
