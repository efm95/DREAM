import torch

from tqdm import tqdm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

class DREAM_gp:
    def __init__(self,
                 models:tuple #tuple of DREAM models
                 ) -> None:
        self.models = models
        

    def train_repetitions(self,
                          x_event:torch.tensor,
                          x_non_event:torch.tensor,
                          stat_test:torch.tensor,
                          repetitions:int=5,
                          batch_size:int= 2**12,
                          gradient_clipping:float = 1.0,
                          bootstrap:bool=True,
                          bootstrap_sample_size:int=None,
                          model_verbose:bool=False,
                          conv_tol:int=6,
                          conv_jumps:int=15):
        """DREAM trainer used for uncertainty estimate using either non-parametric bootstrap or custom-based GP-process

        Args:
            models (tuple): a tuple containing the number of models that should be trained
            x_non_event (torch.tensor): torch.float32 containing the event covariates
            x_non_event (torch.tensor): torch.float32 containing the event covariates
            stat_test (torch.tensor): a torch.float32 tensor used to check the shape of the fitted curves -----> this wil change in the future
            repetitions (int): number of non-paraemtric bootsrap refits. Defaults to 5.
            batch_size (int): batch size. Defaults to 2**14.
            gradient_clipping (float, optional): loss function clipped. Defaults to 1.0.
            bootstrap (bool): trainig with non-parametric boostrap. Defaults to True.
            bootstrap_sample_size (int): bootstrap sample size. Defaults to None.
            model_verbose (bool): verbose. Defaults to False.
            conv_tol (int): convergence tolerance based on how many digits should the rounding be after the comma on the loss. Defaults to 6.
            conv_jumps (int): convergence tolerance parameter, updates are stopped when no improvement of loss after "conv_jumps" steps. Defaults to 15.
        """
        
        self.stat_test = stat_test
        self.repetitions = repetitions
        self.stat_test_sample_size = self.stat_test.shape[1]
        num_stat = x_event.shape[1]
        self.stat_test_sample_size = self.stat_test.shape[0]
        self.outputs = torch.zeros((num_stat,self.repetitions,self.stat_test_sample_size))
        
        if bootstrap_sample_size is None:
            bootstrap_sample_size:int=x_event.size()[0]
            
        
        if model_verbose:
            iterable = range(self.repetitions)
        else:
            iterable = tqdm(range(self.repetitions))
            
        for rep in iterable:
            #print(f'Repetition: {rep+1}')
            
            #Non-parametric bootsrap
            if bootstrap:
                torch.manual_seed(rep)
                data_indices = torch.randint(0, x_event.size()[0], (bootstrap_sample_size,))
            
                event_set=x_event[data_indices]
                non_event_set=x_non_event[data_indices]
            
            else:
                event_set = x_event
                non_event_set = x_non_event
            
            #Training
            self.models[rep].fit(events=event_set,
                            non_events=non_event_set,
                            batch_size=batch_size,
                            verbose=model_verbose,
                            gradient_clipping=gradient_clipping,
                            conv_tol=conv_tol,
                            conv_jumps=conv_jumps)
            for stat in range(num_stat):
                self.outputs[stat,rep,:] = self.models[rep].feature_out(stat,self.stat_test[:,stat])
    
    def bootsrap_output(self,
                     feature_id:int):
        """Boostrap output --- WORK IN PROGRESS

        Args:
            feature_id (int): feature index

        Returns:
            bootrsap repetitionos (torch.tensor): tensor of size (repetitions,len(stat_test))
        """
        return self.outputs[feature_id]
                
    def gp_feature(self,
                   fetaure_id:int,
                   scale_y:bool=True,
                   scale_x:bool=False):
        """Gaussian Process feature outpout

        Args:
            fetaure_id (int): covariate (feature index)
            scale_y (bool): Rescale y values. Defaults to True.
            scale_x (bool,): Rescale x values. Defaults to False.

        Returns:
            covariate estimated effect, covariate standard error
        """
        
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))+WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-20, 1e+10))
        
        x = self.stat_test[:,fetaure_id].repeat(self.repetitions).reshape(-1,1)
        y = self.outputs[fetaure_id].flatten().reshape(-1,1)
        
        if scale_y:
            m_y = y[:self.stat_test_sample_size].mean()
            s_y = y[:self.stat_test_sample_size].std()
            y = (y-m_y)/s_y
        
        if scale_x:
            m_x = x[:self.stat_test_sample_size].mean()
            s_x = x[:self.stat_test_sample_size].std()
            x = (x-m_x)/s_x

        self.gp = GaussianProcessRegressor(kernel=self.kernel,alpha=0.75**2)
        self.gp.fit(x, y)
        
        y_mean, y_std = self.gp.predict(x[:self.stat_test_sample_size], return_std=True)
        y_mean = torch.tensor(y_mean)
        y_std = torch.tensor(y_std)
        
        if scale_y:
            y_mean = (y_mean*s_y)+m_y
            y_std = y_std*s_y
        
        return y_mean,y_std
    
    #WORK IN PROGRESS 
    
    # def poinwise_gp_feature(self,
    #                         feature_id:int,
    #                         x:torch.tensor,
    #                         scale_y:bool=True,
    #                         scale_x:bool=False):
        
    #     y = torch.stack([model.feature_out(feature_id, x) for model in self.models])
    #     x = x.repeat(self.repetitions).reshape(-1,1)

    #     if scale_y:
    #         m_y = y.mean()
    #         s_y = y.std()
    #         y = (y-m_y)/s_y
        
    #     if scale_x:
    #         m_x = x.mean()
    #         s_x = x.std()
    #         x = (x-m_x)/s_x

    #     self.gp = GaussianProcessRegressor(kernel=self.kernel,alpha=0.75**2)
    #     self.gp.fit(x, y)
        
    #     y_mean, y_std = self.gp.predict(x[:self.stat_test_sample_size], return_std=True)
    #     y_mean = torch.tensor(y_mean)
    #     y_std = torch.tensor(y_std)
        
    #     if scale_y:
    #         y_mean = (y_mean*s_y)+m_y
    #         y_std = y_std*s_y
        
    #     return y_mean,y_std