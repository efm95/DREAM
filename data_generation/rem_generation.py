import numpy as np
from tqdm import tqdm
import logging
logging.basicConfig(format='%(asctime)s [%(filename)s] %(message)s',
    datefmt='%Y-%m-%d-%H:%M:%S',
    level=logging.INFO,
    encoding="utf-8")

class rem_generation_exogenous:
    def __init__(self,
                 n_actors:int,#n_actors= 10
                 n_events:int,#n_events= 100
                 base_rate:float = 0.1,
                 n_exo_stat:int = 1) -> None:
        """Rem generation exogenous

        Args:
            n_actors (int): Number of actors
            n_events (int): Number of events
            n_exo_stat (int): Number of exogenous statistics (if 1 --> 1 sender statistic and 1 receiver statistic,
                                                              if 3 --> 3 sender statistics and 3 receiver statistics, ...). 
                                                              Defaults to 1.
        """
                
        self.n_exo_stat = n_exo_stat
        
        self.n_actors = n_actors
        self.n_events = n_events
        self.base_rate = base_rate
        
        self.activity_basis_sender = list()
        self.activity_basis_receiver = list()
        
        self.activity_list_sender = list()
        self.activity_list_receiver = list()
        
        for i in range(n_exo_stat):
            samp_sender = np.random.uniform(0,1,size=self.n_actors)
            samp_receiver = np.random.uniform(0,1,size=self.n_actors)
            self.activity_basis_sender.append(samp_sender)
            self.activity_basis_receiver.append(samp_receiver)
            
            self.activity_list_sender.append(self.function_sender(samp_sender)) 
            self.activity_list_receiver.append(self.function_receiver(samp_receiver))
    
    def function_sender(self,x):#support between 0 and 1 
        return np.sin(2 * np.pi * x) + 0.5 * np.sin(4 * np.pi * x) + 0.25 * np.sin(8 * np.pi * x)
    
    def function_receiver(self,x):#support between 0 and 1
        return -np.sin(2*(4*x-2))-2*np.exp(-(16**2)*((x-0.5)**2))
    
    def generate(self):
        
        self.event_log = np.zeros((self.n_events,3+(2*self.n_exo_stat)))
        self.non_event_log = np.zeros((self.n_events,3+(2*self.n_exo_stat)))
        
                
        self.activity_list_sender = np.array(self.activity_list_sender)
        self.activity_list_receiver = np.array(self.activity_list_receiver)

        # Here, we sum over the actor-specific values for each activity type
        sender_activity_sum = np.sum(self.activity_list_sender, axis=0)
        receiver_activity_sum = np.sum(self.activity_list_receiver, axis=0)

        # Expand into a 2D matrix for each sender/receiver combination
        total_activity_sum = sender_activity_sum[:, None] + receiver_activity_sum[None, :]

        covariate_effects = np.exp(total_activity_sum)
        hazard_rates = self.base_rate * covariate_effects

        # Set hazard rate to zero for self-events
        np.fill_diagonal(hazard_rates, 0)
        current_time = 0
        
        total_rate = np.sum(hazard_rates)
        event_probabilities = hazard_rates / total_rate
        #event_cdf = np.cumsum(event_probabilities.reshape(-1))
        
        logging.info('Initialize REM generation')
        for event in tqdm(range(self.n_events)):
            # Draw next event time
            
            next_event_time = np.random.exponential(1 / total_rate,size=1)
            current_time += next_event_time
            
            # Draw which event occurs
            
            #Old version
            #rand = np.random.uniform(0,1,size=1)
            #event_index = np.searchsorted(event_cdf, rand)
            
            event_index = np.random.choice(self.n_actors*self.n_actors, size=1,p=event_probabilities.reshape(-1))
            
            sender = event_index // self.n_actors
            receiver = event_index % self.n_actors
            
            # Sample a non-event
            non_event_sender=None
            non_event_receiver=None
            while non_event_sender == non_event_receiver:
                non_event_index = np.random.choice(self.n_actors * self.n_actors)
                non_event_sender = non_event_index // self.n_actors
                non_event_receiver = non_event_index % self.n_actors
            
            # Update network log
            #self.network[sender,receiver] +=1
            #self.network_log[event] = self.network
            
            # Update event log and non-event log
            self.event_log[event,0] = sender ; self.event_log[event,1] = receiver ; self.event_log[event,2] = current_time
            self.non_event_log[event,0] = non_event_sender ; self.non_event_log[event,1] = non_event_receiver ; self.non_event_log[event,2] = current_time
            
            # sender statistics
            for pos in range(3,self.n_exo_stat+3):
                self.event_log[event,pos] = self.activity_basis_sender[pos-3][sender]
                self.non_event_log[event,pos] = self.activity_basis_sender[pos-3][non_event_sender]
            
            # receiver statistics
            for pos in range(self.n_exo_stat+3,(self.n_exo_stat+3)+self.n_exo_stat):
                self.event_log[event,pos] = self.activity_basis_receiver[pos-((self.n_exo_stat+3)+self.n_exo_stat)][receiver]
                self.non_event_log[event,pos] = self.activity_basis_receiver[pos-((self.n_exo_stat+3)+self.n_exo_stat)][non_event_receiver]