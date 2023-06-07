import numpy as np
from tqdm import tqdm
import logging
logging.basicConfig(format='%(asctime)s [%(filename)s] %(message)s',
    datefmt='%Y-%m-%d-%H:%M:%S',
    level=logging.INFO,
    encoding="utf-8")

class rem_generation:
    def __init__(self,
                 n_actors:int,#n_actors= 10
                 n_events:int,#n_events= 100
                 base_rate:float = 0.1,
                 sparsity:float = 1.0,
                 n_exo_stat:int = 3,
                 activity_types:list = ['wave','wave','wave']) -> None:
        
        assert n_exo_stat == len(activity_types)
        
        self.n_exo_stat = n_exo_stat
        
        self.n_actors = n_actors
        self.n_events = n_events
        self.base_rate = base_rate
        
        self.activity_basis = list()
        self.activity_list = list()
        
        for i in range(len(activity_types)):
            activity = activity_types[i]
            if activity == 'wave':
                samp = np.random.uniform(0,1,size=self.n_actors)
                self.activity_basis.append(samp)
                self.activity_list.append(self.wave_function(samp)+np.random.normal(0,sparsity,size=self.n_actors))
            else:
                self.activity_list.append(np.random.normal(0,1,size=self.n_actors))
    
    def wave_function(self,x):
        return np.sin(2*(4*x-2))+2*np.exp(-(16**2)*((x-0.5)**2))
    
    def generate(self):
        
        self.event_log = np.zeros((self.n_events,3+(2*self.n_exo_stat)))
        self.non_event_log = np.zeros((self.n_events,3+(2*self.n_exo_stat)))
        
        #self.network = np.zeros((self.n_actors,self.n_actors))
        #self.network_log = np.zeros((self.n_events,self.n_actors,self.n_actors))
        
        
        # Initialize hazard rates for all possible events
        
        # OLD VERSION OF THE CODE
        # hazard_rates = np.full((self.n_actors, self.n_actors), self.base_rate)
        # for sender in range(self.n_actors):
        #     for receiver in range(self.n_actors):
        #         covariate_effect = np.exp(np.sum(self.activity_list[i][sender] for i in range(self.n_exo_stat)) + np.sum(self.activity_list[i][receiver] for i in range(self.n_exo_stat)))
        #         hazard_rates[sender, receiver] = self.base_rate * covariate_effect
                
                
        self.activity_list = np.array(self.activity_list)

        # Here, we sum over the actor-specific values for each activity type
        sender_activity_sum = np.sum(self.activity_list, axis=0)
        receiver_activity_sum = np.sum(self.activity_list, axis=0)

        # Expand into a 2D matrix for each sender/receiver combination
        total_activity_sum = sender_activity_sum[:, None] + receiver_activity_sum[None, :]

        covariate_effects = np.exp(total_activity_sum)
        hazard_rates = self.base_rate * covariate_effects

        # Set hazard rate to zero for self-events
        np.fill_diagonal(hazard_rates, 0)
        current_time = 0
        
        total_rate = np.sum(hazard_rates)
        event_probabilities = hazard_rates / total_rate
        event_cdf = np.cumsum(event_probabilities.reshape(-1))
        
        logging.info('Initialize REM generation')
        for event in tqdm(range(self.n_events)):
            # Draw next event time
            
            next_event_time = np.random.exponential(1 / total_rate,size=1)
            current_time += next_event_time
            
            # Draw which event occurs
            rand = np.random.uniform(0,1,size=1)
            event_index = np.searchsorted(event_cdf, rand)
            
            sender = event_index // self.n_actors
            receiver = event_index % self.n_actors
            
            # Sample a non-event
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
                self.event_log[event,pos] = self.activity_basis[pos-3][sender]
                self.non_event_log[event,pos] = self.activity_basis[pos-3][non_event_sender]
            
            # receiver statistics
            for pos in range(self.n_exo_stat+3,(self.n_exo_stat+3)+self.n_exo_stat):
                self.event_log[event,pos] = self.activity_basis[pos-((self.n_exo_stat+3)+self.n_exo_stat)][receiver]
                self.non_event_log[event,pos] = self.activity_basis[pos-((self.n_exo_stat+3)+self.n_exo_stat)][non_event_receiver]
                
        #return self.event_log, self.non_event_log, self.network_log
