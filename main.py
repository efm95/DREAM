import numpy as np 
import vaex as vx
import torch

from model.NeuREM import *

import logging

logging.basicConfig(format='%(asctime)s [%(filename)s] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO,
    encoding="utf-8")

repetitions = 100

path = '/Users/edoardofilippi-mazzola/Library/CloudStorage/Dropbox/Project_DREAM/effects/effects.hdf5'
df_original = vx.open(path)

df_original['rec_outd'] = df_original['rec_outd'].log()
df_original['rec_outd_2']=df_original['rec_outd_2'].log()

df_original['cumu_cit_rec'] = df_original['cumu_cit_rec'].log()
df_original['cumu_cit_rec_2'] = df_original['cumu_cit_rec_2'].log()

df_original['tfe'] = (df_original['tfe']+1).log()
df_original['tfe_2'] = (df_original['tfe_2']+1).log()


#FOR VISUALIZING EFFECTS 

pub_date = torch.tensor(range(1976,2022),dtype=torch.float32)
sim = torch.tensor(np.linspace(-0.2,1,100),dtype=torch.float32)
lag = torch.tensor(np.linspace(0,16688,100),dtype=torch.float32)
jac = torch.tensor(np.linspace(0,1,100),dtype=torch.float32)
cumu_cit_rec = torch.tensor(np.linspace(0,8.45,100),dtype=torch.float32)
tfe = torch.tensor(np.linspace(0,9.71,100),dtype=torch.float32)
rec_outd = torch.tensor(np.linspace(0,8.67,100),dtype=torch.float32)

pub_date_y = torch.zeros((repetitions,len(pub_date)))
sim_y = torch.zeros((repetitions,100))
lag_y = torch.zeros((repetitions,100))
jac_y = torch.zeros((repetitions,100))
cumu_cit_rec_y = torch.zeros((repetitions,100))
tfe_y = torch.zeros((repetitions,100))
rec_outd_y = torch.zeros((repetitions,100))

if __name__ == '__main__':
    
    for rep in range(repetitions):
        
        logging.info(f'Repetition: {rep+1}')
        df = df_original.sample(frac=1,random_state=rep)
        events = df['rec_pub_year','sim','lag','jac_sim','cumu_cit_rec','rec_outd','tfe']
        non_events = df['rec_pub_year_2','sim_2','lag_2','jac_sim_2','cumu_cit_rec_2','rec_outd_2','tfe_2']

        events = np.array(events.to_arrays()).transpose()
        non_events = np.array(non_events.to_arrays()).transpose()

        events = torch.tensor(events).type(torch.float32)
        non_events = torch.tensor(non_events).type(torch.float32)
        
        ev_mean = torch.mean(events,dim=0)
        ev_std = torch.std(events,dim=0)
        
        events = ((events-ev_mean)/ev_std)
        non_events = ((non_events-ev_mean)/ev_std)
        
        model = NeuREM(7,[256]*7,[256,64,32])
        model.fit(events=events,non_events=non_events,verbose=False)
        
        #Publication year
        pub_date_test = (pub_date-ev_mean[0])/ev_std[0]
        pub_date_y[rep,:] = model.feature_out(nn_id=0,input=pub_date_test)
        
        #Textual similarity
        sim_test = ((sim-ev_mean[1])/ev_std[1])
        sim_y[rep,:] = model.feature_out(nn_id=1,input=sim_test)
        
        #Time lag
        lag_test = (lag-ev_mean[2])/ev_std[2]
        lag_y[rep,:] = model.feature_out(nn_id=2,input=lag_test)
        
        #Jaccard similarity
        jac_test = (jac-ev_mean[3])/ev_std[3]
        jac_y[rep,:] = model.feature_out(nn_id=3,input=jac_test)
        
        #Cumulative citations received
        cumu_cit_rec_test = (cumu_cit_rec-ev_mean[4])/ev_std[4]
        cumu_cit_rec_y[rep,:]= model.feature_out(nn_id=4,input=cumu_cit_rec_test)
        
        #Receiver outdegree
        rec_outd_test = (rec_outd-ev_mean[5])/ev_std[5]
        rec_outd_y[rep,:] = model.feature_out(nn_id=5,input=rec_outd_test)
        
        #Time from last event
        tfe_test = (tfe-ev_mean[6])/ev_std[6]
        tfe_y[rep,:]= model.feature_out(nn_id=6,input=tfe_test)
        

out = {'mean':ev_mean,
       'std':ev_std,
       'pub_date':pub_date_y,
       'sim':sim_y,
       'lag':lag_y,
       'jac':jac_y,
       'cumu_cit':cumu_cit_rec_y,
       'rec_outd':rec_outd_y,
       'tfe':tfe_y}

torch.save(out,'nam_out_rep100.pt')

