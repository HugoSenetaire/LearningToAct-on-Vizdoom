#!/usr/bin/env python3
import sys
import numpy as np
sys.path = ['../..'] + sys.path
from dfp import expt_common

simulator_args = vars(expt_common.run_exp_parse_args())

simulator_args.update({
    "config" : '../../maps/D3_battle.cfg',
    "resolution" : (64,64),
    "frame_skip" : 4,
    "color_mode" : 'GRAY',	
    "use_shaping_reward" : False,
    "maps" : ['MAP01'],
    "switch_maps" : False,
    "logdir" : "logs",
    "test_checkpoint" : 'checkpoints/logs',
    "simulator" : "doom",
    "game_args" : "",
    "gpu": True,
    # i'm not sure why we use 8 simulators here
    "num_simulators" : 8,
})
## Simulator


target_maker_args = {
    'future_steps': [1, 2, 4, 8, 16, 32],
    # 'meas_to_predict': list(range(simulator_args['measure_fun'].num_meas)),
    'meas_to_predict': [0,1,2],
}
targ_scale_coeffs = np.expand_dims((np.expand_dims(np.array([7.5,30.,1.]),1) * np.ones((1,len(target_maker_args['future_steps'])))).flatten(),0)

# targ_scale_coeffs = np.expand_dims((np.expand_dims(np.array([1., 1., 1., 1.]), 1)
                                    # * np.ones((1, len(target_maker_args['future_steps'])))).flatten(), 0)

agent_args = {
    'modalities': ['color', 'measurements', 'depth', "segEnnemies", "segMedkit"],
    'preprocess_input_targets': lambda x: x / targ_scale_coeffs,
    'postprocess_predictions': lambda x: x * targ_scale_coeffs,
    'objective_coeffs_meas': np.array([1, 0.5, 0.5]),
    'infer_modalities': ["segEnnemies","segMedkit"],
    'start_learningtoact_training_iter' : 10000,
    'freezeType' : "None" ,
    #Possibility are "None" : No Freeze; "Partial" : Only the end is trained; "All": Everything freezed 
    # 'coefs_loss' : [(0.,1.0),(0.8,0.2)], # coefs_loss[0] = (target_loss,infer_loss) before training learning to act
    'coefs_loss' : [0.,0.5], # coefs_loss [0] : coefficient for learning to act when not learning to act; coef[1] after
    'test_inference_every' : 500,
    'change_inference_dataset_every':10000,
}
agent_args['unet_params'] = np.array([(4,2,2), (8,2,2), (16,2,2)],
									 dtype = [('out_channels',int), ('kernel',int), ('stride',int)])

train_experience_args = {
    'default_history_length': 1,
    'history_step': 1,
    'history_lengths': {'actions': 12}
}

expt_common.start_experiment(globals())
