#!/usr/bin/env python3
import sys
import numpy as np
sys.path = ['../..'] + sys.path
from dfp import expt_common

simulator_args = vars(expt_common.run_exp_parse_args())

simulator_args.update({
    "config" : '../../maps/D3_battle.cfg',
    "resolution" : (84,84),
    "frame_skip" : 4,
    "color_mode" : 'GRAY',	
    "use_shaping_reward" : False,
    "maps" : ['MAP01'],
    "switch_maps" : False,
    "logdir" : "logs",
    "test_checkpoint" : 'checkpoints/2017_04_09_09_07_45',
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
    'modalities': ['color', 'measurements'],
    'preprocess_input_targets': lambda x: x / targ_scale_coeffs,
    'postprocess_predictions': lambda x: x * targ_scale_coeffs,
    'objective_coeffs_meas': np.array([1, 0.5, 0.5])
}

train_experience_args = {
    'default_history_length': 1,
    'history_step': 1,
    'history_lengths': {'actions': 12}
}

expt_common.start_experiment(globals())
