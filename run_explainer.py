from xai import kernelshap
from sklearn.model_selection import train_test_split
import json
import numpy as np
from env.LP4_distance_berthing.auto_docking_vessel import Vessel
import tensorflow as tf

import pandas

if __name__ == "__main__":
    states_path = "output/trajectory_False.csv"
    actions_path = "output/actions.csv"
    model_path = "src/default_models/LP4_distance_berthing/model_ppo/_sv28/simple_save"

    df_raw_states = pandas.read_csv(states_path)
    df_states = df_raw_states[['eta_x', 'eta_y','eta_psi','nu_u','nu_v','nu_r','on_land','d_obs','psi_tilde_obs']]
    df_raw_actions = pandas.read_csv(actions_path)
    df_actions = df_raw_states[['f_1', 'f_2','f_3','a_1','a_2']]

    kernelshap.create_shap_values(df_states[0:1500],df_states[1501:1505],df_actions[0:1500])
    kernelshap.plot_shap_values(explainer.expected_value[0], shap_values[0], X_test)