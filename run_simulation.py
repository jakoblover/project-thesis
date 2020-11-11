import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json

from env.LP4_distance_berthing.auto_docking_vessel import Vessel, rad2deg
from spinnup_local.download_policy_plot_tools import *
from video.create_video import make_video

from plot_env import Vessel_plot

import random

from xai import kernelshap

random.seed()
import pandas as pd
from collections import namedtuple


def save_state(env_plot, ep_len, save_path=""):
    df = pd.DataFrame()
    df["eta_x"] = env_plot.eta[0, :ep_len]
    df["eta_y"] = env_plot.eta[1, :ep_len]
    df["eta_psi"] = env_plot.eta[2, :ep_len]
    df["nu_u"] = env_plot.nu[0, :ep_len]
    df["nu_v"] = env_plot.nu[1, :ep_len]
    df["nu_r"] = env_plot.nu[2, :ep_len]

    column_names = ["x_tilde", "y_tilde", "on_land", "u", "v", "r",
                    "d_obs", "psi_tilde_obs", "psi_tilde"]

    for i in range(len(column_names)):
        df[column_names[i]] = env_plot.x[i, :ep_len]

    df.to_csv(save_path)


def save_actions(env_plot, ep_len, save_path):
    df = pd.DataFrame()
    df["f_1"] = env_plot.f[0, :ep_len]
    df["f_2"] = env_plot.f[1, :ep_len]
    df["f_3"] = env_plot.f[2, :ep_len]

    df["a_1"] = env_plot.alpha[0, :ep_len]
    df["a_2"] = env_plot.alpha[1, :ep_len]

    df.to_csv(save_path)


def run_policy(env, get_action, num_episodes, normalization_obs=False, stats=None, u_init=None, video=False,
               save_trajectory=False, save_fig=False, save_figure_path="", max_ep_len=1500,
               eta_init=None, nu_init=None, save_trajectory_path="", save_video_path="", save_actions_path=""):
    o, r, d, ep_ret, ep_len, n = env.reset(u_init=u_init), 0, False, 0, 0, 0

    if (eta_init is not None and nu_init is not None):
        o = env.reset(eta_init, nu_init)

    total_ep_r = 0

    env_plot = Vessel_plot(max_ep_len, env=env)

    while (n < num_episodes):

        if (normalization_obs):
            o = stats.normalize(o)

        a = get_action(o)

        if ep_len == 0:
            print("start pos", env.eta.reshape(3), env.nu.reshape(3))

        o, r, d, _ = env.step(a)

        env_plot.update_state(o, a, env, ep_len)

        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):

            if (video):
                make_video(env_plot, 1200, save_path=save_video_path, area=False)

            env_plot.plot_episode(ep_len, folder=save_figure_path, savefig=save_fig)

            if (save_trajectory):
                save_state(env_plot, ep_len, save_trajectory_path)
            if(save_actions_path):
                save_actions(env_plot, ep_len, save_actions_path)

            print('Episode %d \t EpRet %.3f  \t EpLen %d' % (n, ep_ret, ep_len))
            print("env n episodes", env.n_episodes)

            print('EpRet %.0f' % (ep_ret))
            x_tilde = env_plot.x[0, :ep_len]
            y_tilde = env_plot.x[1, :ep_len]
            if (env.state_augmentation):
                y_tilde -= env_plot.y_tilde_ss[:ep_len]
                x_tilde -= env_plot.x_tilde_ss[:ep_len]
            d_d = np.sqrt(x_tilde ** 2 + y_tilde ** 2)
            print("mean d_d,psi_tilde (%.3f,  %.3f)" % (
                np.mean(d_d), np.mean(np.abs(env_plot.x[8, :]))))
            print("min d_d,psi_tilde (%.3f,  %.3f)" % (
                np.min(d_d), np.min(np.abs(env_plot.x[8, :]))))

            print("mean alpha_dot,mean  f_dot (%.3f,  %.3f)" % (
                np.mean(np.sum(np.abs(env_plot.alpha_dot_norm), axis=0)),
                np.mean(np.sum(np.abs(env_plot.f_dot_norm), axis=0))))
            print(np.sum(np.abs(env_plot.f_dot_norm), axis=0))

            o, r, d, ep_ret, ep_len = env.reset(u_init=u_init), 0, False, 0, 0
            n += 1
            env_plot.reset()

            plt.show()
        total_ep_r += r

    print("average EpRet", total_ep_r / num_episodes)


if __name__ == "__main__":

    root_path = "src/default_models/LP4_distance_berthing/model_ppo/"
    folder = "_sv28"
    export_dir = root_path + folder + "/simple_save"
    model_info_path = root_path + folder + "/simple_save/model_info.pkl"
    env_args_path = root_path + folder + "/arguments_env.txt"
    ddpg_args_path = root_path + folder + "/arguments_model.txt"
    model_env_path = root_path + folder + "/vars.pkl"
    stats_path = root_path + folder + "/stats.txt"

    save_fig = False
    save_trajectory = True
    video = True
    #
    save_path = "output"
    save_figure_path = save_path + "/simulation/LP4_ppo_" + str(save_fig)
    save_trajectory_path = save_path + "/trajectory_" + str(save_fig) + ".csv"
    save_actions_path = save_path + "/actions.csv"
    save_video_path = save_path + "/video.mkv"
    save_shap_values_path = save_path + "/explanations/SHAP"

    with open(env_args_path) as json_file:
        env_args = json.load(json_file)

    with open(stats_path) as json_file:
        stats_args = json.load(json_file)

    print(stats_args)

    print(env_args)
    print(folder)

    env_args["x_d"] = 800
    env_args["y_d"] = 517.8

    env_args["backward_psi"] = False
    env_args["state_augmentation"] = False
    env_args["current"] = False
    env_args["beta_c"] = 45 * np.pi / 180
    env_args["U_c"] = 0.15

    if env_args["current"]:
        base_name += "current_"

    env = Vessel(**env_args)

    stats = NormalizeStats(stats_args, env.observation_space.shape[0])
    u_init = None
    num_episodes = 1
    max_ep_len = 2500
    # video = False
    batch_normalization = False
    determenistic = False
    normalization_obs = True
    n_hidden = 2

    eta_init, nu_init = None, None
    eta_init, nu_init = np.array([312.183, 876.256, -0.384]), np.array([0.3, 0, 0])

    sess = tf.Session()
    get_action = load_policy(sess, export_dir, model_info_path, env, deterministic=determenistic, batch_normalization=batch_normalization, n_hidden=n_hidden)

    #run_policy(env, get_action, num_episodes, normalization_obs=normalization_obs, u_init=u_init, video=video,
    #           stats=stats, save_trajectory=save_trajectory, max_ep_len=max_ep_len,
    #           save_fig=save_fig, save_figure_path=save_figure_path, eta_init=eta_init, nu_init=nu_init,
    #           save_trajectory_path=save_trajectory_path,save_video_path=save_video_path,save_actions_path=save_actions_path)

    states_path = "output/trajectory_False.csv"
    actions_path = "output/actions.csv"
    model_path = "src/default_models/LP4_distance_berthing/model_ppo/_sv28/simple_save"

    df_raw_states = pd.read_csv(states_path)
    state_vector = ['x_tilde', 'y_tilde', 'psi_tilde', 'nu_u', 'nu_v', 'nu_r', 'on_land', 'd_obs', 'psi_tilde_obs']
    df_states = df_raw_states[state_vector]
    df_raw_actions = pd.read_csv(actions_path)
    actions_vector = ['f_1', 'f_2', 'f_3', 'a_1', 'a_2']
    df_actions = df_raw_actions[actions_vector]

    #X_train = df_states[0:1500]
    #X_test = df_states[2000:2002]
    explainer, shap_values = kernelshap.create_shap_values(df_states, df_states, get_action,actions_vector,state_vector,save_shap_values_path)

    #kernelshap.plot_shap_values(explainer.expected_value[0], shap_values[0], X_test, "index.htm")


    exit()
