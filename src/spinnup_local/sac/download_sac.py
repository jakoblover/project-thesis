

import gym


import time
import joblib
import os
import os.path as osp
import tensorflow as tf
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph

import sys
sys.path.append("../..") # Adds higher directory to python modules path.

from src_2.download_policy_plot_tools import load_policy



def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):




    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    print("hei", n, num_episodes)
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)


        a = get_action(o)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1


        if d or (ep_len == max_ep_len):

            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1







folder = "sac_sv1"
main_path = "/home/ella-lovise/Dokumenter/master_auto_docking/spinnup_local/sac/model/"
export_dir = main_path  + folder + "/simple_save"
model_info_path = main_path + folder + "/simple_save/model_info.pkl"
model_env_path = main_path+folder+"/vars.pkl"

env = gym.make('HalfCheetah-v2')


# download environment
sess = tf.Session()

get_action = load_policy(sess,export_dir,model_info_path,env,deterministic = True,batch_normalization=False)

#env = gym.make("HalfCheetah-v2")

render = False
num_episodes = 100
max_ep_len = 1000

run_policy(env,get_action,max_ep_len = max_ep_len,num_episodes=num_episodes, render=render)
