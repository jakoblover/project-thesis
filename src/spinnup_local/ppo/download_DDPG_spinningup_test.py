import tensorflow as tf
import numpy as np
import os.path as osp, atexit, os
import joblib
import time
import gym


folder = "v1/v1_s0/"
root = "/home/ella-lovise/Dokumenter/overview_RL_spescialization_assignment/ppo_spinnup/model_ppo/"


export_dir = "/home/ella-lovise/Dokumenter/overview_RL_spescialization_assignment/ppo/model_ppo/v1/v1_s0/simple_save"
model_info_path = "/home/ella-lovise/Dokumenter/overview_RL_spescialization_assignment/ppo/model_ppo/v1/v1_s0/simple_save/model_info.pkl"
model_env_path ="/home/ella-lovise/Dokumenter/overview_RL_spescialization_assignment/ppo/model_ppo/v1/v1_s0/vars.pkl"

## Download neurla network



def load_policy(sess):

    tf.saved_model.loader.load(sess,["serve"],export_dir)


    model_info = joblib.load(model_info_path)

    graph = tf.get_default_graph()

    model = {}
    model.update({k:  graph.get_tensor_by_name(v) for k,v in model_info['inputs'].items()})
    model.update({k:  graph.get_tensor_by_name(v) for k,v in model_info['outputs'].items()})

    print(model)

    action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x: sess.run(action_op,feed_dict ={model['x']: x[None,:]})[0]

    state = joblib.load(model_env_path)
    env = state['env']
    return env,get_action

def run_policy(env,get_action,num_episodes, render):
    o,r,d,ep_ret,ep_len,n = env.reset(),0,False,0,0,0

    total_ep_r = 0

    while(n < num_episodes):
        if render:
            env.render()
            time.sleep(1e-3)
        a = get_action(o)
        o,r,d,_ = env.step(a)
        ep_ret += r
        ep_len +=1

        if d or (ep_len == max_ep_len):
            print('Episode %d \t EpRet %.3f \t EpLen %d' %(n,ep_ret,ep_len))
            o,r,d,ep_ret,ep_len = env.reset(),0,False,0,0
            n+=1
        total_ep_r += r

    print("average EpRet", total_ep_r/num_episodes)

# download environment
sess = tf.Session()

env,get_action = load_policy(sess)

#env = gym.make("HalfCheetah-v2")

render = True
num_episodes = 100
max_ep_len = 1000
render  = True
run_policy(env,get_action,num_episodes, render)
