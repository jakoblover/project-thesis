import numpy as np
import tensorflow as tf
import gym
import time
#from spinup.algos.ddpg import core


from spinnup_local.ddpg import core
from spinnup_local.ddpg.core import get_vars
#from spinup.algos.ddpg.core import get_vars
from spinnup_local.utils.logx import  EpochLogger
#from spinup.utils.logx import  EpochLogger
import os.path as osp
import joblib
import random
import copy
from runstats import Statistics
import json
import pandas as pd

def get_prev_params(path,transfer_scope = "main"):

    #path = "/home/ella-lovise/Dokumenter/overview_RL_spescialization_assignment/psi_tilde_controller_sin/model/ddpg_sv2_b"

    model_path = path + "/simple_save"
    model_info_path = path + "/simple_save/model_info.pkl"

    # fetch trained values
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph = graph) as sess:
            tf.saved_model.loader.load(sess,["serve"],model_path)

            params_tensor = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=transfer_scope)

            params_value = tf.get_default_session().run(params_tensor)

    return params_value

def set_param_values(sess,param_values,scope):
    params_tensor = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)

    assign_plc = [tf.placeholder(tf.float32,shape=param.get_shape(),name="assign_%s" %param.name.replace('/','_').replace(':','_')) for param in params_tensor]
    assign_ops = [tf.assign(params_tensor[i],assign_plc[i]) for i in range(len(params_tensor))]

    print(tf.get_default_session())
    sess.run(assign_ops,feed_dict={assign_plc[i]:param_values[i] for i in range(len(params_tensor))})

    ##
    #model_info = joblib.load(model_info_path)

    #model = {}
    #model.update({k:  graph.get_tensor_by_name(v) for k,v in model_info['inputs'].items()})
    #model.update({k:  graph.get_tensor_by_name(v) for k,v in model_info['outputs'].items()})

    """pi = model['pi']
    q = model['q']
    q_pi = model['q_pi']
    pi_targ = model['pi_targ']
    q_pi_targ = model['q_pi_targ']

    print(pi)
    print(q )
    print(q_pi )
    print(pi_targ)
    print(q_pi_targ)


    return pi,q,q_pi,pi_targ, q_pi_targ"""




class NormalizeStats():
    """docstring forNormalizeObs."""
    def __init__(self, size,path):
        self.stats = [Statistics() for num in range(size)]
        self.size = size
        self.path = path
        self.mean = np.zeros(size, 'float64')
        self.std = np.ones(size, 'float64')


    def __del__(self):
        self.update_mean_std()
        self.save()

    def save(self):
        data = {}
        path = self.path
        for i in range(self.size):
            data[i] = {"mean": self.mean.item(i),"std":self.std.item(i)}
            #print('Index', i, 'mean:', self.mean.item(i), "std: ",self.std.item(i))
        if(path == ""):
            return

        with open(path,'w') as json_file:
            json.dump(data,json_file)


    def update(self,o):
        if(self.size == 1):
            self.stats[0].push(o)
        else:
            for i in range(self.size):
                self.stats[i].push(o.item(i))

    def update_mean_std(self):
        for i in range(self.size):
            self.mean[i] = self.stats[i].mean()
            self.std[i] = self.stats[i].stddev()


    def normalize(self,o):
        self.update_mean_std()
        if(self.size == 1):
            o = normalize(o,self.mean.item(0), self.std.item(0))
        else:
            for i in range(self.size):
                o[i] = normalize(o.item(i),self.mean.item(i), self.std.item(i))
        return o


def normalize(x, mean,std):
    if std == 0:
        return x - mean
    return (x - mean )/ std

class NormalActionNoise():
    def __init__(self,size, mu = 0.0, sigma = 1.0,noise_scale = 0.1):
        self.mu = mu
        self.sigma = sigma
        self.size = size
        self.noise_scale = noise_scale

    def sample(self):
        # times 1/sigma to normalize the output
        return self.noise_scale*np.random.normal(self.mu, self.sigma,self.size)



# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise():
    def __init__(self, size, mu=0., sigma=0.2,theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.size)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size,path):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.path = path

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):

        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


    def save(self):
        df = pd.DataFrame()
        for i in range(self.obs1_buf.shape[1]):
            df["obs1_"+str(i)] = self.obs1_buf[:self.ptr,i]
            df["obs2_" + str(i)] = self.obs2_buf[:self.ptr, i]

        for i in range(self.acts_buf.shape[1]):
            df["a_"+str(i)] = self.acts_buf[:self.ptr,i]

        df["r"] = self.rews_buf[:self.ptr]
        df["d"] = self.done_buf[:self.ptr]

        #print(df.head())

        df.to_csv(self.path)

    #def __del__(self):
    #    self.save()



"""

Deep Deterministic Policy Gradient (DDPG)

"""
def ddpg(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99,
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
         max_ep_len=1000, logger_kwargs=dict(), save_freq=1, noise_type = "norm",update_each_time = 0,
         stats_path = "stats.txt",batch_normalization = False,reward_normalization = False,transfer_learning_path = None,
         transfer_scope = "main",replay_buffer_path="a.csv",
         obs_normalization  = False):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols
            for state, ``x_ph``, and action, ``a_ph``, and returns the main
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ``q``        (batch,)          | Gives the current estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q_pi``     (batch,)          | Gives the composition of ``q`` and
                                           | ``pi`` for states in ``x_ph``:
                                           | q(x, pi(x)).
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        act_noise (float): Stddev for Gaussian exploration noise added to
            policy at training time. (At test time, no noise is added.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)



    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    #act_limit = env.action_space.high[0]


    #OUnoise
    if(noise_type == "OUnoise"):
        print("ounoise")
        noise = OrnsteinUhlenbeckActionNoise(act_dim)
    else:
        print("normal noise")
        noise =  NormalActionNoise(act_dim)

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

    if(batch_normalization):
        phase = tf.placeholder(tf.bool,name = 'phase')
    else:
        phase = None



    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size,path = replay_buffer_path)
    main_scope = 'main'
    target_scope = 'target'


    # Main outputs from computation graph
    with tf.variable_scope(main_scope):
        print(phase)
        pi, q, q_pi = actor_critic(x_ph, a_ph,phase = phase, **ac_kwargs)

    # Target networks
    with tf.variable_scope(target_scope):
        # Note that the action placeholder going to actor_critic here is
        # irrelevant, because we only need q_targ(s, pi_targ(s)).
        pi_targ, _, q_pi_targ  = actor_critic(x2_ph, a_ph,phase = phase, **ac_kwargs)


    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in [main_scope+'/pi', main_scope+'/q',main_scope+'/q_pi', main_scope])
    print('\nmain Number of parameters: \t pi: %d, \t , q: %d, \t ,q_pi: %d, \t total: %d\n'%var_counts)

    var_counts = tuple(core.count_vars(scope) for scope in [target_scope +'/pi',target_scope+ '/q',target_scope +'/q_pi', target_scope])
    print('\ntarget Number of parameters: \t pi: %d, \t ,q: %d, \t ,q_pi: %d, \t total: %d\n'%var_counts)

    # Bellman backup for Q function
    backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*q_pi_targ)

    # DDPG losses
    pi_loss = -tf.reduce_mean(q_pi)
    q_loss = tf.reduce_mean((q-backup)**2)


    if(batch_normalization):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            #ensure that we execute the update_ops before performing the train step:
            pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
            q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
            train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars(main_scope+'/pi'))
            train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars(main_scope +'/q'))
    else:
        # Separate train ops for pi, q
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
        q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
        train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars(main_scope+'/pi'))
        train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars(main_scope+'/q'))

    # Polyak averaging for target variables
    target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                              for v_main, v_targ in zip(get_vars(main_scope), get_vars(target_scope))])



    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars(main_scope), get_vars(target_scope))])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())




    if (transfer_learning_path):

        """vars = tf.trainable_variables()
        print(vars)  # some infos about variables...
        vars_vals = sess.run(vars)
        for var, val in zip(vars, vars_vals):
            if(var.name == "main/q/dense/bias:0"):
                print("var: {}, value: {}".format(var.name, val))
            if (var.name == "main/pi/dense/kernel:0"):
                print("var: {}, value: {}".format(var.name, val))
        print("transfer_learning_path ", transfer_learning_path)"""
        param_values = get_prev_params(transfer_learning_path,transfer_scope)
        set_param_values(sess,param_values,transfer_scope)


        """vars = tf.trainable_variables()
        print(vars)  # some infos about variables...
        vars_vals = sess.run(vars)
        for var, val in zip(vars, vars_vals):
            if(var.name == "main/q/dense/bias:0"):
                print("var: {}, value: {}".format(var.name, val))
            if (var.name == "main/pi/dense/kernel:0"):
                print("var: {}, value: {}".format(var.name, val))
        print("finished")
        exit()"""



    sess.run(target_init)
    if(batch_normalization):
        logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph,'phase':phase},
                              outputs={'pi': pi, 'q': q, 'q_pi':q_pi, 'pi_targ':pi_targ,'q_pi_targ':q_pi_targ})
    else:
        logger.setup_tf_saver(sess, inputs={'x':x_ph, 'a':a_ph},
                              outputs={'pi':pi, 'q':q, 'q_pi':q_pi, 'pi_targ':pi_targ, 'q_pi_targ':q_pi_targ})

    print("finish init")

    def scale_action(a):
        act_mean = (env.action_space.high + env.action_space.low)/2
        act_scale = env.action_space.high - act_mean
        a = act_scale*a + act_mean
        #a += noise_scale *noise.sample()
        return np.clip(a, env.action_space.low, env.action_space.high)


    def get_action(o, noise_scale):
        #act_dim = env.action_space.shape[0]
        #act_limit = env.action_space.high[0]
        if(batch_normalization and noise_scale > 0):
            feed_dict = {x_ph: o.reshape(1,-1),phase:0}
        else:
            feed_dict = {x_ph: o.reshape(1,-1)}
        a = sess.run(pi, feed_dict=feed_dict)[0]
        a += noise_scale *noise.sample()
        return a


    def test_agent(n=10):
        for j in range(n):
            try:
                o, r, d, ep_ret, ep_len = test_env.reset(test =True), 0, False, 0, 0
            except:
                o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                if (not batch_normalization or obs_normalization):
                    o = stats.normalize(o)
                if (reward_normalization):
                    r = stats_reward.normalize(r)
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(scale_action(get_action(o, 0)))
                ep_ret += r
                ep_len += 1
                stats.update(o)
                stats_reward.update(r)

            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    total_steps = steps_per_epoch * epochs

    stats = NormalizeStats(env.observation_space.shape[0], stats_path)
    stats_reward = NormalizeStats(1, "")

    for i in range(10):
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        stats.update(o)
        stats_reward.update(r)

    if (not batch_normalization or obs_normalization):
        o = stats.normalize(o)
    if (reward_normalization):
        r = stats_reward.normalize(r)

    n_episodes = 0
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards,
        use the learned policy (with some noise, via act_noise).
        """
        if t > start_steps:
            a = get_action(o, 1)
        else:
            a = np.random.uniform(size=act_dim).reshape(act_dim)

        # Step the env
        o2, r, d, _ = env.step(scale_action(a))
        ep_ret += r
        ep_len += 1



        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        #normalize observation and reward
        stats.update(o2)
        stats_reward.update(r)
        if(not batch_normalization or obs_normalization):
            o2 = stats.normalize(o2)

        if(reward_normalization):
            r = stats_reward.normalize(r)

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        if(update_each_time and t%update_each_time == 0):
            batch = replay_buffer.sample_batch(batch_size)

            if(batch_normalization):
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done'],
                             phase: 1
                            }
            else:
                feed_dict = {x_ph:batch['obs1'],
                             x2_ph:batch['obs2'],
                             a_ph:batch['acts'],
                             r_ph:batch['rews'],
                             d_ph:batch['done'],
                             }


            # Q-learning update
            outs = sess.run([q_loss, q, train_q_op], feed_dict)
            logger.store(LossQ=outs[0], QVals=outs[1])

            # Policy update
            outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)
            logger.store(LossPi=outs[0])



        if d or (ep_len == max_ep_len):
            """
            Perform all DDPG updates at the end of the trajectory,
            in accordance with tuning done by TD3 paper authors.
            """
            if( not update_each_time):
                for _ in range(ep_len):
                    batch = replay_buffer.sample_batch(batch_size)
                    if (batch_normalization):
                        feed_dict = {x_ph:batch['obs1'],
                                     x2_ph:batch['obs2'],
                                     a_ph:batch['acts'],
                                     r_ph:batch['rews'],
                                     d_ph:batch['done'],
                                     phase:1
                                     }
                    else:
                        feed_dict = {x_ph:batch['obs1'],
                                     x2_ph:batch['obs2'],
                                     a_ph:batch['acts'],
                                     r_ph:batch['rews'],
                                     d_ph:batch['done'],
                                     }

                    # Q-learning update
                    outs = sess.run([q_loss, q, train_q_op], feed_dict)
                    logger.store(LossQ=outs[0], QVals=outs[1])

                    # Policy update
                    outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)
                    logger.store(LossPi=outs[0])

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n_episodes += 1

            stats.update(o)
            stats_reward.update(r)
            if (not batch_normalization or obs_normalization):
                o = stats.normalize(o)

            if (reward_normalization):
                r = stats_reward.normalize(r)

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, None)
                stats.save()

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.log_tabular('N episodes', n_episodes)
            logger.dump_tabular()








