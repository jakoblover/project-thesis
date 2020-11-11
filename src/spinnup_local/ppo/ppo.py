import numpy as np
import tensorflow as tf
import gym
import time


import spinnup_local.ppo.core as core
from spinnup_local.utils.logx import EpochLogger
from spinnup_local.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinnup_local.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

from runstats import Statistics
import json

def get_prev_params(path,transfer_scope = None):

    #path = "/home/ella-lovise/Dokumenter/overview_RL_spescialization_assignment/psi_tilde_controller_sin/model/ddpg_sv2"

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
            data[i] = {"mean":self.mean.item(i), "std":self.std.item(i)}
            #print('Index', i, 'mean:', self.mean.item(i), "std: ", self.std.item(i))
        if (path == ""):
            return

        with open(path, 'w') as json_file:
            json.dump(data, json_file)




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


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf,
                self.ret_buf, self.logp_buf]


"""

Proximal Policy Optimization (by clipping),

with early stopping based on approximate KL

"""
def ppo(env, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10,stats_path ="stats.txt",normalize = False,
        transfer_learning_path = None,
        transfer_scope = "main"):
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
            ``pi``       (batch, act_dim)  | Samples actions from policy given
                                           | states.
            ``logp``     (batch,)          | Gives log probability, according to
                                           | the policy, of taking actions ``a_ph``
                                           | in states ``x_ph``.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``.
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. (Critical: make sure
                                           | to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)



    #env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape



    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
    adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

    # Main outputs from computation graph
    pi, logp, logp_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi]

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # PPO objectives
    ratio = tf.exp(logp - logp_old_ph)          # pi(a|s) / pi_old(a|s)
    min_adv = tf.where(adv_ph>0, (1+clip_ratio)*adv_ph, (1-clip_ratio)*adv_ph)
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
    v_loss = tf.reduce_mean((ret_ph - v)**2)

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - logp)      # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-logp)                  # a sample estimate for entropy, also easy to compute
    clipped = tf.logical_or(ratio > (1+clip_ratio), ratio < (1-clip_ratio))
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

    # Optimizers
    train_pi = MpiAdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    train_v = MpiAdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    if (transfer_learning_path):
        """vars = tf.trainable_variables()
        print(vars)  # some infos about variables...
        vars_vals = sess.run(vars)
        for var, val in zip(vars, vars_vals):
            if(var.name == "v/dense/kernel:0"):
                print("var: {}, value: {}".format(var.name, val[0]))
            if (var.name == "pi/dense/kernel:0"):
                print("var: {}, value: {}".format(var.name, val[0]))
        print("transfer_learning_path ", transfer_learning_path)"""
        param_values = get_prev_params(transfer_learning_path,transfer_scope)
        set_param_values(sess,param_values,transfer_scope)

        """vars = tf.trainable_variables()
        print(vars)  # some infos about variables...
        vars_vals = sess.run(vars)
        for var, val in zip(vars, vars_vals):
            if(var.name == "v/dense/kernel:0"):
                print("var: {}, value: {}".format(var.name, val[0]))
            if (var.name == "pi/dense/kernel:0"):
                print("var: {}, value: {}".format(var.name, val[0]))
        print("finished")
        exit()"""

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x':x_ph}, outputs={'pi':pi, 'v':v})

    def update():
        inputs = {k:v for k,v in zip(all_phs, buf.get())}
        pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)

        # Training
        for i in range(train_pi_iters):
            _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
            kl = mpi_avg(kl)
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
        logger.store(StopIter=i)
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, v_l_new, kl, cf = sess.run([pi_loss, v_loss, approx_kl, clipfrac], feed_dict=inputs)
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))




    stats = NormalizeStats(env.observation_space.shape[0],stats_path)
    stats_reward = NormalizeStats(1,"")
    for i in range(10):
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        stats.update(o)
        stats_reward.update(r)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    stats.update(o)
    stats_reward.update(r)

    if(normalize):
        o = stats.normalize(o)

    def scale_action(a):

        act_mean = (env.action_space.high + env.action_space.low)/2
        act_scale = env.action_space.high - act_mean
        a = act_scale*a + act_mean
        a= np.clip(a, env.action_space.low, env.action_space.high)

        return a

    # Main loop: collect experience in env and update/log each epoch
    n_episodes = 0
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):


            a, v_t, logp_t = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1,-1)})

            # save and log

            buf.store(o, a, r, v_t, logp_t)
            logger.store(VVals=v_t)

            o, r, d, _ = env.step(scale_action(a[0]))
            ep_ret += r
            ep_len += 1
            stats.update(o)
            stats_reward.update(r)
            if (normalize):
                o = stats.normalize(o)


            terminal = d or (ep_len == max_ep_len)
            if terminal or (t==local_steps_per_epoch-1):



                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = r if d else sess.run(v, feed_dict={x_ph: o.reshape(1,-1)})
                buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                n_episodes += 1

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)
            stats.save()

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.log_tabular('N episodes', n_episodes)
        logger.dump_tabular()

