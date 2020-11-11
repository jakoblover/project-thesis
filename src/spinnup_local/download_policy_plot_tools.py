
import tensorflow as tf
#from spinup.algos.ddpg import core
import matplotlib as mpl
from matplotlib.transforms import Bbox
from matplotlib.ticker import StrMethodFormatter
import joblib



class NormalizeStats():
    """docstring forNormalizeObs."""
    def __init__(self, dic,size):

        self.mean = [0]*size
        self.std = [0]*size
        self.size = size

        for i in range(size):
            self.mean[i] = dic[str(i)]["mean"]
            self.std[i] = dic[str(i)]["std"]


    def normalize(self,o):
        for i in range(self.size):
            o[i] = (o.item(i) - self.mean[i]) / self.std[i]
        return o


def load_policy(sess,export_dir,model_info_path,env,deterministic = False,batch_normalization=False,
                unscaled = False,load_layers = False,n_hidden = 0):

    tf.saved_model.loader.load(sess,["serve"],export_dir)


    model_info = joblib.load(model_info_path)

    graph = tf.get_default_graph()

    model = {}
    model.update({k:  graph.get_tensor_by_name(v) for k,v in model_info['inputs'].items()})
    model.update({k:  graph.get_tensor_by_name(v) for k,v in model_info['outputs'].items()})

    print(model)

    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']

    else:
        print('Using default action op.')
        action_op = model['pi']  # loads entire stochastic policy, with noise
        if n_hidden:
            action_op = action_op.graph.get_tensor_by_name('pi/dense_{}/BiasAdd:0'.format(n_hidden))  # This extracts the mean components of the gaussian policy - like setting all noise to zero!


    act_mean = (env.action_space.high + env.action_space.low)/2
    act_scale = env.action_space.high - act_mean

    print(action_op)



    if (unscaled == True):
        if (batch_normalization):
            get_action = lambda x:sess.run(action_op, feed_dict={model['x']:x[None, :], model['phase']:0})[0]
        else:
            get_action = lambda x:sess.run(action_op, feed_dict={model['x']:x[None, :]})[0]
    else:
        if (batch_normalization):
            get_action = lambda x:act_scale * sess.run(action_op, feed_dict={model['x']:x[None, :], model['phase']:0})[
                0] + act_mean
        else:
            get_action = lambda x:act_scale * sess.run(action_op, feed_dict={model['x']:x[None, :]})[0] + act_mean


    # make function for producing an action given a single state
    #get_action = lambda x: act_scale*sess.run(action_op,feed_dict = feed_dict ={model['x']: x[None,:]})[0] + act_mean

    if(load_layers):
        return get_action,model,act_mean,act_scale
    else:
        return get_action