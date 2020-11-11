
import matplotlib.pyplot as plt
import numpy as np

def calc_mean_global_shap(shap_values):

    n_states = shap_values[0].shape[1]
    n_a = len(shap_values)
    mean_shap = np.zeros((n_a,n_states))

    for n in range(n_a):
        for i in range(n_states):
            mean_shap[n,i] = np.mean(np.abs(shap_values[n][:, i]))

    return mean_shap




def plot_mean_shap(mean_shap,o_name, actuator_unity, actuator_name,
                    path, figsize = (22, 15)):

    fig, ax = plt.subplots(2, 3, figsize = figsize)
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    ax = ax.flatten()
    ax_place = [0, 1, 2, 3, 4]

    n_a  = mean_shap.shape[0]

    y = np.arange(len(o_name))

    for n in range(n_a):
        ax[ax_place[n]].barh(y, mean_shap[n,:], align='center')
        ax[ax_place[n]].set_yticks(y)
        ax[ax_place[n]].set_yticklabels(o_name)
        ax[ax_place[n]].invert_yaxis()  # labels read top-to-bottom
        ax[ax_place[n]].set_xlabel('Mean abs SHAP value ' + actuator_unity[n])
        ax[ax_place[n]].set_title('Control input ' + actuator_name[n])
        ax[ax_place[n]].set_ylabel("State")
        ax[ax_place[n]].yaxis.set_tick_params(labelsize=35)

    fig.delaxes(ax[5])  # The indexing is zero-based here
    fig.savefig(path)


def plot_accurac_shap_model(a_shap, a_pred,actuator_name, actuator_unity,path,figsize =(20, 27) ):

    fig, ax = plt.subplots(nrows = 3, ncols = 2, figsize = figsize)
    ax = ax.flatten()
    ax_place = [0, 2, 4,1,3]
    plt.subplots_adjust(hspace=0.5, wspace=0.4)

    for n in range(a_shap.shape[1]):
        ax[ax_place[n]].plot(a_pred[:, n], label="PPO");
        ax[ax_place[n]].plot(a_shap[:, n], label="SHAP");
        ax[ax_place[n]].grid()
        ax[ax_place[n]].set_ylabel('Control input ' + actuator_name[n] + " " + actuator_unity[n]);
        ax[ax_place[n]].set_xlabel("Instance nr.")

    fig.delaxes(ax[5])  # The indexing is zero-based here

    fig.savefig(path)

def plot_shap_values_per_state_entire_episode(shap_values, o_name, actuator_name,actuator_unity,path, figsize=(22, 22)):

    n_a = len(shap_values)
    fig, ax = plt.subplots(figsize=figsize)
    #plt.subplots_adjust(hspace=0.3, wspace=0.4)
    plt.subplots_adjust(hspace=0.5, wspace=0.4)

    fig_num = 0
    for n in range(n_a):
        plt.subplot(321 + fig_num)

        for i in range(len(o_name)):
            plt.plot(shap_values[n][:, i], label=o_name[i])

        plt.xlabel('Time [s]')
        plt.ylabel('SHAP value ' + actuator_unity[n])
        plt.title("Control input " + actuator_name[n])
        plt.grid()
        fig_num += 1

    plt.legend(loc="lower right", bbox_to_anchor=(3.0, 0.0), prop={'size':37}, ncol=2)
    fig.savefig(path)

def plot_relative_shap_values_per_state_per_episode(relative_shap, o_name, actuator_name, path,figsize =(24, 30)):
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=0.3, wspace=0.4)
    n_a = len(relative_shap)
    fig_num = 0
    #test = shap_values[n][:, 0] * 0
    state_place = [0, 1, 0, 3, 4, 5, 6, 7, 2]
    for i in range(len(o_name)):
        if (o_name[i] == "$l$"):
            continue
        plt.subplot(421 + state_place[i])

        for n in range(n_a):
            plt.plot(relative_shap[n,:, i], label=actuator_name[n])
            plt.ylabel('Relativ \n contribution')
            plt.title("State " + o_name[i])
        if (state_place[i] > 5):
            plt.xlabel("Time")
        if (state_place[i] % 2 != 0):
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size':30})
        fig_num += 1

    fig.savefig(path)


def calc_relative_shap_values(shap_values):

    #test = shap_values[n][:, 0] * 0
    n_a = len(shap_values)
    n_states = shap_values[0].shape[1]
    relative_shap = np.zeros((len(shap_values),shap_values[0].shape[0],shap_values[0].shape[1]))

    for i in range(n_states):
        for n in range(n_a):
            max_n = np.sum(np.abs(shap_values[n]), axis=1)

            relative_shap[n,:,i] = abs(shap_values[n][:, i]) / max_n
            #test += shap_values[n][:, i] / np.max(abs(shap_values[n]))
    return relative_shap
