
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['lines.linewidth'] = 2
mpl.rc('xtick', labelsize=21)
mpl.rc('ytick', labelsize=21)
mpl.rc('axes', titlesize = 28)
mpl.rc('axes', labelsize = 21)
mpl.rcParams['axes.labelpad'] = 10
mpl.rc('legend', fontsize = 16)
mpl.rcParams['figure.figsize'] = 11,8

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'


def make_video(env_plot,ep_len ,step = 1,save_path = "", area = False,speed_limit = False):

    T = np.arange(0, ep_len) * env_plot.env.h
    N = env_plot.eta[0, :ep_len]
    E = env_plot.eta[1, :ep_len]
    psi = env_plot.eta[2, :ep_len]

    step = max(step ,1)



    arr1 = np.arange(0, ep_len + step, step)
    fig = plt.figure()

    # legend = plt.legend(dpi = 600)

    def animate(i):
        plt.cla()
        plt.clf()
        # legend.remove()
        plt.subplots_adjust(bottom=0.18)

        print("iiii" ,i)
        i = arr1[i]
        print("iiii", i)

        if (i >= ep_len):
            i = ep_len - 1


        plt.scatter(E.item(0), N.item(0), label="Start point", s=110, color="yellowgreen")
        plt.scatter(env_plot.env.eta_d.item(1), env_plot.env.eta_d.item(0), label="Target point", color="teal", s=110)
        x_init, y_init = env_plot.env.map.get_docking_space_init_boarder(draw=True)

        if(area):
            x_doc, y_doc = env_plot.env.map.get_docking_space_boarder(draw=True)
            plt.plot(y_doc, x_doc, linewidth=4, color="teal", label="Target-rectangle")

        if(speed_limit):
            speed_reg_x = np.arange(np.min(y_init), np.max(y_init))
            plt.plot(speed_reg_x, np.ones(len(speed_reg_x)) * env_plot.env.limit_speed_regulation_x, label="Speed limit")

        plt.imshow(env_plot.env.map.im);
        x_s, y_s = env_plot.env.map.get_dock_boarder(draw=True)
        plt.plot(y_s, x_s, linewidth=2, color="black", label="Harbour constrains")
        plt.plot(E[:i], N[:i], label='Path', color='r');

        x_ned_b, y_ned_b = env_plot.env.map.calc_vessel_vertex_pos(N.item(i), E.item(i), psi.item(i), draw=True)
        plt.plot(y_ned_b, x_ned_b, color="purple")


        # plt.plot(y_init, x_init, color="brown", label="init space")
        plt.title("Time " + str( i *env_plot.env.h) + " s" )

        plt.xlim(max(min(np.min(y_init), env_plot.env.eta_d.item(1)) - 200,0),
                 max(np.max(y_init), env_plot.env.eta_d.item(1)) + 50)
        plt.ylim(max(min(np.min(N),np.min(x_init), env_plot.env.eta_d.item(0)) - 50,0), min(max(np.max(x_init), env_plot.env.eta_d.item(0)) + 50,env_plot.env.map.im.shape[0]))
        plt.xlabel("East [m]")
        plt.ylabel("North [m]")
        plt.legend(loc = "lower left")



    anni = animation.FuncAnimation(fig ,animate ,frames = len(arr1) ,interval = 100)
    anni.save(save_path)

    plt.show()
    exit()