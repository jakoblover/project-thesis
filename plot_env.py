import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json

from env.LP4_distance_berthing.auto_docking_vessel import Vessel,rad2deg

from spinnup_local.download_policy_plot_tools import *
from config.config_simulation_plot import *

import random
random.seed()


class Vessel_plot(object):
    """docstring for ."""
    def __init__(self, N_iterations,env):
        self.N_iterations= N_iterations
        self.env = env

        self.reset()


    def reset(self):
        self.eta = np.zeros((3, self.N_iterations))
        self.nu = np.zeros((3, self.N_iterations))
        self.nu_r = np.zeros((3, self.N_iterations))
        self.a = np.zeros((self.env.action_space.shape[0], self.N_iterations))

        self.x = np.zeros((self.env.observation_space.shape[0],self.N_iterations))

        self.reward = np.zeros(self.N_iterations)
        self.reward_u = np.zeros(self.N_iterations)
        self.reward_psi_tilde = np.zeros(self.N_iterations)
        self.reward_dock = np.zeros(self.N_iterations)
        self.reward_collision = np.zeros(self.N_iterations)
        self.reward_t = np.zeros(self.N_iterations)
        self.terminal = np.zeros(self.N_iterations)
        self.f = np.zeros((3,self.N_iterations))
        self.alpha = np.zeros((3,self.N_iterations))
        self.f_dot = np.zeros((3, self.N_iterations))
        self.alpha_dot = np.zeros((3, self.N_iterations))

        self.f_dot_norm = np.zeros((3, self.N_iterations))
        self.alpha_dot_norm = np.zeros((2, self.N_iterations))

        self.d_obss = np.zeros((5, self.N_iterations))
        self.theta_obss = np.zeros((5, self.N_iterations))

        self.reward_cumulativ = np.zeros(self.N_iterations)
        self.d_obs = np.zeros(self.N_iterations)
        self.d_edge = np.zeros(self.N_iterations)

        self.d_doc = np.zeros((4, self.N_iterations))
        self.psi_tilde_doc = np.zeros((4, self.N_iterations))
        self.d_d_dot = np.zeros(self.N_iterations)

        self.x_tilde_ss = np.zeros(self.N_iterations)
        self.y_tilde_ss = np.zeros(self.N_iterations)





    def update_state(self,o, a, env,n):
        self.x[:,n] = np.around(o.reshape(env.observation_space.shape[0]),decimals=3)
        self.eta[:,n] = np.around(env.eta.reshape(3),decimals=3)
        self.nu[:,n] = np.around(env.nu.reshape(3),decimals=6)
        self.nu_r[:, n] = np.around(env.nu_r.reshape(3), decimals=6)
        self.a[:,n] = np.around(a.reshape(env.action_space.shape[0]),decimals=3)
        self.reward[n] = np.around(env.tot_reward,decimals=3)
        self.reward_u[n] = np.around(env.reward_u,decimals=3)
        self.reward_dock[n] = np.around(env.reward_dock,decimals=3)
        self.reward_psi_tilde[n] = np.around(env.reward_psi_tilde, decimals=3)
        self.reward_collision[n] = np.around(env.reward_collision,decimals=3)
        self.reward_t[n] = np.around(env.reward_t, decimals=3)
        self.terminal[n] = np.around(env.terminal,decimals=1)
        self.f[:,n] = np.around(env.f.reshape(len(env.f)),decimals=3)
        self.alpha[:, n] = np.around(env.alpha.reshape(len(env.alpha)),decimals=3)
        self.f_dot[:, n] = np.around(env.f_dot.reshape(len(env.f)), decimals=3)
        self.alpha_dot[:, n] = np.around(env.alpha_dot.reshape(len(env.alpha)), decimals=3)

        self.f_dot_norm[:, n] = np.around(env.f_dot_norm.reshape(len(env.f)), decimals=3)
        self.alpha_dot_norm[:, n] = np.around(env.alpha_dot_norm.reshape(len(env.alpha_dot_norm)), decimals=3)

        #self.d_obss[:,n] = np.around(env.d_obss.reshape(len(env.d_obss)),decimals=3)
        #self.theta_obss[:, n] = np.around(env.theta_obss.reshape(len(env.theta_obss)), decimals=3)
        if(n == 0):
            self.reward_cumulativ[n] =  np.around(env.tot_reward,decimals=4)
        else:
            self.reward_cumulativ[n] = np.around(0.99*self.reward_cumulativ[n - 1] + env.tot_reward,decimals=4)

        self.d_obs[n] = np.around(env.d_obs,decimals=3)
        self.d_edge[n] = np.around(env.d_edge, decimals=3)
        self.d_d_dot[n] = np.around(env.d_d_dot, decimals=3)
        self.x_tilde_ss[n] = np.around(env.x_tilde_ss, decimals=3)
        self.y_tilde_ss[n] = np.around(env.y_tilde_ss, decimals=3)

        #self.d_doc[:,n] = np.around(env.d_doc.reshape(len(env.d_doc)),decimals=3)
        #self.psi_tilde_doc[:,n] = np.around(env.psi_tilde_doc.reshape(len(env.psi_tilde_doc)),decimals=3)

    def make_video_frames(self,ep_len):

        T = np.arange(0, ep_len) * self.env.h
        N = self.eta[0, :ep_len]
        E = self.eta[1, :ep_len]
        psi = self.eta[2, :ep_len]

        step = max(50,1)

        for i in range(0, ep_len + step, step):

            fig = plt.figure(dpi = 600)

            if(i >= ep_len):
                i = ep_len - 1

            print(i)

            plt.plot(E[:i], N[:i], label='Vessel path', color='r');
            plt.imshow(self.env.map.im);
            x_s, y_s = self.env.map.get_dock_boarder(draw=True)
            plt.plot(y_s, x_s, linewidth=2, color="black", label="Harbour constrains")
            x_doc, y_doc = self.env.map.get_docking_space_boarder(draw=True)
            plt.plot(y_doc, x_doc, linewidth=2, color="purple", label="Target box")
            plt.scatter(self.env.eta_d.item(1), self.env.eta_d.item(0), label="Target point", color="deeppink", s=100)

            if (self.env.docking_init_space_circle):
                x_init = self.env.max_rad_init_circle * np.sin(np.arange(0, 2 * np.pi, 0.01)) + self.env.eta_d.item(0)
                y_init = self.env.max_rad_init_circle * np.cos(np.arange(0, 2 * np.pi, 0.01)) + self.env.eta_d.item(1)

            else:
                x_init, y_init = self.env.map.get_docking_space_init_boarder(draw=True)


            x_ned_b, y_ned_b = self.env.map.calc_vessel_vertex_pos(N.item(i), E.item(i), psi.item(i), draw=True)
            plt.plot(y_ned_b, x_ned_b, color="blue",label="Vessel")
            x_min = -100 + max(min(np.min(y_init), self.env.eta_d.item(1)),200)
            x_max = min(max(np.max(y_init), self.env.eta_d.item(1)),1200) + 100
            plt.xlim(x_min,x_max)
            y_min = -100 + max(min(np.min(x_init), self.env.eta_d.item(0)),200)
            y_max = min(max(np.max(x_init), self.env.eta_d.item(0)),1200) + 100

            plt.ylim(y_min,y_max)

            # plt.ylim(550, 850)
            # plt.xlim(200, 1200)
            # plt.ylim(200, 1200)

            plt.legend()

            fig.savefig("figure/video/" + str(i) + ".png")
            plt.cla()
            plt.clf()
            plt.close()


    def plot_episode(self,ep_len, model_num = 0,savefig = False,folder = ""):

        T = np.arange(0, ep_len) * self.env.h
        N = self.eta[0,:ep_len]
        E = self.eta[1,:ep_len]
        psi = self.eta[2,:ep_len]
        psi_d = np.ones(len(psi))*self.env.eta_d.item(2)

        u = self.nu[0,:ep_len]
        v = self.nu[1,:ep_len]
        r = self.nu[2,:ep_len]

        u_r = self.nu_r[0, :ep_len]
        v_r = self.nu_r[1, :ep_len]
        r_r = self.nu_r[2, :ep_len]



        if(self.env.state_func == 1 or self.env.state_func == 2  ):



            x_tilde = np.around(self.x[0, :ep_len], decimals=3)
            y_tilde = np.around(self.x[1, :ep_len], decimals=3)
            if (self.env.state_augmentation):
                y_tilde -= self.y_tilde_ss[:ep_len]
                x_tilde -= self.x_tilde_ss[:ep_len]

            on_land = np.around(self.x[2, :ep_len], decimals=3)
            # u = np.around(x_array[3, :], decimals=3)
            # v = np.around(x_array[4, :], decimals=3)
            # r = np.around(x_array[5, :], decimals=3)
            d_obs = np.around(self.x[6, :ep_len], decimals=3)
            theta_obs = np.around(self.x[7, :ep_len], decimals=3) + psi
            d_obs[d_obs >= np.around(self.env.map.d_max, decimals=3)] = np.inf
            psi_tilde = np.around(self.x[8, :ep_len], decimals=3)


            d_obs = self.d_obs[:ep_len]
            d_edge = self.d_edge[:ep_len]

            #psi_tilde = np.around(self.x[8, :ep_len], decimals=3)
            d_d = np.sqrt(x_tilde ** 2 + y_tilde ** 2)
            psi_tilde_d = np.arctan2(y_tilde, x_tilde)
            x_obs = N + (d_obs+d_edge) * np.cos(theta_obs)
            y_obs = E + (d_obs+d_edge) * np.sin(theta_obs)

            if(self.env.state_func==2):
                alpha_dot_norm = np.around(self.x[9:11, :ep_len], decimals=3)
                alpha_norm = np.around(self.x[11:13, :ep_len], decimals=3)
                f_dot_norm = np.around(self.x[13:16, :ep_len], decimals=3)
                f_norm = np.around(self.x[16:19, :ep_len], decimals=3)

                fig = plt.figure()
                plt.subplot(221); plt.plot(T,alpha_dot_norm.T);plt.title("alpha dot norm")
                plt.subplot(222);plt.plot(T,alpha_norm.T);plt.title("alpha norm")
                plt.subplot(223); plt.plot(T, f_dot_norm.T);plt.title("f_dot_norm")
                plt.subplot(224); plt.plot(T, f_norm.T);plt.title("f_norm")

        d_d_dot = self.d_d_dot[:ep_len]


        psi_tilde_doc = self.psi_tilde_doc[:,:ep_len]
        d_doc = self.d_doc[:,:ep_len]


        alpha = self.alpha[:,:ep_len]
        f = self.f[:,:ep_len]

        f_dot_max = (self.env.f_max - self.env.f_min) / self.env.h
        alpha_dot_max = (self.env.alpha_max - self.env.alpha_min) / self.env.h

        #print(f_dot_max,alpha_dot_max)

        f_dot = self.f_dot[:,:ep_len]/f_dot_max
        alpha_dot = self.alpha_dot[:,:ep_len]/alpha_dot_max

        f_dot[np.isnan(f_dot)] = 0
        alpha_dot[np.isnan(alpha_dot)] = 0
        ##f_dot = self.f_dot[:, :ep_len]
        #alpha_dot = self.alpha_dot[:, :ep_len]


        fig = plt.figure()
        plt.subplot(211);plt.plot(self.y_tilde_ss[:ep_len]);plt.title("y_tilde_ss")
        plt.subplot(212); plt.plot(self.x_tilde_ss[:ep_len]);plt.title("x_tilde_ss")


        f_0 = f[0,:]
        f_1 = f[1,:]
        f_2 = f[2,:]

        alpha_0 = alpha[0,:]
        alpha_1 = alpha[1,:]
        alpha_2 = alpha[2,:]

        lim_max = max([max(N),max(E)]) + 50
        lim_min = min([min(E),min(N)]) - 50

        fig = plt.figure()
        cmap = matplotlib.cm.get_cmap('Reds')
        color = cmap(0.3 + np.arange(1, ep_len + 1) / (4 / 3 * (ep_len + 1)))

        plt.scatter(E.item(0), N.item(0), label="Start point", s=110, color="yellowgreen")
        plt.scatter(self.env.eta_d.item(1), self.env.eta_d.item(0), label="Berth point", color="teal", s=110)

        x_s, y_s = self.env.map.get_dock_boarder(draw=True)
        plt.plot(y_s, x_s, linewidth=2, color="black", label="Harbour constraints")
        # x_doc, y_doc = self.env.map.get_docking_space_boarder(draw=True)
        # plt.plot(y_doc, x_doc, linewidth=4, color="teal", label="Target-rectangle")

        i = 0
        print(np.sqrt(x_tilde.item(i) ** 2 + y_tilde.item(i) ** 2), self.env.d_dock)
        x_ned_b, y_ned_b = self.env.map.calc_vessel_vertex_pos(N.item(i), E.item(i), psi.item(i), draw=True)
        plt.plot(y_ned_b, x_ned_b, color=color[i])

        old_dist = np.sqrt(x_tilde.item(i) ** 2 + y_tilde.item(i) ** 2)
        i = ep_len - 1
        closest_dist = np.sqrt(x_tilde.item(i) ** 2 + y_tilde.item(i) ** 2)
        for i in range(0, ep_len, 1):
            new_dist = np.sqrt(x_tilde.item(i) ** 2 + y_tilde.item(i) ** 2)
            if abs(new_dist - old_dist) > 100 and abs(new_dist - closest_dist) > 80 :
                old_dist = new_dist
                print(np.sqrt(x_tilde.item(i)**2 + y_tilde.item(i)**2))
                x_ned_b, y_ned_b = self.env.map.calc_vessel_vertex_pos(N.item(i), E.item(i), psi.item(i),draw = True)
                plt.plot(y_ned_b, x_ned_b,color = color[i])


        i = ep_len - 1
        x_ned_b, y_ned_b = self.env.map.calc_vessel_vertex_pos(N.item(i), E.item(i), psi.item(i),draw=True)
        print("min distans",np.min(np.sqrt(x_tilde ** 2 + y_tilde ** 2)), self.env.d_dock)
        plt.plot(y_ned_b, x_ned_b,color = color[i])




        plt.plot(E, N, label='Path',color = 'r');
        plt.imshow(self.env.map.im);

        if(self.env.docking_init_space_circle):
            x_init = self.env.max_rad_init_circle * np.sin(np.arange(0, 2 * np.pi, 0.01)) + self.env.eta_d.item(0)
            y_init = self.env.max_rad_init_circle * np.cos(np.arange(0, 2 * np.pi, 0.01)) + self.env.eta_d.item(1)

        else:
            x_init,y_init = self.env.map.get_docking_space_init_boarder(draw=True)
        #plt.plot(y_init,x_init,color ="brown",label="init space")


        #print("init space",env.map.docking_init_space_x - env.eta_d.item(0),env.map.docking_init_space_y - env.eta_d.item(1))
        plt.ylim(-100 + min(np.min(N), self.env.eta_d.item(0)), max(np.max(N), self.env.eta_d.item(0)) + 100)

        # plt.ylim(550, 850)
        plt.xlim(200, max(850, np.max(E) + 50))
        #plt.scatter(y_obs, x_obs, label="obs",color = color)
        plt.ylabel("North [m]")
        plt.xlabel("East [m]")

        plt.legend()

        if (savefig):
            fig.savefig(folder + "_maps.eps")

        fig = plt.figure(figsize=(23, 17))
        plt.subplots_adjust(hspace=0.2, wspace=0.6)
        plt.subplot(331);
        plt.plot(T,u,label="$u$");plt.plot(T,u_r,label="$u_r$");
        plt.ylabel("Surge [m/s]"); plt.grid(); plt.legend(); plt.xlim(0, max(T));
        plt.subplot(332);
        plt.plot(T,v,label="$v$");plt.plot(T,v_r,label="$v_r$");
        plt.ylabel("Sway [m/s]"); plt.grid(); plt.legend();plt.xlim(0, max(T));
        plt.subplot(333);
        plt.plot(T, r, label="$r$");  plt.plot(T, r_r, label="$r_r$")
        plt.ylabel("Yaw rate [rad/s]"); plt.grid();plt.legend(); plt.xlim(0, max(T));
        plt.subplot(334);
        plt.plot(T, x_tilde);
        plt.ylabel(r'Distance $\tilde{x}$ [m]'); plt.grid(); plt.xlim(0, max(T));
        plt.subplot(335);
        plt.plot(T, y_tilde);
        plt.ylabel(r'Distance $\tilde{y}$ [m]'); plt.grid(); plt.xlim(0, max(T));
        plt.subplot(336);
        plt.plot(T, psi_tilde);
        plt.ylabel(r'Yaw error $\tilde{\psi}$ [rad]');
        plt.grid();
        plt.xlim(0, max(T));
        plt.subplot(337);
        plt.plot(T, d_obs);
        plt.ylabel("Distance $d_{obs} [m] $");
        plt.grid();
        plt.xlim(0, max(T));
        plt.xlabel("Time [s]");



        if (savefig):
            fig.savefig(folder + "_states.eps")

        fig = plt.figure(figsize=MEDIUM_FIGSIZE)
        plt.subplots_adjust(left = 0.17)
        plt.subplot(311) ; plt.plot(T,u,label="$u$");plt.plot(T,u_r,label="$u_r$");plt.ylabel("Surge [m/s]");   plt.grid(); plt.legend()
        plt.subplot(312) ; plt.plot(T,v,label="$v$");plt.plot(T,v_r,label="$v_r$");  plt.ylabel("Sway [m/s]");plt.grid();  plt.legend()
        plt.subplot(313) ; plt.plot(T,r,label="$r$");plt.plot(T,r_r,label="$r_r$"); plt.ylabel("Yaw rate [rad/s]");  plt.grid();plt.xlabel("Time [s]"); plt.legend()
        plt.xlabel("Time [s]");

        if (savefig):
            fig.savefig(folder + "_vel.eps")

        fig = plt.figure(figsize=(22, 13))
        plt.subplots_adjust(bottom=0.2, wspace=0.4)
        plt.subplot(231);
        plt.plot(T,self.reward[:ep_len],label="reward"); plt.plot(T,np.ones(len(T))*(self.env.C_dock + self.env.C_psi_tilde),label="max")
        plt.ylabel("Total reward");  plt.grid();plt.legend();plt.xlim(0, max(T));
        plt.subplot(232);
        plt.plot(T,self.reward_t[:ep_len],label="reward");plt.plot(T,np.ones(len(T))*(self.env.C_d_d_dot),label="max")
        plt.ylabel("Reward $r_{\dot{d}_d}$");plt.grid();plt.legend();plt.xlim(0, max(T));
        plt.subplot(233);
        plt.plot(T,self.reward_dock[:ep_len],label="reward");plt.plot(T,np.ones(len(T))*(self.env.C_dock),label="max")
        plt.ylabel("Reward $r_{d_d}$");plt.grid();plt.legend();plt.xlim(0, max(T));
        plt.subplot(234);
        plt.plot(T,self.reward_collision[:ep_len],label="reward");plt.plot(T,np.ones(len(T))*(0),label="max")
        plt.ylabel("Reward $r_{d_{obs}}$");plt.grid();plt.legend();plt.xlim(0, max(T));
        plt.subplot(235);
        plt.plot(T, self.reward_psi_tilde[:ep_len],label="reward"); plt.plot(T,np.ones(len(T))*(self.env.C_psi_tilde),label="max")
        plt.ylabel(r'Reward $r_{\tilde{\psi}}$');plt.grid();plt.legend();plt.xlim(0, max(T));
        plt.xlabel("Time [s]");
        if(self.env.reward_func == 6):
            plt.subplot(236);
            plt.plot(T, self.reward_u[:ep_len], label="reward");plt.plot(T, np.ones(len(T)) * (0), label="max")
            plt.ylabel(r'Reward $r_{\dot{a}}$');plt.grid(); plt.legend();   plt.xlim(0, max(T));  plt.xlabel("Time [s]");
            plt.xlabel("Time [s]");

        if (savefig):
            fig.savefig(folder + "_reward.eps")

        fig = plt.figure(figsize=(22,13))
        # plt.tight_layout()
        plt.subplots_adjust(wspace=0.4)
        alpha_label = [r'$\alpha_1$', r'$\alpha_1$', r'$\alpha_1$']
        f_label = ["$f_1$", "$f_2$", "$f_3$"]
        plt.subplot(211);
        plt.plot(T, alpha_0, label=alpha_label[0]);  plt.plot(T, alpha_1, label=alpha_label[1]);
        plt.ylabel("Angle [rad]");plt.xlabel("Time [s]");plt.legend();plt.grid();
        plt.xlim(0, max(T));
        plt.subplot(212);
        plt.plot(T, f_0, label=f_label[0]); plt.plot(T, f_1, label=f_label[1]); plt.plot(T, f_2, label=f_label[2]);
        plt.ylabel("Force [kN]");plt.xlabel("Time [s]");  plt.legend();plt.grid();plt.xlim(0, max(T));
        plt.yticks(np.arange(min(min(f_0),min(f_1)), max(max(f_0),max(f_1))+1, 25.0))
        if (savefig):
            fig.savefig(folder + "_act.eps")