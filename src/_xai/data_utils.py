

import numpy as np
import pandas as pd




def copy_df_data(env,xai_save_path,xai_load_path,stats = None,normalize_obs = False,o_name = [" "]):
    df_org = pd.read_csv(xai_load_path)
    df_org = df_org.set_index("Unnamed: 0")
    df_copy = df_org.copy()

    column_names = ["eta_x","eta_y","eta_psi","nu_u","nu_v","nu_r","x_tilde","y_tilde","on_land","u","v","r","d_obs","psi_tilde_obs","psi_tilde"]
    x = df_org["eta_x"]
    y = df_org["eta_y"]
    psi = df_org["eta_psi"]
    u = df_org["nu_u"]
    v = df_org["nu_v"]
    r = df_org["nu_r"]


    for i in range(len(df_org)):

        eta = np.array([x[i],y[i],psi[i]])
        nu = np.array([u[i], v[i], r[i]])

        o = env.reset(eta = eta, nu = nu)
        eta = env.eta
        nu = env.nu
        if (normalize_obs):
            o = stats.normalize(o)
        print(i)
        df_copy[column_names].iloc[i] = eta.reshape(len(eta)).tolist() + nu.reshape(len(nu)).tolist() + o.reshape(len(o)).tolist()

    # df = df[df["d_obs"] != 0]
    df = df_copy.dropna()

    df.to_csv(xai_save_path)
    return df

def copy_df_data_trajectory(env,df_org,stats = None,normalize_obs = False,o_name = [" "]):

    column_names = ["eta_x","eta_y","eta_psi","nu_u","nu_v","nu_r"] + o_name

    try:
        x = df_org["x"]
        y = df_org["y"]
        psi = df_org["psi"]
        u = df_org["u"]
        v = df_org["v"]
        r = df_org["r"]
    except:
        x = df_org["eta_x"]
        y = df_org["eta_y"]
        psi = df_org["eta_psi"]
        u = df_org["nu_u"]
        v = df_org["nu_v"]
        r = df_org["nu_r"]


    df_copy = pd.DataFrame(index=np.arange(0, len(x)), columns=column_names)


    for i in range(len(x)):

        eta = np.array([x[i],y[i],psi[i]])
        nu = np.array([u[i], v[i], r[i]])

        o = env.reset(eta = eta, nu = nu)
        eta = env.eta
        nu = env.nu
        if (normalize_obs):
            o = stats.normalize(o)
        print(i)

        df_copy["eta_x"].iloc[i] = eta.item(0)
        df_copy["eta_y"].iloc[i] = eta.item(1)
        df_copy["eta_psi"].iloc[i] = eta.item(2)

        df_copy["nu_u"].iloc[i] = nu.item(0)
        df_copy["nu_v"].iloc[i] = nu.item(1)
        df_copy["nu_r"].iloc[i] = nu.item(2)

        print(len(o),len(o_name))

        for j in range(len(o_name)):
            df_copy[o_name[j]].iloc[i] = o.item(j)



    return df_copy.copy()

def transform_trajectory_state(env,xai_load_path,stats = None,
                          normalize_obs = False,o_name = [" "]):
    df_org = pd.read_csv(xai_load_path)
    df_org = df_org.set_index("Unnamed: 0")


    column_names = ["eta_x", "eta_y", "eta_psi", "nu_u", "nu_v", "nu_r"] + o_name

    print(df_org.columns)

    df_copy = df_org.copy()


    x_tilde = df_org["x_tilde"]
    y_tilde = df_org["y_tilde"]
    on_land = df_org["on_land"]
    u = df_org["u"]
    v = df_org["v"]
    r = df_org["r"]
    d_osb = df_org["d_obs"]
    psi_tilde_obs = df_org["psi_tilde_obs"]
    psi_tilde = df_org["psi_tilde"]

    x = df_org["eta_x"]
    y = df_org["eta_y"]
    psi = df_org["eta_psi"]


    for i in range(len(df_org)):

        eta = np.array([x[i], y[i], psi[i]])
        nu = np.array([u[i], v[i], r[i]])

        o = np.array([x_tilde[i],y_tilde[i],on_land[i],u[i],v[i],r[i],d_osb[i],psi_tilde_obs[i],psi_tilde[i]])
        if (normalize_obs):
            o = stats.normalize(o)

        df_copy[column_names].iloc[i] = eta.reshape(len(eta)).tolist() + nu.reshape(len(nu)).tolist() + o.reshape(len(o)).tolist()

    # df = df[df["d_obs"] != 0]
    df = df_copy

    return df.copy()


def run_policy(env,get_action,num_episodes,normalization_obs  = False,stats = None,u_init = None,max_ep_len = 1500,
               trajectory_path = "",Vessel_plot = None):

    df_org = pd.read_csv(trajectory_path)

    eta_x = df_org["x"]
    eta_y = df_org["y"]
    eta_psi = df_org["psi"]
    nu_u = df_org["u"]
    nu_v = df_org["v"]
    nu_r = df_org["r"]

    r,d,ep_ret,ep_len,n = 0,False,0,0,0

    o = env.reset(np.array([312.183, 876.256, -0.384]), np.array([0.3, 0, 0]))
    o = env.reset(np.array([eta_x[0], eta_y[0], eta_psi[0]]), np.array([nu_u[0], nu_v[0], nu_r[0]]))

    env_plot = Vessel_plot(max_ep_len,env = env)


    for n in range(max_ep_len):

        o = env.reset(np.array([eta_x[n], eta_y[n], eta_psi[n]]), np.array([nu_u[n], nu_v[n], nu_r[n]]))
        if (normalization_obs):
            o = stats.normalize(o)

        a = get_action(o)

        if ep_len == 0:
            print("start pos", env.eta.reshape(3))

        alpha = np.array([a.item(3), a.item(4), env.alpha_max.item(2)]).reshape(3,1)
        f = np.array([a.item(0), a.item(1), a.item(2)]).reshape(3,1)

        alpha,f = env.saturation(alpha,f)


        env.alpha = alpha
        env.f = f

        env_plot.update_state(o, a, env,n)

        ep_ret += r
        ep_len +=1



    env_plot.plot_episode(max_ep_len)
    f_data = env_plot.f
    alpha_data = env_plot.alpha
    env_plot.reset()

    o = env.reset(np.array([eta_x[0], eta_y[0], eta_psi[0]]), np.array([nu_u[0], nu_v[0], nu_r[0]]))

    for n in range(max_ep_len):


        if (normalization_obs):
            o = stats.normalize(o)

        a = get_action(o)

        if ep_len == 0:
            print("start pos", env.eta.reshape(3))

        o,r,d,_ = env.step(a)

        env_plot.update_state(o, a, env,n)

    print("f",np.mean(np.abs(f_data - env_plot.f)))
    print("alpha,",np.mean(np.abs(alpha_data - env_plot.alpha)))


    env_plot.plot_episode(max_ep_len)
    env_plot.reset()

def create_random_data(env,xai_path,stats = None,normalize_obs = False,number_dp = 100,o_name = []):

    ex_col = ["eta_x","eta_y","eta_psi","nu_u","nu_v","nu_r"]
    df = pd.DataFrame(index = np.arange(0,number_dp),columns = ex_col + o_name)


    for i in range(number_dp):

            o = env.reset()
            eta = env.eta
            nu = env.nu
            if (normalize_obs):
                o = stats.normalize(o)

            df.iloc[i] = eta.reshape(len(eta)).tolist()+ nu.reshape(len(nu)).tolist()+ o.reshape(len(o)).tolist()

            print(i,number_dp)
    #df = df[df["d_obs"] != 0]
    df = df.dropna()
    df.to_csv(xai_path)
    return df

def split_train_test(df, percent, o_name):

    train_index = df.iloc[0:int(percent * len(df))].index

    test_index = df.iloc[int(percent * len(df)):len(df)].index

    train = df[o_name].loc[train_index].to_numpy(dtype=float)
    test = df[o_name].loc[test_index].to_numpy(dtype=float)
    return train, train_index, test, test_index


