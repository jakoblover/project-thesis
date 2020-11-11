
import shap
import pandas as pd
import numpy as np

def create_shap_values(train,test,test_index,o_name, n_a,model, n_neigbours = 100):
    def f(X):
        ret = np.zeros((X.shape[0],n_a))
        for i in range(X.shape[0]):
            ret[i,:] = model(X[i,:])[:n_a]
        return ret

    train_summary = shap.kmeans(train,n_neigbours)

    explainer = shap.KernelExplainer(f, train_summary)
    shap_values = explainer.shap_values(test)

    df_shap = pd.DataFrame(index=test_index)

    for n in range(n_a):
        for i in range(len(o_name)):
            df_shap["shap_value_" + str(n) + "_" + o_name[i]] = np.nan
            df_shap["shap_value_" +str(n) +"_"+ o_name[i]].loc[test_index] = shap_values[n][:,i]
            df_shap["expected_value_" + str(n)] = np.nan
            df_shap["expected_value_" + str(n)].loc[test_index] = explainer.expected_value[n]


    expected_value = explainer.expected_value
    return shap_values, expected_value,df_shap

def load_shap_values(df_shap,o_name,n_a):

    shap_values = [np.nan]*n_a
    expected_value = [np.nan]*n_a
    for n in range(n_a):
        shap_values[n] = np.zeros((len(df_shap), len(o_name)))
        expected_value[n] = df_shap["expected_value_" + str(n)].iloc[0]
        for i in range(len(o_name)):
            shap_values[n][:,i] = df_shap["shap_value_" +str(n) +"_"+ o_name[i]]


    return shap_values, expected_value


def estimate_a_from_shap_values(shap_values, expected_value):
    n_states = shap_values[0].shape[1]
    n_a = len(shap_values)
    n = shap_values[0].shape[0]

    a_shap = np.zeros((n, n_a))

    for n in range(n_a):
        a_shap[:, n] += expected_value[n]
        y = np.arange(n_states)
        x = np.zeros(n_states)
        for i in range(n_states):
            x[i] = np.mean(np.abs(shap_values[n][:, i]))
            a_shap[:, n] += shap_values[n][:, i]
    return a_shap

def make_pred_model_data(env,stats = None,model = None,df = pd.DataFrame()):

    try:
        x = df["x"]
        y = df["y"]
        psi = df["psi"]
        u = df["u"]
        v = df["v"]
        r = df["r"]
    except:
        x = df["eta_x"].values
        y = df["eta_y"].values
        psi = df["eta_psi"].values
        u = df["nu_u"].values
        v = df["nu_v"].values
        r = df["nu_r"].values

    a_ret = np.zeros((len(df),5))

    for i in range(len(df)):

        eta = np.array([x[i],y[i],psi[i]])
        nu = np.array([u[i], v[i], r[i]])

        o = env.reset(eta = eta, nu = nu)

        if(stats):
            o = stats.normalize(o)

        a = model(o)
        #alpha,f = env.saturation(np.array([a.item(3),a.item(4)]), np.array([a.item(0),a.item(1),a.item(2)]))

        a_ret[i,:] = a.reshape(5)

    return a_ret