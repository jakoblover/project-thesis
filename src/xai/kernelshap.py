
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_shap_values(X_train,X_test,model, action_vector_names, state_vector_names, output_folder, n_neigbours = 100):
    def f(X):
        ret = np.zeros((X.shape[0], 5))
        for i in range(X.shape[0]):
            ret[i, :] = model(X[i, :])[:5]
        return ret

    shap.initjs()
    train_summary = shap.kmeans(X_train,n_neigbours)
    explainer = shap.KernelExplainer(f, train_summary)
    shap_values = explainer.shap_values(X_test)

    print("i did it")

    df_shap = pd.DataFrame(columns=state_vector_names)
    for i in range(0,len(action_vector_names)):
        for n in range(0,len(state_vector_names)):
            df_shap[state_vector_names[n]] = pd.Series(shap_values[i][:,n])
        df_shap.to_csv(output_folder + "/" + action_vector_names[i] + ".csv")

    #for i in range(0,len(action_vector_names)):
        #f = shap.force_plot(explainer.expected_value[i], shap_values[i][0, :], X_test.iloc[0, :], link="logit", show=False)
        #f = shap.plots.bar(shap_values[i])
        #shap.save_html(f"index{i}.htm", f)



    #xpl = shap.DeepExplainer

    #df_shap = pd.DataFrame(index=test_index)

    # for n in range(n_a):
    #     for i in range(len(o_name)):
    #         df_shap["shap_value_" + str(n) + "_" + o_name[i]] = np.nan
    #         df_shap["shap_value_" +str(n) +"_"+ o_name[i]].loc[test_index] = shap_values[n][:,i]
    #         df_shap["expected_value_" + str(n)] = np.nan
    #         df_shap["expected_value_" + str(n)].loc[test_index] = explainer.expected_value[n]


    #expected_value = explainer.expected_value
    return explainer, shap_values

def load_shap_values(df_shap,o_name,n_a):

    shap_values = [np.nan]*n_a
    expected_value = [np.nan]*n_a
    for n in range(n_a):
        shap_values[n] = np.zeros((len(df_shap), len(o_name)))
        expected_value[n] = df_shap["expected_value_" + str(n)].iloc[0]
        for i in range(len(o_name)):
            shap_values[n][:,i] = df_shap["shap_value_" +str(n) +"_"+ o_name[i]]


    return shap_values, expected_value