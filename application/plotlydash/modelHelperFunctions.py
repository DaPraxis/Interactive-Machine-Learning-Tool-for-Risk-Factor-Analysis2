import ast
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoLars, ElasticNet, BayesianRidge, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import json
import plotly
import seaborn as sns
import matplotlib.pyplot as plt

def load_info_dict(file):
    f = open(file, 'r')
    cont = f.read()
    f.close()
    return ast.literal_eval(cont)


VAR_PATH = 'data/var_info.txt'
var_info = load_info_dict(VAR_PATH)


def regression_performance(y_true, y_pred):
    """Helper function to evaluate performance of regresson models
    Args:
        y_true (list): True label
        y_pred [list): Predicted label
    Return: MAE,MSE, R-squared
    """

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2


# def classification_performance(y_true, y_pred):
#     """Helper function to evaluate performance of classification models
#     Args:
#         y_true (list): True label
#         y_pred (list): Predicted label
#     Returns:
#         Accuracy, ROC_AUC score, Precision, Recall, F1-Score
#     """
#     #roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovo')
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)

#     # return accuracy, roc_auc, precision, recall, f1
#     return accuracy, precision, recall, f1

def classification_performance(y_true, y_pred):
    """Helper function to evaluate performance of classification models
    Args:
        y_true (list): True label
        y_pred (list): Predicted label
    Returns:
        Accuracy, ROC_AUC score, Precision, Recall, F1-Score
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="micro")
    recall = recall_score(y_true, y_pred, average="micro")
    f1 = f1_score(y_true, y_pred, average="micro")

    return accuracy, precision, recall, f1


def regression_models(X, y, model_type, norm=False, alpha=1.0):
    """Regression models 
    Args:
        X (2D array): Regressor
        y (list): label
        model_type (String): "Linear", "Lasso", "Ridge", "LassoLars", "Bayesian Ridge", "Elastic Net"
        normalize (boolean): Normalize or not (if applicable) 
    Returns:
        regression model, MAE, MSE, R-squared 
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    if model_type == "Linear":
        reg = LinearRegression(normalize=norm).fit(X_train, y_train)
    elif model_type == "Lasso":
        reg = Lasso(alpha=alpha, normalize=norm).fit(X_train, y_train)
    elif model_type == "Ridge":
        reg = Ridge(alpha=alpha, normalize=norm).fit(X_train, y_train)
    elif model_type == "LassoLars":
        reg = LassoLars(alpha=alpha, normalize=norm).fit(X_train, y_train)
    elif model_type == "Bayesian Ridge":
        reg = BayesianRidge(normalize=norm).fit(X_train, y_train)
    elif model_type == "Elastic Net":
        reg = ElasticNet(alpha=alpha, normalize=norm).fit(X_train, y_train)
    else:
        return None
    y_pred = reg.predict(X_test)
    mae, mse, r2 = regression_performance(y_test, y_pred)
    return reg, mae, mse, r2


def classification_models(X, y, model_type, norm=False, C=1.0):
    """Classification models
    Args:
         X (2D array): Regressor
        y (list): label
        model_type (String): "Logistic", "LDA"
        normalize (boolean): Normalize or not (if applicable) 
    Returns:
        classification model, accuracy, roc_auc, precision, recall, f1
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    if norm:
        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit_transform(X_test)
    if model_type == "Logistic":
        clf = LogisticRegression(C=1/C).fit(X_train, y_train)
    elif model_type == "LDA":
        clf = LinearDiscriminantAnalysis().fit(X_train, y_train)
    else:
        return None
    y_pred = clf.predict(X_test)
    accuracy, precision, recall, f1 = classification_performance(
        y_test, y_pred)

    cfn_matrix = confusion_matrix(y_test, y_pred)
    heatmap_filename = confusion_matrix_heatmap(y_test, y_pred)
    # return clf, accuracy, roc_auc, precision, recall, f1
    return clf, accuracy, precision, recall, f1, cfn_matrix, heatmap_filename

'''
def confusion_matrix_heatmap(y_true, y_pred):
    cfn_matrx = confusion_matrix(y_true, y_pred)
    heatmap = px.imshow(cfn_matrx)
    heatmapJSON = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)
    return heatmapJSON
'''
def confusion_matrix_heatmap(y_true, y_pred):
    cfn_matrix = confusion_matrix(y_true, y_pred)
    heatmap = sns.heatmap(cfn_matrix)
    plt.savefig('heatmap.png')
    plt.clf()
    return 'heatmap.png'
    


def reg_risk_factor_analysis(model, cols, nof):
    """Return Risk Factor Analysis table for regression model
        Args:
            model (model object): risk factor analysis  regression model
            cols (list): feature name list
            nof (int): number of factors to be displayed
        Returns:
            list of dict: Risk Factor Analysis table
        """
    coef = model.coef_.reshape(-1)
    sign = np.where(np.array(coef) > 0, '+', '-').tolist()
    sort_coef = sorted(np.abs(coef), reverse=True)
    sort_index = sorted(
        range(len(coef)), key=lambda k: np.abs(coef)[k], reverse=True)
    return [{"Rank": i+1, "Factor": "{}:{}".format(cols[sort_index[i]], var_info.get(cols[sort_index[i]]).get('Label')),
             "Absolute Weight": round(sort_coef[i], 5), "Sign": sign[sort_index[i]]} for i in range(nof)]

def encoding(number, variable):
    f = open("final_codebook.txt", "r", encoding="utf-8")
    lines = f.readlines()
    string = "SAS Variable Name: " + variable
    for i in range(len(lines)):
        if string in lines[i]:
            for j in range(i, i+10):
                if "Yes" in lines[j] and '“Yes”' not in lines[j]: #this means this is a case where there is only 1,2,7,9    such as HIVTST6 or CHCKDNY1
                    if number == 1:
                        output = lines[j]
                    elif number == 2:
                        output = lines[j+1]
                    elif number == 7:
                        output = lines[j+2]
                    elif number == 9:
                        output = lines[j+3]
                    if "—" not in output:
                        return output
                    elif "—" in output:
                        idx = output.find("—")
                        return output[:idx]
            #now in this case, this is a case where it's not only 1,2,7,9      such as PERSDOC2
            for a in range(i, i+20):
                if "Yes" in lines[a]:
                    numbers = []
                    for k in range(i, i+20):
                        if lines[k][0].isdigit() == True:
                            numbers.append(int(lines[k][0]))
                    for x in range(len(numbers)):
                        if number == numbers[x]:
                            output = lines[a + x]
                            if "—" not in output:
                                return output
                            elif "—" in output:
                                idx = output.find("—")
                                return output[:idx]


def clf_risk_factor_analysis(model, cols, nof, variable):
    """Return Risk Factor Analysis table for classification model 
        Args:
            model (model object): risk factor analysis classification model
            cols (list): feature name list
            nof (int): number of factors to be displayed
            variable (str): the variable type
        Returns:
            list of dict: Risk Factor Analysis table
        """
    classes = model.classes_
    rfa_tab = []
    class_idx = 0
    for lst in model.coef_:
        sign = np.where(np.array(lst) > 0, '+', '-').tolist()
        sort_coef = sorted(np.abs(lst), reverse=True)
        sort_index = sorted(
            range(len(lst)), key=lambda k: np.abs(lst)[k], reverse=True)
        '''
        rfa_tab += [{"Rank": classes[class_idx],
                     "Factor": "", "Absolute Weight": "", "Sign": ""}]
        '''
        number = classes[class_idx]
        unencoded = encoding(number, variable)
        unencoded = "Category: " + unencoded
        empty_cell = [{"Rank": "",
                    "Factor": "", "Absolute Weight": "", "Sign": ""}]
        if class_idx != 0:
            rfa_tab += empty_cell
            rfa_tab += empty_cell
            rfa_tab += empty_cell           
        rfa_tab += [{"Rank": unencoded,
                     "Factor": "", "Absolute Weight": "", "Sign": ""}]
        try:
            rfa_tab += [{"Rank": i+1, "Factor": "{}:{}".format(cols[sort_index[i]], var_info.get(cols[sort_index[i]]).get(
                'Label')), "Absolute Weight": round(sort_coef[i], 5), "Sign": sign[sort_index[i]]} for i in range(nof)]
        except AttributeError:
            print("Variable info error")
        class_idx += 1
    return rfa_tab
