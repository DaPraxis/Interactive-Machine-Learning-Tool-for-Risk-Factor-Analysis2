import ast
import numpy as np
from numpy.lib.function_base import percentile
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoLars, ElasticNet, BayesianRidge, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, log_loss, precision_recall_fscore_support
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
    accuracy = round(accuracy_score(y_true, y_pred), 4)
    precision = list(precision_score(y_true, y_pred, average = None))    #first column (actually the first one is the categories one)
    for a in range(len(precision)):
        precision[a] = round(precision[a], 4)
    recall = list(recall_score(y_true, y_pred, average = None))   #second column
    for b in range(len(recall)):
        recall[b] = round(recall[b], 4)
    f1 = list(f1_score(y_true, y_pred, average = None))    #third column
    for c in range(len(f1)):
        f1[c] = round(f1[c], 4)
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
    heatmap = sns.heatmap(cfn_matrix, annot=True)
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
    if variable == "_RFHLTH":
        dict = {1:"Good or Better Health", 2: "Fair or Poor Health", 9: "Don't know/Not Sure or Refused/Missing"}
        return dict[number]
    if variable == "_PHYS14D":
        dict = {1:"Zero days when physical health not good", 2: "1-13 days when physical health not good", 3: "14+ days when physical health not good", 9: "Don't know/Refused/Missing"}
        return dict[number]
    if variable == "_MENT14D":
        dict = {1:"Zero days when mental health not good", 2: "1-13 days when mental health not good", 3: "14+ days when mental health not good", 9: "Don't know/Refused/Missing"}
        return dict[number]
    if variable == "_HCVU651":
        dict = {1: "Have health care coverage", 2: "Do not have health care coverage", 9: "Don't know/Not Sure, Refused or Missing"}
        return dict[number]
    if variable == "_TOTINDA":
        dict = {1: "Had physical activity or exercise", 2: "No physical activity or exercise in last 30 days", 9: "Don't know/Reefused/Missing"}
        return dict[number]
    if variable == "_MICHD":
        dict = {1: "Reported having MI or CHD", 2: "Did not report having MI or CHD"}
        return dict[number]
    if variable == "_LTASTH1" or variable == "_CASTHM1":
        dict = {1: "No", 2: "Yes", 9: "Don't know/Not Sure or Refused/Missing"}
        return dict[number]
    if variable == "_ASTHMS1":
        dict = {1: "Current", 2: "Former", 3: "Never", 9: "Don't know/Not Sure or Refused/Missing"}
        return dict[number]
    if variable == "_DRDXAR1":
        dict = {1: "Diagnosed with arthiritis", 2: "Not diagnosed with arthritis"}
        return dict[number]
    if variable == "_EXTETH3":
        dict = {1: "Not at risk", 2: "At risk", 9: "Don't know/Not Sure or Refused/Missing"}
        return dict[number]
    if variable == "_DENVST3":
        dict = {1: "Yes", 2: "No", 9: "Don't know/Not Sure or Refused/Missing"}
        return dict[number]
    if variable == "_AGEG5YR":
        dict = {1: "Age 18 to 24", 2: "Age 25 to 29", 3: "Age 30 to 34", 4: "Age 35 to 39", 5: "Age 40 to 44", 6: "Age 45 to 49", 7: "Age 50 to 54", 8: "Age 55 to 59", 9: "Age 60 to 64", 10: "Age 65 to 69", 11: "Age 70 to 74", 12: "Age 75 to 79", 13: "Age 80 or older", 14: "Don't know/Refused/Missing"}
        return dict[number]
    if variable == "_AGE65YR":
        dict = {1: "Age 18 to 64", 2: "Age 65 or older", 3: "Don't Know/Refused/Missing"}
        return dict[number]
    if variable == "_AGE_G":
        dict = {1: "Age 18 to 24", 2: "Age 25 to 34", 3: "Age 35 to 44", 4: "Age 45 to 54", 5: "Age 55 to 64", 6: "Age 65 or older"}
        return dict[number]
    if variable == "_BMI5CAT":
        dict = {1: "Underweight", 2: "Normal Weight", 3: "Overweight", 4: "Obese"}
        return dict[number]
    if variable == "_RFBMI5":
        dict = {1: "No", 2: "Yes", 9: "Don't Know/Refused/Missing"}
        return dict[number]
    if variable == "_CHLDCNT":
        dict = {1: "No children in household", 2: "One child in household", 3: "Two children in household", 4: "Three children in household", 5: "Four children in household", 6: "Five or more children in household"}
        return dict[number]
    if variable == "_EDUCAG":
        dict = {1: "Did not graduate High School", 2: "Graduated High School", 3: "Attended College or Technical School", 4: "Graduated from College or Technical School", 9: "Don't know/Not Sure/Missing"}
        return dict[number]
    if variable == "_INCOMG":
        dict = {1: "Less than $15,000", 2: "$15,000 to less than $25,000", 3: "$25,000 to less than $35,000", 4: "$35,000 to less than $50,000", 5: "$50,000 or more", 9: "Don’t know/Not sure/Missing"}
        return dict[number]
    if variable == "_SMOKER3":
        dict = {1: "Current smoker - now smokes every day", 2: "Current smoker - now smokes some days", 3: "Former smoker", 4: "Never smoked", 9: "Don’t know/Refused/Missing"}
        return dict[number]
    if variable == "_RFSMOK3":
        dict = {1: "No", 2: "Yes", 9: "Don’t know/Refused/Missing"}
        return dict[number]
    if variable == "_RFBING5":
        dict = {1: "No", 2: "Yes", 9: "Don’t know/Refused/Missing"}
        return dict[number]
    if variable == "_RFDRHV6":
        dict = {1: "No", 2: "Yes", 9: "Don’t know/Refused/Missing"}
        return dict[number]
    if variable == "_RFSEAT2":
        dict = {1: "Always or Almost Always Wear Seat Belt", 2: "Sometimes, Seldom, or Never Wear Seat Belt", 9: "Don't Know/Not Sure or Refused/Missing"}
        return dict[number]
    if variable == "_RFSEAT3":
        dict = {1: "Always Wear Seat Belt", 2: "Don't Always Wear Seat Belt", 9: "Don't know/Not Sure or Refused/Missing"}
        return dict[number]
    if variable == "_DRNKDRV":
        dict = {1: "Have driven after having too much to drink", 2: "Have not driven after having too much to drink", 9: "Don't know/Not Sure/Refused/Missing"}
        return dict[number]
    if variable == "CHECKUP1":
        dict = {1: "Within past year", 2: "Within past 2 years", 3: "Within past 5 years", 4: "5 or more years ago", 7: "Don't know/Not sure", 8: "Never", 9: "Refused"}
        return dict[number]
    if variable == "LASTDEN4":
        dict = {1: "Within the past year", 2: "Within the past 2 years", 3: "Within the past 5 years", 4: "5 or more years ago", 7: "Don't Know/Not sure", 8: "Never", 9: "Refused"}
        return dict[number]
    if variable == "RMVTETH4":
        dict = {1: "1 to 5", 2: "6 or more, but not all", 3: "All", 7: "Don't know/Not sure", 8: "None", 9: "Refused"}
        return dict[number]
    if variable == "GENHLTH":
        dict = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor", 7: "Don't know/Not sure", 9: "Refused"}
        return dict[number]
    if variable == "USENOW3":
        dict = {1: "Every day", 2: "Some days", 3: "Not at all", 7: "Don't know/Not sure", 9: "Refused"}
        return dict[number]
    if variable == "_METSTAT":
        dict = {1: "Metropolitan counties", 2: "Nonmetropolitan counties"}
        return dict[number]
    if variable == "_URBSTAT":
        dict = {1: "Urban counties", 2: "Rural counties"}
    if variable == "_IMPRACE":
        dict = {1: "White, Non-Hispanic", 2: "Black, Non-Hispanic", 3: "Asian, Non-Hispanic", 4: "American Indian/Alaskan Native, Non-Hispanic", 5: "Hispanic", 6: "Other race, Non-Hispanic"}
        return dict[number]
    if variable == "_DUALUSE":
        dict = {1: "Landline frame with a cell phone", 2: "Cell phone frame with a landline", 9: "No dual phone use"}
        return dict[number]
    if variable == "SEATBELT":
        dict = {1: "Always", 2: "Nearly always", 3: "Sometimes", 4: "Seldom", 5: "Never", 7: "Don't know/Not sure", 8: "Never drive or ride in a car", 9: "Refused"}
        return dict[number]
    if variable == "_PRACE1":
        dict = {1: "White", 2: "Black or African American", 3: "American Indian or Alaskan Native", 4: "Asian", 5: "Native Hawaiian or other Pacific Islander", 6: "Other race", 7: "No preferred race", 77: "Don't know/Not sure", 99: "Refused"}
        return dict[number]
    if variable == "_MRACE1":
        dict = {1: "White only", 2: "Black or African American only", 3: "American Indian or Alaskan Native only", 4: "Asian only", 5: "Native Hawaiian or other Pacific Islander only", 6: "Other race only", 7: "Multiracial", 77: "Don't know/not sure", 99: "Refused"}
        return dict[number]
    if variable == "_HISPANC":
        dict = {1: "Hispanic, Latino/a, or Spanish origin", 2: "Not of Hispanic, Latino/a, or Spanish origin", 9: "Don't know, Refused or Missing"}
        return dict[number]
    if variable == "_RACE":
        dict = {1: "White only, non-Hispanic", 2: "Black only, non-Hispanic", 3: "American Indian or Alaskan Native only, non-Hispanic", 4: "Asian only, non-Hispanic", 5: "Native Hawaiian or other Pacific Islander only, Non-Hispanic", 6: "Other race only, non-Hispanic", 7: "Multiracial, non-Hispanic", 8: "Hispanic", 9: "Don't know/Not sure/Refused"}
        return dict[number]
    if variable == "_RACEG21": 
        dict = {1: "Non-Hispanic White", 2: "Non-White or Hispanic", 9: "Don't know/Not sure/Refused"}
        return dict[number]
    if variable == "_RACEGR3":
        dict = {1: "White only, Non-Hispanic", 2: "Black only, Non-Hispanic", 3: "Other race only, Non-Hispanic", 4: "Multiracial, non-Hispanic", 5: "Hispanic", 9: "Don't know/Not sure/Refused"}
        return dict[number]
    if variable == "_RACE_G1":
        dict = {1: "White - Non-Hispanic", 2: "Black - Non-Hispanic", 3: "Hispanic", 4: "Other race only, non-Hispanic", 5: "Multiracial, non-Hispanic"}
        return dict[number]
    if variable == "MARITAL":
        dict = {1: "Married", 2: "Divorced", 3: "Widowed", 4: "Separated", 5: "Never married", 6: "A member of an unmarried couple", 9: "Refused"}
        return dict[number]
    if variable == "EDUCA":
        dict = {1: "Never attended school or only kindergarten", 2: "Grades 1 through 8 (elementary)", 3: "Grades 9 through 11 (some high school)", 4: "Grade 12 or GED (high school graduate)", 5: "College 1 year to 3 years (some college or technical school)", 6: "College 4 years or more (college graduate)", 9: "Refused"}
        return dict[number]
    if variable == "RENTHOM1":
        dict = {1: "Own", 2: "Rent", 3: "Other arrangment", 7: "Don't know/not sure", 9: "Refused"}
        return dict[number]
    if variable == "EMPLOY1":
        dict = {1: "Emplyed for wages", 2: "Self-employed", 3: "Out of work for 1 year or more", 4: "Out of work for less than 1 year", 5: "A homemaker", 6: "A Student", 7: "Retired", 8: "Unable to work", 9: "Refused"}
        return dict[number]
    if variable == "INCOME2":
        dict = {1: "Less than $10,000", 2: "Less than $15,000 ($10,000 to less than $15,000)", 3: "Less than $20,000 ($15,000 to less than $20,000)", 4: "Less than $25,000 ($20,000 to less than $25,000)", 5: "Less than $35,000 ($25,000 to less than $35,000)", 6: "Less than $50,000 ($35,000 to less than $50,000)", 7: "Less than $75,000 ($50,000 to less than $75,000)", 8: " $75,000 or more", 77: "Don't know/ not sure", 99: "Refused"}
        return dict[number]
    if variable == "ALCDAY5":
        number = int(number)
        str_num = str(number)
        #print (str_num)
        if str_num[0] == '1':
            return str_num[2] + " Days per week"
        elif str_num[0] == '2':
            return str_num[1:] + " Days in past 30"
        elif number == 777:
            return "Don't know/Not sure"
        elif number == 888:
            return "No drinks in past 30 days"
        elif number == 999:
            return "Refused"
    
    
    f = open("final_codebook.txt", "r", encoding="utf-8")
    lines = f.readlines()
    string = "SAS Variable Name: " + variable
    for i in range(len(lines)):
        if string in lines[i]:
            for j in range(i, i+8):
                if "Yes" in lines[j] and '“Yes”' not in lines[j] and "´Yes´" not in lines[j]: #this means this is a case where there is only 1,2,7,9    such as HIVTST6 or CHCKDNY1 or CVDINFR4
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
            #now in this case, this is a case where it's not only 1,2,7,9      such as PERSDOC2 or DIABETE3 
            for a in range(i, i+20):
                if "Yes" in lines[a] and '“Yes”' not in lines[a] and "´Yes´" not in lines[a]:
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
    categories = []
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
        unencoded = unencoded.rstrip('\n')
        categories.append(unencoded)
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
    return rfa_tab, categories