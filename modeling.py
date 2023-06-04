# Python file that holds all the functions used in the modelling part of the model.ipynb notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import random
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import shap

import time
import pickle as pkl
from scipy import stats

import xgboost as xgb

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_regression, RFE

import category_encoders as ce

# for classification
from sklearn.metrics import confusion_matrix, precision_recall_curve, classification_report, accuracy_score, precision_score, recall_score, f1_score,multilabel_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# for regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Model 1

## Model performance 

def classification_model_testing(df, X_train, X_test, y_train, y_test):
    '''

    Function that testes all the classification model imported into the notebook to have a global idea of the best model to use
    It should output a pandas dataframe called df_classification_result where each row represents the result of one model
    The columns should be the name of the model, the accuracy score, the precision score, the recall score and the f1 score

    Parameters
    ----------
    df : pandas dataframe
    The dataframe to use to train the model

    '''

    # Import the models

    models = {'Logistic Regression': LogisticRegression(), 'KNN': KNeighborsClassifier(), #'SVM': SVC(), 
              'Decision Tree': DecisionTreeClassifier(), 'Random Forest': RandomForestClassifier( n_jobs=-1),
              'Gradient Boosting': GradientBoostingClassifier(), 'HIstGradientBoosting': HistGradientBoostingClassifier(),
              'XGBoost': XGBClassifier(verbosity=0, n_jobs=-1, tree_method='gpu_hist', gpu_id=0, random_state=42, objective='binary:logistic'),
              'LightGBM': LGBMClassifier(boosting_type='gbdt', objective='binary', n_jobs=-1, random_state=42, is_unbalance= True)}
              #'CatBoost': CatBoostClassifier(verbose=0, random_state=42, task_type='GPU', eval_metric='F1')}
    
    df_classification_result = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1'])

    for name, model in models.items():
        start = time.time()

        cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)

        # Perform 7-fold cross validation

        metrics = ['accuracy', 'precision', 'recall', 'f1']

        cv_score = cross_validate(model, X_train, y_train, cv=cv, scoring=metrics, return_train_score=True)

        # Get the mean of the scores

        accuracy = cv_score['test_accuracy'].mean()
        precision = cv_score['test_precision'].mean()
        precision_std = cv_score['test_precision'].std()
        recall = cv_score['test_recall'].mean()
        recall_std = cv_score['test_recall'].std()
        f1 = cv_score['test_f1'].mean()
        f1_std = cv_score['test_f1'].std()

        # Append the scores to the df_classification_result dataframe

        df_classification_result = df_classification_result.append({'Model': name, 'Accuracy': accuracy, 'Precision': precision, 'Precision_Std': precision_std, 'Recall': recall, 'Recall_Std': recall_std, 'F1': f1, 'F1_std': f1_std}, ignore_index=True)
        end = time.time()
        print(f'{name} has been fitted and evaluated')
        print(f'It took {end-start} seconds to fit and evaluate the model')
        print('--------------------------------------')

    print('Finished fitting and evaluating all the models')

    print('--------------------------------------')

    df_classification_result = df_classification_result.sort_values(by='F1', ascending=False)

    df_classification_result.reset_index(drop=True, inplace=True)

    # plot the f1 score of each model to compare them 

    plt.figure(figsize=(20, 10))
    sns.barplot(x='Model', y='F1', data=df_classification_result)
    plt.title('F1 score Comparison')
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    plt.show()

    return df_classification_result      

## Hyperparameter Tuning 

# function for hyper parameter tuning LightGBM 
from skopt import BayesSearchCV
from skopt.space import Real, Integer

def hyperparameter_tuning_lgbm_class(X_train, X_test, y_train, y_test, colnames):
    '''
    Function that performs hyperparameter tuning on LightGBM model

    Parameters
    ----------
    X_train : pandas dataframe
        The train set

    X_test : pandas dataframe
        The test set

    y_train : pandas dataframe
        The train set

    y_test : pandas dataframe
        The test set

    Returns
    -------
    best_model : LightGBM model
        The best model after hyperparameter tuning

    '''
    # Create the parameter grid
    param_space = {
        'learning_rate': Real(0.01, 1, prior='log-uniform'),
        'n_estimators': Integer(100, 10000),
        'num_leaves': Integer(2, 100),
        'min_data_in_leaf': Integer(0, 300),
        'max_depth': Integer(1, 16),
        'min_child_samples': Integer(10, 100),
        'subsample': Real(0.01, 1.0),
        'subsample_freq': Integer(0, 10),
        'colsample_bytree': Real(0.5, 1.0),
        'reg_alpha': Real(0.0, 0.5),
        'reg_lambda': Real(0.0, 0.5)
    }

    # Create the model to use for hyperparameter tuning
    model = LGBMClassifier(boosting_type='gbdt', objective='binary', n_jobs=-1, random_state=42, is_unbalance= True)

    start = time.time()

    cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)

    # Setup RandomizedSearchCV
    bayes_cv_tuner = BayesSearchCV(
        model,
        param_space,
        n_iter=100,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        random_state=42,
        scoring='recall'
    )

    # Fit the RandomizedSearchCV version of model

    bayes_cv_tuner.fit(X_train, y_train)

    # Find the best model hyperparameters
    bayes_cv_tuner.best_params_

    end = time.time()

    print(f'It took {end-start} seconds or {(end-start) / 60} minutes to fit and evaluate the model')

    print('--------------------------------------')
    #print best score and params

    print('Best score: ', bayes_cv_tuner.best_score_)
    print('Best params: ', bayes_cv_tuner.best_params_)

    print('--------------------------------------')

    # Evaluate the RandomizedSearchCV model

    y_pred = bayes_cv_tuner.predict(X_test)

    # Keep the the predicted probabilty in an array for later use
    
    y_pred_proba = bayes_cv_tuner.predict_proba(X_test)[:,1]

    # Save the predicted probabilty in a numpy array in the preds folder

    np.save('preds/REPEATER_LightGBM.npy', y_pred_proba)

    # Evaluate the model

    accuracy = accuracy_score(y_test, y_pred)

    print(f'Accuracy score: {accuracy}')

    precision = precision_score(y_test, y_pred)

    print(f'Precision score: {precision}')

    recall = recall_score(y_test, y_pred)

    print(f'Recall score: {recall}')

    f1 = f1_score(y_test, y_pred)

    print(f'F1 score: {f1}')
    print('--------------------------------------')

    # put results in a dataframe

    df_classification_result = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1'])

    df_classification_result = df_classification_result.append({'Model': 'LightGBM_tuned', 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}, ignore_index=True)

    # Create a confusion matrix

    print('Confusion matrix: \n', confusion_matrix(y_test, y_pred))

    print('--------------------------------------')

    # Create a classification report

    print('Classification report: \n', classification_report(y_test, y_pred))

    print('--------------------------------------')
    
    # plot confusion matrix

    plt.figure(figsize=(10, 10))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    print('--------------------------------------')

    # Create a dataframe with the feature importance

    feature_importance = pd.DataFrame({'Feature': colnames, 'Importance': bayes_cv_tuner.best_estimator_.feature_importances_})

    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    # Plot the feature importance

    plt.figure(figsize=(10, 10))

    sns.barplot(x=feature_importance['Importance'], y=feature_importance['Feature'])

    plt.title('Feature Importance')

    plt.show()

    print('--------------------------------------')

    # plot SHAP values

    print('SHAP values plot: ')

    explainer = shap.TreeExplainer(bayes_cv_tuner.best_estimator_)

    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=colnames)

    print('--------------------------------------')

        # plot precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='b')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid(True)
    plt.show()

    # save the model as a pickle file in the models folder

    with open('models/REPEATER_LightGBM.pkl', 'wb') as file:
        pkl.dump(bayes_cv_tuner.best_estimator_, file)
    

    return bayes_cv_tuner.best_estimator_, df_classification_result, y_pred_proba

## Composite Score

from sklearn.preprocessing import MinMaxScaler

def calculate_composite_score(repeater_pred_proba, X_test_repeater_scaled, df_model, weight_repeater=0.75, weight_cv=0.25):
    # Create a DataFrame to store the results
    composite_df = pd.DataFrame()

    # Get the index of the test samples
    id_buyer_test = X_test_repeater_scaled.index

    # Retrieve CLTV values for samples in X_test
    test_cltv = df_model.loc[id_buyer_test, 'CLTV']

    # Add the test_CLTV to X_test_repeater_scaled
    X_test_repeater_scaled['CLTV'] = test_cltv.values

    # Calculate composite score
    composite_df['ID_BUYER'] = id_buyer_test
    composite_df['REPEATER_pred_proba'] = repeater_pred_proba
    composite_df['CLTV'] = test_cltv.values

    # Normalise the CLTV values between 0 and 1
    scaler = MinMaxScaler()
    composite_df['CLTV'] = scaler.fit_transform(composite_df[['CLTV']])

    composite_df['COMPOSITE_SCORE'] = (weight_repeater * composite_df['REPEATER_pred_proba']) + (weight_cv * composite_df['CLTV'])

    # Sort the composite_df by COMPOSITE_SCORE in descending order
    composite_df = composite_df.sort_values(by='COMPOSITE_SCORE', ascending=False)
    composite_df.reset_index(drop=True, inplace=True)

    return composite_df

# Model 2

## Model performance 

def regression_model_testing(df, X_train, X_test, y_train, y_test):
    '''

    Function that testes all the regression model imported into the notebook to have a global idea of the best model to use
    It should output a pandas dataframe called df_regression_result where each row represents the result of one model
    The columns should be the name of the model, the mean absolute error, the mean squared error and the root mean squared error

    Parameters
    ----------
    df : pandas dataframe
    The dataframe to use to train the model

    '''

    # Import the models

    models = {'Linear Regression': LinearRegression(), 'Ridge Regression': Ridge(), 'Lasso Regression': Lasso(),
                'ElasticNet Regression': ElasticNet(), 'Decision Tree': DecisionTreeRegressor(), 'Random Forest': RandomForestRegressor( n_jobs=-1),
                'Gradient Boosting': GradientBoostingRegressor(), #'CatBoost': CatBoostRegressor(verbose=0), 
                'XGBoost': XGBRegressor(n_jobs=-1, random_state=42, verbosity=0, objective='reg:squarederror', booster='gbtree', tree_method='gpu_hist'),
                'LightGBM': LGBMRegressor(n_jobs=-1, random_state=42, boosting_type='gbdt', objective='regression'),
                'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42)}
    
    df_regression_result = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2'])

    for name, model in models.items():
        start = time.time()

        cv = KFold(n_splits=7, shuffle=True, random_state=42)

        # Perform 7-fold cross validation

        metrics = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']

        cv_score = cross_validate(model, X_train, y_train, cv=cv, scoring=metrics, return_train_score=True)

        # Get the mean of the scores

        mae = -cv_score['test_neg_mean_absolute_error'].mean()
        mse = -cv_score['test_neg_mean_squared_error'].mean()
        rmse = np.sqrt(-cv_score['test_neg_mean_squared_error'].mean())
        r2 = cv_score['test_r2'].mean()

        # Append the scores to the df_regression_result dataframe

        df_regression_result = df_regression_result.append({'Model': name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}, ignore_index=True)
        end = time.time()
        print(f'{name} has been fitted and evaluated')
        print(f'It took {end-start} seconds to fit and evaluate the model')
        print('--------------------------------------')


    print('Finished fitting and evaluating all the models')
    
    df_regression_result = df_regression_result.sort_values(by='R2', ascending=False)

    df_regression_result.reset_index(drop=True, inplace=True)

    # plot the R2 and RMSE of each model to compare them

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    sns.barplot(x='Model', y='R2', data=df_regression_result, ax=ax[0])

    sns.barplot(x='Model', y='RMSE', data=df_regression_result, ax=ax[1])

    ax[0].set_title('R2 score Comparison')

    ax[1].set_title('RMSE score Comparison')

    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)

    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)

    plt.show()
    
    return df_regression_result


## Hyperparameter Tuning 

### XGBoost

from skopt import BayesSearchCV
from skopt.space import Real, Integer

def xgb_evaluation(df, X_train, X_test, y_train, y_test, colnames):
    '''
    df : dataframe qui contient les données à analyser

    outliers: booléen qui permet de supprimer les outliers ou non
    -> outliers = True : on supprime les outliers
    -> outliers = False : on ne supprime pas les outliers
    
    Fonction qui permet d'effectuer une Grid Search sur les hyperparamètres du modèle XGBoost
    Ensuite, on évalue le modèle avec les meilleurs hyperparamètres

    return: y_pred : les prédictions du modèle, model : le modèle avec les meilleurs hyperparamètres

    '''
    # Grid Search
    # choix des hyperparamètres à tester pour le modèle XGBoost
    param_grid = {
    'learning_rate': Real(0.01, 1.0, prior='log-uniform'),
    'n_estimators': Integer(50, 1000),
    'max_depth': Integer(2, 10),
    'subsample': Real(0.1, 1.0, prior='uniform'),
    'colsample_bytree': Real(0.1, 1.0, prior='uniform'),
    'reg_lambda': Real(1e-9, 1000, prior='log-uniform'),
    'reg_alpha': Real(1e-9, 1.0, prior='log-uniform'),
    'gamma': Real(1e-9, 0.5, prior='log-uniform'),
    'min_child_weight': Integer(1, 10)}


    # cross validation avec 5 folds et mélange des données à chaque itération
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # création du modèle
    xgb_model = XGBRegressor(objective ='reg:squarederror', n_jobs=-1, tree_method='gpu_hist', random_state=42)

    start = time.time() # début du chronomètre
    
    bayes_cv_tuner = BayesSearchCV(
        xgb_model,
        param_grid,
        n_iter=100,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        random_state=42)

    # fit the model with the training data
    bayes_cv_tuner.fit(X_train, y_train)

    # print the best parameters and score
    print(bayes_cv_tuner.best_params_)
    print(bayes_cv_tuner.best_score_)

    # Evaluation du modèle avec les meilleurs hyperparamètres
    xgtrain = xgb.DMatrix(X_train, label=y_train)
    xgtest = xgb.DMatrix(X_test, label=y_test)

    # create a new model with the best parameters
    gbr = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, tree_method='gpu_hist', random_state=42, **bayes_cv_tuner.best_params_)

    # train the model
    model_test = xgb.train(gbr.get_xgb_params(), xgtrain, num_boost_round=1000, early_stopping_rounds=10, evals=[(xgtest, 'test')], verbose_eval=False)

    # make predictions with the trained model
    y_pred = model_test.predict(xgtest)

    # save the preds in numpy array

    np.save('preds/CLTV_pred_xgb.npy', y_pred)

    end= time.time() # fin du chronomètre

    print(f'It took {end-start} seconds to tune, fit and evaluate the model')

    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    MAE = mean_absolute_error(y_test, y_pred)
    R2 = r2_score(y_test, y_pred)

    print('---------')

    print(f'MSE = {MSE}')
    print(f'RMSE = {RMSE}')
    print(f'MAE = {MAE}')
    print(f'R2 = {R2}')

    print('---------')

    # create a dataframe with the model and scores 

    df_model = pd.DataFrame({'Model': ['XGBoost_Tuned'], 'MSE': [MSE], 'RMSE': [RMSE], 'MAE': [MAE], 'R2': [R2]})


    print('Evaluation finie !')

    print('---------')

    # plot feature importance pour le modèle XGBoost

    feature_importance = model_test.get_score(importance_type='total_gain')
    feature_importance = list(feature_importance.values())
    sorted_feature_importance = sorted(feature_importance, reverse=False)
    
    plt.figure(figsize=(10, 10))
    sns.barplot(x=sorted_feature_importance, y=colnames, orient='h')
    plt.title('XGBoost Feature importance')
    plt.show()

    print('---------')

    # plot residual plot pour le modèle XGBoost

    plt.figure(figsize=(10, 10))
    sns.regplot(x=y_test, y=y_pred)
    plt.xlabel('Valeur réelle')
    plt.ylabel('Valeur prédite')

    plt.title(f'Régression du modèle XGBoost, R2 = {np.round(r2_score(y_test, y_pred),4)} %')

    plt.show()

    print('-------')
    print('Fin de l\'évaluation du modèle XGBoost !') 

    # save the model as a pickle file in the models folder

    with open('models/CLTV_xgb.pkl', 'wb') as file:
        pkl.dump(model_test, file)

    return df_model, model_test, y_pred

### LightGBM

# hyperparamter tuning for LightGBM
from skopt import BayesSearchCV
from skopt.space import Real, Integer

def hyperparameter_tuning_lgbm_reg(X_train, X_test, y_train, y_test, colnames):
    '''
    Function that performs hyperparameter tuning on LightGBM model

    Parameters
    ----------
    X_train : pandas dataframe
        The train set

    X_test : pandas dataframe
        The test set

    y_train : pandas dataframe
        The train set

    y_test : pandas dataframe
        The test set

    Returns
    -------
    best_model : LightGBM model
        The best model after hyperparameter tuning

    '''

    # Create the parameter grid
    param_space = {
        'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
        'n_estimators': Integer(100, 10000),
        'num_leaves': Integer(2, 100),
        'min_data_in_leaf': Integer(0, 300),
        'max_depth': Integer(1, 16),
        'min_child_samples': Integer(10, 100),
        'subsample': Real(0.01, 1.0),
        'subsample_freq': Integer(0, 10),
        'colsample_bytree': Real(0.5, 1.0),
        'reg_alpha': Real(0.0, 0.5),
        'reg_lambda': Real(0.0, 0.5)
    }
        
    # Create the model to use for hyperparameter tuning
    model = LGBMRegressor(boosting_type='gbdt', n_jobs=-1, random_state=42)

    cv = KFold(n_splits=7, shuffle=True, random_state=42)

    start= time.time()
    # Setup RandomizedSearchCV
    bayes_cv_tuner = BayesSearchCV(
        model,
        param_space,
        n_iter=100,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        random_state=42,
        scoring='neg_mean_squared_error'
    )
    
    # Fit the RandomizedSearchCV version of model

    bayes_cv_tuner.fit(X_train, y_train)

    # Find the best model hyperparameters
    bayes_cv_tuner.best_params_

    end = time.time()

    print(f'Bayesian Search took {end - start} seconds')

    #print best score and params

    print('Best score: ', bayes_cv_tuner.best_score_)
    print('Best params: ', bayes_cv_tuner.best_params_)

    # Evaluate the RandomizedSearchCV model

    y_pred = bayes_cv_tuner.predict(X_test)

    #keep the y_pred in a file for later use 

    np.save('preds/CLTV_pred_lgbm.npy', y_pred)

    # Evaluate the model

    print(f'MSE score: {mean_squared_error(y_test, y_pred)}')
    print(f'RMSE score: {np.sqrt(mean_squared_error(y_test, y_pred))}')
    print(f'MAE score: {mean_absolute_error(y_test, y_pred)}')
    print(f'R2 score: {r2_score(y_test, y_pred)}')


    # Create a dataframe with the feature importance

    feature_importance = pd.DataFrame({'Feature': colnames, 'Importance': bayes_cv_tuner.best_estimator_.feature_importances_})

    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    # Plot the feature importance

    plt.figure(figsize=(10, 10))

    sns.barplot(x=feature_importance['Importance'], y=feature_importance['Feature'])

    plt.title('Feature Importance')

    plt.show()


    # plot residual plot pour le modèle LightGBM

    plt.figure(figsize=(10, 10))
    sns.regplot(x=y_test, y=y_pred)
    plt.xlabel('Valeur réelle')
    plt.ylabel('Valeur prédite')
    plt.title(f'Régression du modèle LightGBM, R2 = {np.round(r2_score(y_test, y_pred),4)} %')
    plt.show()

    # plot SHAP values

    explainer = shap.TreeExplainer(bayes_cv_tuner.best_estimator_)

    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=colnames)

    # save the model as a pickle file in the models folder

    with open('models/CLTV_lgbm.pkl', 'wb') as file:
        pkl.dump(bayes_cv_tuner.best_estimator_, file)

    return bayes_cv_tuner.best_estimator_, y_pred

## Composite Score

# Use the predicted Repeater rate & the predicted CLTV  

def WA_Scoring(X_test_REPEATER,repeater_pred_proba, cltv_pred_lgbm, weight_repeater=0.8, weight_CLTV = 0.2):

    WA_df = X_test_REPEATER.copy()

    # add repeater_pred_proba and cltv_pred_lgbm to WA_df

    WA_df['repeater_pred_proba'] = repeater_pred_proba

    WA_df['cltv_pred_lgbm'] = cltv_pred_lgbm

    # Normalize the CLTV values between 0 and 1 using MinMaxScaler

    scaler = MinMaxScaler()

    WA_df['cltv_pred_lgbm'] = scaler.fit_transform(WA_df[['cltv_pred_lgbm']])

    composite_score = weight_repeater * WA_df['repeater_pred_proba'] + weight_CLTV * WA_df['cltv_pred_lgbm']

    WA_df['Composite_Score'] = composite_score

    columns_to_keep = ['repeater_pred_proba', 'cltv_pred_lgbm', 'Composite_Score']

    WA_df = WA_df[columns_to_keep]

    WA_df = WA_df.sort_values(by='Composite_Score', ascending=False)

    return WA_df
