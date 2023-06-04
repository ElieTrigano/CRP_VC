# Corporate Research Project - DSBA x  Vestiaire Collective

## Objective 

**Enhancing customer retention through predictive analysis of repeat behavior.**

## Context 

Vestiaire Collective is a leading second hand luxury marketplace with a community of more than 15M people around the world. Vestiaire Collective aims at changing the fashion industry by bringing circularity.

Today, one of the most important challenge that Vestiaire Collective faces is transforming first-time buyers into loyal repeat customers.
To stimulate a subsequent purchase vouchers are recognised as an effective lever, and we aim to tailor the voucher value based on each customer's potential.

This project is in the scope of the Data Science & Business Analytics Masters at ESSEC & CentraleSupélec. 
We are a group of 5 students working on this task part time for a 6 month period. 

## Project

Our project will focus on Vestiaire Collective's pressing pain point.

We have approached the business issue by providing several options:
- A Weighted Composite Score composed of the probability of repeatness of a customers and current value 
- A Weighted Composite Score composed of the probability of repeatness and CLTV prediction 

Our final product is the first point above as it suits better the data we had at disposal and Vestiaire Collective's current need.
Also, we have provided an Uplift Model to integrate after a first test phase in order to assess the effectiveness of our model.

## Data 

The data at diposal was 50k unique buyers from Vestiaire Collective in the past year of 2022 extracted directly from Snowflake.

We have features that are concentrated around several main areas of the platform:
- Likes & Comment: the interaction between the buyers and products
- Transactions: dataset that contains important information such as demographic, payment type, user segment, clustering, current customer value, etc. 

## Pre-requisites 

You can download the libraries from the requirement.txt file in our repository.
## Files 

An overview of each files:
- app.py: main python file for our Flask app
- data_processing.ipynb : notebook used in order to merge the 3 raw datasets pulled from Vestiaire Collective's datalake
- data_processing.py: functions used in data processing step used for the app
- df_model.csv: final dataset used for the modelling purposes
- EDA_model.ipynb: EDA on the final dataset df_model
- model_preprocessing.py: functions used in the modelling part of the project, it contains our methodology for Feature Selection & Engineering, Scaling and others
- modeling.py: functions used for Model Selection, Hyperparameter Tuning, Final model evaluation and implementation
- model_products.ipynb: notebook of the modelling implementation phase with the two models and the demonstration of a Uplift Model
- requirements.txt: file with packages used in our project

## Contributors:

- Shubham AGARWAL,  b00802820@essec.edu
- Vanshika JAIN, b00798733@essec.edu
- Thomas TAYLOR, thomas.taylor@essec.edu
- Elie TRIGANO, b00792972@essec.edu
- Shiyun WANG, b00802405@essec.edu

Feel free to contact us for more information.
