***
# Supervised Learning : Exploring Parametric Accelerated Failure Time Models for Estimating Lifetimes in Survival Data

***
### John Pauline Pineda <br> <br> *August 3, 2024*
***

* [**1. Table of Contents**](#TOC)
    * [1.1 Data Background](#1.1)
    * [1.2 Data Description](#1.2)
    * [1.3 Data Quality Assessment](#1.3)
    * [1.4 Data Preprocessing](#1.4)
        * [1.4.1 Data Cleaning](#1.4.1)
        * [1.4.2 Missing Data Imputation](#1.4.2)
        * [1.4.3 Outlier Treatment](#1.4.3)
        * [1.4.4 Collinearity](#1.4.4)
        * [1.4.5 Shape Transformation](#1.4.5)
        * [1.4.6 Centering and Scaling](#1.4.6)
        * [1.4.7 Data Encoding](#1.4.7)
        * [1.4.8 Preprocessed Data Description](#1.4.8)
    * [1.5 Data Exploration](#1.5)
        * [1.5.1 Exploratory Data Analysis](#1.5.1)
        * [1.5.2 Hypothesis Testing](#1.5.2)
    * [1.6 Predictive Model Development](#1.6)
        * [1.6.1 Premodelling Data Description](#1.6.1)
        * [1.6.2 Weibull Accelerated Failure Time Model](#1.6.2)
        * [1.6.3 Log-Normal Accelerated Failure Time Model](#1.6.3)
        * [1.6.4 Log-Logistic Accelerated Failure Time Model](#1.6.4)
    * [1.7 Consolidated Findings](#1.7)
* [**2. Summary**](#Summary)   
* [**3. References**](#References)

***

# 1. Table of Contents <a class="anchor" id="TOC"></a>

This project explores parametric **Accelerated Failure Time** models with error distributions following the **Weibull**, **Log-Normal** and **Log-Logistic** distributions using various helpful packages in <mark style="background-color: #CCECFF"><b>Python</b></mark> to analyze time-to-event data by directly modelling  the time until an event of interest occurs. The resulting predictions derived from the candidate models were evaluated in terms of their discrimination power using the Harrel's concordance index metric, and their model fit using the brier score and mean absolute error (MAE) metrics. Additionally, feature impact on model output were estimated using **Shapley Additive Explanations**. All results were consolidated in a [<span style="color: #FF0000"><b>Summary</b></span>](#Summary) presented at the end of the document. 

[Survival Analysis](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) deals with the analysis of time-to-event data. It focuses on the expected duration of time until one or more events of interest occur, such as death, failure, or relapse. This type of analysis is used to study and model the time until the occurrence of an event, taking into account that the event might not have occurred for all subjects during the study period. Several key aspects of survival analysis include the survival function which refers to the probability that an individual survives longer than a certain time, hazard function which describes the instantaneous rate at which events occur, given no prior event, and censoring pertaining to a condition where the event of interest has not occurred for some subjects during the observation period.

[Right-Censored Survival Data](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) occurs when the event of interest has not happened for some subjects by the end of the study period or the last follow-up time. This type of censoring is common in survival analysis because not all individuals may experience the event before the study ends, or they might drop out or be lost to follow-up. Right-censored data is crucial in survival analysis as it allows the inclusion of all subjects in the analysis, providing more accurate and reliable estimates.

[Accelerated Failure Time Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) are a class of survival analysis models used to analyze time-to-event data by directly modelling the survival time itself. An AFT model assumes that the effect of covariates accelerates or decelerates the life time of an event by some constant factor. The mathematical equation is represented by the logarithm of the survival time being equal to the sum of the vector of covariates multiplied to the vector of regression coefficients; and the product of the scale parameter and a random variable with a specified distribution. In an AFT model, the coefficients represent the multiplicative effect on the survival time. An exponentiated regression coefficient greater than one prolongs survival time, while a value less than one shortens it. The scale parameter determines the spread or variability of survival times. AFT models assume that the effect of covariates on survival time is multiplicative and that the survival times can be transformed to follow a specific distribution.

## 1.1. Data Background <a class="anchor" id="1.1"></a>

An open [Liver Cirrhosis Dataset](https://www.kaggle.com/code/arjunbhaybhang/liver-cirrhosis-prediction-with-xgboost-eda) from [Kaggle](https://www.kaggle.com/) (with all credits attributed to [Arjun Bhaybhang](https://www.kaggle.com/arjunbhaybhang)) was used for the analysis as consolidated from the following primary sources: 
1. Reference Book entitled **Counting Processes and Survival Analysis** from [Wiley](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118150672)
2. Research Paper entitled **Efficacy of Liver Transplantation in Patients with Primary Biliary Cirrhosis** from the [New England Journal of Medicine](https://www.nejm.org/doi/abs/10.1056/NEJM198906293202602)
3. Research Paper entitled **Prognosis in Primary Biliary Cirrhosis: Model for Decision Making** from the [Hepatology](https://aasldpubs.onlinelibrary.wiley.com/doi/10.1002/hep.1840100102)

This study hypothesized that the evaluated drug, liver profile test biomarkers and various clinicopathological characteristics influence liver cirrhosis survival between patients.

The event status and survival duration variables for the study are:
* <span style="color: #FF0000">Status</span> - Status of the patient (C, censored | CL, censored due to liver transplant | D, death)
* <span style="color: #FF0000">N_Days</span> - Number of days between registration and the earlier of death, transplantation, or study analysis time (1986)

The predictor variables for the study are:
* <span style="color: #FF0000">Drug</span> - Type of administered drug to the patient (D-Penicillamine | Placebo)
* <span style="color: #FF0000">Age</span> - Patient's age (Days)
* <span style="color: #FF0000">Sex</span> - Patient's sex (Male | Female)
* <span style="color: #FF0000">Ascites</span> - Presence of ascites (Yes | No)
* <span style="color: #FF0000">Hepatomegaly</span> - Presence of hepatomegaly (Yes | No)
* <span style="color: #FF0000">Spiders</span> - Presence of spiders (Yes | No)
* <span style="color: #FF0000">Edema</span> - Presence of edema ( N, No edema and no diuretic therapy for edema | S, Edema present without diuretics or edema resolved by diuretics) | Y, Edema despite diuretic therapy)
* <span style="color: #FF0000">Bilirubin</span> - Serum bilirubin (mg/dl)
* <span style="color: #FF0000">Cholesterol</span> - Serum cholesterol (mg/dl)
* <span style="color: #FF0000">Albumin</span> - Albumin (gm/dl)
* <span style="color: #FF0000">Copper</span> - Urine copper (ug/day)
* <span style="color: #FF0000">Alk_Phos</span> - Alkaline phosphatase (U/liter)
* <span style="color: #FF0000">SGOT</span> - Serum glutamic-oxaloacetic transaminase (U/ml)
* <span style="color: #FF0000">Triglycerides</span> - Triglicerides (mg/dl)
* <span style="color: #FF0000">Platelets</span> - Platelets (cubic ml/1000)
* <span style="color: #FF0000">Prothrombin</span> - Prothrombin time (seconds)
* <span style="color: #FF0000">Stage</span> - Histologic stage of disease (Stage I | Stage II | Stage III | Stage IV)


## 1.2. Data Description <a class="anchor" id="1.2"></a>

1. The dataset is comprised of:
    * **418 rows** (observations)
    * **20 columns** (variables)
        * **1/20 metadata** (object)
            * <span style="color: #FF0000">ID</span>
        * **2/20 event | duration** (object | numeric)
             * <span style="color: #FF0000">Status</span>
             * <span style="color: #FF0000">N_Days</span>
        * **10/20 predictor** (numeric)
             * <span style="color: #FF0000">Age</span>
             * <span style="color: #FF0000">Bilirubin</span>
             * <span style="color: #FF0000">Cholesterol</span>
             * <span style="color: #FF0000">Albumin</span>
             * <span style="color: #FF0000">Copper</span>
             * <span style="color: #FF0000">Alk_Phos</span>
             * <span style="color: #FF0000">SGOT</span>
             * <span style="color: #FF0000">Triglycerides</span>
             * <span style="color: #FF0000">Platelets</span>
             * <span style="color: #FF0000">Prothrombin</span>
        * **7/20 predictor** (object)
             * <span style="color: #FF0000">Drug</span>
             * <span style="color: #FF0000">Sex</span>
             * <span style="color: #FF0000">Ascites</span>
             * <span style="color: #FF0000">Hepatomegaly</span>
             * <span style="color: #FF0000">Spiders</span>
             * <span style="color: #FF0000">Edema</span>
             * <span style="color: #FF0000">Stage</span>


```python
##################################
# Loading Python Libraries
##################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
%matplotlib inline

from operator import add,mul,truediv
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, brier_score_loss

from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency

from lifelines import KaplanMeierFitter
from lifelines.fitters.weibull_fitter import WeibullFitter
from lifelines.fitters.log_normal_fitter import LogNormalFitter
from lifelines.fitters.log_logistic_fitter import LogLogisticFitter
from lifelines.fitters.weibull_aft_fitter import WeibullAFTFitter
from lifelines.fitters.log_normal_aft_fitter import LogNormalAFTFitter
from lifelines.fitters.log_logistic_aft_fitter import LogLogisticAFTFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from lifelines.plotting import qq_plot
import shap

import warnings
warnings.filterwarnings('ignore')
```


```python
##################################
# Loading the dataset
##################################
cirrhosis_survival = pd.read_csv('Cirrhosis_Survival.csv')
```


```python
##################################
# Performing a general exploration of the dataset
##################################
print('Dataset Dimensions: ')
display(cirrhosis_survival.shape)
```

    Dataset Dimensions: 
    


    (418, 20)



```python
##################################
# Listing the column names and data types
##################################
print('Column Names and Data Types:')
display(cirrhosis_survival.dtypes)
```

    Column Names and Data Types:
    


    ID                 int64
    N_Days             int64
    Status            object
    Drug              object
    Age                int64
    Sex               object
    Ascites           object
    Hepatomegaly      object
    Spiders           object
    Edema             object
    Bilirubin        float64
    Cholesterol      float64
    Albumin          float64
    Copper           float64
    Alk_Phos         float64
    SGOT             float64
    Tryglicerides    float64
    Platelets        float64
    Prothrombin      float64
    Stage            float64
    dtype: object



```python
##################################
# Taking a snapshot of the dataset
##################################
cirrhosis_survival.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>N_Days</th>
      <th>Status</th>
      <th>Drug</th>
      <th>Age</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
      <th>Stage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>400</td>
      <td>D</td>
      <td>D-penicillamine</td>
      <td>21464</td>
      <td>F</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>14.5</td>
      <td>261.0</td>
      <td>2.60</td>
      <td>156.0</td>
      <td>1718.0</td>
      <td>137.95</td>
      <td>172.0</td>
      <td>190.0</td>
      <td>12.2</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4500</td>
      <td>C</td>
      <td>D-penicillamine</td>
      <td>20617</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>Y</td>
      <td>N</td>
      <td>1.1</td>
      <td>302.0</td>
      <td>4.14</td>
      <td>54.0</td>
      <td>7394.8</td>
      <td>113.52</td>
      <td>88.0</td>
      <td>221.0</td>
      <td>10.6</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1012</td>
      <td>D</td>
      <td>D-penicillamine</td>
      <td>25594</td>
      <td>M</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>S</td>
      <td>1.4</td>
      <td>176.0</td>
      <td>3.48</td>
      <td>210.0</td>
      <td>516.0</td>
      <td>96.10</td>
      <td>55.0</td>
      <td>151.0</td>
      <td>12.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1925</td>
      <td>D</td>
      <td>D-penicillamine</td>
      <td>19994</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>Y</td>
      <td>S</td>
      <td>1.8</td>
      <td>244.0</td>
      <td>2.54</td>
      <td>64.0</td>
      <td>6121.8</td>
      <td>60.63</td>
      <td>92.0</td>
      <td>183.0</td>
      <td>10.3</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1504</td>
      <td>CL</td>
      <td>Placebo</td>
      <td>13918</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>Y</td>
      <td>N</td>
      <td>3.4</td>
      <td>279.0</td>
      <td>3.53</td>
      <td>143.0</td>
      <td>671.0</td>
      <td>113.15</td>
      <td>72.0</td>
      <td>136.0</td>
      <td>10.9</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Taking the ID column as the index
##################################
cirrhosis_survival.set_index(['ID'], inplace=True)
```


```python
##################################
# Changing the data type for Stage
##################################
cirrhosis_survival['Stage'] = cirrhosis_survival['Stage'].astype('object')
```


```python
##################################
# Changing the data type for Status
##################################
cirrhosis_survival['Status'] = cirrhosis_survival['Status'].replace({'C':False, 'CL':False, 'D':True}) 
```


```python
##################################
# Performing a general exploration of the numeric variables
##################################
print('Numeric Variable Summary:')
display(cirrhosis_survival.describe(include='number').transpose())
```

    Numeric Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>N_Days</th>
      <td>418.0</td>
      <td>1917.782297</td>
      <td>1104.672992</td>
      <td>41.00</td>
      <td>1092.7500</td>
      <td>1730.00</td>
      <td>2613.50</td>
      <td>4795.00</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>418.0</td>
      <td>18533.351675</td>
      <td>3815.845055</td>
      <td>9598.00</td>
      <td>15644.5000</td>
      <td>18628.00</td>
      <td>21272.50</td>
      <td>28650.00</td>
    </tr>
    <tr>
      <th>Bilirubin</th>
      <td>418.0</td>
      <td>3.220813</td>
      <td>4.407506</td>
      <td>0.30</td>
      <td>0.8000</td>
      <td>1.40</td>
      <td>3.40</td>
      <td>28.00</td>
    </tr>
    <tr>
      <th>Cholesterol</th>
      <td>284.0</td>
      <td>369.510563</td>
      <td>231.944545</td>
      <td>120.00</td>
      <td>249.5000</td>
      <td>309.50</td>
      <td>400.00</td>
      <td>1775.00</td>
    </tr>
    <tr>
      <th>Albumin</th>
      <td>418.0</td>
      <td>3.497440</td>
      <td>0.424972</td>
      <td>1.96</td>
      <td>3.2425</td>
      <td>3.53</td>
      <td>3.77</td>
      <td>4.64</td>
    </tr>
    <tr>
      <th>Copper</th>
      <td>310.0</td>
      <td>97.648387</td>
      <td>85.613920</td>
      <td>4.00</td>
      <td>41.2500</td>
      <td>73.00</td>
      <td>123.00</td>
      <td>588.00</td>
    </tr>
    <tr>
      <th>Alk_Phos</th>
      <td>312.0</td>
      <td>1982.655769</td>
      <td>2140.388824</td>
      <td>289.00</td>
      <td>871.5000</td>
      <td>1259.00</td>
      <td>1980.00</td>
      <td>13862.40</td>
    </tr>
    <tr>
      <th>SGOT</th>
      <td>312.0</td>
      <td>122.556346</td>
      <td>56.699525</td>
      <td>26.35</td>
      <td>80.6000</td>
      <td>114.70</td>
      <td>151.90</td>
      <td>457.25</td>
    </tr>
    <tr>
      <th>Tryglicerides</th>
      <td>282.0</td>
      <td>124.702128</td>
      <td>65.148639</td>
      <td>33.00</td>
      <td>84.2500</td>
      <td>108.00</td>
      <td>151.00</td>
      <td>598.00</td>
    </tr>
    <tr>
      <th>Platelets</th>
      <td>407.0</td>
      <td>257.024570</td>
      <td>98.325585</td>
      <td>62.00</td>
      <td>188.5000</td>
      <td>251.00</td>
      <td>318.00</td>
      <td>721.00</td>
    </tr>
    <tr>
      <th>Prothrombin</th>
      <td>416.0</td>
      <td>10.731731</td>
      <td>1.022000</td>
      <td>9.00</td>
      <td>10.0000</td>
      <td>10.60</td>
      <td>11.10</td>
      <td>18.00</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Performing a general exploration of the object variables
##################################
print('object Variable Summary:')
display(cirrhosis_survival.describe(include='object').transpose())
```

    object Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Drug</th>
      <td>312</td>
      <td>2</td>
      <td>D-penicillamine</td>
      <td>158</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>418</td>
      <td>2</td>
      <td>F</td>
      <td>374</td>
    </tr>
    <tr>
      <th>Ascites</th>
      <td>312</td>
      <td>2</td>
      <td>N</td>
      <td>288</td>
    </tr>
    <tr>
      <th>Hepatomegaly</th>
      <td>312</td>
      <td>2</td>
      <td>Y</td>
      <td>160</td>
    </tr>
    <tr>
      <th>Spiders</th>
      <td>312</td>
      <td>2</td>
      <td>N</td>
      <td>222</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>418</td>
      <td>3</td>
      <td>N</td>
      <td>354</td>
    </tr>
    <tr>
      <th>Stage</th>
      <td>412.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>155.0</td>
    </tr>
  </tbody>
</table>
</div>


## 1.3. Data Quality Assessment <a class="anchor" id="1.3"></a>

Data quality findings based on assessment are as follows:
1. No duplicated rows observed.
2. Missing data noted for 12 variables with Null.Count>0 and Fill.Rate<1.0.
    * <span style="color: #FF0000">Tryglicerides</span>: Null.Count = 136, Fill.Rate = 0.675
    * <span style="color: #FF0000">Cholesterol</span>: Null.Count = 134, Fill.Rate = 0.679
    * <span style="color: #FF0000">Copper</span>: Null.Count = 108, Fill.Rate = 0.741
    * <span style="color: #FF0000">Drug</span>: Null.Count = 106, Fill.Rate = 0.746
    * <span style="color: #FF0000">Ascites</span>: Null.Count = 106, Fill.Rate = 0.746
    * <span style="color: #FF0000">Hepatomegaly</span>: Null.Count = 106, Fill.Rate = 0.746
    * <span style="color: #FF0000">Spiders</span>: Null.Count = 106, Fill.Rate = 0.746
    * <span style="color: #FF0000">Alk_Phos</span>: Null.Count = 106, Fill.Rate = 0.746
    * <span style="color: #FF0000">SGOT</span>: Null.Count = 106, Fill.Rate = 0.746
    * <span style="color: #FF0000">Platelets</span>: Null.Count = 11, Fill.Rate = 0.974
    * <span style="color: #FF0000">Stage</span>: Null.Count = 6, Fill.Rate = 0.986
    * <span style="color: #FF0000">Prothrombin</span>: Null.Count = 2, Fill.Rate = 0.995
3. 142 observations noted with at least 1 missing data. From this number, 106 observations reported high Missing.Rate>0.4.
    * 91 Observations: Missing.Rate = 0.450 (9 columns)
    * 15 Observations: Missing.Rate = 0.500 (10 columns)
    * 28 Observations: Missing.Rate = 0.100 (2 columns)
    * 8 Observations: Missing.Rate = 0.050 (1 column)
4. Low variance observed for 3 variables with First.Second.Mode.Ratio>5.
    * <span style="color: #FF0000">Ascites</span>: First.Second.Mode.Ratio = 12.000
    * <span style="color: #FF0000">Sex</span>: First.Second.Mode.Ratio = 8.500
    * <span style="color: #FF0000">Edema</span>: First.Second.Mode.Ratio = 8.045
5. No low variance observed for any variable with Unique.Count.Ratio>10.
6. High and marginally high skewness observed for 2 variables with Skewness>3 or Skewness<(-3).
    * <span style="color: #FF0000">Cholesterol</span>: Skewness = +3.409
    * <span style="color: #FF0000">Alk_Phos</span>: Skewness = +2.993


```python
##################################
# Counting the number of duplicated rows
##################################
cirrhosis_survival.duplicated().sum()
```




    0




```python
##################################
# Gathering the data types for each column
##################################
data_type_list = list(cirrhosis_survival.dtypes)
```


```python
##################################
# Gathering the variable names for each column
##################################
variable_name_list = list(cirrhosis_survival.columns)
```


```python
##################################
# Gathering the number of observations for each column
##################################
row_count_list = list([len(cirrhosis_survival)] * len(cirrhosis_survival.columns))
```


```python
##################################
# Gathering the number of missing data for each column
##################################
null_count_list = list(cirrhosis_survival.isna().sum(axis=0))
```


```python
##################################
# Gathering the number of non-missing data for each column
##################################
non_null_count_list = list(cirrhosis_survival.count())
```


```python
##################################
# Gathering the missing data percentage for each column
##################################
fill_rate_list = map(truediv, non_null_count_list, row_count_list)
```


```python
##################################
# Formulating the summary
# for all columns
##################################
all_column_quality_summary = pd.DataFrame(zip(variable_name_list,
                                              data_type_list,
                                              row_count_list,
                                              non_null_count_list,
                                              null_count_list,
                                              fill_rate_list), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count',                                                 
                                                 'Fill.Rate'])
display(all_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>N_Days</td>
      <td>int64</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Status</td>
      <td>bool</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Drug</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Age</td>
      <td>int64</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sex</td>
      <td>object</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ascites</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hepatomegaly</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Spiders</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Edema</td>
      <td>object</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Bilirubin</td>
      <td>float64</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cholesterol</td>
      <td>float64</td>
      <td>418</td>
      <td>284</td>
      <td>134</td>
      <td>0.679426</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Albumin</td>
      <td>float64</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Copper</td>
      <td>float64</td>
      <td>418</td>
      <td>310</td>
      <td>108</td>
      <td>0.741627</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Alk_Phos</td>
      <td>float64</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SGOT</td>
      <td>float64</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Tryglicerides</td>
      <td>float64</td>
      <td>418</td>
      <td>282</td>
      <td>136</td>
      <td>0.674641</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Platelets</td>
      <td>float64</td>
      <td>418</td>
      <td>407</td>
      <td>11</td>
      <td>0.973684</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Prothrombin</td>
      <td>float64</td>
      <td>418</td>
      <td>416</td>
      <td>2</td>
      <td>0.995215</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Stage</td>
      <td>object</td>
      <td>418</td>
      <td>412</td>
      <td>6</td>
      <td>0.985646</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of columns
# with Fill.Rate < 1.00
##################################
print('Number of Columns with Missing Data:', str(len(all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<1)])))
```

    Number of Columns with Missing Data: 12
    


```python
##################################
# Identifying the columns
# with Fill.Rate < 1.00
##################################
print('Columns with Missing Data:')
display(all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<1)].sort_values(by=['Fill.Rate'], ascending=True))
```

    Columns with Missing Data:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>Tryglicerides</td>
      <td>float64</td>
      <td>418</td>
      <td>282</td>
      <td>136</td>
      <td>0.674641</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cholesterol</td>
      <td>float64</td>
      <td>418</td>
      <td>284</td>
      <td>134</td>
      <td>0.679426</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Copper</td>
      <td>float64</td>
      <td>418</td>
      <td>310</td>
      <td>108</td>
      <td>0.741627</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Drug</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ascites</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hepatomegaly</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Spiders</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Alk_Phos</td>
      <td>float64</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SGOT</td>
      <td>float64</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Platelets</td>
      <td>float64</td>
      <td>418</td>
      <td>407</td>
      <td>11</td>
      <td>0.973684</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Stage</td>
      <td>object</td>
      <td>418</td>
      <td>412</td>
      <td>6</td>
      <td>0.985646</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Prothrombin</td>
      <td>float64</td>
      <td>418</td>
      <td>416</td>
      <td>2</td>
      <td>0.995215</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Identifying the rows
# with Fill.Rate < 1.00
##################################
column_low_fill_rate = all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<1.00)]
```


```python
##################################
# Gathering the metadata labels for each observation
##################################
row_metadata_list = cirrhosis_survival.index.values.tolist()
```


```python
##################################
# Gathering the number of columns for each observation
##################################
column_count_list = list([len(cirrhosis_survival.columns)] * len(cirrhosis_survival))
```


```python
##################################
# Gathering the number of missing data for each row
##################################
null_row_list = list(cirrhosis_survival.isna().sum(axis=1))
```


```python
##################################
# Gathering the missing data percentage for each column
##################################
missing_rate_list = map(truediv, null_row_list, column_count_list)
```


```python
##################################
# Exploring the rows
# for missing data
##################################
all_row_quality_summary = pd.DataFrame(zip(row_metadata_list,
                                           column_count_list,
                                           null_row_list,
                                           missing_rate_list), 
                                        columns=['Row.Name',
                                                 'Column.Count',
                                                 'Null.Count',                                                 
                                                 'Missing.Rate'])
display(all_row_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Row.Name</th>
      <th>Column.Count</th>
      <th>Null.Count</th>
      <th>Missing.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>413</th>
      <td>414</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>414</th>
      <td>415</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>415</th>
      <td>416</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>416</th>
      <td>417</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>417</th>
      <td>418</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
  </tbody>
</table>
<p>418 rows × 4 columns</p>
</div>



```python
##################################
# Counting the number of rows
# with Fill.Rate < 1.00
##################################
print('Number of Rows with Missing Data:',str(len(all_row_quality_summary[all_row_quality_summary['Missing.Rate']>0])))
```

    Number of Rows with Missing Data: 142
    


```python
##################################
# Identifying the rows
# with Fill.Rate < 1.00
##################################
print('Rows with Missing Data:')
display(all_row_quality_summary[all_row_quality_summary['Missing.Rate']>0])
```

    Rows with Missing Data:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Row.Name</th>
      <th>Column.Count</th>
      <th>Null.Count</th>
      <th>Missing.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>19</td>
      <td>1</td>
      <td>0.052632</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>19</td>
      <td>2</td>
      <td>0.105263</td>
    </tr>
    <tr>
      <th>39</th>
      <td>40</td>
      <td>19</td>
      <td>2</td>
      <td>0.105263</td>
    </tr>
    <tr>
      <th>40</th>
      <td>41</td>
      <td>19</td>
      <td>2</td>
      <td>0.105263</td>
    </tr>
    <tr>
      <th>41</th>
      <td>42</td>
      <td>19</td>
      <td>2</td>
      <td>0.105263</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>413</th>
      <td>414</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>414</th>
      <td>415</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>415</th>
      <td>416</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>416</th>
      <td>417</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>417</th>
      <td>418</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
  </tbody>
</table>
<p>142 rows × 4 columns</p>
</div>



```python
##################################
# Counting the number of rows
# based on different Fill.Rate categories
##################################
missing_rate_categories = all_row_quality_summary['Missing.Rate'].value_counts().reset_index()
missing_rate_categories.columns = ['Missing.Rate.Category','Missing.Rate.Count']
display(missing_rate_categories.sort_values(['Missing.Rate.Category'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing.Rate.Category</th>
      <th>Missing.Rate.Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>0.526316</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.473684</td>
      <td>91</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.105263</td>
      <td>28</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.052632</td>
      <td>8</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>276</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Identifying the rows
# with Missing.Rate > 0.40
##################################
row_high_missing_rate = all_row_quality_summary[(all_row_quality_summary['Missing.Rate']>0.40)]
```


```python
##################################
# Formulating the dataset
# with numeric columns only
##################################
cirrhosis_survival_numeric = cirrhosis_survival.select_dtypes(include='number')
```


```python
##################################
# Gathering the variable names for each numeric column
##################################
numeric_variable_name_list = cirrhosis_survival_numeric.columns
```


```python
##################################
# Gathering the minimum value for each numeric column
##################################
numeric_minimum_list = cirrhosis_survival_numeric.min()
```


```python
##################################
# Gathering the mean value for each numeric column
##################################
numeric_mean_list = cirrhosis_survival_numeric.mean()
```


```python
##################################
# Gathering the median value for each numeric column
##################################
numeric_median_list = cirrhosis_survival_numeric.median()
```


```python
##################################
# Gathering the maximum value for each numeric column
##################################
numeric_maximum_list = cirrhosis_survival_numeric.max()
```


```python
##################################
# Gathering the first mode values for each numeric column
##################################
numeric_first_mode_list = [cirrhosis_survival[x].value_counts(dropna=True).index.tolist()[0] for x in cirrhosis_survival_numeric]
```


```python
##################################
# Gathering the second mode values for each numeric column
##################################
numeric_second_mode_list = [cirrhosis_survival[x].value_counts(dropna=True).index.tolist()[1] for x in cirrhosis_survival_numeric]
```


```python
##################################
# Gathering the count of first mode values for each numeric column
##################################
numeric_first_mode_count_list = [cirrhosis_survival_numeric[x].isin([cirrhosis_survival[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in cirrhosis_survival_numeric]
```


```python
##################################
# Gathering the count of second mode values for each numeric column
##################################
numeric_second_mode_count_list = [cirrhosis_survival_numeric[x].isin([cirrhosis_survival[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in cirrhosis_survival_numeric]
```


```python
##################################
# Gathering the first mode to second mode ratio for each numeric column
##################################
numeric_first_second_mode_ratio_list = map(truediv, numeric_first_mode_count_list, numeric_second_mode_count_list)
```


```python
##################################
# Gathering the count of unique values for each numeric column
##################################
numeric_unique_count_list = cirrhosis_survival_numeric.nunique(dropna=True)
```


```python
##################################
# Gathering the number of observations for each numeric column
##################################
numeric_row_count_list = list([len(cirrhosis_survival_numeric)] * len(cirrhosis_survival_numeric.columns))
```


```python
##################################
# Gathering the unique to count ratio for each numeric column
##################################
numeric_unique_count_ratio_list = map(truediv, numeric_unique_count_list, numeric_row_count_list)
```


```python
##################################
# Gathering the skewness value for each numeric column
##################################
numeric_skewness_list = cirrhosis_survival_numeric.skew()
```


```python
##################################
# Gathering the kurtosis value for each numeric column
##################################
numeric_kurtosis_list = cirrhosis_survival_numeric.kurtosis()
```


```python
numeric_column_quality_summary = pd.DataFrame(zip(numeric_variable_name_list,
                                                numeric_minimum_list,
                                                numeric_mean_list,
                                                numeric_median_list,
                                                numeric_maximum_list,
                                                numeric_first_mode_list,
                                                numeric_second_mode_list,
                                                numeric_first_mode_count_list,
                                                numeric_second_mode_count_list,
                                                numeric_first_second_mode_ratio_list,
                                                numeric_unique_count_list,
                                                numeric_row_count_list,
                                                numeric_unique_count_ratio_list,
                                                numeric_skewness_list,
                                                numeric_kurtosis_list), 
                                        columns=['Numeric.Column.Name',
                                                 'Minimum',
                                                 'Mean',
                                                 'Median',
                                                 'Maximum',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio',
                                                 'Skewness',
                                                 'Kurtosis'])
display(numeric_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Minimum</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Maximum</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>N_Days</td>
      <td>41.00</td>
      <td>1917.782297</td>
      <td>1730.00</td>
      <td>4795.00</td>
      <td>1434.00</td>
      <td>3445.00</td>
      <td>2</td>
      <td>2</td>
      <td>1.000000</td>
      <td>399</td>
      <td>418</td>
      <td>0.954545</td>
      <td>0.472602</td>
      <td>-0.482139</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Age</td>
      <td>9598.00</td>
      <td>18533.351675</td>
      <td>18628.00</td>
      <td>28650.00</td>
      <td>19724.00</td>
      <td>18993.00</td>
      <td>7</td>
      <td>6</td>
      <td>1.166667</td>
      <td>344</td>
      <td>418</td>
      <td>0.822967</td>
      <td>0.086850</td>
      <td>-0.616730</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bilirubin</td>
      <td>0.30</td>
      <td>3.220813</td>
      <td>1.40</td>
      <td>28.00</td>
      <td>0.70</td>
      <td>0.60</td>
      <td>33</td>
      <td>31</td>
      <td>1.064516</td>
      <td>98</td>
      <td>418</td>
      <td>0.234450</td>
      <td>2.717611</td>
      <td>8.065336</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cholesterol</td>
      <td>120.00</td>
      <td>369.510563</td>
      <td>309.50</td>
      <td>1775.00</td>
      <td>260.00</td>
      <td>316.00</td>
      <td>4</td>
      <td>4</td>
      <td>1.000000</td>
      <td>201</td>
      <td>418</td>
      <td>0.480861</td>
      <td>3.408526</td>
      <td>14.337870</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Albumin</td>
      <td>1.96</td>
      <td>3.497440</td>
      <td>3.53</td>
      <td>4.64</td>
      <td>3.35</td>
      <td>3.50</td>
      <td>11</td>
      <td>8</td>
      <td>1.375000</td>
      <td>154</td>
      <td>418</td>
      <td>0.368421</td>
      <td>-0.467527</td>
      <td>0.566745</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Copper</td>
      <td>4.00</td>
      <td>97.648387</td>
      <td>73.00</td>
      <td>588.00</td>
      <td>52.00</td>
      <td>67.00</td>
      <td>8</td>
      <td>7</td>
      <td>1.142857</td>
      <td>158</td>
      <td>418</td>
      <td>0.377990</td>
      <td>2.303640</td>
      <td>7.624023</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Alk_Phos</td>
      <td>289.00</td>
      <td>1982.655769</td>
      <td>1259.00</td>
      <td>13862.40</td>
      <td>601.00</td>
      <td>794.00</td>
      <td>2</td>
      <td>2</td>
      <td>1.000000</td>
      <td>295</td>
      <td>418</td>
      <td>0.705742</td>
      <td>2.992834</td>
      <td>9.662553</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SGOT</td>
      <td>26.35</td>
      <td>122.556346</td>
      <td>114.70</td>
      <td>457.25</td>
      <td>71.30</td>
      <td>137.95</td>
      <td>6</td>
      <td>5</td>
      <td>1.200000</td>
      <td>179</td>
      <td>418</td>
      <td>0.428230</td>
      <td>1.449197</td>
      <td>4.311976</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Tryglicerides</td>
      <td>33.00</td>
      <td>124.702128</td>
      <td>108.00</td>
      <td>598.00</td>
      <td>118.00</td>
      <td>90.00</td>
      <td>7</td>
      <td>6</td>
      <td>1.166667</td>
      <td>146</td>
      <td>418</td>
      <td>0.349282</td>
      <td>2.523902</td>
      <td>11.802753</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Platelets</td>
      <td>62.00</td>
      <td>257.024570</td>
      <td>251.00</td>
      <td>721.00</td>
      <td>344.00</td>
      <td>269.00</td>
      <td>6</td>
      <td>5</td>
      <td>1.200000</td>
      <td>243</td>
      <td>418</td>
      <td>0.581340</td>
      <td>0.627098</td>
      <td>0.863045</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Prothrombin</td>
      <td>9.00</td>
      <td>10.731731</td>
      <td>10.60</td>
      <td>18.00</td>
      <td>10.60</td>
      <td>11.00</td>
      <td>39</td>
      <td>32</td>
      <td>1.218750</td>
      <td>48</td>
      <td>418</td>
      <td>0.114833</td>
      <td>2.223276</td>
      <td>10.040773</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the dataset
# with object column only
##################################
cirrhosis_survival_object = cirrhosis_survival.select_dtypes(include='object')
```


```python
##################################
# Gathering the variable names for the object column
##################################
object_variable_name_list = cirrhosis_survival_object.columns
```


```python
##################################
# Gathering the first mode values for the object column
##################################
object_first_mode_list = [cirrhosis_survival[x].value_counts().index.tolist()[0] for x in cirrhosis_survival_object]
```


```python
##################################
# Gathering the second mode values for each object column
##################################
object_second_mode_list = [cirrhosis_survival[x].value_counts().index.tolist()[1] for x in cirrhosis_survival_object]
```


```python
##################################
# Gathering the count of first mode values for each object column
##################################
object_first_mode_count_list = [cirrhosis_survival_object[x].isin([cirrhosis_survival[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in cirrhosis_survival_object]
```


```python
##################################
# Gathering the count of second mode values for each object column
##################################
object_second_mode_count_list = [cirrhosis_survival_object[x].isin([cirrhosis_survival[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in cirrhosis_survival_object]
```


```python
##################################
# Gathering the first mode to second mode ratio for each object column
##################################
object_first_second_mode_ratio_list = map(truediv, object_first_mode_count_list, object_second_mode_count_list)
```


```python
##################################
# Gathering the count of unique values for each object column
##################################
object_unique_count_list = cirrhosis_survival_object.nunique(dropna=True)
```


```python
##################################
# Gathering the number of observations for each object column
##################################
object_row_count_list = list([len(cirrhosis_survival_object)] * len(cirrhosis_survival_object.columns))
```


```python
##################################
# Gathering the unique to count ratio for each object column
##################################
object_unique_count_ratio_list = map(truediv, object_unique_count_list, object_row_count_list)
```


```python
object_column_quality_summary = pd.DataFrame(zip(object_variable_name_list,
                                                 object_first_mode_list,
                                                 object_second_mode_list,
                                                 object_first_mode_count_list,
                                                 object_second_mode_count_list,
                                                 object_first_second_mode_ratio_list,
                                                 object_unique_count_list,
                                                 object_row_count_list,
                                                 object_unique_count_ratio_list), 
                                        columns=['Object.Column.Name',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio'])
display(object_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Object.Column.Name</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Drug</td>
      <td>D-penicillamine</td>
      <td>Placebo</td>
      <td>158</td>
      <td>154</td>
      <td>1.025974</td>
      <td>2</td>
      <td>418</td>
      <td>0.004785</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sex</td>
      <td>F</td>
      <td>M</td>
      <td>374</td>
      <td>44</td>
      <td>8.500000</td>
      <td>2</td>
      <td>418</td>
      <td>0.004785</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ascites</td>
      <td>N</td>
      <td>Y</td>
      <td>288</td>
      <td>24</td>
      <td>12.000000</td>
      <td>2</td>
      <td>418</td>
      <td>0.004785</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hepatomegaly</td>
      <td>Y</td>
      <td>N</td>
      <td>160</td>
      <td>152</td>
      <td>1.052632</td>
      <td>2</td>
      <td>418</td>
      <td>0.004785</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Spiders</td>
      <td>N</td>
      <td>Y</td>
      <td>222</td>
      <td>90</td>
      <td>2.466667</td>
      <td>2</td>
      <td>418</td>
      <td>0.004785</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Edema</td>
      <td>N</td>
      <td>S</td>
      <td>354</td>
      <td>44</td>
      <td>8.045455</td>
      <td>3</td>
      <td>418</td>
      <td>0.007177</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Stage</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>155</td>
      <td>144</td>
      <td>1.076389</td>
      <td>4</td>
      <td>418</td>
      <td>0.009569</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of object columns
# with First.Second.Mode.Ratio > 5.00
##################################
len(object_column_quality_summary[(object_column_quality_summary['First.Second.Mode.Ratio']>5)])
```




    3




```python
##################################
# Counting the number of object columns
# with Unique.Count.Ratio > 10.00
##################################
len(object_column_quality_summary[(object_column_quality_summary['Unique.Count.Ratio']>10)])
```




    0



## 1.4. Data Preprocessing <a class="anchor" id="1.4"></a>

### 1.4.1 Data Cleaning <a class="anchor" id="1.4.1"></a>

1. Subsets of rows with high rates of missing data were removed from the dataset:
    * 106 rows with Missing.Rate>0.4 were exluded for subsequent analysis.
2. No variables were removed due to zero or near-zero variance.


```python
##################################
# Performing a general exploration of the original dataset
##################################
print('Dataset Dimensions: ')
display(cirrhosis_survival.shape)
```

    Dataset Dimensions: 
    


    (418, 19)



```python
##################################
# Filtering out the rows with
# with Missing.Rate > 0.40
##################################
cirrhosis_survival_filtered_row = cirrhosis_survival.drop(cirrhosis_survival[cirrhosis_survival.index.isin(row_high_missing_rate['Row.Name'].values.tolist())].index)
```


```python
##################################
# Performing a general exploration of the filtered dataset
##################################
print('Dataset Dimensions: ')
display(cirrhosis_survival_filtered_row.shape)
```

    Dataset Dimensions: 
    


    (312, 19)



```python
##################################
# Gathering the missing data percentage for each column
# from the filtered data
##################################
data_type_list = list(cirrhosis_survival_filtered_row.dtypes)
variable_name_list = list(cirrhosis_survival_filtered_row.columns)
null_count_list = list(cirrhosis_survival_filtered_row.isna().sum(axis=0))
non_null_count_list = list(cirrhosis_survival_filtered_row.count())
row_count_list = list([len(cirrhosis_survival_filtered_row)] * len(cirrhosis_survival_filtered_row.columns))
fill_rate_list = map(truediv, non_null_count_list, row_count_list)
all_column_quality_summary = pd.DataFrame(zip(variable_name_list,
                                              data_type_list,
                                              row_count_list,
                                              non_null_count_list,
                                              null_count_list,
                                              fill_rate_list), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count',                                                 
                                                 'Fill.Rate'])
display(all_column_quality_summary.sort_values(['Fill.Rate'], ascending=True))

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>Tryglicerides</td>
      <td>float64</td>
      <td>312</td>
      <td>282</td>
      <td>30</td>
      <td>0.903846</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cholesterol</td>
      <td>float64</td>
      <td>312</td>
      <td>284</td>
      <td>28</td>
      <td>0.910256</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Platelets</td>
      <td>float64</td>
      <td>312</td>
      <td>308</td>
      <td>4</td>
      <td>0.987179</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Copper</td>
      <td>float64</td>
      <td>312</td>
      <td>310</td>
      <td>2</td>
      <td>0.993590</td>
    </tr>
    <tr>
      <th>0</th>
      <td>N_Days</td>
      <td>int64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SGOT</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Alk_Phos</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Albumin</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Prothrombin</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Bilirubin</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Spiders</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hepatomegaly</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ascites</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sex</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Age</td>
      <td>int64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Drug</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Status</td>
      <td>bool</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Edema</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Stage</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating a new dataset object
# for the cleaned data
##################################
cirrhosis_survival_cleaned = cirrhosis_survival_filtered_row
```

### 1.4.2 Missing Data Imputation <a class="anchor" id="1.4.2"></a>

1. To prevent data leakage, the original dataset was divided into training and testing subsets prior to imputation.
2. Missing data in the training subset for float variables were imputed using the iterative imputer algorithm with a  linear regression estimator.
    * <span style="color: #FF0000">Tryglicerides</span>: Null.Count = 20
    * <span style="color: #FF0000">Cholesterol</span>: Null.Count = 18
    * <span style="color: #FF0000">Platelets</span>: Null.Count = 2
    * <span style="color: #FF0000">Copper</span>: Null.Count = 1
3. Missing data in the testing subset for float variables will be treated with iterative imputing downstream using a pipeline involving the final preprocessing steps.



```python
##################################
# Formulating the summary
# for all cleaned columns
##################################
cleaned_column_quality_summary = pd.DataFrame(zip(list(cirrhosis_survival_cleaned.columns),
                                                  list(cirrhosis_survival_cleaned.dtypes),
                                                  list([len(cirrhosis_survival_cleaned)] * len(cirrhosis_survival_cleaned.columns)),
                                                  list(cirrhosis_survival_cleaned.count()),
                                                  list(cirrhosis_survival_cleaned.isna().sum(axis=0))), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count'])
display(cleaned_column_quality_summary.sort_values(by=['Null.Count'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>Tryglicerides</td>
      <td>float64</td>
      <td>312</td>
      <td>282</td>
      <td>30</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cholesterol</td>
      <td>float64</td>
      <td>312</td>
      <td>284</td>
      <td>28</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Platelets</td>
      <td>float64</td>
      <td>312</td>
      <td>308</td>
      <td>4</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Copper</td>
      <td>float64</td>
      <td>312</td>
      <td>310</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>N_Days</td>
      <td>int64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Prothrombin</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SGOT</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Alk_Phos</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Albumin</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Bilirubin</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Status</td>
      <td>bool</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Edema</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Spiders</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hepatomegaly</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ascites</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sex</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Age</td>
      <td>int64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Drug</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Stage</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Creating training and testing data
##################################
cirrhosis_survival_train, cirrhosis_survival_test = train_test_split(cirrhosis_survival_cleaned, 
                                                                     test_size=0.30, 
                                                                     stratify=cirrhosis_survival_cleaned['Status'], 
                                                                     random_state=88888888)
cirrhosis_survival_X_train_cleaned = cirrhosis_survival_train.drop(columns=['Status', 'N_Days'])
cirrhosis_survival_y_train_cleaned = cirrhosis_survival_train[['Status', 'N_Days']]
cirrhosis_survival_X_test_cleaned = cirrhosis_survival_test.drop(columns=['Status', 'N_Days'])
cirrhosis_survival_y_test_cleaned = cirrhosis_survival_test[['Status', 'N_Days']]
```


```python
##################################
# Gathering the training data information
##################################
print(f'Training Dataset Dimensions: Predictors: {cirrhosis_survival_X_train_cleaned.shape}, Event|Duration: {cirrhosis_survival_y_train_cleaned.shape}')
```

    Training Dataset Dimensions: Predictors: (218, 17), Event|Duration: (218, 2)
    


```python
##################################
# Gathering the testing data information
##################################
print(f'Testing Dataset Dimensions: Predictors: {cirrhosis_survival_X_test_cleaned.shape}, Event|Duration: {cirrhosis_survival_y_test_cleaned.shape}')
```

    Testing Dataset Dimensions: Predictors: (94, 17), Event|Duration: (94, 2)
    


```python
##################################
# Formulating the summary
# for all cleaned columns
# from the training data
##################################
X_train_cleaned_column_quality_summary = pd.DataFrame(zip(list(cirrhosis_survival_X_train_cleaned.columns),
                                                  list(cirrhosis_survival_X_train_cleaned.dtypes),
                                                  list([len(cirrhosis_survival_X_train_cleaned)] * len(cirrhosis_survival_X_train_cleaned.columns)),
                                                  list(cirrhosis_survival_X_train_cleaned.count()),
                                                  list(cirrhosis_survival_X_train_cleaned.isna().sum(axis=0))), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count'])
display(X_train_cleaned_column_quality_summary.sort_values(by=['Null.Count'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>Tryglicerides</td>
      <td>float64</td>
      <td>218</td>
      <td>200</td>
      <td>18</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Cholesterol</td>
      <td>float64</td>
      <td>218</td>
      <td>202</td>
      <td>16</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Platelets</td>
      <td>float64</td>
      <td>218</td>
      <td>215</td>
      <td>3</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Copper</td>
      <td>float64</td>
      <td>218</td>
      <td>217</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Albumin</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Prothrombin</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>SGOT</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Alk_Phos</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Drug</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Age</td>
      <td>int64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Bilirubin</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Edema</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Spiders</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hepatomegaly</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ascites</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sex</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Stage</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the summary
# for all cleaned columns
# from the testing data
##################################
X_test_cleaned_column_quality_summary = pd.DataFrame(zip(list(cirrhosis_survival_X_test_cleaned.columns),
                                                  list(cirrhosis_survival_X_test_cleaned.dtypes),
                                                  list([len(cirrhosis_survival_X_test_cleaned)] * len(cirrhosis_survival_X_test_cleaned.columns)),
                                                  list(cirrhosis_survival_X_test_cleaned.count()),
                                                  list(cirrhosis_survival_X_test_cleaned.isna().sum(axis=0))), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count'])
display(X_test_cleaned_column_quality_summary.sort_values(by=['Null.Count'], ascending=False))

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>Cholesterol</td>
      <td>float64</td>
      <td>94</td>
      <td>82</td>
      <td>12</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Tryglicerides</td>
      <td>float64</td>
      <td>94</td>
      <td>82</td>
      <td>12</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Platelets</td>
      <td>float64</td>
      <td>94</td>
      <td>93</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Copper</td>
      <td>float64</td>
      <td>94</td>
      <td>93</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Albumin</td>
      <td>float64</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Prothrombin</td>
      <td>float64</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>SGOT</td>
      <td>float64</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Alk_Phos</td>
      <td>float64</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Drug</td>
      <td>object</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Age</td>
      <td>int64</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Bilirubin</td>
      <td>float64</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Edema</td>
      <td>object</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Spiders</td>
      <td>object</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hepatomegaly</td>
      <td>object</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ascites</td>
      <td>object</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sex</td>
      <td>object</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Stage</td>
      <td>object</td>
      <td>94</td>
      <td>94</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the cleaned training dataset
# with object columns only
##################################
cirrhosis_survival_X_train_cleaned_object = cirrhosis_survival_X_train_cleaned.select_dtypes(include='object')
cirrhosis_survival_X_train_cleaned_object.reset_index(drop=True, inplace=True)
cirrhosis_survival_X_train_cleaned_object.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>D-penicillamine</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>D-penicillamine</td>
      <td>M</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>D-penicillamine</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Placebo</td>
      <td>F</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Placebo</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>N</td>
      <td>N</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Formulating the cleaned training dataset
# with integer columns only
##################################
cirrhosis_survival_X_train_cleaned_int = cirrhosis_survival_X_train_cleaned.select_dtypes(include='int')
cirrhosis_survival_X_train_cleaned_int.reset_index(drop=True, inplace=True)
cirrhosis_survival_X_train_cleaned_int.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13329</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12912</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17180</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17884</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15177</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Formulating the cleaned training dataset
# with float columns only
##################################
cirrhosis_survival_X_train_cleaned_float = cirrhosis_survival_X_train_cleaned.select_dtypes(include='float')
cirrhosis_survival_X_train_cleaned_float.reset_index(drop=True, inplace=True)
cirrhosis_survival_X_train_cleaned_float.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.4</td>
      <td>450.0</td>
      <td>3.37</td>
      <td>32.0</td>
      <td>1408.0</td>
      <td>116.25</td>
      <td>118.0</td>
      <td>313.0</td>
      <td>11.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.4</td>
      <td>646.0</td>
      <td>3.83</td>
      <td>102.0</td>
      <td>855.0</td>
      <td>127.00</td>
      <td>194.0</td>
      <td>306.0</td>
      <td>10.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.9</td>
      <td>346.0</td>
      <td>3.77</td>
      <td>59.0</td>
      <td>794.0</td>
      <td>125.55</td>
      <td>56.0</td>
      <td>336.0</td>
      <td>10.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.5</td>
      <td>188.0</td>
      <td>3.67</td>
      <td>57.0</td>
      <td>1273.0</td>
      <td>119.35</td>
      <td>102.0</td>
      <td>110.0</td>
      <td>11.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.7</td>
      <td>296.0</td>
      <td>3.44</td>
      <td>114.0</td>
      <td>9933.2</td>
      <td>206.40</td>
      <td>101.0</td>
      <td>195.0</td>
      <td>10.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Defining the estimator to be used
# at each step of the round-robin imputation
##################################
lr = LinearRegression()
```


```python
##################################
# Defining the parameter of the
# iterative imputer which will estimate 
# the columns with missing values
# as a function of the other columns
# in a round-robin fashion
##################################
iterative_imputer = IterativeImputer(
    estimator = lr,
    max_iter = 10,
    tol = 1e-10,
    imputation_order = 'ascending',
    random_state=88888888
)
```


```python
##################################
# Implementing the iterative imputer 
##################################
cirrhosis_survival_X_train_imputed_float_array = iterative_imputer.fit_transform(cirrhosis_survival_X_train_cleaned_float)
```


```python
##################################
# Transforming the imputed training data
# from an array to a dataframe
##################################
cirrhosis_survival_X_train_imputed_float = pd.DataFrame(cirrhosis_survival_X_train_imputed_float_array, 
                                                        columns = cirrhosis_survival_X_train_cleaned_float.columns)
cirrhosis_survival_X_train_imputed_float.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.4</td>
      <td>450.0</td>
      <td>3.37</td>
      <td>32.0</td>
      <td>1408.0</td>
      <td>116.25</td>
      <td>118.0</td>
      <td>313.0</td>
      <td>11.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.4</td>
      <td>646.0</td>
      <td>3.83</td>
      <td>102.0</td>
      <td>855.0</td>
      <td>127.00</td>
      <td>194.0</td>
      <td>306.0</td>
      <td>10.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.9</td>
      <td>346.0</td>
      <td>3.77</td>
      <td>59.0</td>
      <td>794.0</td>
      <td>125.55</td>
      <td>56.0</td>
      <td>336.0</td>
      <td>10.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.5</td>
      <td>188.0</td>
      <td>3.67</td>
      <td>57.0</td>
      <td>1273.0</td>
      <td>119.35</td>
      <td>102.0</td>
      <td>110.0</td>
      <td>11.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.7</td>
      <td>296.0</td>
      <td>3.44</td>
      <td>114.0</td>
      <td>9933.2</td>
      <td>206.40</td>
      <td>101.0</td>
      <td>195.0</td>
      <td>10.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Formulating the imputed training dataset
##################################
cirrhosis_survival_X_train_imputed = pd.concat([cirrhosis_survival_X_train_cleaned_int,
                                                cirrhosis_survival_X_train_cleaned_object,
                                                cirrhosis_survival_X_train_imputed_float], 
                                               axis=1, 
                                               join='inner')  
```


```python
##################################
# Formulating the summary
# for all imputed columns
##################################
X_train_imputed_column_quality_summary = pd.DataFrame(zip(list(cirrhosis_survival_X_train_imputed.columns),
                                                         list(cirrhosis_survival_X_train_imputed.dtypes),
                                                         list([len(cirrhosis_survival_X_train_imputed)] * len(cirrhosis_survival_X_train_imputed.columns)),
                                                         list(cirrhosis_survival_X_train_imputed.count()),
                                                         list(cirrhosis_survival_X_train_imputed.isna().sum(axis=0))), 
                                                     columns=['Column.Name',
                                                              'Column.Type',
                                                              'Row.Count',
                                                              'Non.Null.Count',
                                                              'Null.Count'])
display(X_train_imputed_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>int64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Drug</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sex</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ascites</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hepatomegaly</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Spiders</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Edema</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Stage</td>
      <td>object</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Bilirubin</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Cholesterol</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Albumin</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Copper</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Alk_Phos</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>SGOT</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Tryglicerides</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Platelets</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Prothrombin</td>
      <td>float64</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


### 1.4.3 Outlier Detection <a class="anchor" id="1.4.3"></a>

1. High number of outliers observed in the training subset for 4 numeric variables with Outlier.Ratio>0.05 and marginal to high Skewness.
    * <span style="color: #FF0000">Alk_Phos</span>: Outlier.Count = 25, Outlier.Ratio = 0.114, Skewness=+3.035
    * <span style="color: #FF0000">Bilirubin</span>: Outlier.Count = 18, Outlier.Ratio = 0.083, Skewness=+3.121
    * <span style="color: #FF0000">Cholesterol</span>: Outlier.Count = 17, Outlier.Ratio = 0.078, Skewness=+3.761
    * <span style="color: #FF0000">Prothrombin</span>: Outlier.Count = 12, Outlier.Ratio = 0.055, Skewness=+1.009
2. Minimal number of outliers observed in the training subset for 5 numeric variables with Outlier.Ratio>0.00 but <0.05 and normal to marginal Skewness.
    * <span style="color: #FF0000">Copper</span>: Outlier.Count = 8, Outlier.Ratio = 0.037, Skewness=+1.485
    * <span style="color: #FF0000">Albumin</span>: Outlier.Count = 6, Outlier.Ratio = 0.027, Skewness=-0.589
    * <span style="color: #FF0000">SGOT</span>: Outlier.Count = 4, Outlier.Ratio = 0.018, Skewness=+0.934
    * <span style="color: #FF0000">Tryglicerides</span>: Outlier.Count = 4, Outlier.Ratio = 0.018, Skewness=+2.817
    * <span style="color: #FF0000">Platelets</span>: Outlier.Count = 4, Outlier.Ratio = 0.018, Skewness=+0.374
    * <span style="color: #FF0000">Age</span>: Outlier.Count = 1, Outlier.Ratio = 0.004, Skewness=+0.223


```python
##################################
# Formulating the imputed dataset
# with numeric columns only
##################################
cirrhosis_survival_X_train_imputed_numeric = cirrhosis_survival_X_train_imputed.select_dtypes(include='number')
```


```python
##################################
# Gathering the variable names for each numeric column
##################################
X_train_numeric_variable_name_list = list(cirrhosis_survival_X_train_imputed_numeric.columns)
```


```python
##################################
# Gathering the skewness value for each numeric column
##################################
X_train_numeric_skewness_list = cirrhosis_survival_X_train_imputed_numeric.skew()
```


```python
##################################
# Computing the interquartile range
# for all columns
##################################
cirrhosis_survival_X_train_imputed_numeric_q1 = cirrhosis_survival_X_train_imputed_numeric.quantile(0.25)
cirrhosis_survival_X_train_imputed_numeric_q3 = cirrhosis_survival_X_train_imputed_numeric.quantile(0.75)
cirrhosis_survival_X_train_imputed_numeric_iqr = cirrhosis_survival_X_train_imputed_numeric_q3 - cirrhosis_survival_X_train_imputed_numeric_q1
```


```python
##################################
# Gathering the outlier count for each numeric column
# based on the interquartile range criterion
##################################
X_train_numeric_outlier_count_list = ((cirrhosis_survival_X_train_imputed_numeric < (cirrhosis_survival_X_train_imputed_numeric_q1 - 1.5 * cirrhosis_survival_X_train_imputed_numeric_iqr)) | (cirrhosis_survival_X_train_imputed_numeric > (cirrhosis_survival_X_train_imputed_numeric_q3 + 1.5 * cirrhosis_survival_X_train_imputed_numeric_iqr))).sum()
```


```python
##################################
# Gathering the number of observations for each column
##################################
X_train_numeric_row_count_list = list([len(cirrhosis_survival_X_train_imputed_numeric)] * len(cirrhosis_survival_X_train_imputed_numeric.columns))
```


```python
##################################
# Gathering the unique to count ratio for each object column
##################################
X_train_numeric_outlier_ratio_list = map(truediv, X_train_numeric_outlier_count_list, X_train_numeric_row_count_list)
```


```python
##################################
# Formulating the outlier summary
# for all numeric columns
##################################
X_train_numeric_column_outlier_summary = pd.DataFrame(zip(X_train_numeric_variable_name_list,
                                                          X_train_numeric_skewness_list,
                                                          X_train_numeric_outlier_count_list,
                                                          X_train_numeric_row_count_list,
                                                          X_train_numeric_outlier_ratio_list), 
                                                      columns=['Numeric.Column.Name',
                                                               'Skewness',
                                                               'Outlier.Count',
                                                               'Row.Count',
                                                               'Outlier.Ratio'])
display(X_train_numeric_column_outlier_summary.sort_values(by=['Outlier.Count'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Skewness</th>
      <th>Outlier.Count</th>
      <th>Row.Count</th>
      <th>Outlier.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Alk_Phos</td>
      <td>3.035777</td>
      <td>25</td>
      <td>218</td>
      <td>0.114679</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bilirubin</td>
      <td>3.121255</td>
      <td>18</td>
      <td>218</td>
      <td>0.082569</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cholesterol</td>
      <td>3.760943</td>
      <td>17</td>
      <td>218</td>
      <td>0.077982</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Prothrombin</td>
      <td>1.009263</td>
      <td>12</td>
      <td>218</td>
      <td>0.055046</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Copper</td>
      <td>1.485547</td>
      <td>8</td>
      <td>218</td>
      <td>0.036697</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albumin</td>
      <td>-0.589651</td>
      <td>6</td>
      <td>218</td>
      <td>0.027523</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SGOT</td>
      <td>0.934535</td>
      <td>4</td>
      <td>218</td>
      <td>0.018349</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Tryglicerides</td>
      <td>2.817187</td>
      <td>4</td>
      <td>218</td>
      <td>0.018349</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Platelets</td>
      <td>0.374251</td>
      <td>4</td>
      <td>218</td>
      <td>0.018349</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>0.223080</td>
      <td>1</td>
      <td>218</td>
      <td>0.004587</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the individual boxplots
# for all numeric columns
##################################
for column in cirrhosis_survival_X_train_imputed_numeric:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=cirrhosis_survival_X_train_imputed_numeric, x=column)
```


    
![png](output_99_0.png)
    



    
![png](output_99_1.png)
    



    
![png](output_99_2.png)
    



    
![png](output_99_3.png)
    



    
![png](output_99_4.png)
    



    
![png](output_99_5.png)
    



    
![png](output_99_6.png)
    



    
![png](output_99_7.png)
    



    
![png](output_99_8.png)
    



    
![png](output_99_9.png)
    


### 1.4.4 Collinearity <a class="anchor" id="1.4.4"></a>

[Pearson’s Correlation Coefficient](https://royalsocietypublishing.org/doi/10.1098/rsta.1896.0007) is a parametric measure of the linear correlation for a pair of features by calculating the ratio between their covariance and the product of their standard deviations. The presence of high absolute correlation values indicate the univariate association between the numeric predictors and the numeric response.

1. All numeric variables in the training subset were retained since majority reported sufficiently moderate and statistically significant correlation with no excessive multicollinearity.
2. Among pairwise combinations of numeric variables in the training subset, the highest Pearson.Correlation.Coefficient values were noted for:
    * <span style="color: #FF0000">Birilubin</span> and <span style="color: #FF0000">Copper</span>: Pearson.Correlation.Coefficient = +0.503
    * <span style="color: #FF0000">Birilubin</span> and <span style="color: #FF0000">SGOT</span>: Pearson.Correlation.Coefficient = +0.444
    * <span style="color: #FF0000">Birilubin</span> and <span style="color: #FF0000">Tryglicerides</span>: Pearson.Correlation.Coefficient = +0.389
    * <span style="color: #FF0000">Birilubin</span> and <span style="color: #FF0000">Cholesterol</span>: Pearson.Correlation.Coefficient = +0.348
    * <span style="color: #FF0000">Birilubin</span> and <span style="color: #FF0000">Prothrombin</span>: Pearson.Correlation.Coefficient = +0.344


```python
##################################
# Formulating a function 
# to plot the correlation matrix
# for all pairwise combinations
# of numeric columns
##################################
def plot_correlation_matrix(corr, mask=None):
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, 
                ax=ax,
                mask=mask,
                annot=True, 
                vmin=-1, 
                vmax=1, 
                center=0,
                cmap='coolwarm', 
                linewidths=1, 
                linecolor='gray', 
                cbar_kws={'orientation': 'horizontal'}) 
```


```python
##################################
# Computing the correlation coefficients
# and correlation p-values
# among pairs of numeric columns
##################################
cirrhosis_survival_X_train_imputed_numeric_correlation_pairs = {}
cirrhosis_survival_X_train_imputed_numeric_columns = cirrhosis_survival_X_train_imputed_numeric.columns.tolist()
for numeric_column_a, numeric_column_b in itertools.combinations(cirrhosis_survival_X_train_imputed_numeric_columns, 2):
    cirrhosis_survival_X_train_imputed_numeric_correlation_pairs[numeric_column_a + '_' + numeric_column_b] = stats.pearsonr(
        cirrhosis_survival_X_train_imputed_numeric.loc[:, numeric_column_a], 
        cirrhosis_survival_X_train_imputed_numeric.loc[:, numeric_column_b])
```


```python
##################################
# Formulating the pairwise correlation summary
# for all numeric columns
##################################
cirrhosis_survival_X_train_imputed_numeric_summary = cirrhosis_survival_X_train_imputed_numeric.from_dict(cirrhosis_survival_X_train_imputed_numeric_correlation_pairs, orient='index')
cirrhosis_survival_X_train_imputed_numeric_summary.columns = ['Pearson.Correlation.Coefficient', 'Correlation.PValue']
display(cirrhosis_survival_X_train_imputed_numeric_summary.sort_values(by=['Pearson.Correlation.Coefficient'], ascending=False).head(20))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pearson.Correlation.Coefficient</th>
      <th>Correlation.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bilirubin_SGOT</th>
      <td>0.503007</td>
      <td>2.210899e-15</td>
    </tr>
    <tr>
      <th>Bilirubin_Copper</th>
      <td>0.444366</td>
      <td>5.768566e-12</td>
    </tr>
    <tr>
      <th>Bilirubin_Tryglicerides</th>
      <td>0.389493</td>
      <td>2.607951e-09</td>
    </tr>
    <tr>
      <th>Bilirubin_Cholesterol</th>
      <td>0.348174</td>
      <td>1.311597e-07</td>
    </tr>
    <tr>
      <th>Bilirubin_Prothrombin</th>
      <td>0.344724</td>
      <td>1.775156e-07</td>
    </tr>
    <tr>
      <th>Copper_SGOT</th>
      <td>0.305052</td>
      <td>4.475849e-06</td>
    </tr>
    <tr>
      <th>Cholesterol_SGOT</th>
      <td>0.280530</td>
      <td>2.635566e-05</td>
    </tr>
    <tr>
      <th>Alk_Phos_Tryglicerides</th>
      <td>0.265538</td>
      <td>7.199789e-05</td>
    </tr>
    <tr>
      <th>Cholesterol_Tryglicerides</th>
      <td>0.257973</td>
      <td>1.169491e-04</td>
    </tr>
    <tr>
      <th>Copper_Tryglicerides</th>
      <td>0.256448</td>
      <td>1.287335e-04</td>
    </tr>
    <tr>
      <th>Copper_Prothrombin</th>
      <td>0.232051</td>
      <td>5.528189e-04</td>
    </tr>
    <tr>
      <th>Copper_Alk_Phos</th>
      <td>0.215001</td>
      <td>1.404964e-03</td>
    </tr>
    <tr>
      <th>Alk_Phos_Platelets</th>
      <td>0.182762</td>
      <td>6.814702e-03</td>
    </tr>
    <tr>
      <th>SGOT_Tryglicerides</th>
      <td>0.176605</td>
      <td>8.972028e-03</td>
    </tr>
    <tr>
      <th>SGOT_Prothrombin</th>
      <td>0.170928</td>
      <td>1.147644e-02</td>
    </tr>
    <tr>
      <th>Albumin_Platelets</th>
      <td>0.170836</td>
      <td>1.152154e-02</td>
    </tr>
    <tr>
      <th>Cholesterol_Copper</th>
      <td>0.165834</td>
      <td>1.422873e-02</td>
    </tr>
    <tr>
      <th>Cholesterol_Alk_Phos</th>
      <td>0.165814</td>
      <td>1.424066e-02</td>
    </tr>
    <tr>
      <th>Age_Prothrombin</th>
      <td>0.157493</td>
      <td>1.999022e-02</td>
    </tr>
    <tr>
      <th>Cholesterol_Platelets</th>
      <td>0.152235</td>
      <td>2.458130e-02</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting the correlation matrix
# for all pairwise combinations
# of numeric columns
##################################
cirrhosis_survival_X_train_imputed_numeric_correlation = cirrhosis_survival_X_train_imputed_numeric.corr()
mask = np.triu(cirrhosis_survival_X_train_imputed_numeric_correlation)
plot_correlation_matrix(cirrhosis_survival_X_train_imputed_numeric_correlation,mask)
plt.show()
```


    
![png](output_104_0.png)
    



```python
##################################
# Formulating a function 
# to plot the correlation matrix
# for all pairwise combinations
# of numeric columns
# with significant p-values only
##################################
def correlation_significance(df=None):
    p_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    for col in df.columns:
        for col2 in df.drop(col,axis=1).columns:
            _ , p = stats.pearsonr(df[col],df[col2])
            p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = p
    return p_matrix
```


```python
##################################
# Plotting the correlation matrix
# for all pairwise combinations
# of numeric columns
# with significant p-values only
##################################
cirrhosis_survival_X_train_imputed_numeric_correlation_p_values = correlation_significance(cirrhosis_survival_X_train_imputed_numeric)                     
mask = np.invert(np.tril(cirrhosis_survival_X_train_imputed_numeric_correlation_p_values<0.05)) 
plot_correlation_matrix(cirrhosis_survival_X_train_imputed_numeric_correlation,mask)
```


    
![png](output_106_0.png)
    


### 1.4.5 Shape Transformation <a class="anchor" id="1.4.5"></a>

[Yeo-Johnson Transformation](https://academic.oup.com/biomet/article-abstract/87/4/954/232908?redirectedFrom=fulltext&login=false) applies a new family of distributions that can be used without restrictions, extending many of the good properties of the Box-Cox power family. Similar to the Box-Cox transformation, the method also estimates the optimal value of lambda but has the ability to transform both positive and negative values by inflating low variance data and deflating high variance data to create a more uniform data set. While there are no restrictions in terms of the applicable values, the interpretability of the transformed values is more diminished as compared to the other methods.

1. A Yeo-Johnson transformation was applied to all numeric variables in the training subset to improve distributional shape.
2. Most variables in the training subset achieved symmetrical distributions with minimal outliers after transformation.
    * <span style="color: #FF0000">Cholesterol</span>: Outlier.Count = 9, Outlier.Ratio = 0.041, Skewness=-0.083
    * <span style="color: #FF0000">Albumin</span>: Outlier.Count = 4, Outlier.Ratio = 0.018, Skewness=+0.006
    * <span style="color: #FF0000">Platelets</span>: Outlier.Count = 2, Outlier.Ratio = 0.009, Skewness=-0.019
    * <span style="color: #FF0000">Age</span>: Outlier.Count = 1, Outlier.Ratio = 0.004, Skewness=+0.223
    * <span style="color: #FF0000">Copper</span>: Outlier.Count = 1, Outlier.Ratio = 0.004, Skewness=-0.010
    * <span style="color: #FF0000">Alk_Phos</span>: Outlier.Count = 1, Outlier.Ratio = 0.004, Skewness=+0.027
    * <span style="color: #FF0000">SGOT</span>: Outlier.Count = 1, Outlier.Ratio = 0.004, Skewness=-0.001
    * <span style="color: #FF0000">Tryglicerides</span>: Outlier.Count = 1, Outlier.Ratio = 0.004, Skewness=+0.000
3. Outlier data in the testing subset for numeric variables will be treated with Yeo-Johnson transformation downstream using a pipeline involving the final preprocessing steps.



```python
##################################
# Formulating a data subset containing
# variables with noted outliers
##################################
X_train_predictors_with_outliers = ['Bilirubin','Cholesterol','Albumin','Copper','Alk_Phos','SGOT','Tryglicerides','Platelets','Prothrombin']
cirrhosis_survival_X_train_imputed_numeric_with_outliers = cirrhosis_survival_X_train_imputed_numeric[X_train_predictors_with_outliers]
```


```python
##################################
# Conducting a Yeo-Johnson Transformation
# to address the distributional
# shape of the variables
##################################
yeo_johnson_transformer = PowerTransformer(method='yeo-johnson',
                                          standardize=False)
cirrhosis_survival_X_train_imputed_numeric_with_outliers_array = yeo_johnson_transformer.fit_transform(cirrhosis_survival_X_train_imputed_numeric_with_outliers)
```


```python
##################################
# Formulating a new dataset object
# for the transformed data
##################################
cirrhosis_survival_X_train_transformed_numeric_with_outliers = pd.DataFrame(cirrhosis_survival_X_train_imputed_numeric_with_outliers_array,
                                                                            columns=cirrhosis_survival_X_train_imputed_numeric_with_outliers.columns)
cirrhosis_survival_X_train_transformed_numeric = pd.concat([cirrhosis_survival_X_train_imputed_numeric[['Age']],
                                                            cirrhosis_survival_X_train_transformed_numeric_with_outliers], 
                                                           axis=1)
```


```python
cirrhosis_survival_X_train_transformed_numeric.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13329</td>
      <td>0.830251</td>
      <td>1.528771</td>
      <td>25.311621</td>
      <td>4.367652</td>
      <td>2.066062</td>
      <td>7.115310</td>
      <td>3.357597</td>
      <td>58.787709</td>
      <td>0.236575</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12912</td>
      <td>0.751147</td>
      <td>1.535175</td>
      <td>34.049208</td>
      <td>6.244827</td>
      <td>2.047167</td>
      <td>7.303237</td>
      <td>3.581345</td>
      <td>57.931137</td>
      <td>0.236572</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17180</td>
      <td>0.491099</td>
      <td>1.523097</td>
      <td>32.812930</td>
      <td>5.320861</td>
      <td>2.043970</td>
      <td>7.278682</td>
      <td>2.990077</td>
      <td>61.554228</td>
      <td>0.236573</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17884</td>
      <td>0.760957</td>
      <td>1.505628</td>
      <td>30.818146</td>
      <td>5.264915</td>
      <td>2.062590</td>
      <td>7.170942</td>
      <td>3.288822</td>
      <td>29.648190</td>
      <td>0.236575</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15177</td>
      <td>0.893603</td>
      <td>1.519249</td>
      <td>26.533792</td>
      <td>6.440904</td>
      <td>2.109170</td>
      <td>8.385199</td>
      <td>3.284118</td>
      <td>43.198326</td>
      <td>0.236572</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Formulating the individual boxplots
# for all transformed numeric columns
##################################
for column in cirrhosis_survival_X_train_transformed_numeric:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=cirrhosis_survival_X_train_transformed_numeric, x=column)
```


    
![png](output_112_0.png)
    



    
![png](output_112_1.png)
    



    
![png](output_112_2.png)
    



    
![png](output_112_3.png)
    



    
![png](output_112_4.png)
    



    
![png](output_112_5.png)
    



    
![png](output_112_6.png)
    



    
![png](output_112_7.png)
    



    
![png](output_112_8.png)
    



    
![png](output_112_9.png)
    



```python
##################################
# Formulating the outlier summary
# for all numeric columns
##################################
X_train_numeric_variable_name_list = list(cirrhosis_survival_X_train_transformed_numeric.columns)
X_train_numeric_skewness_list = cirrhosis_survival_X_train_transformed_numeric.skew()
cirrhosis_survival_X_train_transformed_numeric_q1 = cirrhosis_survival_X_train_transformed_numeric.quantile(0.25)
cirrhosis_survival_X_train_transformed_numeric_q3 = cirrhosis_survival_X_train_transformed_numeric.quantile(0.75)
cirrhosis_survival_X_train_transformed_numeric_iqr = cirrhosis_survival_X_train_transformed_numeric_q3 - cirrhosis_survival_X_train_transformed_numeric_q1
X_train_numeric_outlier_count_list = ((cirrhosis_survival_X_train_transformed_numeric < (cirrhosis_survival_X_train_transformed_numeric_q1 - 1.5 * cirrhosis_survival_X_train_transformed_numeric_iqr)) | (cirrhosis_survival_X_train_transformed_numeric > (cirrhosis_survival_X_train_transformed_numeric_q3 + 1.5 * cirrhosis_survival_X_train_transformed_numeric_iqr))).sum()
X_train_numeric_row_count_list = list([len(cirrhosis_survival_X_train_transformed_numeric)] * len(cirrhosis_survival_X_train_transformed_numeric.columns))
X_train_numeric_outlier_ratio_list = map(truediv, X_train_numeric_outlier_count_list, X_train_numeric_row_count_list)

X_train_numeric_column_outlier_summary = pd.DataFrame(zip(X_train_numeric_variable_name_list,
                                                          X_train_numeric_skewness_list,
                                                          X_train_numeric_outlier_count_list,
                                                          X_train_numeric_row_count_list,
                                                          X_train_numeric_outlier_ratio_list),                                                      
                                        columns=['Numeric.Column.Name',
                                                 'Skewness',
                                                 'Outlier.Count',
                                                 'Row.Count',
                                                 'Outlier.Ratio'])
display(X_train_numeric_column_outlier_summary.sort_values(by=['Outlier.Count'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Skewness</th>
      <th>Outlier.Count</th>
      <th>Row.Count</th>
      <th>Outlier.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Cholesterol</td>
      <td>-0.083072</td>
      <td>9</td>
      <td>218</td>
      <td>0.041284</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albumin</td>
      <td>0.006523</td>
      <td>4</td>
      <td>218</td>
      <td>0.018349</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Platelets</td>
      <td>-0.019323</td>
      <td>2</td>
      <td>218</td>
      <td>0.009174</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>0.223080</td>
      <td>1</td>
      <td>218</td>
      <td>0.004587</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Copper</td>
      <td>-0.010240</td>
      <td>1</td>
      <td>218</td>
      <td>0.004587</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Alk_Phos</td>
      <td>0.027977</td>
      <td>1</td>
      <td>218</td>
      <td>0.004587</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Tryglicerides</td>
      <td>-0.000881</td>
      <td>1</td>
      <td>218</td>
      <td>0.004587</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Prothrombin</td>
      <td>0.000000</td>
      <td>1</td>
      <td>218</td>
      <td>0.004587</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bilirubin</td>
      <td>0.263101</td>
      <td>0</td>
      <td>218</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SGOT</td>
      <td>-0.008416</td>
      <td>0</td>
      <td>218</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


### 1.4.6 Centering and Scaling <a class="anchor" id="1.4.6"></a>

1. All numeric variables in the training subset were transformed using the standardization method to achieve a comparable scale between values.
2. Original data in the testing subset for numeric variables will be treated with standardization scaling downstream using a pipeline involving the final preprocessing steps.


```python
##################################
# Conducting standardization
# to transform the values of the 
# variables into comparable scale
##################################
standardization_scaler = StandardScaler()
cirrhosis_survival_X_train_transformed_numeric_array = standardization_scaler.fit_transform(cirrhosis_survival_X_train_transformed_numeric)
```


```python
##################################
# Formulating a new dataset object
# for the scaled data
##################################
cirrhosis_survival_X_train_scaled_numeric = pd.DataFrame(cirrhosis_survival_X_train_transformed_numeric_array,
                                                         columns=cirrhosis_survival_X_train_transformed_numeric.columns)
```


```python
##################################
# Formulating the individual boxplots
# for all transformed numeric columns
##################################
for column in cirrhosis_survival_X_train_scaled_numeric:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=cirrhosis_survival_X_train_scaled_numeric, x=column)
```


    
![png](output_117_0.png)
    



    
![png](output_117_1.png)
    



    
![png](output_117_2.png)
    



    
![png](output_117_3.png)
    



    
![png](output_117_4.png)
    



    
![png](output_117_5.png)
    



    
![png](output_117_6.png)
    



    
![png](output_117_7.png)
    



    
![png](output_117_8.png)
    



    
![png](output_117_9.png)
    


### 1.4.7 Data Encoding <a class="anchor" id="1.4.7"></a>

1. Binary encoding was applied to the predictor object columns in the training subset:
    * <span style="color: #FF0000">Status</span>
    * <span style="color: #FF0000">Drug</span>
    * <span style="color: #FF0000">Sex</span>
    * <span style="color: #FF0000">Ascites</span>
    * <span style="color: #FF0000">Hepatomegaly</span>
    * <span style="color: #FF0000">Spiders</span>
    * <span style="color: #FF0000">Edema</span>
1. One-hot encoding was applied to the <span style="color: #FF0000">Stage</span> variable resulting to 4 additional columns in the training subset:
    * <span style="color: #FF0000">Stage_1.0</span>
    * <span style="color: #FF0000">Stage_2.0</span>
    * <span style="color: #FF0000">Stage_3.0</span>
    * <span style="color: #FF0000">Stage_4.0</span>
3. Original data in the testing subset for object variables will be treated with binary and one-hot encoding downstream using a pipeline involving the final preprocessing steps.


```python
##################################
# Applying a binary encoding transformation
# for the two-level object columns
##################################
cirrhosis_survival_X_train_cleaned_object['Sex'] = cirrhosis_survival_X_train_cleaned_object['Sex'].replace({'M':0, 'F':1}) 
cirrhosis_survival_X_train_cleaned_object['Ascites'] = cirrhosis_survival_X_train_cleaned_object['Ascites'].replace({'N':0, 'Y':1}) 
cirrhosis_survival_X_train_cleaned_object['Drug'] = cirrhosis_survival_X_train_cleaned_object['Drug'].replace({'Placebo':0, 'D-penicillamine':1}) 
cirrhosis_survival_X_train_cleaned_object['Hepatomegaly'] = cirrhosis_survival_X_train_cleaned_object['Hepatomegaly'].replace({'N':0, 'Y':1}) 
cirrhosis_survival_X_train_cleaned_object['Spiders'] = cirrhosis_survival_X_train_cleaned_object['Spiders'].replace({'N':0, 'Y':1}) 
cirrhosis_survival_X_train_cleaned_object['Edema'] = cirrhosis_survival_X_train_cleaned_object['Edema'].replace({'N':0, 'Y':1, 'S':1}) 
```


```python
##################################
# Formulating the multi-level object column stage
# for encoding transformation
##################################
cirrhosis_survival_X_train_cleaned_object_stage_encoded = pd.DataFrame(cirrhosis_survival_X_train_cleaned_object.loc[:, 'Stage'].to_list(),
                                                                       columns=['Stage'])
```


```python
##################################
# Applying a one-hot encoding transformation
# for the multi-level object column stage
##################################
cirrhosis_survival_X_train_cleaned_object_stage_encoded = pd.get_dummies(cirrhosis_survival_X_train_cleaned_object_stage_encoded, columns=['Stage'])
```


```python
##################################
# Applying a one-hot encoding transformation
# for the multi-level object column stage
##################################
cirrhosis_survival_X_train_cleaned_encoded_object = pd.concat([cirrhosis_survival_X_train_cleaned_object.drop(['Stage'], axis=1), 
                                                               cirrhosis_survival_X_train_cleaned_object_stage_encoded], axis=1)
cirrhosis_survival_X_train_cleaned_encoded_object.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage_1.0</th>
      <th>Stage_2.0</th>
      <th>Stage_3.0</th>
      <th>Stage_4.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### 1.4.8 Preprocessed Data Description <a class="anchor" id="1.4.8"></a>

1. A preprocessing pipeline was formulated to standardize the data transformation methods applied to both the training and testing subsets.
2. The preprocessed training subset is comprised of:
    * **218 rows** (observations)
    * **22 columns** (variables)
        * **2/22 event | duration** (boolean | numeric)
             * <span style="color: #FF0000">Status</span>
             * <span style="color: #FF0000">N_Days</span>
        * **10/22 predictor** (numeric)
             * <span style="color: #FF0000">Age</span>
             * <span style="color: #FF0000">Bilirubin</span>
             * <span style="color: #FF0000">Cholesterol</span>
             * <span style="color: #FF0000">Albumin</span>
             * <span style="color: #FF0000">Copper</span>
             * <span style="color: #FF0000">Alk_Phos</span>
             * <span style="color: #FF0000">SGOT</span>
             * <span style="color: #FF0000">Triglycerides</span>
             * <span style="color: #FF0000">Platelets</span>
             * <span style="color: #FF0000">Prothrombin</span>
        * **10/22 predictor** (object)
             * <span style="color: #FF0000">Drug</span>
             * <span style="color: #FF0000">Sex</span>
             * <span style="color: #FF0000">Ascites</span>
             * <span style="color: #FF0000">Hepatomegaly</span>
             * <span style="color: #FF0000">Spiders</span>
             * <span style="color: #FF0000">Edema</span>
             * <span style="color: #FF0000">Stage_1.0</span>
             * <span style="color: #FF0000">Stage_2.0</span>
             * <span style="color: #FF0000">Stage_3.0</span>
             * <span style="color: #FF0000">Stage_4.0</span>
3. The preprocessed testing subset is comprised of:
    * **94 rows** (observations)
    * **22 columns** (variables)
        * **2/22 event | duration** (boolean | numeric)
             * <span style="color: #FF0000">Status</span>
             * <span style="color: #FF0000">N_Days</span>
        * **10/22 predictor** (numeric)
             * <span style="color: #FF0000">Age</span>
             * <span style="color: #FF0000">Bilirubin</span>
             * <span style="color: #FF0000">Cholesterol</span>
             * <span style="color: #FF0000">Albumin</span>
             * <span style="color: #FF0000">Copper</span>
             * <span style="color: #FF0000">Alk_Phos</span>
             * <span style="color: #FF0000">SGOT</span>
             * <span style="color: #FF0000">Triglycerides</span>
             * <span style="color: #FF0000">Platelets</span>
             * <span style="color: #FF0000">Prothrombin</span>
        * **10/22 predictor** (object)
             * <span style="color: #FF0000">Drug</span>
             * <span style="color: #FF0000">Sex</span>
             * <span style="color: #FF0000">Ascites</span>
             * <span style="color: #FF0000">Hepatomegaly</span>
             * <span style="color: #FF0000">Spiders</span>
             * <span style="color: #FF0000">Edema</span>
             * <span style="color: #FF0000">Stage_1.0</span>
             * <span style="color: #FF0000">Stage_2.0</span>
             * <span style="color: #FF0000">Stage_3.0</span>
             * <span style="color: #FF0000">Stage_4.0</span>


```python
##################################
# Consolidating all preprocessed
# numeric and object predictors
# for the training subset
##################################
cirrhosis_survival_X_train_preprocessed = pd.concat([cirrhosis_survival_X_train_scaled_numeric,
                                                     cirrhosis_survival_X_train_cleaned_encoded_object], 
                                                     axis=1)
cirrhosis_survival_X_train_preprocessed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage_1.0</th>
      <th>Stage_2.0</th>
      <th>Stage_3.0</th>
      <th>Stage_4.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.296446</td>
      <td>0.863802</td>
      <td>0.885512</td>
      <td>-0.451884</td>
      <td>-0.971563</td>
      <td>0.140990</td>
      <td>0.104609</td>
      <td>0.155256</td>
      <td>0.539120</td>
      <td>0.747580</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.405311</td>
      <td>0.516350</td>
      <td>1.556983</td>
      <td>0.827618</td>
      <td>0.468389</td>
      <td>-0.705337</td>
      <td>0.301441</td>
      <td>1.275281</td>
      <td>0.472266</td>
      <td>-0.315794</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.291081</td>
      <td>-0.625875</td>
      <td>0.290561</td>
      <td>0.646582</td>
      <td>-0.240371</td>
      <td>-0.848544</td>
      <td>0.275723</td>
      <td>-1.684460</td>
      <td>0.755044</td>
      <td>0.087130</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.107291</td>
      <td>0.559437</td>
      <td>-1.541148</td>
      <td>0.354473</td>
      <td>-0.283286</td>
      <td>-0.014525</td>
      <td>0.162878</td>
      <td>-0.189015</td>
      <td>-1.735183</td>
      <td>0.649171</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.813996</td>
      <td>1.142068</td>
      <td>-0.112859</td>
      <td>-0.272913</td>
      <td>0.618797</td>
      <td>2.071847</td>
      <td>1.434674</td>
      <td>-0.212560</td>
      <td>-0.677612</td>
      <td>-0.315794</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Creating a pre-processing pipeline
# for numeric predictors
##################################
cirrhosis_survival_numeric_predictors = ['Age', 'Bilirubin','Cholesterol', 'Albumin','Copper', 'Alk_Phos','SGOT', 'Tryglicerides','Platelets', 'Prothrombin']
numeric_transformer = Pipeline(steps=[
    ('imputer', IterativeImputer(estimator = lr,
                                 max_iter = 10,
                                 tol = 1e-10,
                                 imputation_order = 'ascending',
                                 random_state=88888888)),
    ('yeo_johnson', PowerTransformer(method='yeo-johnson',
                                    standardize=False)),
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, cirrhosis_survival_numeric_predictors)])
```


```python
##################################
# Fitting and transforming 
# training subset numeric predictors
##################################
cirrhosis_survival_X_train_numeric_preprocessed = preprocessor.fit_transform(cirrhosis_survival_X_train_cleaned)
cirrhosis_survival_X_train_numeric_preprocessed = pd.DataFrame(cirrhosis_survival_X_train_numeric_preprocessed,
                                                                columns=cirrhosis_survival_numeric_predictors)
cirrhosis_survival_X_train_numeric_preprocessed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.342097</td>
      <td>0.863802</td>
      <td>0.886087</td>
      <td>-0.451884</td>
      <td>-0.972098</td>
      <td>0.140990</td>
      <td>0.104609</td>
      <td>0.155130</td>
      <td>0.540960</td>
      <td>0.747580</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.470901</td>
      <td>0.516350</td>
      <td>1.554523</td>
      <td>0.827618</td>
      <td>0.467579</td>
      <td>-0.705337</td>
      <td>0.301441</td>
      <td>1.275222</td>
      <td>0.474140</td>
      <td>-0.315794</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.239814</td>
      <td>-0.625875</td>
      <td>0.293280</td>
      <td>0.646582</td>
      <td>-0.241205</td>
      <td>-0.848544</td>
      <td>0.275723</td>
      <td>-1.684460</td>
      <td>0.756741</td>
      <td>0.087130</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.052733</td>
      <td>0.559437</td>
      <td>-1.534283</td>
      <td>0.354473</td>
      <td>-0.284113</td>
      <td>-0.014525</td>
      <td>0.162878</td>
      <td>-0.189139</td>
      <td>-1.735375</td>
      <td>0.649171</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.795010</td>
      <td>1.142068</td>
      <td>-0.108933</td>
      <td>-0.272913</td>
      <td>0.618030</td>
      <td>2.071847</td>
      <td>1.434674</td>
      <td>-0.212684</td>
      <td>-0.675951</td>
      <td>-0.315794</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Performing pre-processing operations
# for object predictors
# in the training subset
##################################
cirrhosis_survival_object_predictors = ['Drug', 'Sex','Ascites', 'Hepatomegaly','Spiders', 'Edema','Stage']
cirrhosis_survival_X_train_object = cirrhosis_survival_X_train_cleaned.copy()
cirrhosis_survival_X_train_object = cirrhosis_survival_X_train_object[cirrhosis_survival_object_predictors]
cirrhosis_survival_X_train_object.reset_index(drop=True, inplace=True)
cirrhosis_survival_X_train_object.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>D-penicillamine</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>D-penicillamine</td>
      <td>M</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>D-penicillamine</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Placebo</td>
      <td>F</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Placebo</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>N</td>
      <td>N</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Applying a binary encoding transformation
# for the two-level object columns
# in the training subset
##################################
cirrhosis_survival_X_train_object['Sex'].replace({'M':0, 'F':1}, inplace=True) 
cirrhosis_survival_X_train_object['Ascites'].replace({'N':0, 'Y':1}, inplace=True) 
cirrhosis_survival_X_train_object['Drug'].replace({'Placebo':0, 'D-penicillamine':1}, inplace=True) 
cirrhosis_survival_X_train_object['Hepatomegaly'].replace({'N':0, 'Y':1}, inplace=True) 
cirrhosis_survival_X_train_object['Spiders'].replace({'N':0, 'Y':1}, inplace=True) 
cirrhosis_survival_X_train_object['Edema'].replace({'N':0, 'Y':1, 'S':1}, inplace=True) 
cirrhosis_survival_X_train_object_stage_encoded = pd.DataFrame(cirrhosis_survival_X_train_object.loc[:, 'Stage'].to_list(),
                                                                       columns=['Stage'])
cirrhosis_survival_X_train_object_stage_encoded = pd.get_dummies(cirrhosis_survival_X_train_object_stage_encoded, columns=['Stage'])
cirrhosis_survival_X_train_object_preprocessed = pd.concat([cirrhosis_survival_X_train_object.drop(['Stage'], axis=1), 
                                                            cirrhosis_survival_X_train_object_stage_encoded], 
                                                           axis=1)
cirrhosis_survival_X_train_object_preprocessed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage_1.0</th>
      <th>Stage_2.0</th>
      <th>Stage_3.0</th>
      <th>Stage_4.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Consolidating the preprocessed
# training subset
##################################
cirrhosis_survival_X_train_preprocessed = pd.concat([cirrhosis_survival_X_train_numeric_preprocessed, cirrhosis_survival_X_train_object_preprocessed], 
                                                    axis=1)
cirrhosis_survival_X_train_preprocessed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage_1.0</th>
      <th>Stage_2.0</th>
      <th>Stage_3.0</th>
      <th>Stage_4.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.342097</td>
      <td>0.863802</td>
      <td>0.886087</td>
      <td>-0.451884</td>
      <td>-0.972098</td>
      <td>0.140990</td>
      <td>0.104609</td>
      <td>0.155130</td>
      <td>0.540960</td>
      <td>0.747580</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.470901</td>
      <td>0.516350</td>
      <td>1.554523</td>
      <td>0.827618</td>
      <td>0.467579</td>
      <td>-0.705337</td>
      <td>0.301441</td>
      <td>1.275222</td>
      <td>0.474140</td>
      <td>-0.315794</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.239814</td>
      <td>-0.625875</td>
      <td>0.293280</td>
      <td>0.646582</td>
      <td>-0.241205</td>
      <td>-0.848544</td>
      <td>0.275723</td>
      <td>-1.684460</td>
      <td>0.756741</td>
      <td>0.087130</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.052733</td>
      <td>0.559437</td>
      <td>-1.534283</td>
      <td>0.354473</td>
      <td>-0.284113</td>
      <td>-0.014525</td>
      <td>0.162878</td>
      <td>-0.189139</td>
      <td>-1.735375</td>
      <td>0.649171</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.795010</td>
      <td>1.142068</td>
      <td>-0.108933</td>
      <td>-0.272913</td>
      <td>0.618030</td>
      <td>2.071847</td>
      <td>1.434674</td>
      <td>-0.212684</td>
      <td>-0.675951</td>
      <td>-0.315794</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Verifying the dimensions of the
# preprocessed training subset
##################################
cirrhosis_survival_X_train_preprocessed.shape
```




    (218, 20)




```python
##################################
# Fitting and transforming 
# testing subset numeric predictors
##################################
cirrhosis_survival_X_test_numeric_preprocessed = preprocessor.transform(cirrhosis_survival_X_test_cleaned)
cirrhosis_survival_X_test_numeric_preprocessed = pd.DataFrame(cirrhosis_survival_X_test_numeric_preprocessed,
                                                                columns=cirrhosis_survival_numeric_predictors)
cirrhosis_survival_X_test_numeric_preprocessed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.043704</td>
      <td>0.744396</td>
      <td>0.922380</td>
      <td>0.240951</td>
      <td>0.045748</td>
      <td>0.317282</td>
      <td>-0.078335</td>
      <td>2.671950</td>
      <td>1.654815</td>
      <td>-0.948196</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.936476</td>
      <td>-0.764558</td>
      <td>0.160096</td>
      <td>-0.600950</td>
      <td>-0.179138</td>
      <td>-0.245613</td>
      <td>0.472422</td>
      <td>-0.359800</td>
      <td>0.348533</td>
      <td>0.439089</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.749033</td>
      <td>0.371523</td>
      <td>0.558115</td>
      <td>0.646582</td>
      <td>-0.159024</td>
      <td>0.339454</td>
      <td>0.685117</td>
      <td>-3.109146</td>
      <td>-0.790172</td>
      <td>-0.617113</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.485150</td>
      <td>-0.918484</td>
      <td>-0.690904</td>
      <td>1.629765</td>
      <td>0.028262</td>
      <td>1.713791</td>
      <td>-1.387751</td>
      <td>0.155130</td>
      <td>0.679704</td>
      <td>0.087130</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.815655</td>
      <td>1.286438</td>
      <td>2.610501</td>
      <td>-0.722153</td>
      <td>0.210203</td>
      <td>0.602860</td>
      <td>3.494936</td>
      <td>-0.053214</td>
      <td>-0.475634</td>
      <td>-1.714435</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Performing pre-processing operations
# for object predictors
# in the testing subset
##################################
cirrhosis_survival_object_predictors = ['Drug', 'Sex','Ascites', 'Hepatomegaly','Spiders', 'Edema','Stage']
cirrhosis_survival_X_test_object = cirrhosis_survival_X_test_cleaned.copy()
cirrhosis_survival_X_test_object = cirrhosis_survival_X_test_object[cirrhosis_survival_object_predictors]
cirrhosis_survival_X_test_object.reset_index(drop=True, inplace=True)
cirrhosis_survival_X_test_object.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>D-penicillamine</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>Y</td>
      <td>S</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Placebo</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>D-penicillamine</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>N</td>
      <td>N</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D-penicillamine</td>
      <td>M</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Placebo</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>N</td>
      <td>N</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Applying a binary encoding transformation
# for the two-level object columns
# in the testing subset
##################################
cirrhosis_survival_X_test_object['Sex'].replace({'M':0, 'F':1}, inplace=True) 
cirrhosis_survival_X_test_object['Ascites'].replace({'N':0, 'Y':1}, inplace=True) 
cirrhosis_survival_X_test_object['Drug'].replace({'Placebo':0, 'D-penicillamine':1}, inplace=True) 
cirrhosis_survival_X_test_object['Hepatomegaly'].replace({'N':0, 'Y':1}, inplace=True) 
cirrhosis_survival_X_test_object['Spiders'].replace({'N':0, 'Y':1}, inplace=True) 
cirrhosis_survival_X_test_object['Edema'].replace({'N':0, 'Y':1, 'S':1}, inplace=True) 
cirrhosis_survival_X_test_object_stage_encoded = pd.DataFrame(cirrhosis_survival_X_test_object.loc[:, 'Stage'].to_list(),
                                                                       columns=['Stage'])
cirrhosis_survival_X_test_object_stage_encoded = pd.get_dummies(cirrhosis_survival_X_test_object_stage_encoded, columns=['Stage'])
cirrhosis_survival_X_test_object_preprocessed = pd.concat([cirrhosis_survival_X_test_object.drop(['Stage'], axis=1), 
                                                            cirrhosis_survival_X_test_object_stage_encoded], 
                                                           axis=1)
cirrhosis_survival_X_test_object_preprocessed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage_1.0</th>
      <th>Stage_2.0</th>
      <th>Stage_3.0</th>
      <th>Stage_4.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Consolidating the preprocessed
# testing subset
##################################
cirrhosis_survival_X_test_preprocessed = pd.concat([cirrhosis_survival_X_test_numeric_preprocessed, cirrhosis_survival_X_test_object_preprocessed], 
                                                    axis=1)
cirrhosis_survival_X_test_preprocessed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage_1.0</th>
      <th>Stage_2.0</th>
      <th>Stage_3.0</th>
      <th>Stage_4.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.043704</td>
      <td>0.744396</td>
      <td>0.922380</td>
      <td>0.240951</td>
      <td>0.045748</td>
      <td>0.317282</td>
      <td>-0.078335</td>
      <td>2.671950</td>
      <td>1.654815</td>
      <td>-0.948196</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.936476</td>
      <td>-0.764558</td>
      <td>0.160096</td>
      <td>-0.600950</td>
      <td>-0.179138</td>
      <td>-0.245613</td>
      <td>0.472422</td>
      <td>-0.359800</td>
      <td>0.348533</td>
      <td>0.439089</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.749033</td>
      <td>0.371523</td>
      <td>0.558115</td>
      <td>0.646582</td>
      <td>-0.159024</td>
      <td>0.339454</td>
      <td>0.685117</td>
      <td>-3.109146</td>
      <td>-0.790172</td>
      <td>-0.617113</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.485150</td>
      <td>-0.918484</td>
      <td>-0.690904</td>
      <td>1.629765</td>
      <td>0.028262</td>
      <td>1.713791</td>
      <td>-1.387751</td>
      <td>0.155130</td>
      <td>0.679704</td>
      <td>0.087130</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.815655</td>
      <td>1.286438</td>
      <td>2.610501</td>
      <td>-0.722153</td>
      <td>0.210203</td>
      <td>0.602860</td>
      <td>3.494936</td>
      <td>-0.053214</td>
      <td>-0.475634</td>
      <td>-1.714435</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Verifying the dimensions of the
# preprocessed testing subset
##################################
cirrhosis_survival_X_test_preprocessed.shape
```




    (94, 20)



## 1.5. Data Exploration <a class="anchor" id="1.5"></a>

### 1.5.1 Exploratory Data Analysis <a class="anchor" id="1.5.1"></a>

1. The estimated baseline survival plot indicated a 50% survival rate at <span style="color: #FF0000">N_Days=3358</span>.
2. Bivariate analysis identified individual predictors with potential association to the event status based on visual inspection.
    * Higher values for the following numeric predictors are associated with <span style="color: #FF0000">Status=True</span>: 
        * <span style="color: #FF0000">Age</span>
        * <span style="color: #FF0000">Bilirubin</span>   
        * <span style="color: #FF0000">Copper</span>
        * <span style="color: #FF0000">Alk_Phos</span> 
        * <span style="color: #FF0000">SGOT</span>   
        * <span style="color: #FF0000">Tryglicerides</span> 
        * <span style="color: #FF0000">Prothrombin</span>    
    * Higher counts for the following object predictors are associated with better differentiation between <span style="color: #FF0000">Status=True</span> and <span style="color: #FF0000">Status=False</span>:  
        * <span style="color: #FF0000">Drug</span>
        * <span style="color: #FF0000">Sex</span>
        * <span style="color: #FF0000">Ascites</span>
        * <span style="color: #FF0000">Hepatomegaly</span>
        * <span style="color: #FF0000">Spiders</span>
        * <span style="color: #FF0000">Edema</span>
        * <span style="color: #FF0000">Stage_1.0</span>
        * <span style="color: #FF0000">Stage_2.0</span>
        * <span style="color: #FF0000">Stage_3.0</span>
        * <span style="color: #FF0000">Stage_4.0</span>
2. Bivariate analysis identified individual predictors with potential association to the survival time based on visual inspection.
    * Higher values for the following numeric predictors are positively associated with <span style="color: #FF0000">N_Days</span>: 
        * <span style="color: #FF0000">Albumin</span>        
        * <span style="color: #FF0000">Platelets</span>
    * Levels for the following object predictors are associated with differences in <span style="color: #FF0000">N_Days</span> between <span style="color: #FF0000">Status=True</span> and <span style="color: #FF0000">Status=False</span>:  
        * <span style="color: #FF0000">Drug</span>
        * <span style="color: #FF0000">Sex</span>
        * <span style="color: #FF0000">Ascites</span>
        * <span style="color: #FF0000">Hepatomegaly</span>
        * <span style="color: #FF0000">Spiders</span>
        * <span style="color: #FF0000">Edema</span>
        * <span style="color: #FF0000">Stage_1.0</span>
        * <span style="color: #FF0000">Stage_2.0</span>
        * <span style="color: #FF0000">Stage_3.0</span>
        * <span style="color: #FF0000">Stage_4.0</span>


```python
##################################
# Formulating a complete dataframe
# from the training subset for EDA
##################################
cirrhosis_survival_y_train_cleaned.reset_index(drop=True, inplace=True)
cirrhosis_survival_train_EDA = pd.concat([cirrhosis_survival_y_train_cleaned,
                                          cirrhosis_survival_X_train_preprocessed],
                                         axis=1)
cirrhosis_survival_train_EDA.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Status</th>
      <th>N_Days</th>
      <th>Age</th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>...</th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage_1.0</th>
      <th>Stage_2.0</th>
      <th>Stage_3.0</th>
      <th>Stage_4.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>2475</td>
      <td>-1.342097</td>
      <td>0.863802</td>
      <td>0.886087</td>
      <td>-0.451884</td>
      <td>-0.972098</td>
      <td>0.140990</td>
      <td>0.104609</td>
      <td>0.155130</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>877</td>
      <td>-1.470901</td>
      <td>0.516350</td>
      <td>1.554523</td>
      <td>0.827618</td>
      <td>0.467579</td>
      <td>-0.705337</td>
      <td>0.301441</td>
      <td>1.275222</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>3050</td>
      <td>-0.239814</td>
      <td>-0.625875</td>
      <td>0.293280</td>
      <td>0.646582</td>
      <td>-0.241205</td>
      <td>-0.848544</td>
      <td>0.275723</td>
      <td>-1.684460</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>True</td>
      <td>110</td>
      <td>-0.052733</td>
      <td>0.559437</td>
      <td>-1.534283</td>
      <td>0.354473</td>
      <td>-0.284113</td>
      <td>-0.014525</td>
      <td>0.162878</td>
      <td>-0.189139</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>True</td>
      <td>3839</td>
      <td>-0.795010</td>
      <td>1.142068</td>
      <td>-0.108933</td>
      <td>-0.272913</td>
      <td>0.618030</td>
      <td>2.071847</td>
      <td>1.434674</td>
      <td>-0.212684</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
##################################
# Plotting the baseline survival curve
# and computing the survival rates
##################################
kmf = KaplanMeierFitter()
kmf.fit(durations=cirrhosis_survival_train_EDA['N_Days'], event_observed=cirrhosis_survival_train_EDA['Status'])
plt.figure(figsize=(17, 8))
kmf.plot_survival_function()
plt.title('Kaplan-Meier Baseline Survival Plot')
plt.ylim(0, 1.05)
plt.xlabel('N_Days')
plt.ylabel('Event Survival Probability')

##################################
# Determing the at-risk numbers
##################################
at_risk_counts = kmf.event_table.at_risk
survival_probabilities = kmf.survival_function_.values.flatten()
time_points = kmf.survival_function_.index
for time, prob, at_risk in zip(time_points, survival_probabilities, at_risk_counts):
    if time % 50 == 0: 
        plt.text(time, prob, f'{prob:.2f} : {at_risk}', ha='left', fontsize=10)
median_survival_time = kmf.median_survival_time_
plt.axvline(x=median_survival_time, color='r', linestyle='--')
plt.axhline(y=0.5, color='r', linestyle='--')
plt.text(3400, 0.52, f'Median: {median_survival_time}', ha='left', va='bottom', color='r', fontsize=10)
plt.show()
```


    
![png](output_140_0.png)
    



```python
##################################
# Computing the median survival time
##################################
median_survival_time = kmf.median_survival_time_
print(f'Median Survival Time: {median_survival_time}')
```

    Median Survival Time: 3358.0
    


```python
##################################
# Exploring the relationships between
# the numeric predictors and event status
##################################
cirrhosis_survival_numeric_predictors = ['Age', 'Bilirubin','Cholesterol', 'Albumin','Copper', 'Alk_Phos','SGOT', 'Tryglicerides','Platelets', 'Prothrombin']
plt.figure(figsize=(17, 12))
for i in range(1, 11):
    plt.subplot(2, 5, i)
    sns.boxplot(x='Status', y=cirrhosis_survival_numeric_predictors[i-1], data=cirrhosis_survival_train_EDA)
    plt.title(f'{cirrhosis_survival_numeric_predictors[i-1]} vs Event Status')
plt.tight_layout()
plt.show()
```


    
![png](output_142_0.png)
    



```python
##################################
# Exploring the relationships between
# the object predictors and event status
##################################
cirrhosis_survival_object_predictors = ['Drug', 'Sex','Ascites', 'Hepatomegaly','Spiders', 'Edema','Stage_1.0','Stage_2.0','Stage_3.0','Stage_4.0']
plt.figure(figsize=(17, 12))
for i in range(1, 11):
    plt.subplot(2, 5, i)
    sns.countplot(x=cirrhosis_survival_object_predictors[i-1], hue='Status', data=cirrhosis_survival_train_EDA)
    plt.title(f'{cirrhosis_survival_object_predictors[i-1]} vs Event Status')
    plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
```


    
![png](output_143_0.png)
    



```python
##################################
# Exploring the relationships between
# the numeric predictors and survival time
##################################
plt.figure(figsize=(17, 12))
for i in range(1, 11):
    plt.subplot(2, 5, i)
    sns.scatterplot(x='N_Days', y=cirrhosis_survival_numeric_predictors[i-1], data=cirrhosis_survival_train_EDA, hue='Status')
    loess_smoothed = lowess(cirrhosis_survival_train_EDA['N_Days'], cirrhosis_survival_train_EDA[cirrhosis_survival_numeric_predictors[i-1]], frac=0.3)
    plt.plot(loess_smoothed[:, 1], loess_smoothed[:, 0], color='red')
    plt.title(f'{cirrhosis_survival_numeric_predictors[i-1]} vs Survival Time')
    plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
```


    
![png](output_144_0.png)
    



```python
##################################
# Exploring the relationships between
# the object predictors and survival time
##################################
plt.figure(figsize=(17, 12))
for i in range(1, 11):
    plt.subplot(2, 5, i)
    sns.boxplot(x=cirrhosis_survival_object_predictors[i-1], y='N_Days', hue='Status', data=cirrhosis_survival_train_EDA)
    plt.title(f'{cirrhosis_survival_object_predictors[i-1]} vs Survival Time')
    plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
```


    
![png](output_145_0.png)
    


### 1.5.2 Hypothesis Testing <a class="anchor" id="1.5.2"></a>

1. The relationship between the numeric predictors to the <span style="color: #FF0000">Status</span> event variable was statistically evaluated using the following hypotheses:
    * **Null**: Difference in the means between groups True and False is equal to zero  
    * **Alternative**: Difference in the means between groups True and False is not equal to zero   
2. There is sufficient evidence to conclude of a statistically significant difference between the means of the numeric measurements obtained from the <span style="color: #FF0000">Status</span> groups in 10 numeric predictors given their high t-test statistic values with reported low p-values less than the significance level of 0.05.
    * <span style="color: #FF0000">Bilirubin</span>: T.Test.Statistic=-8.031, T.Test.PValue=0.000
    * <span style="color: #FF0000">Prothrombin</span>: T.Test.Statistic=-7.062, T.Test.PValue=0.000 
    * <span style="color: #FF0000">Copper</span>: T.Test.Statistic=-5.699, T.Test.PValue=0.000  
    * <span style="color: #FF0000">Alk_Phos</span>: T.Test.Statistic=-4.638, T.Test.PValue=0.000 
    * <span style="color: #FF0000">SGOT</span>: T.Test.Statistic=-4.207, T.Test.PValue=0.000 
    * <span style="color: #FF0000">Albumin</span>: T.Test.Statistic=+3.871, T.Test.PValue=0.000  
    * <span style="color: #FF0000">Tryglicerides</span>: T.Test.Statistic=-3.575, T.Test.PValue=0.000   
    * <span style="color: #FF0000">Age</span>: T.Test.Statistic=-3.264, T.Test.PValue=0.001
    * <span style="color: #FF0000">Platelets</span>: T.Test.Statistic=+3.261, T.Test.PValue=0.001
    * <span style="color: #FF0000">Cholesterol</span>: T.Test.Statistic=-2.256, T.Test.PValue=0.025
3. The relationship between the object predictors to the <span style="color: #FF0000">Status</span> event variable was statistically evaluated using the following hypotheses:
    * **Null**: The object predictor is independent of the event variable 
    * **Alternative**: The object predictor is dependent on the event variable   
4. There is sufficient evidence to conclude of a statistically significant relationship between the individual categories and the <span style="color: #FF0000">Status</span> groups in 8 object predictors given their high chisquare statistic values with reported low p-values less than the significance level of 0.05.
    * <span style="color: #FF0000">Ascites</span>: ChiSquare.Test.Statistic=16.854, ChiSquare.Test.PValue=0.000
    * <span style="color: #FF0000">Hepatomegaly</span>: ChiSquare.Test.Statistic=14.206, ChiSquare.Test.PValue=0.000   
    * <span style="color: #FF0000">Edema</span>: ChiSquare.Test.Statistic=12.962, ChiSquare.Test.PValue=0.001 
    * <span style="color: #FF0000">Stage_4.0</span>: ChiSquare.Test.Statistic=11.505, ChiSquare.Test.PValue=0.00
    * <span style="color: #FF0000">Sex</span>: ChiSquare.Test.Statistic=6.837, ChiSquare.Test.PValue=0.008
    * <span style="color: #FF0000">Stage_2.0</span>: ChiSquare.Test.Statistic=4.024, ChiSquare.Test.PValue=0.045   
    * <span style="color: #FF0000">Stage_1.0</span>: ChiSquare.Test.Statistic=3.978, ChiSquare.Test.PValue=0.046 
    * <span style="color: #FF0000">Spiders</span>: ChiSquare.Test.Statistic=3.953, ChiSquare.Test.PValue=0.047
5. The relationship between the object predictors to the <span style="color: #FF0000">Status</span> and <span style="color: #FF0000">N_Days</span> variables was statistically evaluated using the following hypotheses:
    * **Null**: There is no difference in survival probabilities among cases belonging to each category of the object predictor.
    * **Alternative**: There is a difference in survival probabilities among cases belonging to each category of the object predictor.
6. There is sufficient evidence to conclude of a statistically significant difference in survival probabilities between the individual categories and the <span style="color: #FF0000">Status</span> groups with respect to the survival duration <span style="color: #FF0000">N_Days</span> in 8 object predictors given their high log-rank test statistic values with reported low p-values less than the significance level of 0.05.
    * <span style="color: #FF0000">Ascites</span>: LR.Test.Statistic=37.792, LR.Test.PValue=0.000
    * <span style="color: #FF0000">Edema</span>: LR.Test.Statistic=31.619, LR.Test.PValue=0.000 
    * <span style="color: #FF0000">Stage_4.0</span>: LR.Test.Statistic=26.482, LR.Test.PValue=0.000
    * <span style="color: #FF0000">Hepatomegaly</span>: LR.Test.Statistic=20.350, LR.Test.PValue=0.000   
    * <span style="color: #FF0000">Spiders</span>: LR.Test.Statistic=10.762, LR.Test.PValue=0.001
    * <span style="color: #FF0000">Stage_2.0</span>: LR.Test.Statistic=6.775, LR.Test.PValue=0.009   
    * <span style="color: #FF0000">Sex</span>: LR.Test.Statistic=5.514, LR.Test.PValue=0.018
    * <span style="color: #FF0000">Stage_1.0</span>: LR.Test.Statistic=5.473, LR.Test.PValue=0.019 
7. The relationship between the binned numeric predictors to the <span style="color: #FF0000">Status</span> and <span style="color: #FF0000">N_Days</span> variables was statistically evaluated using the following hypotheses:
    * **Null**: There is no difference in survival probabilities among cases belonging to each category of the binned numeric predictor.
    * **Alternative**: There is a difference in survival probabilities among cases belonging to each category of the binned numeric predictor.
8. There is sufficient evidence to conclude of a statistically significant difference in survival probabilities between the individual categories and the <span style="color: #FF0000">Status</span> groups with respect to the survival duration <span style="color: #FF0000">N_Days</span> in 9 binned numeric predictors given their high log-rank test statistic values with reported low p-values less than the significance level of 0.05.
    * <span style="color: #FF0000">Binned_Bilirubin</span>: LR.Test.Statistic=62.559, LR.Test.PValue=0.000
    * <span style="color: #FF0000">Binned_Albumin</span>: LR.Test.Statistic=29.444, LR.Test.PValue=0.000 
    * <span style="color: #FF0000">Binned_Copper</span>: LR.Test.Statistic=27.452, LR.Test.PValue=0.000
    * <span style="color: #FF0000">Binned_Prothrombin</span>: LR.Test.Statistic=21.695, LR.Test.PValue=0.000   
    * <span style="color: #FF0000">Binned_SGOT</span>: LR.Test.Statistic=16.178, LR.Test.PValue=0.000
    * <span style="color: #FF0000">Binned_Tryglicerides</span>: LR.Test.Statistic=11.512, LR.Test.PValue=0.000   
    * <span style="color: #FF0000">Binned_Age</span>: LR.Test.Statistic=9.012, LR.Test.PValue=0.002
    * <span style="color: #FF0000">Binned_Platelets</span>: LR.Test.Statistic=6.741, LR.Test.PValue=0.009 
    * <span style="color: #FF0000">Binned_Alk_Phos</span>: LR.Test.Statistic=5.503, LR.Test.PValue=0.018 



```python
##################################
# Computing the t-test 
# statistic and p-values
# between the event variable
# and numeric predictor columns
##################################
cirrhosis_survival_numeric_ttest_event = {}
for numeric_column in cirrhosis_survival_numeric_predictors:
    group_0 = cirrhosis_survival_train_EDA[cirrhosis_survival_train_EDA.loc[:,'Status']==False]
    group_1 = cirrhosis_survival_train_EDA[cirrhosis_survival_train_EDA.loc[:,'Status']==True]
    cirrhosis_survival_numeric_ttest_event['Status_' + numeric_column] = stats.ttest_ind(
        group_0[numeric_column], 
        group_1[numeric_column], 
        equal_var=True)
```


```python
##################################
# Formulating the pairwise ttest summary
# between the event variable
# and numeric predictor columns
##################################
cirrhosis_survival_numeric_ttest_summary = cirrhosis_survival_train_EDA.from_dict(cirrhosis_survival_numeric_ttest_event, orient='index')
cirrhosis_survival_numeric_ttest_summary.columns = ['T.Test.Statistic', 'T.Test.PValue']
display(cirrhosis_survival_numeric_ttest_summary.sort_values(by=['T.Test.PValue'], ascending=True))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>T.Test.Statistic</th>
      <th>T.Test.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Status_Bilirubin</th>
      <td>-8.030789</td>
      <td>6.198797e-14</td>
    </tr>
    <tr>
      <th>Status_Prothrombin</th>
      <td>-7.062933</td>
      <td>2.204961e-11</td>
    </tr>
    <tr>
      <th>Status_Copper</th>
      <td>-5.699409</td>
      <td>3.913575e-08</td>
    </tr>
    <tr>
      <th>Status_Alk_Phos</th>
      <td>-4.638524</td>
      <td>6.077981e-06</td>
    </tr>
    <tr>
      <th>Status_SGOT</th>
      <td>-4.207123</td>
      <td>3.791642e-05</td>
    </tr>
    <tr>
      <th>Status_Albumin</th>
      <td>3.871216</td>
      <td>1.434736e-04</td>
    </tr>
    <tr>
      <th>Status_Tryglicerides</th>
      <td>-3.575779</td>
      <td>4.308371e-04</td>
    </tr>
    <tr>
      <th>Status_Age</th>
      <td>-3.264563</td>
      <td>1.274679e-03</td>
    </tr>
    <tr>
      <th>Status_Platelets</th>
      <td>3.261042</td>
      <td>1.289850e-03</td>
    </tr>
    <tr>
      <th>Status_Cholesterol</th>
      <td>-2.256073</td>
      <td>2.506758e-02</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Computing the chisquare
# statistic and p-values
# between the event variable
# and categorical predictor columns
##################################
cirrhosis_survival_object_chisquare_event = {}
for object_column in cirrhosis_survival_object_predictors:
    contingency_table = pd.crosstab(cirrhosis_survival_train_EDA[object_column], 
                                    cirrhosis_survival_train_EDA['Status'])
    cirrhosis_survival_object_chisquare_event['Status_' + object_column] = stats.chi2_contingency(
        contingency_table)[0:2]
```


```python
##################################
# Formulating the pairwise chisquare summary
# between the event variable
# and categorical predictor columns
##################################
cirrhosis_survival_object_chisquare_event_summary = cirrhosis_survival_train_EDA.from_dict(cirrhosis_survival_object_chisquare_event, orient='index')
cirrhosis_survival_object_chisquare_event_summary.columns = ['ChiSquare.Test.Statistic', 'ChiSquare.Test.PValue']
display(cirrhosis_survival_object_chisquare_event_summary.sort_values(by=['ChiSquare.Test.PValue'], ascending=True))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ChiSquare.Test.Statistic</th>
      <th>ChiSquare.Test.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Status_Ascites</th>
      <td>16.854134</td>
      <td>0.000040</td>
    </tr>
    <tr>
      <th>Status_Hepatomegaly</th>
      <td>14.206045</td>
      <td>0.000164</td>
    </tr>
    <tr>
      <th>Status_Edema</th>
      <td>12.962303</td>
      <td>0.000318</td>
    </tr>
    <tr>
      <th>Status_Stage_4.0</th>
      <td>11.505826</td>
      <td>0.000694</td>
    </tr>
    <tr>
      <th>Status_Sex</th>
      <td>6.837272</td>
      <td>0.008928</td>
    </tr>
    <tr>
      <th>Status_Stage_2.0</th>
      <td>4.024677</td>
      <td>0.044839</td>
    </tr>
    <tr>
      <th>Status_Stage_1.0</th>
      <td>3.977918</td>
      <td>0.046101</td>
    </tr>
    <tr>
      <th>Status_Spiders</th>
      <td>3.953826</td>
      <td>0.046765</td>
    </tr>
    <tr>
      <th>Status_Stage_3.0</th>
      <td>0.082109</td>
      <td>0.774459</td>
    </tr>
    <tr>
      <th>Status_Drug</th>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Exploring the relationships between
# the object predictors with
# survival event and duration
##################################
plt.figure(figsize=(17, 25))
for i in range(0, len(cirrhosis_survival_object_predictors)):
    ax = plt.subplot(5, 2, i+1)
    for group in [0,1]:
        kmf.fit(durations=cirrhosis_survival_train_EDA[cirrhosis_survival_train_EDA[cirrhosis_survival_object_predictors[i]] == group]['N_Days'],
                event_observed=cirrhosis_survival_train_EDA[cirrhosis_survival_train_EDA[cirrhosis_survival_object_predictors[i]] == group]['Status'], label=group)
        kmf.plot_survival_function(ax=ax)
    plt.title(f'Survival Probabilities by {cirrhosis_survival_object_predictors[i]} Categories')
    plt.xlabel('N_Days')
    plt.ylabel('Event Survival Probability')
plt.tight_layout()
plt.show()
```


    
![png](output_151_0.png)
    



```python
##################################
# Computing the log-rank test
# statistic and p-values
# between the event and duration variables
# with the object predictor columns
##################################
cirrhosis_survival_object_lrtest_event = {}
for object_column in cirrhosis_survival_object_predictors:
    groups = [0,1]
    group_0_event = cirrhosis_survival_train_EDA[cirrhosis_survival_train_EDA[object_column] == groups[0]]['Status']
    group_1_event = cirrhosis_survival_train_EDA[cirrhosis_survival_train_EDA[object_column] == groups[1]]['Status']
    group_0_duration = cirrhosis_survival_train_EDA[cirrhosis_survival_train_EDA[object_column] == groups[0]]['N_Days']
    group_1_duration = cirrhosis_survival_train_EDA[cirrhosis_survival_train_EDA[object_column] == groups[1]]['N_Days']
    lr_test = logrank_test(group_0_duration, group_1_duration,event_observed_A=group_0_event, event_observed_B=group_1_event)
    cirrhosis_survival_object_lrtest_event['Status_NDays_' + object_column] = (lr_test.test_statistic, lr_test.p_value)
```


```python
##################################
# Formulating the log-rank test summary
# between the event and duration variables
# with the object predictor columns
##################################
cirrhosis_survival_object_lrtest_summary = cirrhosis_survival_train_EDA.from_dict(cirrhosis_survival_object_lrtest_event, orient='index')
cirrhosis_survival_object_lrtest_summary.columns = ['LR.Test.Statistic', 'LR.Test.PValue']
display(cirrhosis_survival_object_lrtest_summary.sort_values(by=['LR.Test.PValue'], ascending=True))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LR.Test.Statistic</th>
      <th>LR.Test.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Status_NDays_Ascites</th>
      <td>37.792220</td>
      <td>7.869499e-10</td>
    </tr>
    <tr>
      <th>Status_NDays_Edema</th>
      <td>31.619652</td>
      <td>1.875223e-08</td>
    </tr>
    <tr>
      <th>Status_NDays_Stage_4.0</th>
      <td>26.482676</td>
      <td>2.659121e-07</td>
    </tr>
    <tr>
      <th>Status_NDays_Hepatomegaly</th>
      <td>20.360210</td>
      <td>6.414988e-06</td>
    </tr>
    <tr>
      <th>Status_NDays_Spiders</th>
      <td>10.762275</td>
      <td>1.035900e-03</td>
    </tr>
    <tr>
      <th>Status_NDays_Stage_2.0</th>
      <td>6.775033</td>
      <td>9.244176e-03</td>
    </tr>
    <tr>
      <th>Status_NDays_Sex</th>
      <td>5.514094</td>
      <td>1.886385e-02</td>
    </tr>
    <tr>
      <th>Status_NDays_Stage_1.0</th>
      <td>5.473270</td>
      <td>1.930946e-02</td>
    </tr>
    <tr>
      <th>Status_NDays_Stage_3.0</th>
      <td>0.478031</td>
      <td>4.893156e-01</td>
    </tr>
    <tr>
      <th>Status_NDays_Drug</th>
      <td>0.000016</td>
      <td>9.968084e-01</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Creating an alternate copy of the 
# EDA data which will utilize
# binning for numeric predictors
##################################
cirrhosis_survival_train_EDA_binned = cirrhosis_survival_train_EDA.copy()

##################################
# Creating a function to bin
# numeric predictors into two groups
##################################
def bin_numeric_predictor(df, predictor):
    median = df[predictor].median()
    df[f'Binned_{predictor}'] = np.where(df[predictor] <= median, 0, 1)
    return df

##################################
# Binning the numeric predictors
# in the alternate EDA data into two groups
##################################
for numeric_column in cirrhosis_survival_numeric_predictors:
    cirrhosis_survival_train_EDA_binned = bin_numeric_predictor(cirrhosis_survival_train_EDA_binned, numeric_column)
    
##################################
# Formulating the binned numeric predictors
##################################    
cirrhosis_survival_binned_numeric_predictors = ["Binned_" + predictor for predictor in cirrhosis_survival_numeric_predictors]
```


```python
##################################
# Exploring the relationships between
# the binned numeric predictors with
# survival event and duration
##################################
plt.figure(figsize=(17, 25))
for i in range(0, len(cirrhosis_survival_binned_numeric_predictors)):
    ax = plt.subplot(5, 2, i+1)
    for group in [0,1]:
        kmf.fit(durations=cirrhosis_survival_train_EDA_binned[cirrhosis_survival_train_EDA_binned[cirrhosis_survival_binned_numeric_predictors[i]] == group]['N_Days'],
                event_observed=cirrhosis_survival_train_EDA_binned[cirrhosis_survival_train_EDA_binned[cirrhosis_survival_binned_numeric_predictors[i]] == group]['Status'], label=group)
        kmf.plot_survival_function(ax=ax)
    plt.title(f'Survival Probabilities by {cirrhosis_survival_binned_numeric_predictors[i]} Categories')
    plt.xlabel('N_Days')
    plt.ylabel('Event Survival Probability')
plt.tight_layout()
plt.show()
```


    
![png](output_155_0.png)
    



```python
##################################
# Computing the log-rank test
# statistic and p-values
# between the event and duration variables
# with the binned numeric predictor columns
##################################
cirrhosis_survival_binned_numeric_lrtest_event = {}
for binned_numeric_column in cirrhosis_survival_binned_numeric_predictors:
    groups = [0,1]
    group_0_event = cirrhosis_survival_train_EDA_binned[cirrhosis_survival_train_EDA_binned[binned_numeric_column] == groups[0]]['Status']
    group_1_event = cirrhosis_survival_train_EDA_binned[cirrhosis_survival_train_EDA_binned[binned_numeric_column] == groups[1]]['Status']
    group_0_duration = cirrhosis_survival_train_EDA_binned[cirrhosis_survival_train_EDA_binned[binned_numeric_column] == groups[0]]['N_Days']
    group_1_duration = cirrhosis_survival_train_EDA_binned[cirrhosis_survival_train_EDA_binned[binned_numeric_column] == groups[1]]['N_Days']
    lr_test = logrank_test(group_0_duration, group_1_duration,event_observed_A=group_0_event, event_observed_B=group_1_event)
    cirrhosis_survival_binned_numeric_lrtest_event['Status_NDays_' + binned_numeric_column] = (lr_test.test_statistic, lr_test.p_value)
```


```python
##################################
# Formulating the log-rank test summary
# between the event and duration variables
# with the binned numeric predictor columns
##################################
cirrhosis_survival_binned_numeric_lrtest_summary = cirrhosis_survival_train_EDA_binned.from_dict(cirrhosis_survival_binned_numeric_lrtest_event, orient='index')
cirrhosis_survival_binned_numeric_lrtest_summary.columns = ['LR.Test.Statistic', 'LR.Test.PValue']
display(cirrhosis_survival_binned_numeric_lrtest_summary.sort_values(by=['LR.Test.PValue'], ascending=True))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LR.Test.Statistic</th>
      <th>LR.Test.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Status_NDays_Binned_Bilirubin</th>
      <td>62.559303</td>
      <td>2.585412e-15</td>
    </tr>
    <tr>
      <th>Status_NDays_Binned_Albumin</th>
      <td>29.444808</td>
      <td>5.753197e-08</td>
    </tr>
    <tr>
      <th>Status_NDays_Binned_Copper</th>
      <td>27.452421</td>
      <td>1.610072e-07</td>
    </tr>
    <tr>
      <th>Status_NDays_Binned_Prothrombin</th>
      <td>21.695995</td>
      <td>3.194575e-06</td>
    </tr>
    <tr>
      <th>Status_NDays_Binned_SGOT</th>
      <td>16.178483</td>
      <td>5.764520e-05</td>
    </tr>
    <tr>
      <th>Status_NDays_Binned_Tryglicerides</th>
      <td>11.512960</td>
      <td>6.911262e-04</td>
    </tr>
    <tr>
      <th>Status_NDays_Binned_Age</th>
      <td>9.011700</td>
      <td>2.682568e-03</td>
    </tr>
    <tr>
      <th>Status_NDays_Binned_Platelets</th>
      <td>6.741196</td>
      <td>9.421142e-03</td>
    </tr>
    <tr>
      <th>Status_NDays_Binned_Alk_Phos</th>
      <td>5.503850</td>
      <td>1.897465e-02</td>
    </tr>
    <tr>
      <th>Status_NDays_Binned_Cholesterol</th>
      <td>3.773953</td>
      <td>5.205647e-02</td>
    </tr>
  </tbody>
</table>
</div>


### 1.6.1 Premodelling Data Description <a class="anchor" id="1.6.1"></a>

1. To improve interpretation, reduce dimensionality and avoid inducing design matrix singularity, 3 object predictors were dropped prior to modelling:
    * <span style="color: #FF0000">Stage_1.0</span>
    * <span style="color: #FF0000">Stage_2.0</span>
    * <span style="color: #FF0000">Stage_3.0</span>
2. To evaluate the feature selection capabilities of the candidate models, all remaining predictors were accounted during the model development process using the training subset:
    * **218 rows** (observations)
    * **19 columns** (variables)
        * **2/19 event | duration** (boolean | numeric)
             * <span style="color: #FF0000">Status</span>
             * <span style="color: #FF0000">N_Days</span>
        * **10/19 predictor** (numeric)
             * <span style="color: #FF0000">Age</span>
             * <span style="color: #FF0000">Bilirubin</span>
             * <span style="color: #FF0000">Cholesterol</span>
             * <span style="color: #FF0000">Albumin</span>
             * <span style="color: #FF0000">Copper</span>
             * <span style="color: #FF0000">Alk_Phos</span>
             * <span style="color: #FF0000">SGOT</span>
             * <span style="color: #FF0000">Triglycerides</span>
             * <span style="color: #FF0000">Platelets</span>
             * <span style="color: #FF0000">Prothrombin</span>
        * **7/19 predictor** (object)
             * <span style="color: #FF0000">Drug</span>
             * <span style="color: #FF0000">Sex</span>
             * <span style="color: #FF0000">Ascites</span>
             * <span style="color: #FF0000">Hepatomegaly</span>
             * <span style="color: #FF0000">Spiders</span>
             * <span style="color: #FF0000">Edema</span>
             * <span style="color: #FF0000">Stage_4.0</span>
3. Similarly, all remaining predictors were accounted during the model evaluation process using the testing subset:
    * **94 rows** (observations)
    * **19 columns** (variables)
        * **2/19 event | duration** (boolean | numeric)
             * <span style="color: #FF0000">Status</span>
             * <span style="color: #FF0000">N_Days</span>
        * **10/19 predictor** (numeric)
             * <span style="color: #FF0000">Age</span>
             * <span style="color: #FF0000">Bilirubin</span>
             * <span style="color: #FF0000">Cholesterol</span>
             * <span style="color: #FF0000">Albumin</span>
             * <span style="color: #FF0000">Copper</span>
             * <span style="color: #FF0000">Alk_Phos</span>
             * <span style="color: #FF0000">SGOT</span>
             * <span style="color: #FF0000">Triglycerides</span>
             * <span style="color: #FF0000">Platelets</span>
             * <span style="color: #FF0000">Prothrombin</span>
        * **7/19 predictor** (object)
             * <span style="color: #FF0000">Drug</span>
             * <span style="color: #FF0000">Sex</span>
             * <span style="color: #FF0000">Ascites</span>
             * <span style="color: #FF0000">Hepatomegaly</span>
             * <span style="color: #FF0000">Spiders</span>
             * <span style="color: #FF0000">Edema</span>
             * <span style="color: #FF0000">Stage_4.0</span>


```python
##################################
# Formulating a complete dataframe
# from the training subset for modelling
##################################
cirrhosis_survival_y_train_cleaned.reset_index(drop=True, inplace=True)
cirrhosis_survival_train_modeling = pd.concat([cirrhosis_survival_y_train_cleaned,
                                               cirrhosis_survival_X_train_preprocessed],
                                              axis=1)
cirrhosis_survival_train_modeling.drop(columns=['Stage_1.0', 'Stage_2.0', 'Stage_3.0'], axis=1, inplace=True)
cirrhosis_survival_train_modeling.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Status</th>
      <th>N_Days</th>
      <th>Age</th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage_4.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>2475</td>
      <td>-1.342097</td>
      <td>0.863802</td>
      <td>0.886087</td>
      <td>-0.451884</td>
      <td>-0.972098</td>
      <td>0.140990</td>
      <td>0.104609</td>
      <td>0.155130</td>
      <td>0.540960</td>
      <td>0.747580</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>877</td>
      <td>-1.470901</td>
      <td>0.516350</td>
      <td>1.554523</td>
      <td>0.827618</td>
      <td>0.467579</td>
      <td>-0.705337</td>
      <td>0.301441</td>
      <td>1.275222</td>
      <td>0.474140</td>
      <td>-0.315794</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>3050</td>
      <td>-0.239814</td>
      <td>-0.625875</td>
      <td>0.293280</td>
      <td>0.646582</td>
      <td>-0.241205</td>
      <td>-0.848544</td>
      <td>0.275723</td>
      <td>-1.684460</td>
      <td>0.756741</td>
      <td>0.087130</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>True</td>
      <td>110</td>
      <td>-0.052733</td>
      <td>0.559437</td>
      <td>-1.534283</td>
      <td>0.354473</td>
      <td>-0.284113</td>
      <td>-0.014525</td>
      <td>0.162878</td>
      <td>-0.189139</td>
      <td>-1.735375</td>
      <td>0.649171</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>True</td>
      <td>3839</td>
      <td>-0.795010</td>
      <td>1.142068</td>
      <td>-0.108933</td>
      <td>-0.272913</td>
      <td>0.618030</td>
      <td>2.071847</td>
      <td>1.434674</td>
      <td>-0.212684</td>
      <td>-0.675951</td>
      <td>-0.315794</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Formulating a complete dataframe
# from the testing subset for modelling
##################################
cirrhosis_survival_y_test_cleaned.reset_index(drop=True, inplace=True)
cirrhosis_survival_test_modeling = pd.concat([cirrhosis_survival_y_test_cleaned,
                                               cirrhosis_survival_X_test_preprocessed],
                                              axis=1)
cirrhosis_survival_test_modeling.drop(columns=['Stage_1.0', 'Stage_2.0', 'Stage_3.0'], axis=1, inplace=True)
cirrhosis_survival_test_modeling.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Status</th>
      <th>N_Days</th>
      <th>Age</th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Stage_4.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>3336</td>
      <td>1.043704</td>
      <td>0.744396</td>
      <td>0.922380</td>
      <td>0.240951</td>
      <td>0.045748</td>
      <td>0.317282</td>
      <td>-0.078335</td>
      <td>2.671950</td>
      <td>1.654815</td>
      <td>-0.948196</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>1321</td>
      <td>-1.936476</td>
      <td>-0.764558</td>
      <td>0.160096</td>
      <td>-0.600950</td>
      <td>-0.179138</td>
      <td>-0.245613</td>
      <td>0.472422</td>
      <td>-0.359800</td>
      <td>0.348533</td>
      <td>0.439089</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>1435</td>
      <td>-1.749033</td>
      <td>0.371523</td>
      <td>0.558115</td>
      <td>0.646582</td>
      <td>-0.159024</td>
      <td>0.339454</td>
      <td>0.685117</td>
      <td>-3.109146</td>
      <td>-0.790172</td>
      <td>-0.617113</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>4459</td>
      <td>-0.485150</td>
      <td>-0.918484</td>
      <td>-0.690904</td>
      <td>1.629765</td>
      <td>0.028262</td>
      <td>1.713791</td>
      <td>-1.387751</td>
      <td>0.155130</td>
      <td>0.679704</td>
      <td>0.087130</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>2721</td>
      <td>-0.815655</td>
      <td>1.286438</td>
      <td>2.610501</td>
      <td>-0.722153</td>
      <td>0.210203</td>
      <td>0.602860</td>
      <td>3.494936</td>
      <td>-0.053214</td>
      <td>-0.475634</td>
      <td>-1.714435</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### 1.6.2 Weibull Accelerated Failure Time Model <a class="anchor" id="1.6.2"></a>

[Survival Analysis](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) deals with the analysis of time-to-event data. It focuses on the expected duration of time until one or more events of interest occur, such as death, failure, or relapse. This type of analysis is used to study and model the time until the occurrence of an event, taking into account that the event might not have occurred for all subjects during the study period. Several key aspects of survival analysis include the survival function which refers to the probability that an individual survives longer than a certain time, hazard function which describes the instantaneous rate at which events occur, given no prior event, and censoring pertaining to a condition where the event of interest has not occurred for some subjects during the observation period.

[Right-Censored Survival Data](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) occurs when the event of interest has not happened for some subjects by the end of the study period or the last follow-up time. This type of censoring is common in survival analysis because not all individuals may experience the event before the study ends, or they might drop out or be lost to follow-up. Right-censored data is crucial in survival analysis as it allows the inclusion of all subjects in the analysis, providing more accurate and reliable estimates.

[Accelerated Failure Time Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) are a class of survival analysis models used to analyze time-to-event data by directly modelling the survival time itself. An AFT model assumes that the effect of covariates accelerates or decelerates the life time of an event by some constant factor. The mathematical equation is represented by the logarithm of the survival time being equal to the sum of the vector of covariates multiplied to the vector of regression coefficients; and the product of the scale parameter and a random variable with a specified distribution. In an AFT model, the coefficients represent the multiplicative effect on the survival time. An exponentiated regression coefficient greater than one prolongs survival time, while a value less than one shortens it. The scale parameter determines the spread or variability of survival times. AFT models assume that the effect of covariates on survival time is multiplicative and that the survival times can be transformed to follow a specific distribution.

[Weibull Accelerated Failure Time Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) assumes that the survival time errors follow a Weibull distribution. The mathematical equation is represented by the logarithm of the survival time being equal to the sum of the vector of covariates multiplied to the vector of regression coefficients; and the product of the scale parameter and a Weibull-distributed error term. This model is flexible as it can model both increasing and decreasing hazard rates over time and can be used to model various types of survival data. However, the results may be complex to interpret if the shape parameter does not align well with the data, and the model can also be sensitive to the distributional assumptions.

[Weibull Accelerated Failure Time Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) assumes that the survival time errors follow a Weibull distribution. The mathematical equation is represented by the logarithm of the survival time being equal to the sum of the vector of covariates multiplied to the vector of regression coefficients; and the product of the scale parameter and a Weibull-distributed error term. This model is flexible as it can model both increasing and decreasing hazard rates over time and can be used to model various types of survival data. However, the results may be complex to interpret if the shape parameter does not align well with the data, and the model can also be sensitive to the distributional assumptions.

[Concordance Index](https://lifelines.readthedocs.io/en/latest/lifelines.utils.html) measures the model's ability to correctly order pairs of observations based on their predicted survival times. Values range from 0.5 to 1.0 indicating no predictive power (random guessing) and perfect predictions, respectively. As a metric, it provides a measure of discriminative ability and useful for ranking predictions. However, it does not provide information on the magnitude of errors and may be insensitive to the calibration of predicted survival probabilities.

[Mean Absolute Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html) measures the average magnitude of the errors between predicted and actual survival times. Lower MAE values indicate better model performance, while reflecting the average prediction error in the same units as the survival time. As a metric, it is intuitive and easy to interpret while providing a direct measure of prediction accuracy. However, it may be sensitive to outliers and does not consider the probabilistic nature of survival predictions.

[Brier Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html) measures the accuracy of probabilistic predictions of survival at a specific time point. Values range from 0 to 1 with a lower brier scores indicating better accuracy. A Brier score of 0 indicates perfect predictions, while a score of 1 indicates the worst possible predictions. As a metric, it considers both discrimination and calibration, while reflecting the accuracy of predicted survival probabilities. However, it requires specifying a time point and aggregating scores over time points may be less interpretable.

1. The [weibull accelerated failure time model](https://lifelines.readthedocs.io/en/latest/fitters/regression/WeibullAFTFitter.html) from the <mark style="background-color: #CCECFF"><b>lifelines.WeibullAFTFitter</b></mark> Python library API was implemented. 
2. The model implementation used 1 hyperparameter:
    * <span style="color: #FF0000">penalizer</span> = penalty to the size of the coefficients during regression fixed at a value = 0.30
3. Only 3 of the 17 predictors, which were determined by the model as statistically significant, were used for prediction:
    * <span style="color: #FF0000">Bilirubin</span>: Increase in value associated with a decrease in time to event 
    * <span style="color: #FF0000">Prothrombin</span>: Increase in value associated with a decrease in time to event 
    * <span style="color: #FF0000">Age</span>: Increase in value associated with a decrease in time to event 
4. The cross-validated model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8250
    * **Mean Absolute Error** = 2303.6056
    * **Brier Score** = 0.5125
5. The apparent model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8291
    * **Mean Absolute Error** = 2280.7437
    * **Brier Score** = 0.5151
6. The independent test model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8526
    * **Mean Absolute Error** = 1948.8733
    * **Brier Score** = 0.5375
7. Comparable apparent and cross-validated model performance was observed, indicative of the presence of minimal model overfitting.
8. The MAE for event observations were typically lower because the errors were directly tied to the observed event times, which are known and can be more accurately predicted. For censored observations, the prediction error reflects the model's inability to pinpoint the exact event time, leading to higher MAE due to the larger variability and the longer tail of potential event times beyond the censoring point.
9. Survival probability curves estimated for all cases. Shorter median survival times were observed for:
    * Event cases as compared to censored cases
    * Higher values for <span style="color: #FF0000">Bilirubin</span> as compared to lower values
    * Higher values for <span style="color: #FF0000">Prothrombin</span> as compared to lower values
    * Higher values for <span style="color: #FF0000">Age</span> as compared to lower values
10. SHAP values were computed for the significant predictors, with contributions to the model output ranked as follows:
    * Higher values for <span style="color: #FF0000">Bilirubin</span> result to the event expected to occur sooner
    * Higher values for <span style="color: #FF0000">Prothrombin</span> result to the event expected to occur sooner
    * Higher values for <span style="color: #FF0000">Age</span> result to the event expected to occur sooner


```python
##################################
# Assessing the survival probability 
# and hazard function plots
# with a Weibull distribution
##################################
cirrhosis_survival_weibull = WeibullFitter()
cirrhosis_survival_weibull.fit(durations=cirrhosis_survival_train_modeling['N_Days'], 
                               event_observed=cirrhosis_survival_train_modeling['Status'])

##################################
# Fitting a Kaplan-Meier estimation
##################################
cirrhosis_survival_km = KaplanMeierFitter()
cirrhosis_survival_km.fit(durations=cirrhosis_survival_train_modeling['N_Days'],
                          event_observed=cirrhosis_survival_train_modeling['Status'])

##################################
# Generating the survival probability 
# and hazard function plots
##################################
plt.figure(figsize=(17, 8))

##################################
# Generating the Kaplan-Meier
# survival probability plot
##################################
plt.subplot(1, 3, 1)
cirrhosis_survival_km.plot_survival_function()
plt.title("Kaplan-Meier Survival Probability Curve")
plt.xlabel("N_Days")
plt.ylabel("Survival Probability")
plt.legend('',frameon=False)

##################################
# Generating the Weibull
# survival probability plot
##################################
plt.subplot(1, 3, 2)
cirrhosis_survival_weibull.plot_survival_function()
plt.title("Weibull Survival Probability Curve")
plt.xlabel("N_Days")
plt.ylabel("Survival Probability")
plt.legend('',frameon=False)

##################################
# Generating the Weibull
# hazard function plot
##################################
plt.subplot(1, 3, 3)
cirrhosis_survival_weibull.plot_hazard()
plt.title("Weibull Hazard Function")
plt.xlabel("N_Days")
plt.ylabel("Hazard")
plt.legend('',frameon=False)

##################################
# Consolidating all plots
##################################
plt.tight_layout()
plt.show()
```


    
![png](output_162_0.png)
    



```python
##################################
# Formulating the Accelerated Failure Time model
# based on a Weibull distribution
# and generating the summary
##################################
cirrhosis_survival_aft_weibull = WeibullAFTFitter(penalizer=0.30)
cirrhosis_survival_aft_weibull.fit(cirrhosis_survival_train_modeling, duration_col='N_Days', event_col='Status')
cirrhosis_survival_aft_weibull.print_summary()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.WeibullAFTFitter</td>
    </tr>
    <tr>
      <th>duration col</th>
      <td>'N_Days'</td>
    </tr>
    <tr>
      <th>event col</th>
      <td>'Status'</td>
    </tr>
    <tr>
      <th>penalizer</th>
      <td>0.3</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>218</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>87</td>
    </tr>
    <tr>
      <th>log-likelihood</th>
      <td>-777.86</td>
    </tr>
    <tr>
      <th>time fit was run</th>
      <td>2024-08-17 00:55:51 UTC</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">exp(coef)</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">exp(coef) lower 95%</th>
      <th style="min-width: 12px;">exp(coef) upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="18" valign="top">lambda_</th>
      <th>Age</th>
      <td>-0.17</td>
      <td>0.85</td>
      <td>0.07</td>
      <td>-0.30</td>
      <td>-0.03</td>
      <td>0.74</td>
      <td>0.98</td>
      <td>0.00</td>
      <td>-2.31</td>
      <td>0.02</td>
      <td>5.59</td>
    </tr>
    <tr>
      <th>Albumin</th>
      <td>0.10</td>
      <td>1.10</td>
      <td>0.08</td>
      <td>-0.05</td>
      <td>0.25</td>
      <td>0.95</td>
      <td>1.28</td>
      <td>0.00</td>
      <td>1.27</td>
      <td>0.20</td>
      <td>2.29</td>
    </tr>
    <tr>
      <th>Alk_Phos</th>
      <td>-0.04</td>
      <td>0.96</td>
      <td>0.07</td>
      <td>-0.18</td>
      <td>0.11</td>
      <td>0.83</td>
      <td>1.11</td>
      <td>0.00</td>
      <td>-0.52</td>
      <td>0.60</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>Ascites</th>
      <td>-0.20</td>
      <td>0.82</td>
      <td>0.23</td>
      <td>-0.66</td>
      <td>0.26</td>
      <td>0.51</td>
      <td>1.29</td>
      <td>0.00</td>
      <td>-0.87</td>
      <td>0.38</td>
      <td>1.38</td>
    </tr>
    <tr>
      <th>Bilirubin</th>
      <td>-0.29</td>
      <td>0.75</td>
      <td>0.09</td>
      <td>-0.46</td>
      <td>-0.12</td>
      <td>0.63</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>-3.39</td>
      <td>&lt;0.005</td>
      <td>10.47</td>
    </tr>
    <tr>
      <th>Cholesterol</th>
      <td>-0.05</td>
      <td>0.95</td>
      <td>0.08</td>
      <td>-0.20</td>
      <td>0.10</td>
      <td>0.82</td>
      <td>1.11</td>
      <td>0.00</td>
      <td>-0.61</td>
      <td>0.54</td>
      <td>0.89</td>
    </tr>
    <tr>
      <th>Copper</th>
      <td>-0.11</td>
      <td>0.89</td>
      <td>0.08</td>
      <td>-0.27</td>
      <td>0.04</td>
      <td>0.77</td>
      <td>1.04</td>
      <td>0.00</td>
      <td>-1.46</td>
      <td>0.15</td>
      <td>2.79</td>
    </tr>
    <tr>
      <th>Drug</th>
      <td>0.10</td>
      <td>1.10</td>
      <td>0.14</td>
      <td>-0.17</td>
      <td>0.37</td>
      <td>0.84</td>
      <td>1.44</td>
      <td>0.00</td>
      <td>0.71</td>
      <td>0.48</td>
      <td>1.07</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>-0.33</td>
      <td>0.72</td>
      <td>0.18</td>
      <td>-0.68</td>
      <td>0.01</td>
      <td>0.51</td>
      <td>1.01</td>
      <td>0.00</td>
      <td>-1.88</td>
      <td>0.06</td>
      <td>4.05</td>
    </tr>
    <tr>
      <th>Hepatomegaly</th>
      <td>-0.11</td>
      <td>0.90</td>
      <td>0.15</td>
      <td>-0.40</td>
      <td>0.19</td>
      <td>0.67</td>
      <td>1.20</td>
      <td>0.00</td>
      <td>-0.72</td>
      <td>0.47</td>
      <td>1.08</td>
    </tr>
    <tr>
      <th>Platelets</th>
      <td>0.05</td>
      <td>1.06</td>
      <td>0.07</td>
      <td>-0.08</td>
      <td>0.19</td>
      <td>0.92</td>
      <td>1.21</td>
      <td>0.00</td>
      <td>0.77</td>
      <td>0.44</td>
      <td>1.18</td>
    </tr>
    <tr>
      <th>Prothrombin</th>
      <td>-0.19</td>
      <td>0.83</td>
      <td>0.08</td>
      <td>-0.34</td>
      <td>-0.05</td>
      <td>0.71</td>
      <td>0.96</td>
      <td>0.00</td>
      <td>-2.56</td>
      <td>0.01</td>
      <td>6.57</td>
    </tr>
    <tr>
      <th>SGOT</th>
      <td>-0.11</td>
      <td>0.90</td>
      <td>0.08</td>
      <td>-0.26</td>
      <td>0.04</td>
      <td>0.77</td>
      <td>1.04</td>
      <td>0.00</td>
      <td>-1.44</td>
      <td>0.15</td>
      <td>2.74</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>0.02</td>
      <td>1.02</td>
      <td>0.19</td>
      <td>-0.35</td>
      <td>0.40</td>
      <td>0.70</td>
      <td>1.49</td>
      <td>0.00</td>
      <td>0.11</td>
      <td>0.91</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>Spiders</th>
      <td>-0.04</td>
      <td>0.96</td>
      <td>0.16</td>
      <td>-0.35</td>
      <td>0.27</td>
      <td>0.71</td>
      <td>1.30</td>
      <td>0.00</td>
      <td>-0.26</td>
      <td>0.79</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>Stage_4.0</th>
      <td>-0.15</td>
      <td>0.86</td>
      <td>0.16</td>
      <td>-0.46</td>
      <td>0.16</td>
      <td>0.63</td>
      <td>1.17</td>
      <td>0.00</td>
      <td>-0.96</td>
      <td>0.34</td>
      <td>1.56</td>
    </tr>
    <tr>
      <th>Tryglicerides</th>
      <td>-0.06</td>
      <td>0.94</td>
      <td>0.07</td>
      <td>-0.20</td>
      <td>0.08</td>
      <td>0.82</td>
      <td>1.08</td>
      <td>0.00</td>
      <td>-0.88</td>
      <td>0.38</td>
      <td>1.40</td>
    </tr>
    <tr>
      <th>Intercept</th>
      <td>8.52</td>
      <td>5006.58</td>
      <td>0.21</td>
      <td>8.10</td>
      <td>8.94</td>
      <td>3292.49</td>
      <td>7613.01</td>
      <td>0.00</td>
      <td>39.84</td>
      <td>&lt;0.005</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>rho_</th>
      <th>Intercept</th>
      <td>0.34</td>
      <td>1.41</td>
      <td>0.07</td>
      <td>0.20</td>
      <td>0.49</td>
      <td>1.22</td>
      <td>1.63</td>
      <td>0.00</td>
      <td>4.61</td>
      <td>&lt;0.005</td>
      <td>17.96</td>
    </tr>
  </tbody>
</table><br><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>Concordance</th>
      <td>0.85</td>
    </tr>
    <tr>
      <th>AIC</th>
      <td>1593.72</td>
    </tr>
    <tr>
      <th>log-likelihood ratio test</th>
      <td>101.74 on 17 df</td>
    </tr>
    <tr>
      <th>-log2(p) of ll-ratio test</th>
      <td>44.43</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting the log accelerated failure rate of the
# formulated Accelerated Failure Time model
# based on a Weibull distribution
##################################
cirrhosis_survival_aft_weibull_summary = cirrhosis_survival_aft_weibull.summary
cirrhosis_survival_aft_weibull_summary_params = pd.DataFrame(cirrhosis_survival_aft_weibull.params_)
significant = cirrhosis_survival_aft_weibull_summary['p'] < 0.05
cirrhosis_survival_aft_weibull_summary_log_accelerated_failure_rate = (list(cirrhosis_survival_aft_weibull_summary_params.iloc[:,0].values))
plt.figure(figsize=(17, 8))
colors = ['#993300' if sig else '#CC9966' for sig in significant]

plt.barh([(index[0] + index[1]) for index in cirrhosis_survival_aft_weibull_summary_params.index[0:17]], 
         cirrhosis_survival_aft_weibull_summary_log_accelerated_failure_rate[0:17], 
         xerr=cirrhosis_survival_aft_weibull_summary['se(coef)'][0:17], 
         color=colors)
plt.xlabel('Log(Accelerated Failure Rate)')
plt.ylabel('Variables')
plt.title('AFT_WEIBULL Log(Accelerated Failure Rate) Forest Plot')
plt.axvline(x=0, color='k', linestyle='--')
plt.gca().invert_yaxis()
plt.show()
```


    
![png](output_164_0.png)
    



```python
##################################
# Determining the number of
# significant predictors
##################################
cirrhosis_survival_aft_weibull_significant = sum(cirrhosis_survival_aft_weibull_summary['p'] < 0.05)
display(f"Number of Significant Predictors: {cirrhosis_survival_aft_weibull_significant-2}")
```


    'Number of Significant Predictors: 3'



```python
##################################
# Formulating the Accelerated Failure Time model
# based on a Weibull distribution,
# using the significant predictors only
# and generating the summary
##################################
feature_subset = ['Bilirubin','Prothrombin','Age','N_Days','Status']
cirrhosis_survival_aft_weibull = WeibullAFTFitter(penalizer=0.30)
cirrhosis_survival_aft_weibull.fit(cirrhosis_survival_train_modeling[feature_subset], duration_col='N_Days', event_col='Status')
cirrhosis_survival_aft_weibull.print_summary()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.WeibullAFTFitter</td>
    </tr>
    <tr>
      <th>duration col</th>
      <td>'N_Days'</td>
    </tr>
    <tr>
      <th>event col</th>
      <td>'Status'</td>
    </tr>
    <tr>
      <th>penalizer</th>
      <td>0.3</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>218</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>87</td>
    </tr>
    <tr>
      <th>log-likelihood</th>
      <td>-789.68</td>
    </tr>
    <tr>
      <th>time fit was run</th>
      <td>2024-08-17 00:55:52 UTC</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">exp(coef)</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">exp(coef) lower 95%</th>
      <th style="min-width: 12px;">exp(coef) upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">lambda_</th>
      <th>Age</th>
      <td>-0.19</td>
      <td>0.83</td>
      <td>0.06</td>
      <td>-0.31</td>
      <td>-0.06</td>
      <td>0.73</td>
      <td>0.94</td>
      <td>0.00</td>
      <td>-2.97</td>
      <td>&lt;0.005</td>
      <td>8.40</td>
    </tr>
    <tr>
      <th>Bilirubin</th>
      <td>-0.51</td>
      <td>0.60</td>
      <td>0.07</td>
      <td>-0.64</td>
      <td>-0.38</td>
      <td>0.53</td>
      <td>0.68</td>
      <td>0.00</td>
      <td>-7.56</td>
      <td>&lt;0.005</td>
      <td>44.52</td>
    </tr>
    <tr>
      <th>Prothrombin</th>
      <td>-0.27</td>
      <td>0.77</td>
      <td>0.07</td>
      <td>-0.40</td>
      <td>-0.13</td>
      <td>0.67</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>-3.85</td>
      <td>&lt;0.005</td>
      <td>13.03</td>
    </tr>
    <tr>
      <th>Intercept</th>
      <td>8.36</td>
      <td>4272.85</td>
      <td>0.09</td>
      <td>8.19</td>
      <td>8.53</td>
      <td>3594.18</td>
      <td>5079.66</td>
      <td>0.00</td>
      <td>94.73</td>
      <td>&lt;0.005</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>rho_</th>
      <th>Intercept</th>
      <td>0.35</td>
      <td>1.42</td>
      <td>0.07</td>
      <td>0.21</td>
      <td>0.50</td>
      <td>1.23</td>
      <td>1.64</td>
      <td>0.00</td>
      <td>4.74</td>
      <td>&lt;0.005</td>
      <td>18.81</td>
    </tr>
  </tbody>
</table><br><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>Concordance</th>
      <td>0.83</td>
    </tr>
    <tr>
      <th>AIC</th>
      <td>1589.35</td>
    </tr>
    <tr>
      <th>log-likelihood ratio test</th>
      <td>78.11 on 3 df</td>
    </tr>
    <tr>
      <th>-log2(p) of ll-ratio test</th>
      <td>53.51</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting the log accelerated failure rate of the
# formulated Accelerated Failure Time model
# using the significant predictors only
# based on a Weibull distribution
##################################
cirrhosis_survival_aft_weibull_summary = cirrhosis_survival_aft_weibull.summary
cirrhosis_survival_aft_weibull_summary_params = pd.DataFrame(cirrhosis_survival_aft_weibull.params_)
significant = cirrhosis_survival_aft_weibull_summary['p'] < 0.05
cirrhosis_survival_aft_weibull_summary_log_accelerated_failure_rate = (list(cirrhosis_survival_aft_weibull_summary_params.iloc[:,0].values))
plt.figure(figsize=(17, 8))
colors = ['#993300' if sig else '#CC9966' for sig in significant]

plt.barh([(index[0] + index[1]) for index in cirrhosis_survival_aft_weibull_summary_params.index[0:3]], 
         cirrhosis_survival_aft_weibull_summary_log_accelerated_failure_rate[0:3], 
         xerr=cirrhosis_survival_aft_weibull_summary['se(coef)'][0:3], 
         color=colors)
plt.xlabel('Log(Accelerated Failure Rate)')
plt.ylabel('Variables')
plt.title('AFT_WEIBULL Log(Accelerated Failure Rate) Forest Plot')
plt.axvline(x=0, color='k', linestyle='--')
plt.gca().invert_yaxis()
plt.show()
```


    
![png](output_167_0.png)
    



```python
##################################
# Gathering the apparent model performance
# as baseline for evaluating overfitting
##################################
cirrhosis_survival_aft_weibull.fit(cirrhosis_survival_train_modeling[feature_subset], duration_col='N_Days', event_col='Status')
train_predictions = cirrhosis_survival_aft_weibull.predict_median(cirrhosis_survival_train_modeling)
cirrhosis_survival_aft_weibull_train_ci = concordance_index(cirrhosis_survival_train_modeling['N_Days'], 
                                                            train_predictions, 
                                                            cirrhosis_survival_train_modeling['Status'])
time_point = cirrhosis_survival_train_modeling['N_Days'].median()
cirrhosis_survival_aft_weibull_train_mae = mean_absolute_error(cirrhosis_survival_train_modeling['N_Days'], train_predictions)
cirrhosis_survival_aft_weibull_train_brier = brier_score_loss(cirrhosis_survival_train_modeling['Status'], 
                                                              cirrhosis_survival_aft_weibull.predict_survival_function(cirrhosis_survival_train_modeling, 
                                                                                                                       times=[time_point]).T[time_point])
display(f"Apparent Concordance Index: {cirrhosis_survival_aft_weibull_train_ci}")
display(f"Apparent MAE: {cirrhosis_survival_aft_weibull_train_mae}")
display(f"Apparent Brier Score: {cirrhosis_survival_aft_weibull_train_brier}")
```


    'Apparent Concordance Index: 0.8290799739921977'



    'Apparent MAE: 2280.743783352583'



    'Apparent Brier Score: 0.5151484140783107'



```python
##################################
# Performing 5-Fold Cross-Validation
# on the training data
##################################
kf = KFold(n_splits=5, shuffle=True, random_state=88888888)
ci_scores = []
mae_scores = []
brier_scores = []

for train_index, val_index in kf.split(cirrhosis_survival_train_modeling):
    df_train_fold = cirrhosis_survival_train_modeling.iloc[train_index]
    df_val_fold = cirrhosis_survival_train_modeling.iloc[val_index]
    
    cirrhosis_survival_aft_weibull.fit(df_train_fold[feature_subset], duration_col='N_Days', event_col='Status')
    val_predictions = cirrhosis_survival_aft_weibull.predict_median(df_val_fold)
    time_point = df_val_fold['N_Days'].median()
    ci = concordance_index(df_val_fold['N_Days'], val_predictions, df_val_fold['Status'])
    mae = mean_absolute_error(df_val_fold['N_Days'], val_predictions)
    brier = brier_score_loss(df_val_fold['Status'],
                             cirrhosis_survival_aft_weibull.predict_survival_function(df_val_fold, 
                                                                                      times=[time_point]).T[time_point])
    ci_scores.append(ci)
    mae_scores.append(mae)
    brier_scores.append(brier)

cirrhosis_survival_aft_weibull_cv_ci_mean = np.mean(ci_scores)
cirrhosis_survival_aft_weibull_cv_ci_std = np.std(ci_scores)
cirrhosis_survival_aft_weibull_cv_mae_mean = np.mean(mae_scores)
cirrhosis_survival_aft_weibull_cv_brier_mean = np.mean(brier_scores)

display(f"Cross-Validated Concordance Index: {cirrhosis_survival_aft_weibull_cv_ci_mean}")
display(f"Cross-Validated MAE: {cirrhosis_survival_aft_weibull_cv_mae_mean}")
display(f"Cross-Validated Brier Score: {cirrhosis_survival_aft_weibull_cv_brier_mean}")
```


    'Cross-Validated Concordance Index: 0.82500812019991'



    'Cross-Validated MAE: 2303.6056275460082'



    'Cross-Validated Brier Score: 0.5125825238516043'



```python
##################################
# Evaluating the model performance
# on test data
##################################
test_predictions = cirrhosis_survival_aft_weibull.predict_median(cirrhosis_survival_test_modeling)
cirrhosis_survival_aft_weibull_test_ci = concordance_index(cirrhosis_survival_test_modeling['N_Days'], 
                                                           test_predictions, 
                                                           cirrhosis_survival_test_modeling['Status'])
time_point = cirrhosis_survival_test_modeling['N_Days'].median()
cirrhosis_survival_aft_weibull_test_mae = mean_absolute_error(cirrhosis_survival_test_modeling['N_Days'], test_predictions)
cirrhosis_survival_aft_weibull_test_brier = brier_score_loss(cirrhosis_survival_test_modeling['Status'], 
                                                              cirrhosis_survival_aft_weibull.predict_survival_function(cirrhosis_survival_test_modeling, 
                                                                                                                       times=[time_point]).T[time_point])
display(f"Apparent Concordance Index: {cirrhosis_survival_aft_weibull_test_ci}")
display(f"Apparent MAE: {cirrhosis_survival_aft_weibull_test_mae}")
display(f"Apparent Brier Score: {cirrhosis_survival_aft_weibull_test_brier}")
```


    'Apparent Concordance Index: 0.8526077097505669'



    'Apparent MAE: 1948.87338022389'



    'Apparent Brier Score: 0.5375559341601057'



```python
##################################
# Gathering the model performance metrics
# from training, cross-validation and test
##################################
aft_weibull_set = pd.DataFrame(["Train","Cross-Validation","Test"]*3)
aft_weibull_metric = pd.DataFrame((["Concordance.Index"]*3) + (["MAE"]*3) + (["Brier.Score"]*3))
aft_weibull_metric_values = pd.DataFrame([cirrhosis_survival_aft_weibull_train_ci,
                                           cirrhosis_survival_aft_weibull_cv_ci_mean,
                                           cirrhosis_survival_aft_weibull_test_ci,
                                           cirrhosis_survival_aft_weibull_train_mae,
                                           cirrhosis_survival_aft_weibull_cv_mae_mean,
                                           cirrhosis_survival_aft_weibull_test_mae,
                                           cirrhosis_survival_aft_weibull_train_brier,
                                           cirrhosis_survival_aft_weibull_cv_brier_mean,
                                           cirrhosis_survival_aft_weibull_test_brier])
aft_weibull_method = pd.DataFrame(["AFT_WEIBULL"]*9)
aft_weibull_summary = pd.concat([aft_weibull_set,
                                       aft_weibull_metric,
                                       aft_weibull_metric_values,
                                       aft_weibull_method], 
                                      axis=1)
aft_weibull_summary.columns = ['Set', 'Metric', 'Value', 'Method']
aft_weibull_summary.reset_index(inplace=True, drop=True)
display(aft_weibull_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Set</th>
      <th>Metric</th>
      <th>Value</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Train</td>
      <td>Concordance.Index</td>
      <td>0.829080</td>
      <td>AFT_WEIBULL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cross-Validation</td>
      <td>Concordance.Index</td>
      <td>0.825008</td>
      <td>AFT_WEIBULL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Test</td>
      <td>Concordance.Index</td>
      <td>0.852608</td>
      <td>AFT_WEIBULL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Train</td>
      <td>MAE</td>
      <td>2280.743783</td>
      <td>AFT_WEIBULL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cross-Validation</td>
      <td>MAE</td>
      <td>2303.605628</td>
      <td>AFT_WEIBULL</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Test</td>
      <td>MAE</td>
      <td>1948.873380</td>
      <td>AFT_WEIBULL</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Train</td>
      <td>Brier.Score</td>
      <td>0.515148</td>
      <td>AFT_WEIBULL</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Cross-Validation</td>
      <td>Brier.Score</td>
      <td>0.512583</td>
      <td>AFT_WEIBULL</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Test</td>
      <td>Brier.Score</td>
      <td>0.537556</td>
      <td>AFT_WEIBULL</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Evaluating the predicted
# and actual survival times
##################################
predicted_survival_times = cirrhosis_survival_aft_weibull.predict_median(cirrhosis_survival_test_modeling)
fig, ax = plt.subplots(figsize=(17, 8))
for status, color, label in zip([True, False], ['#FF7F0E','#1F77B4'], ['Death', 'Censored']):
    subset = cirrhosis_survival_test_modeling[cirrhosis_survival_test_modeling['Status'] == status]
    ax.scatter(subset['N_Days'], predicted_survival_times.iloc[subset.index], c=color, label=label, alpha=0.8)
ax.set_xlabel('Actual Survival Time')
ax.set_ylabel('Predicted Survival Time')
ax.set_title('AFT_WEIBULL: Predicted Versus Actual Survival Times')
ax.legend()
plt.plot([0, cirrhosis_survival_test_modeling['N_Days'].max()], 
         [0, cirrhosis_survival_test_modeling['N_Days'].max()], 
         color='black', linestyle='--')
plt.show()
```


    
![png](output_172_0.png)
    



```python
##################################
# Plotting the individual
# survival probability profiles
##################################
plt.figure(figsize=(17, 8))
for status, color, label in zip([True, False], ['#FF7F0E','#1F77B4'], ['Death', 'Censored']):
    subset = cirrhosis_survival_test_modeling[cirrhosis_survival_test_modeling['Status'] == status]
    for i, row in subset.iterrows():
        survival_function = cirrhosis_survival_aft_weibull.predict_survival_function(row)
        plt.plot(survival_function.index, survival_function.iloc[:, 0], c=color, alpha=0.8)
plt.title('AFT_WEIBULL: Survival Probability Profiles')
plt.xlabel('N_Days')
plt.ylabel('Survival Probability')
death_patch = plt.Line2D([0], [0], color='#FF7F0E', lw=2, label='Death')
censored_patch = plt.Line2D([0], [0], color='#1F77B4', lw=2, label='Censored')
plt.legend(handles=[death_patch, censored_patch])
plt.show()
```


    
![png](output_173_0.png)
    



```python
##################################
# Plotting the individual
# survival probability profiles
# for the discretized Bilirubin predictor
##################################
cirrhosis_survival_test_modeling['Bilirubin_Level'] = pd.qcut(cirrhosis_survival_test_modeling['Bilirubin'], 3, labels=['Low','Moderate','High'])
plt.figure(figsize=(17, 8))
for bilirubin_level, color, label in zip(['Low', 'Moderate', 'High'], ['#FA8000','#E50000', '#8C000F'], ['Low', 'Moderate', 'High']):
    subset = cirrhosis_survival_test_modeling[cirrhosis_survival_test_modeling['Bilirubin_Level'] == bilirubin_level]
    for i, row in subset.iterrows():
        survival_function = cirrhosis_survival_aft_weibull.predict_survival_function(row)
        plt.plot(survival_function.index, survival_function.iloc[:, 0], c=color, alpha=0.8)
plt.title('AFT_WEIBULL: Survival Probability Profiles by Bilirubin Level')
plt.xlabel('N_Days')
plt.ylabel('Survival Probability')
low_patch = plt.Line2D([0], [0], color='#FA8000', lw=2, label='Low')
moderate_patch = plt.Line2D([0], [0], color='#E50000', lw=2, label='Moderate')
high_patch = plt.Line2D([0], [0], color='#8C000F', lw=2, label='High')
plt.legend(handles=[low_patch, moderate_patch, high_patch])
plt.show()
```


    
![png](output_174_0.png)
    



```python
##################################
# Plotting the individual
# survival probability profiles
# for the discretized Prothrombin predictor
##################################
cirrhosis_survival_test_modeling['Prothrombin_Level'] = pd.qcut(cirrhosis_survival_test_modeling['Prothrombin'], 3, labels=['Low','Moderate','High'])
plt.figure(figsize=(17, 8))
for prothrombin_level, color, label in zip(['Low', 'Moderate', 'High'], ['#FA8000','#E50000', '#8C000F'], ['Low', 'Moderate', 'High']):
    subset = cirrhosis_survival_test_modeling[cirrhosis_survival_test_modeling['Prothrombin_Level'] == prothrombin_level]
    for i, row in subset.iterrows():
        survival_function = cirrhosis_survival_aft_weibull.predict_survival_function(row)
        plt.plot(survival_function.index, survival_function.iloc[:, 0], c=color, alpha=0.8)
plt.title('AFT_WEIBULL: Survival Probability Profiles by Prothrombin Level')
plt.xlabel('N_Days')
plt.ylabel('Survival Probability')
low_patch = plt.Line2D([0], [0], color='#FA8000', lw=2, label='Low')
moderate_patch = plt.Line2D([0], [0], color='#E50000', lw=2, label='Moderate')
high_patch = plt.Line2D([0], [0], color='#8C000F', lw=2, label='High')
plt.legend(handles=[low_patch, moderate_patch, high_patch])
plt.show()
```


    
![png](output_175_0.png)
    



```python
##################################
# Plotting the individual
# survival probability profiles
# for the discretized Age predictor
##################################
cirrhosis_survival_test_modeling['Age_Level'] = pd.qcut(cirrhosis_survival_test_modeling['Age'], 3, labels=['Low','Moderate','High'])
plt.figure(figsize=(17, 8))
for age_level, color, label in zip(['Low', 'Moderate', 'High'], ['#FA8000','#E50000', '#8C000F'], ['Low', 'Moderate', 'High']):
    subset = cirrhosis_survival_test_modeling[cirrhosis_survival_test_modeling['Age_Level'] == age_level]
    for i, row in subset.iterrows():
        survival_function = cirrhosis_survival_aft_weibull.predict_survival_function(row)
        plt.plot(survival_function.index, survival_function.iloc[:, 0], c=color, alpha=0.8)
plt.title('AFT_WEIBULL: Survival Probability Profiles by Age Level')
plt.xlabel('N_Days')
plt.ylabel('Survival Probability')
low_patch = plt.Line2D([0], [0], color='#FA8000', lw=2, label='Low')
moderate_patch = plt.Line2D([0], [0], color='#E50000', lw=2, label='Moderate')
high_patch = plt.Line2D([0], [0], color='#8C000F', lw=2, label='High')
plt.legend(handles=[low_patch, moderate_patch, high_patch])
plt.show()
```


    
![png](output_176_0.png)
    



```python
##################################
# Defining a prediction function
# for SHAP value estimation
##################################
def aft_predict(fitter, df):
    return fitter.predict_expectation(df)

##################################
# Creating the explainer object
##################################
explainer_weibull = shap.Explainer(lambda x: aft_predict(cirrhosis_survival_aft_weibull, 
                                                         pd.DataFrame(x, columns=cirrhosis_survival_train_modeling.columns[2:])), 
                                   cirrhosis_survival_train_modeling.iloc[:, 2:])
shap_values_weibull = explainer_weibull(cirrhosis_survival_train_modeling.iloc[:, 2:])
```

    PermutationExplainer explainer: 219it [00:27,  5.91it/s]                         
    


```python
##################################
# Plotting the SHAP summary plot
##################################
shap.summary_plot(shap_values_weibull, 
                  cirrhosis_survival_train_modeling.iloc[:, 2:])
```


    
![png](output_178_0.png)
    


### 1.6.3 Log-Normal Accelerated Failure Time Model <a class="anchor" id="1.6.3"></a>

[Survival Analysis](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) deals with the analysis of time-to-event data. It focuses on the expected duration of time until one or more events of interest occur, such as death, failure, or relapse. This type of analysis is used to study and model the time until the occurrence of an event, taking into account that the event might not have occurred for all subjects during the study period. Several key aspects of survival analysis include the survival function which refers to the probability that an individual survives longer than a certain time, hazard function which describes the instantaneous rate at which events occur, given no prior event, and censoring pertaining to a condition where the event of interest has not occurred for some subjects during the observation period.

[Right-Censored Survival Data](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) occurs when the event of interest has not happened for some subjects by the end of the study period or the last follow-up time. This type of censoring is common in survival analysis because not all individuals may experience the event before the study ends, or they might drop out or be lost to follow-up. Right-censored data is crucial in survival analysis as it allows the inclusion of all subjects in the analysis, providing more accurate and reliable estimates.

[Accelerated Failure Time Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) are a class of survival analysis models used to analyze time-to-event data by directly modelling the survival time itself. An AFT model assumes that the effect of covariates accelerates or decelerates the life time of an event by some constant factor. The mathematical equation is represented by the logarithm of the survival time being equal to the sum of the vector of covariates multiplied to the vector of regression coefficients; and the product of the scale parameter and a random variable with a specified distribution. In an AFT model, the coefficients represent the multiplicative effect on the survival time. An exponentiated regression coefficient greater than one prolongs survival time, while a value less than one shortens it. The scale parameter determines the spread or variability of survival times. AFT models assume that the effect of covariates on survival time is multiplicative and that the survival times can be transformed to follow a specific distribution.

[Log-Normal Accelerated Failure Time Model](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) assumes that the logarithm of survival time errors follows a normal distribution. The mathematical equation is represented by the logarithm of the survival time being equal to the sum of the vector of covariates multiplied to the vector of regression coefficients; and the product of the scale parameter and a standard normal error term. This model can be a good fit for data with a log-normal distribution, with well-established statistical properties and straightforward interpretation on the log scale. However, the model can have limited flexibility in hazard rate shapes compared to the Weibull model and can be less intuitive to interpret on the original time scale.

[Concordance Index](https://lifelines.readthedocs.io/en/latest/lifelines.utils.html) measures the model's ability to correctly order pairs of observations based on their predicted survival times. Values range from 0.5 to 1.0 indicating no predictive power (random guessing) and perfect predictions, respectively. As a metric, it provides a measure of discriminative ability and useful for ranking predictions. However, it does not provide information on the magnitude of errors and may be insensitive to the calibration of predicted survival probabilities.

[Mean Absolute Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html) measures the average magnitude of the errors between predicted and actual survival times. Lower MAE values indicate better model performance, while reflecting the average prediction error in the same units as the survival time. As a metric, it is intuitive and easy to interpret while providing a direct measure of prediction accuracy. However, it may be sensitive to outliers and does not consider the probabilistic nature of survival predictions.

[Brier Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html) measures the accuracy of probabilistic predictions of survival at a specific time point. Values range from 0 to 1 with a lower brier scores indicating better accuracy. A Brier score of 0 indicates perfect predictions, while a score of 1 indicates the worst possible predictions. As a metric, it considers both discrimination and calibration, while reflecting the accuracy of predicted survival probabilities. However, it requires specifying a time point and aggregating scores over time points may be less interpretable.

1. The [log-normal accelerated failure time model](https://lifelines.readthedocs.io/en/latest/fitters/regression/LogNormalAFTFitter.html) from the <mark style="background-color: #CCECFF"><b>lifelines.LogNormalAFTFitter</b></mark> Python library API was implemented. 
2. The model implementation used 1 hyperparameter:
    * <span style="color: #FF0000">penalizer</span> = penalty to the size of the coefficients during regression fixed at a value = 0.30
3. Only 5 of the 17 predictors, which were determined by the model as statistically significant, were used for prediction:
    * <span style="color: #FF0000">Edema</span>: Presence associated with a decrease in time to event 
    * <span style="color: #FF0000">Bilirubin</span>: Increase in value associated with a decrease in time to event 
    * <span style="color: #FF0000">Age</span>: Increase in value associated with a decrease in time to event 
    * <span style="color: #FF0000">Prothrombin</span>: Increase in value associated with a decrease in time to event 
    * <span style="color: #FF0000">Copper</span>: Increase in value associated with a decrease in time to event 
4. The cross-validated model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8255
    * **Mean Absolute Error** = 2502.6369
    * **Brier Score** = 0.5425
5. The apparent model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8413
    * **Mean Absolute Error** = 2518.3594
    * **Brier Score** = 0.5470
6. The independent test model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8753
    * **Mean Absolute Error** = 1904.9879
    * **Brier Score** = 0.5775
7. Comparable apparent and cross-validated model performance was observed, indicative of the presence of minimal model overfitting.
8. The MAE for event observations were typically lower because the errors were directly tied to the observed event times, which are known and can be more accurately predicted. For censored observations, the prediction error reflects the model's inability to pinpoint the exact event time, leading to higher MAE due to the larger variability and the longer tail of potential event times beyond the censoring point.
9. Survival probability curves estimated for all cases. Shorter median survival times were observed for:
    * Event cases as compared to censored cases
    * Presence of <span style="color: #FF0000">Edema</span> as compared to absence
    * Higher values for <span style="color: #FF0000">Bilirubin</span> as compared to lower values
    * Higher values for <span style="color: #FF0000">Age</span> as compared to lower values
    * Higher values for <span style="color: #FF0000">Prothrombin</span> as compared to lower values
    * Higher values for <span style="color: #FF0000">Copper</span> as compared to lower values
10. SHAP values were computed for the significant predictors, with contributions to the model output ranked as follows:
    * Higher values for <span style="color: #FF0000">Bilirubin</span> result to the event expected to occur sooner
    * Higher values for <span style="color: #FF0000">Prothrombin</span> result to the event expected to occur sooner
    * Higher values for <span style="color: #FF0000">Copper</span> result to the event expected to occur sooner
    * Higher values for <span style="color: #FF0000">Age</span> result to the event expected to occur sooner
    * Presence of <span style="color: #FF0000">Edema</span> results to the event expected to occur sooner


```python
##################################
# Assessing the survival probability 
# and hazard function plots
# with a Log-Normal distribution
##################################
cirrhosis_survival_lognormal = LogNormalFitter()
cirrhosis_survival_lognormal.fit(durations=cirrhosis_survival_train_modeling['N_Days'], 
                                 event_observed=cirrhosis_survival_train_modeling['Status'])

##################################
# Fitting a Kaplan-Meier estimation
##################################
cirrhosis_survival_km = KaplanMeierFitter()
cirrhosis_survival_km.fit(durations=cirrhosis_survival_train_modeling['N_Days'],
                          event_observed=cirrhosis_survival_train_modeling['Status'])

##################################
# Generating the survival probability 
# and hazard function plots
##################################
plt.figure(figsize=(17, 8))

##################################
# Generating the Kaplan-Meier
# survival probability plot
##################################
plt.subplot(1, 3, 1)
cirrhosis_survival_km.plot_survival_function()
plt.title("Kaplan-Meier Survival Probability Curve")
plt.xlabel("N_Days")
plt.ylabel("Survival Probability")
plt.legend('',frameon=False)

##################################
# Generating the Weibull
# survival probability plot
##################################
plt.subplot(1, 3, 2)
cirrhosis_survival_lognormal.plot_survival_function()
plt.title("Log-Normal Survival Probability Curve")
plt.xlabel("N_Days")
plt.ylabel("Survival Probability")
plt.legend('',frameon=False)

##################################
# Generating the Weibull
# hazard function plot
##################################
plt.subplot(1, 3, 3)
cirrhosis_survival_lognormal.plot_hazard()
plt.title("Log-Normal Hazard Function")
plt.xlabel("N_Days")
plt.ylabel("Hazard")
plt.legend('',frameon=False)

##################################
# Consolidating all plots
##################################
plt.tight_layout()
plt.show()
```


    
![png](output_180_0.png)
    



```python
##################################
# Formulating the Accelerated Failure Time model
# based on a Log-Normal distribution
# and generating the summary
##################################
cirrhosis_survival_aft_lognormal = LogNormalAFTFitter(penalizer=0.30)
cirrhosis_survival_aft_lognormal.fit(cirrhosis_survival_train_modeling, duration_col='N_Days', event_col='Status')
cirrhosis_survival_aft_lognormal.print_summary()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.LogNormalAFTFitter</td>
    </tr>
    <tr>
      <th>duration col</th>
      <td>'N_Days'</td>
    </tr>
    <tr>
      <th>event col</th>
      <td>'Status'</td>
    </tr>
    <tr>
      <th>penalizer</th>
      <td>0.3</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>218</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>87</td>
    </tr>
    <tr>
      <th>log-likelihood</th>
      <td>-769.81</td>
    </tr>
    <tr>
      <th>time fit was run</th>
      <td>2024-08-17 00:56:28 UTC</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">exp(coef)</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">exp(coef) lower 95%</th>
      <th style="min-width: 12px;">exp(coef) upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="18" valign="top">mu_</th>
      <th>Age</th>
      <td>-0.18</td>
      <td>0.83</td>
      <td>0.07</td>
      <td>-0.32</td>
      <td>-0.05</td>
      <td>0.73</td>
      <td>0.95</td>
      <td>0.00</td>
      <td>-2.62</td>
      <td>0.01</td>
      <td>6.85</td>
    </tr>
    <tr>
      <th>Albumin</th>
      <td>0.08</td>
      <td>1.09</td>
      <td>0.07</td>
      <td>-0.06</td>
      <td>0.22</td>
      <td>0.95</td>
      <td>1.25</td>
      <td>0.00</td>
      <td>1.16</td>
      <td>0.24</td>
      <td>2.03</td>
    </tr>
    <tr>
      <th>Alk_Phos</th>
      <td>-0.03</td>
      <td>0.97</td>
      <td>0.07</td>
      <td>-0.17</td>
      <td>0.11</td>
      <td>0.85</td>
      <td>1.12</td>
      <td>0.00</td>
      <td>-0.40</td>
      <td>0.69</td>
      <td>0.54</td>
    </tr>
    <tr>
      <th>Ascites</th>
      <td>-0.33</td>
      <td>0.72</td>
      <td>0.25</td>
      <td>-0.81</td>
      <td>0.16</td>
      <td>0.44</td>
      <td>1.17</td>
      <td>0.00</td>
      <td>-1.32</td>
      <td>0.19</td>
      <td>2.43</td>
    </tr>
    <tr>
      <th>Bilirubin</th>
      <td>-0.26</td>
      <td>0.77</td>
      <td>0.08</td>
      <td>-0.42</td>
      <td>-0.09</td>
      <td>0.66</td>
      <td>0.91</td>
      <td>0.00</td>
      <td>-3.06</td>
      <td>&lt;0.005</td>
      <td>8.83</td>
    </tr>
    <tr>
      <th>Cholesterol</th>
      <td>0.01</td>
      <td>1.01</td>
      <td>0.07</td>
      <td>-0.13</td>
      <td>0.16</td>
      <td>0.88</td>
      <td>1.17</td>
      <td>0.00</td>
      <td>0.18</td>
      <td>0.86</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>Copper</th>
      <td>-0.16</td>
      <td>0.85</td>
      <td>0.07</td>
      <td>-0.30</td>
      <td>-0.01</td>
      <td>0.74</td>
      <td>0.99</td>
      <td>0.00</td>
      <td>-2.12</td>
      <td>0.03</td>
      <td>4.89</td>
    </tr>
    <tr>
      <th>Drug</th>
      <td>0.09</td>
      <td>1.10</td>
      <td>0.13</td>
      <td>-0.16</td>
      <td>0.35</td>
      <td>0.85</td>
      <td>1.42</td>
      <td>0.00</td>
      <td>0.70</td>
      <td>0.48</td>
      <td>1.05</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>-0.37</td>
      <td>0.69</td>
      <td>0.18</td>
      <td>-0.73</td>
      <td>-0.01</td>
      <td>0.48</td>
      <td>0.99</td>
      <td>0.00</td>
      <td>-2.04</td>
      <td>0.04</td>
      <td>4.60</td>
    </tr>
    <tr>
      <th>Hepatomegaly</th>
      <td>-0.06</td>
      <td>0.94</td>
      <td>0.14</td>
      <td>-0.34</td>
      <td>0.22</td>
      <td>0.71</td>
      <td>1.24</td>
      <td>0.00</td>
      <td>-0.43</td>
      <td>0.67</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>Platelets</th>
      <td>0.08</td>
      <td>1.08</td>
      <td>0.07</td>
      <td>-0.06</td>
      <td>0.21</td>
      <td>0.94</td>
      <td>1.24</td>
      <td>0.00</td>
      <td>1.11</td>
      <td>0.27</td>
      <td>1.92</td>
    </tr>
    <tr>
      <th>Prothrombin</th>
      <td>-0.18</td>
      <td>0.83</td>
      <td>0.07</td>
      <td>-0.33</td>
      <td>-0.04</td>
      <td>0.72</td>
      <td>0.96</td>
      <td>0.00</td>
      <td>-2.53</td>
      <td>0.01</td>
      <td>6.47</td>
    </tr>
    <tr>
      <th>SGOT</th>
      <td>-0.09</td>
      <td>0.91</td>
      <td>0.07</td>
      <td>-0.23</td>
      <td>0.05</td>
      <td>0.79</td>
      <td>1.05</td>
      <td>0.00</td>
      <td>-1.24</td>
      <td>0.22</td>
      <td>2.21</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>0.04</td>
      <td>1.04</td>
      <td>0.19</td>
      <td>-0.32</td>
      <td>0.40</td>
      <td>0.72</td>
      <td>1.49</td>
      <td>0.00</td>
      <td>0.21</td>
      <td>0.83</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>Spiders</th>
      <td>-0.18</td>
      <td>0.84</td>
      <td>0.15</td>
      <td>-0.47</td>
      <td>0.12</td>
      <td>0.62</td>
      <td>1.12</td>
      <td>0.00</td>
      <td>-1.19</td>
      <td>0.24</td>
      <td>2.08</td>
    </tr>
    <tr>
      <th>Stage_4.0</th>
      <td>-0.26</td>
      <td>0.77</td>
      <td>0.15</td>
      <td>-0.57</td>
      <td>0.04</td>
      <td>0.57</td>
      <td>1.04</td>
      <td>0.00</td>
      <td>-1.70</td>
      <td>0.09</td>
      <td>3.50</td>
    </tr>
    <tr>
      <th>Tryglicerides</th>
      <td>-0.09</td>
      <td>0.92</td>
      <td>0.07</td>
      <td>-0.23</td>
      <td>0.05</td>
      <td>0.80</td>
      <td>1.05</td>
      <td>0.00</td>
      <td>-1.24</td>
      <td>0.22</td>
      <td>2.21</td>
    </tr>
    <tr>
      <th>Intercept</th>
      <td>8.26</td>
      <td>3879.36</td>
      <td>0.21</td>
      <td>7.86</td>
      <td>8.67</td>
      <td>2580.42</td>
      <td>5832.16</td>
      <td>0.00</td>
      <td>39.72</td>
      <td>&lt;0.005</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>sigma_</th>
      <th>Intercept</th>
      <td>-0.14</td>
      <td>0.87</td>
      <td>0.07</td>
      <td>-0.28</td>
      <td>-0.01</td>
      <td>0.76</td>
      <td>0.99</td>
      <td>0.00</td>
      <td>-2.15</td>
      <td>0.03</td>
      <td>5.00</td>
    </tr>
  </tbody>
</table><br><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>Concordance</th>
      <td>0.85</td>
    </tr>
    <tr>
      <th>AIC</th>
      <td>1577.63</td>
    </tr>
    <tr>
      <th>log-likelihood ratio test</th>
      <td>122.57 on 17 df</td>
    </tr>
    <tr>
      <th>-log2(p) of ll-ratio test</th>
      <td>57.48</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting the log accelerated failure rate of the
# formulated Accelerated Failure Time model
# based on a Log-Normal distribution
##################################
cirrhosis_survival_aft_lognormal_summary = cirrhosis_survival_aft_lognormal.summary
cirrhosis_survival_aft_lognormal_summary_params = pd.DataFrame(cirrhosis_survival_aft_lognormal.params_)
significant = cirrhosis_survival_aft_lognormal_summary['p'] < 0.05
cirrhosis_survival_aft_lognormal_summary_log_accelerated_failure_rate = (list(cirrhosis_survival_aft_lognormal_summary_params.iloc[:,0].values))
plt.figure(figsize=(17, 8))
colors = ['#993300' if sig else '#CC9966' for sig in significant]

plt.barh([(index[0] + index[1]) for index in cirrhosis_survival_aft_lognormal_summary_params.index[0:17]], 
         cirrhosis_survival_aft_lognormal_summary_log_accelerated_failure_rate[0:17], 
         xerr=cirrhosis_survival_aft_lognormal_summary['se(coef)'][0:17], 
         color=colors)
plt.xlabel('Log(Accelerated Failure Rate)')
plt.ylabel('Variables')
plt.title('AFT_LOGNORMAL Log(Accelerated Failure Rate) Forest Plot')
plt.axvline(x=0, color='k', linestyle='--')
plt.gca().invert_yaxis()
plt.show()
```


    
![png](output_182_0.png)
    



```python
##################################
# Determining the number of
# significant predictors
##################################
cirrhosis_survival_aft_lognormal_significant = sum(cirrhosis_survival_aft_lognormal_summary['p'] < 0.05)
display(f"Number of Significant Predictors: {cirrhosis_survival_aft_lognormal_significant-2}")
```


    'Number of Significant Predictors: 5'



```python
##################################
# Formulating the Accelerated Failure Time model
# based on a Log-Normal distribution
# using the significant predictors only
# and generating the summary
##################################
feature_subset = ['Bilirubin','Prothrombin','Age','Copper','Edema','N_Days','Status']
cirrhosis_survival_aft_lognormal = LogNormalAFTFitter(penalizer=0.30)
cirrhosis_survival_aft_lognormal.fit(cirrhosis_survival_train_modeling[feature_subset], duration_col='N_Days', event_col='Status')
cirrhosis_survival_aft_lognormal.print_summary()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.LogNormalAFTFitter</td>
    </tr>
    <tr>
      <th>duration col</th>
      <td>'N_Days'</td>
    </tr>
    <tr>
      <th>event col</th>
      <td>'Status'</td>
    </tr>
    <tr>
      <th>penalizer</th>
      <td>0.3</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>218</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>87</td>
    </tr>
    <tr>
      <th>log-likelihood</th>
      <td>-779.28</td>
    </tr>
    <tr>
      <th>time fit was run</th>
      <td>2024-08-17 00:56:29 UTC</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">exp(coef)</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">exp(coef) lower 95%</th>
      <th style="min-width: 12px;">exp(coef) upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="6" valign="top">mu_</th>
      <th>Age</th>
      <td>-0.19</td>
      <td>0.83</td>
      <td>0.07</td>
      <td>-0.32</td>
      <td>-0.06</td>
      <td>0.73</td>
      <td>0.94</td>
      <td>0.00</td>
      <td>-2.90</td>
      <td>&lt;0.005</td>
      <td>8.05</td>
    </tr>
    <tr>
      <th>Bilirubin</th>
      <td>-0.37</td>
      <td>0.69</td>
      <td>0.07</td>
      <td>-0.52</td>
      <td>-0.23</td>
      <td>0.60</td>
      <td>0.79</td>
      <td>0.00</td>
      <td>-5.10</td>
      <td>&lt;0.005</td>
      <td>21.51</td>
    </tr>
    <tr>
      <th>Copper</th>
      <td>-0.22</td>
      <td>0.80</td>
      <td>0.07</td>
      <td>-0.36</td>
      <td>-0.08</td>
      <td>0.70</td>
      <td>0.93</td>
      <td>0.00</td>
      <td>-3.02</td>
      <td>&lt;0.005</td>
      <td>8.61</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>-0.56</td>
      <td>0.57</td>
      <td>0.17</td>
      <td>-0.90</td>
      <td>-0.21</td>
      <td>0.41</td>
      <td>0.81</td>
      <td>0.00</td>
      <td>-3.19</td>
      <td>&lt;0.005</td>
      <td>9.47</td>
    </tr>
    <tr>
      <th>Prothrombin</th>
      <td>-0.26</td>
      <td>0.77</td>
      <td>0.07</td>
      <td>-0.40</td>
      <td>-0.12</td>
      <td>0.67</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>-3.73</td>
      <td>&lt;0.005</td>
      <td>12.35</td>
    </tr>
    <tr>
      <th>Intercept</th>
      <td>8.18</td>
      <td>3566.84</td>
      <td>0.09</td>
      <td>8.00</td>
      <td>8.36</td>
      <td>2978.10</td>
      <td>4271.97</td>
      <td>0.00</td>
      <td>88.87</td>
      <td>&lt;0.005</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>sigma_</th>
      <th>Intercept</th>
      <td>-0.11</td>
      <td>0.90</td>
      <td>0.07</td>
      <td>-0.24</td>
      <td>0.02</td>
      <td>0.79</td>
      <td>1.02</td>
      <td>0.00</td>
      <td>-1.66</td>
      <td>0.10</td>
      <td>3.38</td>
    </tr>
  </tbody>
</table><br><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>Concordance</th>
      <td>0.84</td>
    </tr>
    <tr>
      <th>AIC</th>
      <td>1572.55</td>
    </tr>
    <tr>
      <th>log-likelihood ratio test</th>
      <td>103.65 on 5 df</td>
    </tr>
    <tr>
      <th>-log2(p) of ll-ratio test</th>
      <td>66.59</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting the log accelerated failure rate of the
# formulated Accelerated Failure Time model
# using the significant predictors only
# based on a Log-Normal distribution
##################################
cirrhosis_survival_aft_lognormal_summary = cirrhosis_survival_aft_lognormal.summary
cirrhosis_survival_aft_lognormal_summary_params = pd.DataFrame(cirrhosis_survival_aft_lognormal.params_)
significant = cirrhosis_survival_aft_lognormal_summary['p'] < 0.05
cirrhosis_survival_aft_lognormal_summary_log_accelerated_failure_rate = (list(cirrhosis_survival_aft_lognormal_summary_params.iloc[:,0].values))
plt.figure(figsize=(17, 8))
colors = ['#993300' if sig else '#CC9966' for sig in significant]

plt.barh([(index[0] + index[1]) for index in cirrhosis_survival_aft_lognormal_summary_params.index[0:5]], 
         cirrhosis_survival_aft_lognormal_summary_log_accelerated_failure_rate[0:5], 
         xerr=cirrhosis_survival_aft_lognormal_summary['se(coef)'][0:5], 
         color=colors)
plt.xlabel('Log(Accelerated Failure Rate)')
plt.ylabel('Variables')
plt.title('AFT_LOGNORMAL Log(Accelerated Failure Rate) Forest Plot')
plt.axvline(x=0, color='k', linestyle='--')
plt.gca().invert_yaxis()
plt.show()
```


    
![png](output_185_0.png)
    



```python
##################################
# Gathering the apparent model performance
# as baseline for evaluating overfitting
##################################
cirrhosis_survival_aft_lognormal.fit(cirrhosis_survival_train_modeling[feature_subset], duration_col='N_Days', event_col='Status')
train_predictions = cirrhosis_survival_aft_lognormal.predict_median(cirrhosis_survival_train_modeling)
cirrhosis_survival_aft_lognormal_train_ci = concordance_index(cirrhosis_survival_train_modeling['N_Days'], 
                                                            train_predictions, 
                                                            cirrhosis_survival_train_modeling['Status'])
time_point = cirrhosis_survival_train_modeling['N_Days'].median()
cirrhosis_survival_aft_lognormal_train_mae = mean_absolute_error(cirrhosis_survival_train_modeling['N_Days'], train_predictions)
cirrhosis_survival_aft_lognormal_train_brier = brier_score_loss(cirrhosis_survival_train_modeling['Status'], 
                                                              cirrhosis_survival_aft_lognormal.predict_survival_function(cirrhosis_survival_train_modeling, 
                                                                                                                       times=[time_point]).T[time_point])
display(f"Apparent Concordance Index: {cirrhosis_survival_aft_lognormal_train_ci}")
display(f"Apparent MAE: {cirrhosis_survival_aft_lognormal_train_mae}")
display(f"Apparent Brier Score: {cirrhosis_survival_aft_lognormal_train_brier}")
```


    'Apparent Concordance Index: 0.8413524057217165'



    'Apparent MAE: 2518.3593852441504'



    'Apparent Brier Score: 0.5470406779352225'



```python
##################################
# Performing 5-Fold Cross-Validation
# on the training data
##################################
kf = KFold(n_splits=5, shuffle=True, random_state=88888888)
ci_scores = []
mae_scores = []
brier_scores = []

for train_index, val_index in kf.split(cirrhosis_survival_train_modeling):
    df_train_fold = cirrhosis_survival_train_modeling.iloc[train_index]
    df_val_fold = cirrhosis_survival_train_modeling.iloc[val_index]
    
    cirrhosis_survival_aft_lognormal.fit(df_train_fold[feature_subset], duration_col='N_Days', event_col='Status')
    val_predictions = cirrhosis_survival_aft_lognormal.predict_median(df_val_fold)
    time_point = df_val_fold['N_Days'].median()
    ci = concordance_index(df_val_fold['N_Days'], val_predictions, df_val_fold['Status'])
    mae = mean_absolute_error(df_val_fold['N_Days'], val_predictions)
    brier = brier_score_loss(df_val_fold['Status'],
                             cirrhosis_survival_aft_lognormal.predict_survival_function(df_val_fold, 
                                                                                      times=[time_point]).T[time_point])
    ci_scores.append(ci)
    mae_scores.append(mae)
    brier_scores.append(brier)

cirrhosis_survival_aft_lognormal_cv_ci_mean = np.mean(ci_scores)
cirrhosis_survival_aft_lognormal_cv_ci_std = np.std(ci_scores)
cirrhosis_survival_aft_lognormal_cv_mae_mean = np.mean(mae_scores)
cirrhosis_survival_aft_lognormal_cv_brier_mean = np.mean(brier_scores)

display(f"Cross-Validated Concordance Index: {cirrhosis_survival_aft_lognormal_cv_ci_mean}")
display(f"Cross-Validated MAE: {cirrhosis_survival_aft_lognormal_cv_mae_mean}")
display(f"Cross-Validated Brier Score: {cirrhosis_survival_aft_lognormal_cv_brier_mean}")
```


    'Cross-Validated Concordance Index: 0.8255764006037584'



    'Cross-Validated MAE: 2502.6369548831367'



    'Cross-Validated Brier Score: 0.5425832599120203'



```python
##################################
# Evaluating the model performance
# on test data
##################################
test_predictions = cirrhosis_survival_aft_lognormal.predict_median(cirrhosis_survival_test_modeling)
cirrhosis_survival_aft_lognormal_test_ci = concordance_index(cirrhosis_survival_test_modeling['N_Days'], 
                                                           test_predictions, 
                                                           cirrhosis_survival_test_modeling['Status'])
time_point = cirrhosis_survival_test_modeling['N_Days'].median()
cirrhosis_survival_aft_lognormal_test_mae = mean_absolute_error(cirrhosis_survival_test_modeling['N_Days'], test_predictions)
cirrhosis_survival_aft_lognormal_test_brier = brier_score_loss(cirrhosis_survival_test_modeling['Status'], 
                                                              cirrhosis_survival_aft_lognormal.predict_survival_function(cirrhosis_survival_test_modeling, 
                                                                                                                       times=[time_point]).T[time_point])
display(f"Apparent Concordance Index: {cirrhosis_survival_aft_lognormal_test_ci}")
display(f"Apparent MAE: {cirrhosis_survival_aft_lognormal_test_mae}")
display(f"Apparent Brier Score: {cirrhosis_survival_aft_lognormal_test_brier}")
```


    'Apparent Concordance Index: 0.8752834467120182'



    'Apparent MAE: 1904.9879866903511'



    'Apparent Brier Score: 0.5775019785104171'



```python
##################################
# Gathering the model performance metrics
# from training, cross-validation and test
##################################
aft_lognormal_set = pd.DataFrame(["Train","Cross-Validation","Test"]*3)
aft_lognormal_metric = pd.DataFrame((["Concordance.Index"]*3) + (["MAE"]*3) + (["Brier.Score"]*3))
aft_lognormal_metric_values = pd.DataFrame([cirrhosis_survival_aft_lognormal_train_ci,
                                           cirrhosis_survival_aft_lognormal_cv_ci_mean,
                                           cirrhosis_survival_aft_lognormal_test_ci,
                                           cirrhosis_survival_aft_lognormal_train_mae,
                                           cirrhosis_survival_aft_lognormal_cv_mae_mean,
                                           cirrhosis_survival_aft_lognormal_test_mae,
                                           cirrhosis_survival_aft_lognormal_train_brier,
                                           cirrhosis_survival_aft_lognormal_cv_brier_mean,
                                           cirrhosis_survival_aft_lognormal_test_brier])
aft_lognormal_method = pd.DataFrame(["AFT_LOGNORMAL"]*9)
aft_lognormal_summary = pd.concat([aft_lognormal_set,
                                       aft_lognormal_metric,
                                       aft_lognormal_metric_values,
                                       aft_lognormal_method], 
                                      axis=1)
aft_lognormal_summary.columns = ['Set', 'Metric', 'Value', 'Method']
aft_lognormal_summary.reset_index(inplace=True, drop=True)
display(aft_lognormal_summary)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Set</th>
      <th>Metric</th>
      <th>Value</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Train</td>
      <td>Concordance.Index</td>
      <td>0.841352</td>
      <td>AFT_LOGNORMAL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cross-Validation</td>
      <td>Concordance.Index</td>
      <td>0.825576</td>
      <td>AFT_LOGNORMAL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Test</td>
      <td>Concordance.Index</td>
      <td>0.875283</td>
      <td>AFT_LOGNORMAL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Train</td>
      <td>MAE</td>
      <td>2518.359385</td>
      <td>AFT_LOGNORMAL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cross-Validation</td>
      <td>MAE</td>
      <td>2502.636955</td>
      <td>AFT_LOGNORMAL</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Test</td>
      <td>MAE</td>
      <td>1904.987987</td>
      <td>AFT_LOGNORMAL</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Train</td>
      <td>Brier.Score</td>
      <td>0.547041</td>
      <td>AFT_LOGNORMAL</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Cross-Validation</td>
      <td>Brier.Score</td>
      <td>0.542583</td>
      <td>AFT_LOGNORMAL</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Test</td>
      <td>Brier.Score</td>
      <td>0.577502</td>
      <td>AFT_LOGNORMAL</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Evaluating the predicted
# and actual survival times
##################################
predicted_survival_times = cirrhosis_survival_aft_lognormal.predict_median(cirrhosis_survival_test_modeling)
fig, ax = plt.subplots(figsize=(17, 8))
for status, color, label in zip([True, False], ['#FF7F0E','#1F77B4'], ['Death', 'Censored']):
    subset = cirrhosis_survival_test_modeling[cirrhosis_survival_test_modeling['Status'] == status]
    ax.scatter(subset['N_Days'], predicted_survival_times.iloc[subset.index], c=color, label=label, alpha=0.8)
ax.set_xlabel('Actual Survival Time')
ax.set_ylabel('Predicted Survival Time')
ax.set_title('AFT_LOGNORMAL: Predicted Versus Actual Survival Times')
ax.legend()
plt.plot([0, cirrhosis_survival_test_modeling['N_Days'].max()], 
         [0, cirrhosis_survival_test_modeling['N_Days'].max()], 
         color='black', linestyle='--')
plt.show()
```


    
![png](output_190_0.png)
    



```python
##################################
# Plotting the individual
# survival probability profiles
##################################
plt.figure(figsize=(17, 8))
for status, color, label in zip([True, False], ['#FF7F0E','#1F77B4'], ['Death', 'Censored']):
    subset = cirrhosis_survival_test_modeling[cirrhosis_survival_test_modeling['Status'] == status]
    for i, row in subset.iterrows():
        survival_function = cirrhosis_survival_aft_lognormal.predict_survival_function(row)
        plt.plot(survival_function.index, survival_function.iloc[:, 0], c=color, alpha=0.8)
plt.title('AFT_LOGNORMAL: Survival Probability Profiles')
plt.xlabel('N_Days')
plt.ylabel('Survival Probability')
death_patch = plt.Line2D([0], [0], color='#FF7F0E', lw=2, label='Death')
censored_patch = plt.Line2D([0], [0], color='#1F77B4', lw=2, label='Censored')
plt.legend(handles=[death_patch, censored_patch])
plt.show()
```


    
![png](output_191_0.png)
    



```python
##################################
# Plotting the individual
# survival probability profiles
# for the discretized Edema predictor
##################################
plt.figure(figsize=(17, 8))
for edema_level, color, label in zip([0, 1], ['#FA8000', '#8C000F'], ['Not Present', 'Present']):
    subset = cirrhosis_survival_test_modeling[cirrhosis_survival_test_modeling['Edema'] == edema_level]
    for i, row in subset.iterrows():
        survival_function = cirrhosis_survival_aft_lognormal.predict_survival_function(row)
        plt.plot(survival_function.index, survival_function.iloc[:, 0], c=color, alpha=0.8)
plt.title('AFT_LOGNORMAL: Survival Probability Profiles by Edema Level')
plt.xlabel('N_Days')
plt.ylabel('Survival Probability')
not_present_patch = plt.Line2D([0], [0], color='#FA8000', lw=2, label='Not Present')
present_patch = plt.Line2D([0], [0], color='#8C000F', lw=2, label='Present')
plt.legend(handles=[not_present_patch, present_patch])
plt.show()
```


    
![png](output_192_0.png)
    



```python
##################################
# Plotting the individual
# survival probability profiles
# for the discretized Bilirubin predictor
##################################
cirrhosis_survival_test_modeling['Bilirubin_Level'] = pd.qcut(cirrhosis_survival_test_modeling['Bilirubin'], 3, labels=['Low','Moderate','High'])
plt.figure(figsize=(17, 8))
for bilirubin_level, color, label in zip(['Low', 'Moderate', 'High'], ['#FA8000','#E50000', '#8C000F'], ['Low', 'Moderate', 'High']):
    subset = cirrhosis_survival_test_modeling[cirrhosis_survival_test_modeling['Bilirubin_Level'] == bilirubin_level]
    for i, row in subset.iterrows():
        survival_function = cirrhosis_survival_aft_lognormal.predict_survival_function(row)
        plt.plot(survival_function.index, survival_function.iloc[:, 0], c=color, alpha=0.8)
plt.title('AFT_LOGNORMAL: Survival Probability Profiles by Bilirubin Level')
plt.xlabel('N_Days')
plt.ylabel('Survival Probability')
low_patch = plt.Line2D([0], [0], color='#FA8000', lw=2, label='Low')
moderate_patch = plt.Line2D([0], [0], color='#E50000', lw=2, label='Moderate')
high_patch = plt.Line2D([0], [0], color='#8C000F', lw=2, label='High')
plt.legend(handles=[low_patch, moderate_patch, high_patch])
plt.show()
```


    
![png](output_193_0.png)
    



```python
##################################
# Plotting the individual
# survival probability profiles
# for the discretized Prothrombin predictor
##################################
cirrhosis_survival_test_modeling['Prothrombin_Level'] = pd.qcut(cirrhosis_survival_test_modeling['Prothrombin'], 3, labels=['Low','Moderate','High'])
plt.figure(figsize=(17, 8))
for prothrombin_level, color, label in zip(['Low', 'Moderate', 'High'], ['#FA8000','#E50000', '#8C000F'], ['Low', 'Moderate', 'High']):
    subset = cirrhosis_survival_test_modeling[cirrhosis_survival_test_modeling['Prothrombin_Level'] == prothrombin_level]
    for i, row in subset.iterrows():
        survival_function = cirrhosis_survival_aft_lognormal.predict_survival_function(row)
        plt.plot(survival_function.index, survival_function.iloc[:, 0], c=color, alpha=0.8)
plt.title('AFT_LOGNORMAL: Survival Probability Profiles by Prothrombin Level')
plt.xlabel('N_Days')
plt.ylabel('Survival Probability')
low_patch = plt.Line2D([0], [0], color='#FA8000', lw=2, label='Low')
moderate_patch = plt.Line2D([0], [0], color='#E50000', lw=2, label='Moderate')
high_patch = plt.Line2D([0], [0], color='#8C000F', lw=2, label='High')
plt.legend(handles=[low_patch, moderate_patch, high_patch])
plt.show()
```


    
![png](output_194_0.png)
    



```python
##################################
# Plotting the individual
# survival probability profiles
# for the discretized Copper predictor
##################################
cirrhosis_survival_test_modeling['Copper_Level'] = pd.qcut(cirrhosis_survival_test_modeling['Copper'], 3, labels=['Low','Moderate','High'])
plt.figure(figsize=(17, 8))
for copper_level, color, label in zip(['Low', 'Moderate', 'High'], ['#FA8000','#E50000', '#8C000F'], ['Low', 'Moderate', 'High']):
    subset = cirrhosis_survival_test_modeling[cirrhosis_survival_test_modeling['Copper_Level'] == copper_level]
    for i, row in subset.iterrows():
        survival_function = cirrhosis_survival_aft_lognormal.predict_survival_function(row)
        plt.plot(survival_function.index, survival_function.iloc[:, 0], c=color, alpha=0.8)
plt.title('AFT_LOGNORMAL: Survival Probability Profiles by Copper Level')
plt.xlabel('N_Days')
plt.ylabel('Survival Probability')
low_patch = plt.Line2D([0], [0], color='#FA8000', lw=2, label='Low')
moderate_patch = plt.Line2D([0], [0], color='#E50000', lw=2, label='Moderate')
high_patch = plt.Line2D([0], [0], color='#8C000F', lw=2, label='High')
plt.legend(handles=[low_patch, moderate_patch, high_patch])
plt.show()
```


    
![png](output_195_0.png)
    



```python
##################################
# Plotting the individual
# survival probability profiles
# for the discretized Age predictor
##################################
cirrhosis_survival_test_modeling['Age_Level'] = pd.qcut(cirrhosis_survival_test_modeling['Age'], 3, labels=['Low','Moderate','High'])
plt.figure(figsize=(17, 8))
for age_level, color, label in zip(['Low', 'Moderate', 'High'], ['#FA8000','#E50000', '#8C000F'], ['Low', 'Moderate', 'High']):
    subset = cirrhosis_survival_test_modeling[cirrhosis_survival_test_modeling['Age_Level'] == age_level]
    for i, row in subset.iterrows():
        survival_function = cirrhosis_survival_aft_lognormal.predict_survival_function(row)
        plt.plot(survival_function.index, survival_function.iloc[:, 0], c=color, alpha=0.8)
plt.title('AFT_LOGNORMAL: Survival Probability Profiles by Age Level')
plt.xlabel('N_Days')
plt.ylabel('Survival Probability')
low_patch = plt.Line2D([0], [0], color='#FA8000', lw=2, label='Low')
moderate_patch = plt.Line2D([0], [0], color='#E50000', lw=2, label='Moderate')
high_patch = plt.Line2D([0], [0], color='#8C000F', lw=2, label='High')
plt.legend(handles=[low_patch, moderate_patch, high_patch])
plt.show()
```


    
![png](output_196_0.png)
    



```python
##################################
# Defining a prediction function
# for SHAP value estimation
##################################
def aft_predict(fitter, df):
    return fitter.predict_expectation(df)

##################################
# Creating the explainer object
##################################
explainer_lognormal = shap.Explainer(lambda x: aft_predict(cirrhosis_survival_aft_lognormal, 
                                                         pd.DataFrame(x, columns=cirrhosis_survival_train_modeling.columns[2:])), 
                                   cirrhosis_survival_train_modeling.iloc[:, 2:])
shap_values_lognormal = explainer_lognormal(cirrhosis_survival_train_modeling.iloc[:, 2:])

```

    PermutationExplainer explainer: 219it [00:22,  5.42it/s]                         
    


```python
##################################
# Plotting the SHAP summary plot
##################################
shap.summary_plot(shap_values_lognormal, 
                  cirrhosis_survival_train_modeling.iloc[:, 2:])
```


    
![png](output_198_0.png)
    


### 1.6.4 Log-Logistic Accelerated Failure Time Model <a class="anchor" id="1.6.4"></a>

[Survival Analysis](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) deals with the analysis of time-to-event data. It focuses on the expected duration of time until one or more events of interest occur, such as death, failure, or relapse. This type of analysis is used to study and model the time until the occurrence of an event, taking into account that the event might not have occurred for all subjects during the study period. Several key aspects of survival analysis include the survival function which refers to the probability that an individual survives longer than a certain time, hazard function which describes the instantaneous rate at which events occur, given no prior event, and censoring pertaining to a condition where the event of interest has not occurred for some subjects during the observation period.

[Right-Censored Survival Data](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) occurs when the event of interest has not happened for some subjects by the end of the study period or the last follow-up time. This type of censoring is common in survival analysis because not all individuals may experience the event before the study ends, or they might drop out or be lost to follow-up. Right-censored data is crucial in survival analysis as it allows the inclusion of all subjects in the analysis, providing more accurate and reliable estimates.

[Accelerated Failure Time Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) are a class of survival analysis models used to analyze time-to-event data by directly modelling the survival time itself. An AFT model assumes that the effect of covariates accelerates or decelerates the life time of an event by some constant factor. The mathematical equation is represented by the logarithm of the survival time being equal to the sum of the vector of covariates multiplied to the vector of regression coefficients; and the product of the scale parameter and a random variable with a specified distribution. In an AFT model, the coefficients represent the multiplicative effect on the survival time. An exponentiated regression coefficient greater than one prolongs survival time, while a value less than one shortens it. The scale parameter determines the spread or variability of survival times. AFT models assume that the effect of covariates on survival time is multiplicative and that the survival times can be transformed to follow a specific distribution.

[Log-Logistic Accelerated Failure Time Model](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) assumes that the survival time errors follow a log-logistic distribution. The mathematical equation is represented by the logarithm of the survival time being equal to the sum of the vector of covariates multiplied to the vector of regression coefficients; and the product of the scale parameter and a standard logistic error term. This model can estimate various hazard shapes, including non-monotonic hazard functions and may be more flexible than the Log Normal model. However, interpretation of results can be complex due to the nature of the logistic distribution and the model structure may be less robust to outliers compared to other models.

[Concordance Index](https://lifelines.readthedocs.io/en/latest/lifelines.utils.html) measures the model's ability to correctly order pairs of observations based on their predicted survival times. Values range from 0.5 to 1.0 indicating no predictive power (random guessing) and perfect predictions, respectively. As a metric, it provides a measure of discriminative ability and useful for ranking predictions. However, it does not provide information on the magnitude of errors and may be insensitive to the calibration of predicted survival probabilities.

[Mean Absolute Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html) measures the average magnitude of the errors between predicted and actual survival times. Lower MAE values indicate better model performance, while reflecting the average prediction error in the same units as the survival time. As a metric, it is intuitive and easy to interpret while providing a direct measure of prediction accuracy. However, it may be sensitive to outliers and does not consider the probabilistic nature of survival predictions.

[Brier Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html) measures the accuracy of probabilistic predictions of survival at a specific time point. Values range from 0 to 1 with a lower brier scores indicating better accuracy. A Brier score of 0 indicates perfect predictions, while a score of 1 indicates the worst possible predictions. As a metric, it considers both discrimination and calibration, while reflecting the accuracy of predicted survival probabilities. However, it requires specifying a time point and aggregating scores over time points may be less interpretable.

1. The [log-logistic accelerated failure time model](https://lifelines.readthedocs.io/en/latest/fitters/regression/LogLogisticAFTFitter.html) from the <mark style="background-color: #CCECFF"><b>lifelines.LogLogisticAFTFitter</b></mark> Python library API was implemented. 
2. The model implementation used 1 hyperparameter:
    * <span style="color: #FF0000">penalizer</span> = penalty to the size of the coefficients during regression fixed at a value = 0.30
3. Only 5 of the 17 predictors, which were determined by the model as statistically significant, were used for prediction:
    * <span style="color: #FF0000">Bilirubin</span>: Increase in value associated with a decrease in time to event 
    * <span style="color: #FF0000">Prothrombin</span>: Increase in value associated with a decrease in time to event 
    * <span style="color: #FF0000">Age</span>: Increase in value associated with a decrease in time to event 
    * <span style="color: #FF0000">Copper</span>: Increase in value associated with a decrease in time to event 
4. The cross-validated model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8301
    * **Mean Absolute Error** = 2711.6604
    * **Brier Score** = 0.5065
5. The apparent model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8383
    * **Mean Absolute Error** = 2727.4651
    * **Brier Score** = 0.5095
6. The independent test model performance of the model is summarized as follows:
    * **Concordance Index** = 0.8625
    * **Mean Absolute Error** = 2189.9323
    * **Brier Score** = 0.5332
7. Comparable apparent and cross-validated model performance was observed, indicative of the presence of minimal model overfitting.
8. The MAE for event observations were typically lower because the errors were directly tied to the observed event times, which are known and can be more accurately predicted. For censored observations, the prediction error reflects the model's inability to pinpoint the exact event time, leading to higher MAE due to the larger variability and the longer tail of potential event times beyond the censoring point.
9. Survival probability curves estimated for all cases. Shorter median survival times were observed for:
    * Event cases as compared to censored cases
    * Higher values for <span style="color: #FF0000">Bilirubin</span> as compared to lower values
    * Higher values for <span style="color: #FF0000">Prothrombin</span> as compared to lower values
    * Higher values for <span style="color: #FF0000">Age</span> as compared to lower values
    * Higher values for <span style="color: #FF0000">Copper</span> as compared to lower values
10. SHAP values were computed for the significant predictors, with contributions to the model output ranked as follows:
    * Higher values for <span style="color: #FF0000">Bilirubin</span> result to the event expected to occur sooner
    * Higher values for <span style="color: #FF0000">Prothrombin</span> result to the event expected to occur sooner
    * Higher values for <span style="color: #FF0000">Copper</span> result to the event expected to occur sooner
    * Higher values for <span style="color: #FF0000">Age</span> result to the event expected to occur sooner


```python
##################################
# Assessing the survival probability 
# and hazard function plots
# with a Log-Logistic distribution
##################################
cirrhosis_survival_loglogistic = LogLogisticFitter()
cirrhosis_survival_loglogistic.fit(durations=cirrhosis_survival_train_modeling['N_Days'], 
                                   event_observed=cirrhosis_survival_train_modeling['Status'])

##################################
# Fitting a Kaplan-Meier estimation
##################################
cirrhosis_survival_km = KaplanMeierFitter()
cirrhosis_survival_km.fit(durations=cirrhosis_survival_train_modeling['N_Days'],
                          event_observed=cirrhosis_survival_train_modeling['Status'])

##################################
# Generating the survival probability 
# and hazard function plots
##################################
plt.figure(figsize=(17, 8))

##################################
# Generating the Kaplan-Meier
# survival probability plot
##################################
plt.subplot(1, 3, 1)
cirrhosis_survival_km.plot_survival_function()
plt.title("Kaplan-Meier Survival Probability Curve")
plt.xlabel("N_Days")
plt.ylabel("Survival Probability")
plt.legend('',frameon=False)

##################################
# Generating the Log-Logistic
# survival probability plot
##################################
plt.subplot(1, 3, 2)
cirrhosis_survival_loglogistic.plot_survival_function()
plt.title("Log-Logistic Survival Probability Curve")
plt.xlabel("N_Days")
plt.ylabel("Survival Probability")
plt.legend('',frameon=False)

##################################
# Generating the log-Logistic
# hazard function plot
##################################
plt.subplot(1, 3, 3)
cirrhosis_survival_loglogistic.plot_hazard()
plt.title("Log-Logistic Hazard Function")
plt.xlabel("N_Days")
plt.ylabel("Hazard")
plt.legend('',frameon=False)

##################################
# Consolidating all plots
##################################
plt.tight_layout()
plt.show()
```


    
![png](output_200_0.png)
    



```python
##################################
# Formulating the Accelerated Failure Time model
# based on a Log-Logistic distribution
# and generating the summary
##################################
cirrhosis_survival_aft_loglogistic = LogLogisticAFTFitter(penalizer=0.30)
cirrhosis_survival_aft_loglogistic.fit(cirrhosis_survival_train_modeling, duration_col='N_Days', event_col='Status')
cirrhosis_survival_aft_loglogistic.print_summary()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.LogLogisticAFTFitter</td>
    </tr>
    <tr>
      <th>duration col</th>
      <td>'N_Days'</td>
    </tr>
    <tr>
      <th>event col</th>
      <td>'Status'</td>
    </tr>
    <tr>
      <th>penalizer</th>
      <td>0.3</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>218</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>87</td>
    </tr>
    <tr>
      <th>log-likelihood</th>
      <td>-781.35</td>
    </tr>
    <tr>
      <th>time fit was run</th>
      <td>2024-08-17 00:57:02 UTC</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">exp(coef)</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">exp(coef) lower 95%</th>
      <th style="min-width: 12px;">exp(coef) upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="18" valign="top">alpha_</th>
      <th>Age</th>
      <td>-0.18</td>
      <td>0.83</td>
      <td>0.08</td>
      <td>-0.33</td>
      <td>-0.04</td>
      <td>0.72</td>
      <td>0.97</td>
      <td>0.00</td>
      <td>-2.42</td>
      <td>0.02</td>
      <td>6.02</td>
    </tr>
    <tr>
      <th>Albumin</th>
      <td>0.10</td>
      <td>1.10</td>
      <td>0.08</td>
      <td>-0.05</td>
      <td>0.25</td>
      <td>0.95</td>
      <td>1.29</td>
      <td>0.00</td>
      <td>1.27</td>
      <td>0.20</td>
      <td>2.30</td>
    </tr>
    <tr>
      <th>Alk_Phos</th>
      <td>-0.05</td>
      <td>0.95</td>
      <td>0.08</td>
      <td>-0.20</td>
      <td>0.11</td>
      <td>0.82</td>
      <td>1.11</td>
      <td>0.00</td>
      <td>-0.60</td>
      <td>0.55</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>Ascites</th>
      <td>-0.29</td>
      <td>0.75</td>
      <td>0.28</td>
      <td>-0.83</td>
      <td>0.25</td>
      <td>0.43</td>
      <td>1.28</td>
      <td>0.00</td>
      <td>-1.06</td>
      <td>0.29</td>
      <td>1.79</td>
    </tr>
    <tr>
      <th>Bilirubin</th>
      <td>-0.27</td>
      <td>0.77</td>
      <td>0.09</td>
      <td>-0.44</td>
      <td>-0.09</td>
      <td>0.64</td>
      <td>0.91</td>
      <td>0.00</td>
      <td>-3.00</td>
      <td>&lt;0.005</td>
      <td>8.55</td>
    </tr>
    <tr>
      <th>Cholesterol</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.08</td>
      <td>-0.15</td>
      <td>0.16</td>
      <td>0.86</td>
      <td>1.17</td>
      <td>0.00</td>
      <td>0.04</td>
      <td>0.97</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>Copper</th>
      <td>-0.16</td>
      <td>0.85</td>
      <td>0.08</td>
      <td>-0.32</td>
      <td>-0.01</td>
      <td>0.73</td>
      <td>0.99</td>
      <td>0.00</td>
      <td>-2.05</td>
      <td>0.04</td>
      <td>4.64</td>
    </tr>
    <tr>
      <th>Drug</th>
      <td>0.09</td>
      <td>1.10</td>
      <td>0.14</td>
      <td>-0.19</td>
      <td>0.38</td>
      <td>0.83</td>
      <td>1.46</td>
      <td>0.00</td>
      <td>0.66</td>
      <td>0.51</td>
      <td>0.98</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>-0.33</td>
      <td>0.72</td>
      <td>0.20</td>
      <td>-0.73</td>
      <td>0.06</td>
      <td>0.48</td>
      <td>1.07</td>
      <td>0.00</td>
      <td>-1.65</td>
      <td>0.10</td>
      <td>3.32</td>
    </tr>
    <tr>
      <th>Hepatomegaly</th>
      <td>-0.10</td>
      <td>0.90</td>
      <td>0.15</td>
      <td>-0.40</td>
      <td>0.20</td>
      <td>0.67</td>
      <td>1.22</td>
      <td>0.00</td>
      <td>-0.66</td>
      <td>0.51</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>Platelets</th>
      <td>0.08</td>
      <td>1.09</td>
      <td>0.08</td>
      <td>-0.07</td>
      <td>0.23</td>
      <td>0.94</td>
      <td>1.26</td>
      <td>0.00</td>
      <td>1.09</td>
      <td>0.27</td>
      <td>1.87</td>
    </tr>
    <tr>
      <th>Prothrombin</th>
      <td>-0.20</td>
      <td>0.82</td>
      <td>0.08</td>
      <td>-0.35</td>
      <td>-0.04</td>
      <td>0.70</td>
      <td>0.96</td>
      <td>0.00</td>
      <td>-2.52</td>
      <td>0.01</td>
      <td>6.41</td>
    </tr>
    <tr>
      <th>SGOT</th>
      <td>-0.09</td>
      <td>0.91</td>
      <td>0.08</td>
      <td>-0.25</td>
      <td>0.06</td>
      <td>0.78</td>
      <td>1.06</td>
      <td>0.00</td>
      <td>-1.19</td>
      <td>0.23</td>
      <td>2.09</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>0.11</td>
      <td>1.12</td>
      <td>0.20</td>
      <td>-0.29</td>
      <td>0.51</td>
      <td>0.75</td>
      <td>1.66</td>
      <td>0.00</td>
      <td>0.55</td>
      <td>0.59</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>Spiders</th>
      <td>-0.15</td>
      <td>0.86</td>
      <td>0.17</td>
      <td>-0.47</td>
      <td>0.18</td>
      <td>0.62</td>
      <td>1.20</td>
      <td>0.00</td>
      <td>-0.87</td>
      <td>0.39</td>
      <td>1.37</td>
    </tr>
    <tr>
      <th>Stage_4.0</th>
      <td>-0.25</td>
      <td>0.78</td>
      <td>0.17</td>
      <td>-0.57</td>
      <td>0.08</td>
      <td>0.57</td>
      <td>1.08</td>
      <td>0.00</td>
      <td>-1.49</td>
      <td>0.14</td>
      <td>2.87</td>
    </tr>
    <tr>
      <th>Tryglicerides</th>
      <td>-0.09</td>
      <td>0.92</td>
      <td>0.08</td>
      <td>-0.24</td>
      <td>0.07</td>
      <td>0.79</td>
      <td>1.07</td>
      <td>0.00</td>
      <td>-1.11</td>
      <td>0.27</td>
      <td>1.90</td>
    </tr>
    <tr>
      <th>Intercept</th>
      <td>8.28</td>
      <td>3927.15</td>
      <td>0.23</td>
      <td>7.82</td>
      <td>8.73</td>
      <td>2501.84</td>
      <td>6164.47</td>
      <td>0.00</td>
      <td>35.97</td>
      <td>&lt;0.005</td>
      <td>938.99</td>
    </tr>
    <tr>
      <th>beta_</th>
      <th>Intercept</th>
      <td>0.51</td>
      <td>1.67</td>
      <td>0.08</td>
      <td>0.36</td>
      <td>0.67</td>
      <td>1.43</td>
      <td>1.95</td>
      <td>0.00</td>
      <td>6.46</td>
      <td>&lt;0.005</td>
      <td>33.18</td>
    </tr>
  </tbody>
</table><br><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>Concordance</th>
      <td>0.86</td>
    </tr>
    <tr>
      <th>AIC</th>
      <td>1600.71</td>
    </tr>
    <tr>
      <th>log-likelihood ratio test</th>
      <td>95.58 on 17 df</td>
    </tr>
    <tr>
      <th>-log2(p) of ll-ratio test</th>
      <td>40.64</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting the log accelerated failure rate of the
# formulated Accelerated Failure Time model
# based on a Log-Logistic distribution
##################################
cirrhosis_survival_aft_loglogistic_summary = cirrhosis_survival_aft_loglogistic.summary
cirrhosis_survival_aft_loglogistic_summary_params = pd.DataFrame(cirrhosis_survival_aft_loglogistic.params_)
significant = cirrhosis_survival_aft_loglogistic_summary['p'] < 0.05
cirrhosis_survival_aft_loglogistic_summary_log_accelerated_failure_rate = (list(cirrhosis_survival_aft_loglogistic_summary_params.iloc[:,0].values))
plt.figure(figsize=(17, 8))
colors = ['#993300' if sig else '#CC9966' for sig in significant]

plt.barh([(index[0] + index[1]) for index in cirrhosis_survival_aft_loglogistic_summary_params.index[0:17]], 
         cirrhosis_survival_aft_loglogistic_summary_log_accelerated_failure_rate[0:17], 
         xerr=cirrhosis_survival_aft_loglogistic_summary['se(coef)'][0:17], 
         color=colors)
plt.xlabel('Log(Accelerated Failure Rate)')
plt.ylabel('Variables')
plt.title('AFT_LOGLOGISTIC Log(Accelerated Failure Rate) Forest Plot')
plt.axvline(x=0, color='k', linestyle='--')
plt.gca().invert_yaxis()
plt.show()
```


    
![png](output_202_0.png)
    



```python
##################################
# Determining the number of
# significant predictors
##################################
cirrhosis_survival_aft_loglogistic_significant = sum(cirrhosis_survival_aft_loglogistic_summary['p'] < 0.05)
display(f"Number of Significant Predictors: {cirrhosis_survival_aft_loglogistic_significant-2}")
```


    'Number of Significant Predictors: 4'



```python
##################################
# Formulating the Accelerated Failure Time model
# based on a Log-Logistic distribution
# using the significant predictors only
# and generating the summary
##################################
feature_subset = ['Bilirubin','Prothrombin','Age','Copper','N_Days','Status']
cirrhosis_survival_aft_loglogistic = LogLogisticAFTFitter(penalizer=0.30)
cirrhosis_survival_aft_loglogistic.fit(cirrhosis_survival_train_modeling[feature_subset], duration_col='N_Days', event_col='Status')
cirrhosis_survival_aft_loglogistic.print_summary()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.LogLogisticAFTFitter</td>
    </tr>
    <tr>
      <th>duration col</th>
      <td>'N_Days'</td>
    </tr>
    <tr>
      <th>event col</th>
      <td>'Status'</td>
    </tr>
    <tr>
      <th>penalizer</th>
      <td>0.3</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>218</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>87</td>
    </tr>
    <tr>
      <th>log-likelihood</th>
      <td>-792.48</td>
    </tr>
    <tr>
      <th>time fit was run</th>
      <td>2024-08-17 00:57:03 UTC</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">exp(coef)</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">exp(coef) lower 95%</th>
      <th style="min-width: 12px;">exp(coef) upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">alpha_</th>
      <th>Age</th>
      <td>-0.20</td>
      <td>0.82</td>
      <td>0.07</td>
      <td>-0.35</td>
      <td>-0.06</td>
      <td>0.71</td>
      <td>0.94</td>
      <td>0.00</td>
      <td>-2.80</td>
      <td>0.01</td>
      <td>7.60</td>
    </tr>
    <tr>
      <th>Bilirubin</th>
      <td>-0.40</td>
      <td>0.67</td>
      <td>0.08</td>
      <td>-0.56</td>
      <td>-0.25</td>
      <td>0.57</td>
      <td>0.78</td>
      <td>0.00</td>
      <td>-5.13</td>
      <td>&lt;0.005</td>
      <td>21.73</td>
    </tr>
    <tr>
      <th>Copper</th>
      <td>-0.22</td>
      <td>0.80</td>
      <td>0.08</td>
      <td>-0.38</td>
      <td>-0.07</td>
      <td>0.69</td>
      <td>0.93</td>
      <td>0.00</td>
      <td>-2.84</td>
      <td>&lt;0.005</td>
      <td>7.80</td>
    </tr>
    <tr>
      <th>Prothrombin</th>
      <td>-0.29</td>
      <td>0.75</td>
      <td>0.08</td>
      <td>-0.44</td>
      <td>-0.14</td>
      <td>0.65</td>
      <td>0.87</td>
      <td>0.00</td>
      <td>-3.84</td>
      <td>&lt;0.005</td>
      <td>13.00</td>
    </tr>
    <tr>
      <th>Intercept</th>
      <td>8.17</td>
      <td>3530.45</td>
      <td>0.10</td>
      <td>7.98</td>
      <td>8.36</td>
      <td>2909.02</td>
      <td>4284.63</td>
      <td>0.00</td>
      <td>82.70</td>
      <td>&lt;0.005</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>beta_</th>
      <th>Intercept</th>
      <td>0.48</td>
      <td>1.61</td>
      <td>0.08</td>
      <td>0.32</td>
      <td>0.63</td>
      <td>1.38</td>
      <td>1.87</td>
      <td>0.00</td>
      <td>6.08</td>
      <td>&lt;0.005</td>
      <td>29.59</td>
    </tr>
  </tbody>
</table><br><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>Concordance</th>
      <td>0.84</td>
    </tr>
    <tr>
      <th>AIC</th>
      <td>1596.96</td>
    </tr>
    <tr>
      <th>log-likelihood ratio test</th>
      <td>73.33 on 4 df</td>
    </tr>
    <tr>
      <th>-log2(p) of ll-ratio test</th>
      <td>47.66</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting the log accelerated failure rate of the
# formulated Accelerated Failure Time model
# using the significant predictors only
# based on a Log-Logistic distribution
##################################
cirrhosis_survival_aft_loglogistic_summary = cirrhosis_survival_aft_loglogistic.summary
cirrhosis_survival_aft_loglogistic_summary_params = pd.DataFrame(cirrhosis_survival_aft_loglogistic.params_)
significant = cirrhosis_survival_aft_loglogistic_summary['p'] < 0.05
cirrhosis_survival_aft_loglogistic_summary_log_accelerated_failure_rate = (list(cirrhosis_survival_aft_loglogistic_summary_params.iloc[:,0].values))
plt.figure(figsize=(17, 8))
colors = ['#993300' if sig else '#CC9966' for sig in significant]

plt.barh([(index[0] + index[1]) for index in cirrhosis_survival_aft_loglogistic_summary_params.index[0:4]], 
         cirrhosis_survival_aft_loglogistic_summary_log_accelerated_failure_rate[0:4], 
         xerr=cirrhosis_survival_aft_loglogistic_summary['se(coef)'][0:4], 
         color=colors)
plt.xlabel('Log(Accelerated Failure Rate)')
plt.ylabel('Variables')
plt.title('AFT_LOGLOGISTIC Log(Accelerated Failure Rate) Forest Plot')
plt.axvline(x=0, color='k', linestyle='--')
plt.gca().invert_yaxis()
plt.show()
```


    
![png](output_205_0.png)
    



```python
##################################
# Gathering the apparent model performance
# as baseline for evaluating overfitting
##################################
cirrhosis_survival_aft_loglogistic.fit(cirrhosis_survival_train_modeling[feature_subset], duration_col='N_Days', event_col='Status')
train_predictions = cirrhosis_survival_aft_loglogistic.predict_median(cirrhosis_survival_train_modeling)
cirrhosis_survival_aft_loglogistic_train_ci = concordance_index(cirrhosis_survival_train_modeling['N_Days'], 
                                                            train_predictions, 
                                                            cirrhosis_survival_train_modeling['Status'])
time_point = cirrhosis_survival_train_modeling['N_Days'].median()
cirrhosis_survival_aft_loglogistic_train_mae = mean_absolute_error(cirrhosis_survival_train_modeling['N_Days'], train_predictions)
cirrhosis_survival_aft_loglogistic_train_brier = brier_score_loss(cirrhosis_survival_train_modeling['Status'], 
                                                              cirrhosis_survival_aft_loglogistic.predict_survival_function(cirrhosis_survival_train_modeling, 
                                                                                                                       times=[time_point]).T[time_point])
display(f"Apparent Concordance Index: {cirrhosis_survival_aft_loglogistic_train_ci}")
display(f"Apparent MAE: {cirrhosis_survival_aft_loglogistic_train_mae}")
display(f"Apparent Brier Score: {cirrhosis_survival_aft_loglogistic_train_brier}")
```


    'Apparent Concordance Index: 0.8383452535760728'



    'Apparent MAE: 2727.465086218323'



    'Apparent Brier Score: 0.5095276225408752'



```python
##################################
# Performing 5-Fold Cross-Validation
# on the training data
##################################
kf = KFold(n_splits=5, shuffle=True, random_state=88888888)
ci_scores = []
mae_scores = []
brier_scores = []

for train_index, val_index in kf.split(cirrhosis_survival_train_modeling):
    df_train_fold = cirrhosis_survival_train_modeling.iloc[train_index]
    df_val_fold = cirrhosis_survival_train_modeling.iloc[val_index]
    
    cirrhosis_survival_aft_loglogistic.fit(df_train_fold[feature_subset], duration_col='N_Days', event_col='Status')
    val_predictions = cirrhosis_survival_aft_loglogistic.predict_median(df_val_fold)
    time_point = df_val_fold['N_Days'].median()
    ci = concordance_index(df_val_fold['N_Days'], val_predictions, df_val_fold['Status'])
    mae = mean_absolute_error(df_val_fold['N_Days'], val_predictions)
    brier = brier_score_loss(df_val_fold['Status'],
                             cirrhosis_survival_aft_loglogistic.predict_survival_function(df_val_fold, 
                                                                                      times=[time_point]).T[time_point])
    ci_scores.append(ci)
    mae_scores.append(mae)
    brier_scores.append(brier)

cirrhosis_survival_aft_loglogistic_cv_ci_mean = np.mean(ci_scores)
cirrhosis_survival_aft_loglogistic_cv_ci_std = np.std(ci_scores)
cirrhosis_survival_aft_loglogistic_cv_mae_mean = np.mean(mae_scores)
cirrhosis_survival_aft_loglogistic_cv_brier_mean = np.mean(brier_scores)

display(f"Cross-Validated Concordance Index: {cirrhosis_survival_aft_loglogistic_cv_ci_mean}")
display(f"Cross-Validated MAE: {cirrhosis_survival_aft_loglogistic_cv_mae_mean}")
display(f"Cross-Validated Brier Score: {cirrhosis_survival_aft_loglogistic_cv_brier_mean}")
```


    'Cross-Validated Concordance Index: 0.8301281045334907'



    'Cross-Validated MAE: 2711.660486031347'



    'Cross-Validated Brier Score: 0.5065381245204558'



```python
##################################
# Evaluating the model performance
# on test data
##################################
test_predictions = cirrhosis_survival_aft_loglogistic.predict_median(cirrhosis_survival_test_modeling)
cirrhosis_survival_aft_loglogistic_test_ci = concordance_index(cirrhosis_survival_test_modeling['N_Days'], 
                                                           test_predictions, 
                                                           cirrhosis_survival_test_modeling['Status'])
time_point = cirrhosis_survival_test_modeling['N_Days'].median()
cirrhosis_survival_aft_loglogistic_test_mae = mean_absolute_error(cirrhosis_survival_test_modeling['N_Days'], test_predictions)
cirrhosis_survival_aft_loglogistic_test_brier = brier_score_loss(cirrhosis_survival_test_modeling['Status'], 
                                                              cirrhosis_survival_aft_loglogistic.predict_survival_function(cirrhosis_survival_test_modeling, 
                                                                                                                       times=[time_point]).T[time_point])
display(f"Apparent Concordance Index: {cirrhosis_survival_aft_loglogistic_test_ci}")
display(f"Apparent MAE: {cirrhosis_survival_aft_loglogistic_test_mae}")
display(f"Apparent Brier Score: {cirrhosis_survival_aft_loglogistic_test_brier}")
```


    'Apparent Concordance Index: 0.8625850340136054'



    'Apparent MAE: 2189.9323142397434'



    'Apparent Brier Score: 0.5332955064077312'



```python
##################################
# Gathering the model performance metrics
# from training, cross-validation and test
##################################
aft_loglogistic_set = pd.DataFrame(["Train","Cross-Validation","Test"]*3)
aft_loglogistic_metric = pd.DataFrame((["Concordance.Index"]*3) + (["MAE"]*3) + (["Brier.Score"]*3))
aft_loglogistic_metric_values = pd.DataFrame([cirrhosis_survival_aft_loglogistic_train_ci,
                                           cirrhosis_survival_aft_loglogistic_cv_ci_mean,
                                           cirrhosis_survival_aft_loglogistic_test_ci,
                                           cirrhosis_survival_aft_loglogistic_train_mae,
                                           cirrhosis_survival_aft_loglogistic_cv_mae_mean,
                                           cirrhosis_survival_aft_loglogistic_test_mae,
                                           cirrhosis_survival_aft_loglogistic_train_brier,
                                           cirrhosis_survival_aft_loglogistic_cv_brier_mean,
                                           cirrhosis_survival_aft_loglogistic_test_brier])
aft_loglogistic_method = pd.DataFrame(["AFT_LOGLOGISTIC"]*9)
aft_loglogistic_summary = pd.concat([aft_loglogistic_set,
                                       aft_loglogistic_metric,
                                       aft_loglogistic_metric_values,
                                       aft_loglogistic_method], 
                                      axis=1)
aft_loglogistic_summary.columns = ['Set', 'Metric', 'Value', 'Method']
aft_loglogistic_summary.reset_index(inplace=True, drop=True)
display(aft_loglogistic_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Set</th>
      <th>Metric</th>
      <th>Value</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Train</td>
      <td>Concordance.Index</td>
      <td>0.838345</td>
      <td>AFT_LOGLOGISTIC</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cross-Validation</td>
      <td>Concordance.Index</td>
      <td>0.830128</td>
      <td>AFT_LOGLOGISTIC</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Test</td>
      <td>Concordance.Index</td>
      <td>0.862585</td>
      <td>AFT_LOGLOGISTIC</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Train</td>
      <td>MAE</td>
      <td>2727.465086</td>
      <td>AFT_LOGLOGISTIC</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cross-Validation</td>
      <td>MAE</td>
      <td>2711.660486</td>
      <td>AFT_LOGLOGISTIC</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Test</td>
      <td>MAE</td>
      <td>2189.932314</td>
      <td>AFT_LOGLOGISTIC</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Train</td>
      <td>Brier.Score</td>
      <td>0.509528</td>
      <td>AFT_LOGLOGISTIC</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Cross-Validation</td>
      <td>Brier.Score</td>
      <td>0.506538</td>
      <td>AFT_LOGLOGISTIC</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Test</td>
      <td>Brier.Score</td>
      <td>0.533296</td>
      <td>AFT_LOGLOGISTIC</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Evaluating the predicted
# and actual survival times
##################################
predicted_survival_times = cirrhosis_survival_aft_loglogistic.predict_median(cirrhosis_survival_test_modeling)
fig, ax = plt.subplots(figsize=(17, 8))
for status, color, label in zip([True, False], ['#FF7F0E','#1F77B4'], ['Death', 'Censored']):
    subset = cirrhosis_survival_test_modeling[cirrhosis_survival_test_modeling['Status'] == status]
    ax.scatter(subset['N_Days'], predicted_survival_times.iloc[subset.index], c=color, label=label, alpha=0.8)
ax.set_xlabel('Actual Survival Time')
ax.set_ylabel('Predicted Survival Time')
ax.set_title('AFT_LOGLOGISTIC: Predicted Versus Actual Survival Times')
ax.legend()
plt.plot([0, cirrhosis_survival_test_modeling['N_Days'].max()], 
         [0, cirrhosis_survival_test_modeling['N_Days'].max()], 
         color='black', linestyle='--')
plt.show()
```


    
![png](output_210_0.png)
    



```python
##################################
# Plotting the individual
# survival probability profiles
##################################
plt.figure(figsize=(17, 8))
for status, color, label in zip([True, False], ['#FF7F0E','#1F77B4'], ['Death', 'Censored']):
    subset = cirrhosis_survival_test_modeling[cirrhosis_survival_test_modeling['Status'] == status]
    for i, row in subset.iterrows():
        survival_function = cirrhosis_survival_aft_loglogistic.predict_survival_function(row)
        plt.plot(survival_function.index, survival_function.iloc[:, 0], c=color, alpha=0.8)
plt.title('AFT_LOGLOGISTIC: Survival Probability Profiles')
plt.xlabel('N_Days')
plt.ylabel('Survival Probability')
death_patch = plt.Line2D([0], [0], color='#FF7F0E', lw=2, label='Death')
censored_patch = plt.Line2D([0], [0], color='#1F77B4', lw=2, label='Censored')
plt.legend(handles=[death_patch, censored_patch])
plt.show()
```


    
![png](output_211_0.png)
    



```python
##################################
# Plotting the individual
# survival probability profiles
# for the discretized Bilirubin predictor
##################################
cirrhosis_survival_test_modeling['Bilirubin_Level'] = pd.qcut(cirrhosis_survival_test_modeling['Bilirubin'], 3, labels=['Low','Moderate','High'])
plt.figure(figsize=(17, 8))
for bilirubin_level, color, label in zip(['Low', 'Moderate', 'High'], ['#FA8000','#E50000', '#8C000F'], ['Low', 'Moderate', 'High']):
    subset = cirrhosis_survival_test_modeling[cirrhosis_survival_test_modeling['Bilirubin_Level'] == bilirubin_level]
    for i, row in subset.iterrows():
        survival_function = cirrhosis_survival_aft_loglogistic.predict_survival_function(row)
        plt.plot(survival_function.index, survival_function.iloc[:, 0], c=color, alpha=0.8)
plt.title('AFT_LOGLOGISTIC: Survival Probability Profiles by Bilirubin Level')
plt.xlabel('N_Days')
plt.ylabel('Survival Probability')
low_patch = plt.Line2D([0], [0], color='#FA8000', lw=2, label='Low')
moderate_patch = plt.Line2D([0], [0], color='#E50000', lw=2, label='Moderate')
high_patch = plt.Line2D([0], [0], color='#8C000F', lw=2, label='High')
plt.legend(handles=[low_patch, moderate_patch, high_patch])
plt.show()
```


    
![png](output_212_0.png)
    



```python
##################################
# Plotting the individual
# survival probability profiles
# for the discretized Prothrombin predictor
##################################
cirrhosis_survival_test_modeling['Prothrombin_Level'] = pd.qcut(cirrhosis_survival_test_modeling['Prothrombin'], 3, labels=['Low','Moderate','High'])
plt.figure(figsize=(17, 8))
for prothrombin_level, color, label in zip(['Low', 'Moderate', 'High'], ['#FA8000','#E50000', '#8C000F'], ['Low', 'Moderate', 'High']):
    subset = cirrhosis_survival_test_modeling[cirrhosis_survival_test_modeling['Prothrombin_Level'] == prothrombin_level]
    for i, row in subset.iterrows():
        survival_function = cirrhosis_survival_aft_loglogistic.predict_survival_function(row)
        plt.plot(survival_function.index, survival_function.iloc[:, 0], c=color, alpha=0.8)
plt.title('AFT_LOGLOGISTIC: Survival Probability Profiles by Prothrombin Level')
plt.xlabel('N_Days')
plt.ylabel('Survival Probability')
low_patch = plt.Line2D([0], [0], color='#FA8000', lw=2, label='Low')
moderate_patch = plt.Line2D([0], [0], color='#E50000', lw=2, label='Moderate')
high_patch = plt.Line2D([0], [0], color='#8C000F', lw=2, label='High')
plt.legend(handles=[low_patch, moderate_patch, high_patch])
plt.show()
```


    
![png](output_213_0.png)
    



```python
##################################
# Plotting the individual
# survival probability profiles
# for the discretized Copper predictor
##################################
cirrhosis_survival_test_modeling['Copper_Level'] = pd.qcut(cirrhosis_survival_test_modeling['Copper'], 3, labels=['Low','Moderate','High'])
plt.figure(figsize=(17, 8))
for copper_level, color, label in zip(['Low', 'Moderate', 'High'], ['#FA8000','#E50000', '#8C000F'], ['Low', 'Moderate', 'High']):
    subset = cirrhosis_survival_test_modeling[cirrhosis_survival_test_modeling['Copper_Level'] == copper_level]
    for i, row in subset.iterrows():
        survival_function = cirrhosis_survival_aft_loglogistic.predict_survival_function(row)
        plt.plot(survival_function.index, survival_function.iloc[:, 0], c=color, alpha=0.8)
plt.title('AFT_LOGLOGISTIC: Survival Probability Profiles by Copper Level')
plt.xlabel('N_Days')
plt.ylabel('Survival Probability')
low_patch = plt.Line2D([0], [0], color='#FA8000', lw=2, label='Low')
moderate_patch = plt.Line2D([0], [0], color='#E50000', lw=2, label='Moderate')
high_patch = plt.Line2D([0], [0], color='#8C000F', lw=2, label='High')
plt.legend(handles=[low_patch, moderate_patch, high_patch])
plt.show()
```


    
![png](output_214_0.png)
    



```python
##################################
# Plotting the individual
# survival probability profiles
# for the discretized Age predictor
##################################
cirrhosis_survival_test_modeling['Age_Level'] = pd.qcut(cirrhosis_survival_test_modeling['Age'], 3, labels=['Low','Moderate','High'])
plt.figure(figsize=(17, 8))
for age_level, color, label in zip(['Low', 'Moderate', 'High'], ['#FA8000','#E50000', '#8C000F'], ['Low', 'Moderate', 'High']):
    subset = cirrhosis_survival_test_modeling[cirrhosis_survival_test_modeling['Age_Level'] == age_level]
    for i, row in subset.iterrows():
        survival_function = cirrhosis_survival_aft_loglogistic.predict_survival_function(row)
        plt.plot(survival_function.index, survival_function.iloc[:, 0], c=color, alpha=0.8)
plt.title('AFT_LOGLOGISTIC: Survival Probability Profiles by Age Level')
plt.xlabel('N_Days')
plt.ylabel('Survival Probability')
low_patch = plt.Line2D([0], [0], color='#FA8000', lw=2, label='Low')
moderate_patch = plt.Line2D([0], [0], color='#E50000', lw=2, label='Moderate')
high_patch = plt.Line2D([0], [0], color='#8C000F', lw=2, label='High')
plt.legend(handles=[low_patch, moderate_patch, high_patch])
plt.show()
```


    
![png](output_215_0.png)
    



```python
##################################
# Defining a prediction function
# for SHAP value estimation
##################################
def aft_predict(fitter, df):
    return fitter.predict_expectation(df)

##################################
# Creating the explainer object
##################################
explainer_loglogistic = shap.Explainer(lambda x: aft_predict(cirrhosis_survival_aft_loglogistic, 
                                                         pd.DataFrame(x, columns=cirrhosis_survival_train_modeling.columns[2:])), 
                                   cirrhosis_survival_train_modeling.iloc[:, 2:])
shap_values_loglogistic = explainer_loglogistic(cirrhosis_survival_train_modeling.iloc[:, 2:])
```

    PermutationExplainer explainer: 219it [00:22,  5.27it/s]                         
    


```python
##################################
# Plotting the SHAP summary plot
##################################
shap.summary_plot(shap_values_loglogistic, 
                  cirrhosis_survival_train_modeling.iloc[:, 2:])
```


    
![png](output_217_0.png)
    


## 1.7. Consolidated Findings <a class="anchor" id="1.7"></a>

1. In the context of accelerated failure time models, the choice of distribution is crucial as it impacts the estimation of survival times and the interpretation of covariate effects. Factors to consider include the nature of the data, the properties of the distributions, and the model fit.
    * [Weibull distribution](https://lifelines.readthedocs.io/en/latest/fitters/regression/WeibullAFTFitter.html) is suitable for data where the hazard function is either monotonically increasing or decreasing. It can model both increasing hazard (positive shape parameter) and decreasing hazard (negative shape parameter).
    * [Log-Normal distribution](https://lifelines.readthedocs.io/en/latest/fitters/regression/LogNormalAFTFitter.html) is appropriate when the logarithm of the survival times follows a normal distribution. It is useful for data where the hazard rate initially increases and then decreases.
    * [Log-Logistic distribution](https://lifelines.readthedocs.io/en/latest/fitters/regression/LogLogisticAFTFitter.html) is suitable when the hazard function initially increases and then decreases. It can accommodate both heavy-tailed distributions and distributions with a decreasing hazard rate at larger time values. 
2. Several metrics are available for evaluating the performance of accelerated failure time models, each with strengths and weaknesses.
    * [Concordance index](https://lifelines.readthedocs.io/en/latest/lifelines.utils.html) provides a measure of discriminative ability and useful for ranking predictions. However, it does not provide information on the magnitude of errors and may be insensitive to the calibration of predicted survival probabilities.
    * [Mean absolute error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html) is intuitive and easy to interpret while providing a direct measure of prediction accuracy. However, it may be sensitive to outliers and does not consider the probabilistic nature of survival predictions.
    * [Brier score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html) considers both discrimination and calibration, while reflecting the accuracy of predicted survival probabilities. However, it requires specifying a time point and aggregating scores over time points may be less interpretable.
3. Comparing all results from the accelerated failure time models formulated, the most viable model for prediction using the MAE metric was determined as:
    * [Weibull distribution](https://lifelines.readthedocs.io/en/latest/fitters/regression/WeibullAFTFitter.html)
        * Supports a monotonically increasing hazard function which may be more suitable for modeling cirrhosis survival
        * Demonstrated one of the best independent cross-validated (**MAE** = 2303.605628) and test (**MAE** = 1948.873380) model performance 
        * Showed minimal overfit between the train (**MAE** = 2280.743783) and cross-validated (**MAE** = 2303.605628) model performance
        * Selected a sufficient number of predictors (3 out of 17)
        * Identified a sufficient number of statistically significant predictors (3 out of 17):
            * <span style="color: #FF0000">Bilirubin</span>: Increase in value associated with a decrease in time to event 
            * <span style="color: #FF0000">Prothrombin</span>: Increase in value associated with a decrease in time to event 
            * <span style="color: #FF0000">Age</span>: Increase in value associated with a decrease in time to event 
        * Survival probability curves estimated for all cases. Shorter median survival times were observed for:
            * Event cases as compared to censored cases
            * Higher values for <span style="color: #FF0000">Bilirubin</span> as compared to lower values
            * Higher values for <span style="color: #FF0000">Prothrombin</span> as compared to lower values
            * Higher values for <span style="color: #FF0000">Age</span> as compared to lower values
        * Obtained **SHAP values** provided an insightful and clear indication of each significant predictor's impact on the lifetime prediction:
            * Higher values for <span style="color: #FF0000">Bilirubin</span> result to the event expected to occur sooner
            * Higher values for <span style="color: #FF0000">Prothrombin</span> result to the event expected to occur sooner
            * Higher values for <span style="color: #FF0000">Age</span> result to the event expected to occur sooner


```python
##################################
# Consolidating all the
# model performance metrics
##################################
model_performance_comparison = pd.concat([aft_weibull_summary, 
                                          aft_lognormal_summary,
                                          aft_loglogistic_summary], 
                                         axis=0,
                                         ignore_index=True)
print('Accelerated Failure Time Model Comparison: ')
display(model_performance_comparison)
```

    Accelerated Failure Time Model Comparison: 
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Set</th>
      <th>Metric</th>
      <th>Value</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Train</td>
      <td>Concordance.Index</td>
      <td>0.829080</td>
      <td>AFT_WEIBULL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cross-Validation</td>
      <td>Concordance.Index</td>
      <td>0.825008</td>
      <td>AFT_WEIBULL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Test</td>
      <td>Concordance.Index</td>
      <td>0.852608</td>
      <td>AFT_WEIBULL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Train</td>
      <td>MAE</td>
      <td>2280.743783</td>
      <td>AFT_WEIBULL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cross-Validation</td>
      <td>MAE</td>
      <td>2303.605628</td>
      <td>AFT_WEIBULL</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Test</td>
      <td>MAE</td>
      <td>1948.873380</td>
      <td>AFT_WEIBULL</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Train</td>
      <td>Brier.Score</td>
      <td>0.515148</td>
      <td>AFT_WEIBULL</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Cross-Validation</td>
      <td>Brier.Score</td>
      <td>0.512583</td>
      <td>AFT_WEIBULL</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Test</td>
      <td>Brier.Score</td>
      <td>0.537556</td>
      <td>AFT_WEIBULL</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Train</td>
      <td>Concordance.Index</td>
      <td>0.841352</td>
      <td>AFT_LOGNORMAL</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cross-Validation</td>
      <td>Concordance.Index</td>
      <td>0.825576</td>
      <td>AFT_LOGNORMAL</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Test</td>
      <td>Concordance.Index</td>
      <td>0.875283</td>
      <td>AFT_LOGNORMAL</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Train</td>
      <td>MAE</td>
      <td>2518.359385</td>
      <td>AFT_LOGNORMAL</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Cross-Validation</td>
      <td>MAE</td>
      <td>2502.636955</td>
      <td>AFT_LOGNORMAL</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Test</td>
      <td>MAE</td>
      <td>1904.987987</td>
      <td>AFT_LOGNORMAL</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Train</td>
      <td>Brier.Score</td>
      <td>0.547041</td>
      <td>AFT_LOGNORMAL</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Cross-Validation</td>
      <td>Brier.Score</td>
      <td>0.542583</td>
      <td>AFT_LOGNORMAL</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Test</td>
      <td>Brier.Score</td>
      <td>0.577502</td>
      <td>AFT_LOGNORMAL</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Train</td>
      <td>Concordance.Index</td>
      <td>0.838345</td>
      <td>AFT_LOGLOGISTIC</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Cross-Validation</td>
      <td>Concordance.Index</td>
      <td>0.830128</td>
      <td>AFT_LOGLOGISTIC</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Test</td>
      <td>Concordance.Index</td>
      <td>0.862585</td>
      <td>AFT_LOGLOGISTIC</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Train</td>
      <td>MAE</td>
      <td>2727.465086</td>
      <td>AFT_LOGLOGISTIC</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Cross-Validation</td>
      <td>MAE</td>
      <td>2711.660486</td>
      <td>AFT_LOGLOGISTIC</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Test</td>
      <td>MAE</td>
      <td>2189.932314</td>
      <td>AFT_LOGLOGISTIC</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Train</td>
      <td>Brier.Score</td>
      <td>0.509528</td>
      <td>AFT_LOGLOGISTIC</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Cross-Validation</td>
      <td>Brier.Score</td>
      <td>0.506538</td>
      <td>AFT_LOGLOGISTIC</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Test</td>
      <td>Brier.Score</td>
      <td>0.533296</td>
      <td>AFT_LOGLOGISTIC</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Consolidating the concordance indices
# for all sets and models
##################################
set_labels = ['Train','Cross-Validation','Test']
aft_weibull_ci = model_performance_comparison[((model_performance_comparison['Set'] == 'Train') |
                                               (model_performance_comparison['Set'] == 'Cross-Validation') |
                                               (model_performance_comparison['Set'] == 'Test')) & 
                                              (model_performance_comparison['Metric']=='Concordance.Index') &
                                              (model_performance_comparison['Method']=='AFT_WEIBULL')]['Value'].values
aft_lognormal_ci = model_performance_comparison[((model_performance_comparison['Set'] == 'Train') |
                                               (model_performance_comparison['Set'] == 'Cross-Validation') |
                                               (model_performance_comparison['Set'] == 'Test')) & 
                                              (model_performance_comparison['Metric']=='Concordance.Index') &
                                              (model_performance_comparison['Method']=='AFT_LOGNORMAL')]['Value'].values
aft_loglogistic_ci = model_performance_comparison[((model_performance_comparison['Set'] == 'Train') |
                                               (model_performance_comparison['Set'] == 'Cross-Validation') |
                                               (model_performance_comparison['Set'] == 'Test')) & 
                                              (model_performance_comparison['Metric']=='Concordance.Index') &
                                              (model_performance_comparison['Method']=='AFT_LOGLOGISTIC')]['Value'].values
```


```python
##################################
# Plotting the values for the
# concordance indices
# for all models
##################################
ci_plot = pd.DataFrame({'AFT_WEIBULL': list(aft_weibull_ci),
                        'AFT_LOGNORMAL': list(aft_lognormal_ci),
                        'AFT_LOGLOGISTIC': list(aft_loglogistic_ci)},
                       index = set_labels)
display(ci_plot)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AFT_WEIBULL</th>
      <th>AFT_LOGNORMAL</th>
      <th>AFT_LOGLOGISTIC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Train</th>
      <td>0.829080</td>
      <td>0.841352</td>
      <td>0.838345</td>
    </tr>
    <tr>
      <th>Cross-Validation</th>
      <td>0.825008</td>
      <td>0.825576</td>
      <td>0.830128</td>
    </tr>
    <tr>
      <th>Test</th>
      <td>0.852608</td>
      <td>0.875283</td>
      <td>0.862585</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting all the concordance indices
# for all models
##################################
ci_plot = ci_plot.plot.barh(figsize=(10, 6), width=0.90)
ci_plot.set_xlim(0.00,1.00)
ci_plot.set_title("Model Comparison by Concordance Indice")
ci_plot.set_xlabel("Concordance Index")
ci_plot.set_ylabel("Data Set")
ci_plot.grid(False)
ci_plot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
for container in ci_plot.containers:
    ci_plot.bar_label(container, fmt='%.5f', padding=-50, color='white', fontweight='bold')
```


    
![png](output_222_0.png)
    



```python
##################################
# Consolidating the mean absolute errors
# for all sets and models
##################################
set_labels = ['Train','Cross-Validation','Test']
aft_weibull_mae = model_performance_comparison[((model_performance_comparison['Set'] == 'Train') |
                                               (model_performance_comparison['Set'] == 'Cross-Validation') |
                                               (model_performance_comparison['Set'] == 'Test')) & 
                                              (model_performance_comparison['Metric']=='MAE') &
                                              (model_performance_comparison['Method']=='AFT_WEIBULL')]['Value'].values
aft_lognormal_mae = model_performance_comparison[((model_performance_comparison['Set'] == 'Train') |
                                               (model_performance_comparison['Set'] == 'Cross-Validation') |
                                               (model_performance_comparison['Set'] == 'Test')) & 
                                              (model_performance_comparison['Metric']=='MAE') &
                                              (model_performance_comparison['Method']=='AFT_LOGNORMAL')]['Value'].values
aft_loglogistic_mae = model_performance_comparison[((model_performance_comparison['Set'] == 'Train') |
                                               (model_performance_comparison['Set'] == 'Cross-Validation') |
                                               (model_performance_comparison['Set'] == 'Test')) & 
                                              (model_performance_comparison['Metric']=='MAE') &
                                              (model_performance_comparison['Method']=='AFT_LOGLOGISTIC')]['Value'].values
```


```python
##################################
# Plotting the values for the
# mean absolute errors
# for all models
##################################
mae_plot = pd.DataFrame({'AFT_WEIBULL': list(aft_weibull_mae),
                         'AFT_LOGNORMAL': list(aft_lognormal_mae),
                         'AFT_LOGLOGISTIC': list(aft_loglogistic_mae)},
                       index = set_labels)
display(mae_plot)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AFT_WEIBULL</th>
      <th>AFT_LOGNORMAL</th>
      <th>AFT_LOGLOGISTIC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Train</th>
      <td>2280.743783</td>
      <td>2518.359385</td>
      <td>2727.465086</td>
    </tr>
    <tr>
      <th>Cross-Validation</th>
      <td>2303.605628</td>
      <td>2502.636955</td>
      <td>2711.660486</td>
    </tr>
    <tr>
      <th>Test</th>
      <td>1948.873380</td>
      <td>1904.987987</td>
      <td>2189.932314</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting all the mean absolute errors
# for all models
##################################
mae_plot = mae_plot.plot.barh(figsize=(10, 6), width=0.90)
mae_plot.set_xlim(0.00,3000.00)
mae_plot.set_title("Model Comparison by Mean Absolute Error")
mae_plot.set_xlabel("Mean Absolute Error")
mae_plot.set_ylabel("Data Set")
mae_plot.grid(False)
mae_plot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
for container in mae_plot.containers:
    mae_plot.bar_label(container, fmt='%.5f', padding=-75, color='white', fontweight='bold')
```


    
![png](output_225_0.png)
    



```python
##################################
# Consolidating the brier scores
# for all sets and models
##################################
set_labels = ['Train','Cross-Validation','Test']
aft_weibull_brier_score = model_performance_comparison[((model_performance_comparison['Set'] == 'Train') |
                                               (model_performance_comparison['Set'] == 'Cross-Validation') |
                                               (model_performance_comparison['Set'] == 'Test')) & 
                                              (model_performance_comparison['Metric']=='Brier.Score') &
                                              (model_performance_comparison['Method']=='AFT_WEIBULL')]['Value'].values
aft_lognormal_brier_score = model_performance_comparison[((model_performance_comparison['Set'] == 'Train') |
                                               (model_performance_comparison['Set'] == 'Cross-Validation') |
                                               (model_performance_comparison['Set'] == 'Test')) & 
                                              (model_performance_comparison['Metric']=='Brier.Score') &
                                              (model_performance_comparison['Method']=='AFT_LOGNORMAL')]['Value'].values
aft_loglogistic_brier_score = model_performance_comparison[((model_performance_comparison['Set'] == 'Train') |
                                               (model_performance_comparison['Set'] == 'Cross-Validation') |
                                               (model_performance_comparison['Set'] == 'Test')) & 
                                              (model_performance_comparison['Metric']=='Brier.Score') &
                                              (model_performance_comparison['Method']=='AFT_LOGLOGISTIC')]['Value'].values
```


```python
##################################
# Plotting the values for the
# brier scores
# for all models
##################################
brier_score_plot = pd.DataFrame({'AFT_WEIBULL': list(aft_weibull_brier_score),
                                 'AFT_LOGNORMAL': list(aft_lognormal_brier_score),
                                 'AFT_LOGLOGISTIC': list(aft_loglogistic_brier_score)},
                       index = set_labels)
display(brier_score_plot)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AFT_WEIBULL</th>
      <th>AFT_LOGNORMAL</th>
      <th>AFT_LOGLOGISTIC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Train</th>
      <td>0.515148</td>
      <td>0.547041</td>
      <td>0.509528</td>
    </tr>
    <tr>
      <th>Cross-Validation</th>
      <td>0.512583</td>
      <td>0.542583</td>
      <td>0.506538</td>
    </tr>
    <tr>
      <th>Test</th>
      <td>0.537556</td>
      <td>0.577502</td>
      <td>0.533296</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting all the mean absolute errors
# for all models
##################################
brier_score_plot = brier_score_plot.plot.barh(figsize=(10, 6), width=0.90)
brier_score_plot.set_xlim(0.00,1.00)
brier_score_plot.set_title("Model Comparison by Brier Score")
brier_score_plot.set_xlabel("Brier Score")
brier_score_plot.set_ylabel("Data Set")
brier_score_plot.grid(False)
brier_score_plot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
for container in brier_score_plot.containers:
    brier_score_plot.bar_label(container, fmt='%.5f', padding=-75, color='white', fontweight='bold')
```


    
![png](output_228_0.png)
    


# 2. Summary <a class="anchor" id="Summary"></a>

![Project53_Summary.png](attachment:9c73b63a-841a-40e1-9179-83b28e2cc386.png)

# 3. References <a class="anchor" id="References"></a>

* **[Book]** [Clinical Prediction Models](http://clinicalpredictionmodels.org/) by Ewout Steyerberg
* **[Book]** [Survival Analysis: A Self-Learning Text](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) by David Kleinbaum and Mitchel Klein
* **[Book]** [Applied Survival Analysis Using R](https://link.springer.com/book/10.1007/978-3-319-31245-3/) by Dirk Moore
* **[Book]** [Survival Analysis with Python](https://www.taylorfrancis.com/books/mono/10.1201/9781003255499/survival-analysis-python-avishek-nag) by Avishek Nag
* **[Python Library API]** [SciKit-Survival](https://pypi.org/project/scikit-survival/) by SciKit-Survival Team
* **[Python Library API]** [SciKit-Learn](https://scikit-learn.org/stable/index.html) by SciKit-Learn Team
* **[Python Library API]** [StatsModels](https://www.statsmodels.org/stable/index.html) by StatsModels Team
* **[Python Library API]** [SciPy](https://scipy.org/) by SciPy Team
* **[Python Library API]** [Lifelines](https://lifelines.readthedocs.io/en/latest/) by Lifelines Team
* **[Kaggle Project]** [Applied Reliability, Solutions To Problems](https://www.kaggle.com/code/keenanzhuo/applied-reliability-solutions-to-problems) by Keenan Zhuo (Kaggle)
* **[Kaggle Project]** [Survival Models VS ML Models Benchmark - Churn Tel](https://www.kaggle.com/code/caralosal/survival-models-vs-ml-models-benchmark-churn-tel) by Carlos Alonso Salcedo (Kaggle)
* **[Kaggle Project]** [Survival Analysis with Cox Model Implementation](https://www.kaggle.com/code/bryanb/survival-analysis-with-cox-model-implementation/notebook) by Bryan Boulé (Kaggle)
* **[Kaggle Project]** [Survival Analysis](https://www.kaggle.com/code/gunesevitan/survival-analysis/notebook) by Gunes Evitan (Kaggle)
* **[Kaggle Project]** [Survival Analysis of Lung Cancer Patients](https://www.kaggle.com/code/bryanb/survival-analysis-with-cox-model-implementation/notebook) by Sayan Chakraborty (Kaggle)
* **[Kaggle Project]** [COVID-19 Cox Survival Regression](https://www.kaggle.com/code/bryanb/survival-analysis-with-cox-model-implementation/notebook) by Ilias Katsabalos (Kaggle)
* **[Kaggle Project]** [Liver Cirrhosis Prediction with XGboost & EDA](https://www.kaggle.com/code/arjunbhaybhang/liver-cirrhosis-prediction-with-xgboost-eda) by Arjun Bhaybang (Kaggle)
* **[Article]** [Survival Analysis](https://quantdev.ssri.psu.edu/resources/survival-analysis) by Jessica Lougheed and Lizbeth Benson (QuantDev.SSRI.PSU.Edu)
* **[Article]** [Part 1: How to Format Data for Several Types of Survival Analysis Models](https://quantdev.ssri.psu.edu/tutorials/part-1-how-format-data-several-types-survival-analysis-models) by Jessica Lougheed and Lizbeth Benson (QuantDev.SSRI.PSU.Edu)
* **[Article]** [Part 2: Single-Episode Cox Regression Model with Time-Invariant Predictors](https://quantdev.ssri.psu.edu/tutorials/part-2-single-episode-cox-regression-model-time-invariant-predictors) by Jessica Lougheed and Lizbeth Benson (QuantDev.SSRI.PSU.Edu)
* **[Article]** [Part 3: Single-Episode Cox Regression Model with Time-Varying Predictors](https://quantdev.ssri.psu.edu/tutorials/part-3-single-episode-cox-regression-model-time-varying-predictors) by Jessica Lougheed and Lizbeth Benson (QuantDev.SSRI.PSU.Edu)
* **[Article]** [Part 4: Recurring-Episode Cox Regression Model with Time-Invariant Predictors](https://quantdev.ssri.psu.edu/tutorials/part-4-recurring-episode-cox-regression-model-time-invariant-predictors) by Jessica Lougheed and Lizbeth Benson (QuantDev.SSRI.PSU.Edu)
* **[Article]** [Part 5: Recurring-Episode Cox Regression Model with Time-Varying Predictors](https://quantdev.ssri.psu.edu/tutorials/part-5-recurring-episode-cox-regression-model-time-varying-predictors) by Jessica Lougheed and Lizbeth Benson (QuantDev.SSRI.PSU.Edu)
* **[Article]** [Parametric Survival Modeling](https://devinincerti.com/2019/06/18/parametric_survival.html) by Devin Incerti (DevinIncerti.Com)
* **[Article]** [Survival Analysis Simplified: Explaining and Applying with Python](https://medium.com/@zynp.atlii/survival-analysis-simplified-explaining-and-applying-with-python-7efacf86ba32) by Zeynep Atli (Medium)
* **[Article]** [Exploring Time-to-Event with Survival Analysis](https://towardsdatascience.com/exploring-time-to-event-with-survival-analysis-8b0a7a33a7be) by 
Olivia Tanuwidjaja (Medium)
* **[Article]** [Understanding Survival Analysis Models: Bridging the Gap between Parametric and Semiparametric Approaches](https://medium.com/@zynp.atlii/understanding-survival-analysis-models-bridging-the-gap-between-parametric-and-semiparametric-923cdcfc9f05) by Zeynep Atli (Medium)
* **[Article]** [Survival Analysis in Python (KM Estimate, Cox-PH and AFT Model)](https://medium.com/the-researchers-guide/survival-analysis-in-python-km-estimate-cox-ph-and-aft-model-5533843c5d5d) by Rahul Raoniar (Medium)
* **[Article]** [Survival Modeling — Accelerated Failure Time — XGBoost](https://towardsdatascience.com/survival-modeling-accelerated-failure-time-xgboost-971aaa1ba794) by Avinash Barnwal (Medium)
* **[Publication]** [Marginal Likelihoods Based on Cox's Regression and Life Model](https://www.jstor.org/stable/2334538) by Jack Kalbfleisch and Ross Prentice (Biometrika)
* **[Publication]** [Hazard Rate Models with Covariates](https://www.jstor.org/stable/2529934) by Jack Kalbfleisch and Ross Prentice (Biometrics)
* **[Publication]** [Linear Regression with Censored Data](https://www.jstor.org/stable/2335161) by Jonathan Buckley and Ian James (Biometrika)
* **[Publication]** [A Statistical Distribution Function of Wide Applicability](https://www.semanticscholar.org/paper/A-Statistical-Distribution-Function-of-Wide-Weibull/88c37770028e7ed61180a34d6a837a9a4db3b264) by Waloddi Weibull (Journal of Applied Mechanics)
* **[Publication]** [Exponential Survivals with Censoring and Explanatory Variables](https://www.jstor.org/stable/2334539) by Ross Prentice (Biometrika)
* **[Publication]** [The Lognormal Distribution, with Special Reference to its Uses in Economics](https://www.semanticscholar.org/paper/The-Lognormal-Distribution%2C-with-Special-Reference-Corlett-Aitchison/1f59c53ff512fa77e7aee5e6d98b1786c2aaf129) by John Aitchison and James Brown (Economics Applied Statistics)
* **[Course]** [Survival Analysis in Python](https://app.datacamp.com/learn/courses/survival-analysis-in-python) by Shae Wang (DataCamp)


```python
from IPython.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>"))
```


<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>

