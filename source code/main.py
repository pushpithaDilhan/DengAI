import csv
from sklearn import linear_model
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LassoLarsCV
import random
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools import eval_measures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from statsmodels.tsa.stattools import adfuller


###------------------------ Read training features -------------------------------------------###

raw_data = {'ndvi_se':[],
            'ndvi_sw':[],
            'ndvi_ne':[],
            'ndvi_nw':[],
            'avg_air_temp':[],
            'dew_point':[],
            'humidity_percentage':[],
            'total_precipitation':[]
            }

input_file = csv.DictReader(open("training_data_features.csv"))

for row in input_file:
    if row['ndvi_se'] == '':
        raw_data['ndvi_se'].append('')
    else:
        raw_data['ndvi_se'].append(round(float(row['ndvi_se']), 3))
    if row['ndvi_sw'] == '':
        raw_data['ndvi_sw'].append('')
    else:
        raw_data['ndvi_sw'].append(round(float(row['ndvi_sw']), 3))
    if row['ndvi_ne'] == '':
        raw_data['ndvi_ne'].append('')
    else:
        raw_data['ndvi_ne'].append(round(float(row['ndvi_ne']), 3))
    if row['ndvi_nw'] == '':
        raw_data['ndvi_nw'].append('')
    else:
        raw_data['ndvi_nw'].append(round(float(row['ndvi_nw']), 3))

    if row['reanalysis_avg_temp_k'] == '':
        raw_data['avg_air_temp'].append('')
    else:
        raw_data['avg_air_temp'].append(round(float(row['reanalysis_avg_temp_k']), 3))

    if row['reanalysis_dew_point_temp_k'] == '':
        raw_data['dew_point'].append('')
    else:
        raw_data['dew_point'].append(round(float(row['reanalysis_dew_point_temp_k']), 3))

    if row['reanalysis_relative_humidity_percent'] == '':
        raw_data['humidity_percentage'].append('')
    else:
        raw_data['humidity_percentage'].append(round(float(row['reanalysis_relative_humidity_percent']), 3))

    if row['station_precip_mm'] == '':
        raw_data['total_precipitation'].append('')
    else:
        raw_data['total_precipitation'].append(round(float(row['station_precip_mm']), 1))

####----------------- Data Preprocessing ----------------------------------------------####

for i in raw_data:
    # calculate the mean
    sum = 0
    for index in range(len(raw_data[i])):
        if not raw_data[i][index] == '':
            sum += raw_data[i][index]
    mean = sum / float(len(raw_data[i]))

    # fill missing values
    for index in range(len(raw_data[i])):
        if raw_data[i][index] == '':
            if index == len(raw_data[i]) - 1 or raw_data[i][index+1] == '':
                raw_data[i][index] = mean
            else:
                raw_data[i][index] = (raw_data[i][index-1] + raw_data[i][index+1])/2.0

###------------------ Read target file -----------------------------------------####

target = []

input_file = csv.DictReader(open("training_data_labels.csv"))

for row in input_file:
    target.append(row['total_cases'])

target_df =  pd.DataFrame(target)

training_df = pd.DataFrame(raw_data, columns = ['ndvi_se', 'ndvi_sw', 'ndvi_ne', 'ndvi_nw','avg_air_temp','dew_point','humidity_percentage','total_precipitation'])

###------------------------- Machine learning -------------------------------------------------###

# Negative Binomial Regression Model
# alpha_grid = np.linspace(0.01, 2, 100)
#
# best_alpha = 0;
# best_mae = 1000
#
# for alpha in alpha_grid:
#     model = smf.glm(formula=raw_data, data=target,
#                     family=sm.families.NegativeBinomial(alpha=alpha))
#     nb_model = model.fit(training_df,target_df)

# Linear Regression
# lm = linear_model.LinearRegression()
# liner_model = lm.fit(training_df,target_df)

# Poisson model
# model = smf.glm(formula = raw_data, data = target, family = sm.families.Poisson())
# poisson_model = model.fit(training_df,target_df)

# Ridge Regression
# ridge = Ridge(random_state=10)
# param_grid = {'alpha': np.logspace(-5, 3, 50)}
# model = GridSearchCV(ridge, param_grid, scoring='neg_mean_absolute_error', cv=10)
# ridge_reg = model.fit(training_df,target_df)

# Lasso regression
# lasso = Lasso(random_state=10)
# param_grid = {'alpha': np.logspace(-5, 3, 50),}
# model = GridSearchCV(lasso, param_grid, scoring='neg_mean_absolute_error', cv=10)
# lassoreg = model.fit(training_df,target_df)

# Time series regression
# series = map(float, raw_data)
# model = sm.tsa.ARIMA(series, order = (2, 1, 1), exog = target)
# model_fit = model.fit(training_df,target_df)

###------------------------ Read testing features -------------------------------------------###

testing_data = {'ndvi_se':[],
            'ndvi_sw':[],
            'ndvi_ne':[],
            'ndvi_nw':[],
            'avg_air_temp':[],
            'dew_point':[],
            'humidity_percentage':[],
            'total_precipitation':[]
            }

input_file_test = csv.DictReader(open("test_data_features.csv"))

for row in input_file_test:
    if row['ndvi_se'] == '':
        testing_data['ndvi_se'].append('')
    else:
        testing_data['ndvi_se'].append(round(float(row['ndvi_se']), 3))
    if row['ndvi_sw'] == '':
        testing_data['ndvi_sw'].append('')
    else:
        testing_data['ndvi_sw'].append(round(float(row['ndvi_sw']), 3))
    if row['ndvi_ne'] == '':
        testing_data['ndvi_ne'].append('')
    else:
        testing_data['ndvi_ne'].append(round(float(row['ndvi_ne']), 3))
    if row['ndvi_nw'] == '':
        testing_data['ndvi_nw'].append('')
    else:
        testing_data['ndvi_nw'].append(round(float(row['ndvi_nw']), 3))

    if row['reanalysis_avg_temp_k'] == '':
        testing_data['avg_air_temp'].append('')
    else:
        testing_data['avg_air_temp'].append(round(float(row['reanalysis_avg_temp_k']), 3))

    if row['reanalysis_dew_point_temp_k'] == '':
        testing_data['dew_point'].append('')
    else:
        testing_data['dew_point'].append(round(float(row['reanalysis_dew_point_temp_k']), 3))

    if row['reanalysis_relative_humidity_percent'] == '':
        testing_data['humidity_percentage'].append('')
    else:
        testing_data['humidity_percentage'].append(round(float(row['reanalysis_relative_humidity_percent']), 3))

    if row['station_precip_mm'] == '':
        testing_data['total_precipitation'].append('')
    else:
        testing_data['total_precipitation'].append(round(float(row['station_precip_mm']), 1))


####----------------- Data Preprocessing ----------------------------------------------####

for i in testing_data:
    # calculate the mean
    sum = 0
    for index in range(len(testing_data[i])):
        if not testing_data[i][index] == '':
            sum += testing_data[i][index]
    mean = sum / float(len(testing_data[i]))

    # fill missing values
    for index in range(len(testing_data[i])):
        if testing_data[i][index] == '':
            if index == len(testing_data[i]) - 1 or testing_data[i][index+1] == '':
                testing_data[i][index] = mean
            else:
                testing_data[i][index] = (testing_data[i][index-1] + testing_data[i][index+1])/2.0

testing_df = pd.DataFrame(testing_data, columns = ['ndvi_se', 'ndvi_sw', 'ndvi_ne', 'ndvi_nw','avg_air_temp','dew_point','humidity_percentage','total_precipitation'])

predictions = lm.predict(testing_df)
predictions_array = []

for i in predictions:
    predictions_array.append(int(i[0]))

print len(predictions_array)

# predict the test data set
results = [["city","year","weekofyear","total_cases"]]

with open('submission_format.csv', 'rb') as test_file:
    test_reader = csv.reader(test_file)
    next(test_reader, None)
    id = 0
    for row in test_reader:
        data = []
        for index in range(len(row)-1):
            data.append(row[index])
        # append the answer - random number for now
        if predictions_array[id] < 0:
            data.append(0)
        else:
            data.append(predictions_array[id])
        results.append(data)
        id += 1

print len(results)

print predictions_array
# write results to the submit.csv
with open('output/submit.csv', "wb") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for line in results:
        writer.writerow(line)
