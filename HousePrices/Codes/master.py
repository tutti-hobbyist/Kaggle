import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pathlib
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import preprocessing
import lightgbm

currentDir = pathlib.Path().resolve()
compeTitleDir = currentDir.parent

# 使用データのディレクトリ名指定
usedDataDir = 'original'
dataDir = str(compeTitleDir) + f'/LocalData/{usedDataDir}'
train_csv = dataDir + '/train.csv'
test_csv = dataDir + '/test.csv'

df_train = pd.read_csv(train_csv)
df_test_sub = pd.read_csv(test_csv)
print(df_train.shape, df_test_sub.shape)
print(df_train)

le = preprocessing.LabelEncoder()

train_idx = df_train.shape[0]
test_idx = df_test_sub.shape[0]

df_all = pd.concat([df_train, df_test_sub])

# LGBMRegressorは特徴量として、int, float, booleanのみ有効なのでカテゴリ変数をエンコード
def encoder(dataframe):
    for column in dataframe.columns:
        if dataframe[column].dtype == 'object':
            targetCol = dataframe[column]
            encodedCol = le.fit_transform(targetCol)
        else:
            continue
        dataframe[column] = pd.Series(encodedCol)
    return dataframe

df_all = encoder(df_all)

df_train = df_all.iloc[:train_idx, :]
df_test_sub = df_all.iloc[train_idx:, :]
df_test_sub = df_test_sub.dropna(how='all', axis=1)
print(df_test_sub)

df_train_x = df_train.iloc[:, :-1]
df_train_y = df_train.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.2, random_state=0)

regressor = lightgbm.LGBMRegressor()

model = regressor.fit(x_train, y_train)
y_predicted = model.predict(x_test)

plt.scatter(y_test, y_predicted)
plt.xlabel('Test data')
plt.ylabel('Predicted data')

# 検証用のテストデータによる精度確認
r2 = r2_score(y_test, y_predicted)
rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
print(r2, '\n', rmse)

# テストデータに対して予測
y_test_predicted = model.predict(df_test_sub)

df_test_sub_id = df_test_sub.loc[:, 'Id']
df_predict = pd.DataFrame(pd.Series(y_test_predicted), columns=['SalePrice'])
df_submission = pd.concat([df_test_sub_id, df_predict], axis=1)
print(df_submission)

df_submission.to_csv(str(compeTitleDir) + '/submissions/sub_ver1.csv', index=False)

