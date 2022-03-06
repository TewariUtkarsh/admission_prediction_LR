import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split

from pandas_profiling import ProfileReport
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


class lin_reg:
    def __init__(self, x_test):
        self.r_sq = 0.0
        self.adj_r_sq = 0.0
        self.x_test_str = x_test
        self.x_test = []
        self.y_pred=0.0

    def perform(self):

        df = pd.read_csv('Admission_Prediction.csv')

        # 1. GRE Score is having Nan/null value
        df['GRE Score'] = df['GRE Score'].fillna(df['GRE Score'].mean())

        # 2. TOEFL Score is having NaN/null values
        df['TOEFL Score'].fillna(value=df['TOEFL Score'].mean(), inplace=True)

        # 3. University Rating is having NaN/null values:
        df['University Rating'].fillna(df['University Rating'].mean(), inplace=True)

        df.drop('Serial No.', axis=1, inplace=True)

        y = df[['Chance of Admit']]
        x = df.drop(columns=['Chance of Admit'])
        # print(x)

        scaler = StandardScaler()
        std_x1 = scaler.fit_transform(x)
        # print(std_x)

        # np.random.seed
        x_train, x_test, y_train, y_test = train_test_split(std_x1, y, test_size=0.15, random_state=100)

        lr = LinearRegression()
        lr.fit(x_train, y_train)

        std_x = scaler.transform([self.x_test])
        self.y_pred = lr.predict(std_x)
        # self.y_pred = lr.predict(self.x_test)
        self.r_sq = lr.score(x_test, y_test)

        def adj_r2(x, y):
            r2 = lr.score(x, y)
            n = x.shape[0]
            p = y.shape[1]
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            return adjusted_r2

        self.adj_r_sq = adj_r2(x_test, y_test)

        # print(f"R Square: {r_square}\nAdjusted R Square: {adj_r_square}")

        # print(df)
        profile = ProfileReport(df, minimal=True)
        profile.to_file("profile.html")

    def input_std_scalar(self):
        scalar = StandardScaler()
        std_x = scalar.transform([self.x_test])
        return std_x

    def input_parser(self):
        # We can also use eval
        inp = self.x_test_str.split(",")
        self.x_test = [eval(i) for i in inp]
        print(self.x_test)
        # self.x_test = map(int, self.x_test.split(","))