"""实现固定预测规则、线性回归和 Ridge 回归。"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def predict_last_value(X):
    """使用窗口最后一个观测作为预测。"""
    return X[:, -1]


def predict_window_mean(X):
    """使用窗口均值作为预测。"""
    return np.mean(X, axis=1)


def make_lag_linear():
    """建立标准化加普通线性回归管道。"""
    return make_pipeline(StandardScaler(), LinearRegression())


def make_ridge(alpha):
    """建立标准化加 Ridge 回归管道。"""
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def fit_model(model, X, y):
    """在训练窗口上拟合模型。"""
    return model.fit(X, y)


def predict_model(model, X):
    """使用已拟合模型预测新窗口。"""
    return model.predict(X)
