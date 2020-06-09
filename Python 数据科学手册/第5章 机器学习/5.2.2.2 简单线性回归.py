# !/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
Created on Fri Oct 11 19:17:22 2019

@author: SHI YI
"""

# 1.选择模型类
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)

# 2.选择模型超参数
model = LinearRegression(fit_intercept=True)

# 3.将数据整理成「特征矩阵」和「目标数组」
X = x[ : , np.newaxis]

# 4.用模型「拟合」数据
model.fit(X, y)
print(f"model.coef_: {model.coef_[0]}")
print(f"model.intercept_: {model.intercept_}")

# 5.预测新数据的标签
xfit = np.linspace(-1, 11)
Xfit = xfit[ : , np.newaxis]
yfit = model.predict(Xfit)

plt.scatter(x, y)
plt.plot(xfit, yfit)
