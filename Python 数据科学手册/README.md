# 《Python 数据科学手册》

学习《 Python 数据科学手册》的过程中敲的代码，及其官方的源码



### 以下是一些错误的校证

如果诸位在学习《python数据科学手册》的过程中，遇到什么疑难，欢迎留言。
1. scikit-learn.cross_validation 模块变迁
自 `scikit-learn 0.20 `版起，已经用`model_selection`模块代替`cross_validation`模块。因此，复现代码时，`from sklearn.cross_validation import xxx` 时，会报出`ModuleNotFoundError: No module named 'sklearn.cross_validation'`的错误。
P307:
```python
In[15]:from sklearn.cross_validation import train_test_split  # Error
In[15]:from sklearn.model_selection import train_test_split  # Amend
```
P308
```python
In[20]:
from sklearn.mixture import GMM  # Error
from sklearn.mixture import GaussianMixture  # Amend
```
P315
```python
In[5]: from sklearn.cross_validation import train_test_split  # Error
In[5]: from sklearn.model_selection import train_test_split  # Amend
```
P316：
```python
# 用 model_selection 替换 cross_validation
In[7]: from sklearn.cross_validation import cross_val_scroe  # Error
In[7]: from sklearn.model_selection import cross_val_scroe  # Amend

In[8]: from sklearn.cross_validation import cross_val_scroe   # Error
	   scores = cross_val_score(model, X, y, cv=LeaveOneOut(len(X))  # Error
In[8]: from sklearn.model_selection import cross_val_scroe   # Amend
	   scores = cross_val_score(model, X, y, cv=LeaveOneOut()  # Amend，去掉 len(X)
```

2. scikit-learn.learning_curve 模块变迁
自 `scikit-learn 0.20 `版起，已经用`model_selection`模块代替`learning_curve`模块。因此，复现代码时，`from sklearn.learning_curve import xxx` 时，会报出`ModuleNotFoundError: No module named 'sklearn.learning_curve'`的错误。
P321:
```python
In[13]:
from sklearn.learning_curve import validation_curve  # Error
from sklearn.model_selection import validation_curve  # Amend
```
P325
```python
In[17]:
from sklearn.learning_curve import learning_curve  # Error
from sklearn.model_selection import learning_curve  # Amend
```
3. scikit-learn.grid_search 模块变迁
自 `scikit-learn 0.20 `版起，已经用`model_selection`模块代替`grid_search`模块。因此，复现代码时，`from sklearn.grid_search import xxx` 时，会报出`ModuleNotFoundError: No module named 'sklearn.grid_search'`的错误。
P326
```python
In[18]:from sklearn.grid_search import GridSearchCV  # Error
In[18]:from sklearn.model_selection import GridSearchCV  # Amend

In[21]: plt.plot(X_test.ravel(), y_test, hold=True);  # Error
In[21]: plt.plot(X_test.ravel(), y_test);  # Amend, 去掉 hold=True
```
4. 其他错误
P201:
```python
In[4]: from sklearn.gaussian_process import GaussianProcess  # Error
# 计算高斯过程拟合结果
gp = GaussianProcess(corr='cubic', theta=1e-1,....)  # Error
In[4]: from sklearn.gaussian_process import GaussianProcessRegressor  # Amend
# 计算高斯过程拟合结果
gp = GaussianProcessRegressor(corr='cubic', theta=1e-1,....)  # Amend
```
P248:
```python
In[3]:
ax = plt.axes(axisbg='#E6E6E6')  # Error
ax = plt.axes(facecolor='#E6E6E6')  # Amend, axisbg -> facecolor
```
P275:
```python
In[6]:
plt.hist(data[col], normed=True, alpha=0.5)  # Error
plt.hist(data[col], density=True, alpha=0.5)  # Amend, normed -> density
```
P279:
```python
In[13]:
sns.pairplot(iris, hue='species', size=2.5)  # Error
sns.pairplot(iris, hue='species', height=2.5)  # Amend, size -> height
```
P301:
```python
In[2]:
sns.parirplot(iris, hue='species', size=1.5);  # Error
sns.parirplot(iris, hue='species', height=1.5);  # Amend, size 改为 height
```
P349:
```python
In[14]:
weather = pd.read_csv('599021.csv', index_col='DATE', parse_dates=True)  # Error
weather = pd.read_csv('599021.csv', index_col='DATE', parse_dates=True)  # Amend, 599021.csv -> BicycleWeather.csv

In[15]: daily = counts.resample('d', how='sum')  # Error
In[15]: daily = counts.resample('d').sum()  # Amend
```
P361:
```python
In[14]: clf = SVC(kernel='rbf', C=1E6)  # Error
In[14]: clf = SVC(kernel='rbf', C=1E6, gamma='auto')  # Amend, add gamma='auto'
```
P363
```python
In[20]:
from sklearn.decomposition import RandomizedPCA  # Error
pac = RandomizedPCA(n_components=150, whiten=True, random_state=42)  # Error 

from sklearn.decomposition import PCA  # Amend, RandomizedPCA -> PCA
pac = PCA(n_components=150, whiten=True, random_state=42)  # Amend, RandomizedPCA -> PCA
```
P364:
```python
In[21]: from sklearn.cross_validation import train_test_split  # Error
In[21]: from sklearn.model_selection import train_test_split  # Amend

In[22]: from sklearn.grid_search import GridSearchCV  # Error
		grid = GridSearchCV(model, param_grid)        # Error
In[22]: from sklearn.model_selection import GridSearchCV  # Amend
		grid = GridSearchCV(model, param_grid, cv=3)        # Amend, add cv=3
```


