# Logistic-Regression. Gradient Ascent

In this notebook I write a little tutorial on Logisitc Regression (LR). Firstly, we will dive into theory and derive LR optimization problem and then will write the code for the algorithm of gradient ascent. Using gradient ascent we will find parameters of probability distribution (Bernoulli distribution) that maximizes the likelyhood of observed data. After finding the parameters we will plot the logistic function.

<img src="https://render.githubusercontent.com/render/math?math=\textrm{I}">. Our goal is to build a classification model, that predicts class of <img src="https://render.githubusercontent.com/render/math?math=y">, which is <img src="https://render.githubusercontent.com/render/math?math=0"> or <img src="https://render.githubusercontent.com/render/math?math=1">.

We want to use linear model <img src="https://render.githubusercontent.com/render/math?math=w^tx"> to make predictions.

In case of Linear Regression: <img src="https://render.githubusercontent.com/render/math?math=y = w^tx, -\infty \leq x \leq +\infty, -\infty \leq y \leq +\infty">

Since the linear model is unbounded we cannot fit such model as <img src="https://render.githubusercontent.com/render/math?math=y = bias %2B w^tx">, because it would give nonsenical results, when the output is binary.

We need to tranform LHS (the dependent variable) to match the range of RHS (feature variables).

1. First step is to use <img src="https://render.githubusercontent.com/render/math?math=P"> (probability of success, i.e. <img src="https://render.githubusercontent.com/render/math?math=P(y=1|x)"> instead of <img src="https://render.githubusercontent.com/render/math?math=y">. <img src="https://render.githubusercontent.com/render/math?math=0\leq P \leq 1">.

2. Now we introduce the odds: <img src="https://render.githubusercontent.com/render/math?math=O = \frac{P}{1-P}$. $0\leq O \leq %2B \infty">.

3. Finally taking <img src="https://render.githubusercontent.com/render/math?math=log"> of odds will yield a variable that ranges from negative to positive infinity.
    <img src="https://render.githubusercontent.com/render/math?math=-\infty \leq ln(O) \leq %2B \infty">
    
Now we have a model that makes sense:

<img src="https://render.githubusercontent.com/render/math?math=ln(\frac{P}{1-P}) = w^tx">. Both LHS and RHS has the same range.

We can derive <img src="https://render.githubusercontent.com/render/math?math=P"> from this equation:

<img src="https://render.githubusercontent.com/render/math?math=P = \frac{1}{1 %2B e^{-(w^tx)}}">

To sum up: we build a linear model that can predict the logarithm of odds <img src="https://render.githubusercontent.com/render/math?math=ln(O)">, and that can be transformed to probability. Thus our model predict the probability of <img src="https://render.githubusercontent.com/render/math?math=y"> belonging to class 1.

<img src="https://render.githubusercontent.com/render/math?math=\textrm{II}">. Now to estiamte the probability we first need to estimate vector of parameters w of our model. In order to estimate parameters we will maximize the likelyhood of our data. 

First we need to establish by what hypothesis (distribuition) the data was generated and then find values of parameters that maximizes the likelyhood of data. 

We assumed:

<img src="https://render.githubusercontent.com/render/math?math=P(y=1|x) = \frac{1}{1 %2B e^{-(w^tx)}}">, thus:

<img src="https://render.githubusercontent.com/render/math?math=P(y=0|x) = 1 - \frac{1}{1 %2B e^{-(w^tx)}}">, by law of total probability.

We have two classes and we know the probability of each.
A Bernoulli random variable has two possible outcomes: 0 or 1.
(A binomial distribution is the sum of independent and identically distributed Bernoulli random variables.
In general, if there are n Bernoulli trials, then the sum of those trials is binomially distributed with parameters n and p.) So each <img src="https://render.githubusercontent.com/render/math?math=y"> is a Bernoulli random variable, assuming that all <img src="https://render.githubusercontent.com/render/math?math=y">'s are independent we can write the likelyhood of data as: 

### <img src="https://render.githubusercontent.com/render/math?math=L(w) = \prod_{i}^{n}P^{y_i}(1-P)^{1-y_i}">

For more convient further optimization let's take log of likelyhood function:

### <img src="https://render.githubusercontent.com/render/math?math=l(w) = ln(\prod_{i}^{n}P^{y_i}(1-P)^{1-y_i}) = \sum_i^n ln(P^{y_i}(1-P)^{1-y_i})"> 

### <img src="https://render.githubusercontent.com/render/math?math== \sum_i^n y_i ln(P)+(1-y_i)ln(1-P)">

### <img src="https://render.githubusercontent.com/render/math?math=\sum_i^n y_i ln(\frac{1}{1 %2B e^{-w^tx}}) %2B (1-y_i)ln(\frac{1}{1 %2B e^{w^tx}})">

Log likelyhood derivative:

### <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial l(w)}{\partial w_j} = (\frac{y}{g(w^tx)}-\frac{1-y}{1-g(w^tx)})\frac{\partial g(w^tx)}{\partial w_j}"> 

### <img src="https://render.githubusercontent.com/render/math?math==(\frac{y}{g(w^tx)}-\frac{1-y}{1-g(w^tx)})g(w^tx)(1-g(w^tx)) \frac{\partial w^tx}{\partial w_j}">

### <img src="https://render.githubusercontent.com/render/math?math==(\frac{y}{g(w^tx)}-\frac{1-y}{1-g(w^tx)})g(w^tx)(1g(w^tx))x_j">

### <img src="https://render.githubusercontent.com/render/math?math==(y(1-g(w^tx))-(1-y)g(w^tx))x_j">

### <img src="https://render.githubusercontent.com/render/math?math==(y - g(w^tx))x_j">

Update rule:

### <img src="https://render.githubusercontent.com/render/math?math=w_{j%2B1} = w_j %2B \alpha(y - g(w^tx))x_j">

plugging in sigmoid function:

### <img src="https://render.githubusercontent.com/render/math?math=w_{j %2B 1} = w_j %2B \alpha(y - \frac{1}{1 %2B e^{-w^tx}})x_j">


```python
%reload_ext watermark
%watermark -u -d -v -p matplotlib,numpy
%matplotlib inline
```

    last updated: 2019-09-04 
    
    CPython 3.7.3
    IPython 7.6.1
    
    matplotlib 3.1.0
    numpy 1.16.4



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%config IPCompleter.greedy=True
```


```python
import os
os.chdir('/Users/Eddvibe/Developer/python/data')
```


```python
ds = pd.read_csv('data_logistic.csv', header=None)
missing_values_count = ds.isnull().sum()
missing_values_count
```




    0    0
    1    0
    2    0
    dtype: int64



Data has some outliers:


```python
plt.scatter(ds.iloc[:,1], ds.iloc[:,2], c=ds.iloc[:,0].to_list())
plt.show()
```


![png](output_7_0.png)



```python
plt.scatter(ds[ds[2]>-2.5].iloc[:,1], ds[ds[2]>-2.5].iloc[:,2], c=ds[ds[2]>-2.5].iloc[:,0].tolist())
plt.show()
```


![png](output_8_0.png)



```python
ds = pd.read_csv('data_logistic.csv', header=None)
ds[0].replace(to_replace=-1, value=0, inplace=True)
ds = ds[ds[2]>-2.5]
ds = ds.round(2)
ds = ds.values
X = ds[:,1:]
y = ds[:,0]
```


```python
# Likelyhood function with bias

def LL_b(X, y, w, b):
    ll = 0
    for i in range(len(X)):
        if y[i] == 1:
            ll += float( np.log( 1/(1+np.e**(-(np.dot(w, X[i, :])+b )) ) ) )
        else:
            ll += float( (np.log( 1/(1+np.e**(np.dot(w, X[i, :])+b))) ) )
    
    return ll
```


```python
# Likelyhood function w/o bias

def LL(X, y, w):
    ll = 0
    for i in range(len(X)):
        if y[i] == 1:
            ll += float( np.log( 1/(1+np.e**(-(np.dot(w, X[i, :]))) ) ) )
        else:
            ll += float( (np.log( 1/(1+np.e**(np.dot(w, X[i, :])))) ) )
    
    return ll
```


```python
# Algorithm of gradient ascent with bias

def gradient_ascent_bias(X, y, alpha, ep=0.000001, max_iter=10000):
    iter = 0
    converged = False
    #w = np.array([0.05, 0.05])
    #b = 0.05 
    w = np.random.random(X.shape[1])
    b = np.random.random()
    grad_w = 0
    grad_b = 0
    L_old = LL_b(X, y, w, b)
    
    while not converged:
        for i in range(len(X)):
            grad_w += (y[i]-(1/(1+np.e**(-(np.dot(w, X[i, :])+b)))))*X[i,:]
            grad_b += (y[i]-(1/(1+np.e**(-(np.dot(w, X[i, :])+b)))))
        w_new = w + alpha * grad_w
        w_new = w_new.reshape(2,)
        b_new = b + alpha * grad_b
        b_new = b_new.reshape(1,)
        L_new = LL_b(X, y, w_new, b_new)
        
        if abs(L_new - L_old) <= ep:
            print('Converged, iterations:', iter)
            converged = True
            w = w_new
            b = b_new
        
        iter += 1
        L_old = L_new
        w = w_new
        b = b_new
        grad_w = 0
        grad_b = 0
        
        if iter == max_iter:
            print('Max interations exceeded')
            converged = True
        
    return w, b
```


```python
gradient_ascent_bias(X, y, 0.001)
```

    Converged, iterations: 4215





    (array([2.71856672, 2.17691157]), array([-4.68999261]))




```python
# Algorithm of gradient ascent w/o bias
def gradient_ascent(X, y, alpha, ep=0.000001, max_iter=1000):
    iter = 0
    converged = False
    #w = np.array([0.05, 0.05])
    w = np.random.random(X.shape[1])
    grad_w = 0
    L_old = LL(X, y, w)

    while not converged:
        for i in range(len(X)):
            grad_w += (y[i]-(1/(1+np.e**(-np.dot(w, X[i, :])))))*X[i,:]
        w_new = w + alpha * grad_w
        w_new = w_new.reshape(2,)
        L_new = LL(X, y, w_new)
        
        if abs(L_new - L_old) <= ep:
            print('Converged, iterations:', iter)
            converged = True
            w = w_new

        iter += 1
        L_old = L_new
        w = w_new
        grad_w = 0
        
        if iter == max_iter:
            print('Max interations exceeded')
            converged = True
        
    return w
```


```python
gradient_ascent(X, y, 0.001)
```

    Converged, iterations: 168





    array([0.79763801, 0.55227569])



### Combining two algorithms in one function:


```python
def optimizer(LL, gradient_ascent_bias, gradient_ascent, X, y, bias):
    if bias == True:
        return gradient_ascent_bias(X, y, 0.01)
    else:
        return gradient_ascent(X, y, 0.01)    
```


```python
w, b = optimizer(LL, gradient_ascent_bias, gradient_ascent, X, y, bias=True)
print(w, b)
```

    Converged, iterations: 538
    [2.72861453 2.18488675] [-4.70863302]


### Comparing our results with sklearn library:


```python
from sklearn.linear_model import LogisticRegression

clf_bias = LogisticRegression(penalty='none', fit_intercept=True, solver='sag')
clf_bias.fit(X,y)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='none',
                       random_state=None, solver='sag', tol=0.0001, verbose=0,
                       warm_start=False)




```python
print(clf_bias.coef_, clf_bias.intercept_)
```

    [[2.73451569 2.18958694]] [-4.71958526]



```python
clf = LogisticRegression(penalty='none', fit_intercept=False, solver='sag')
clf.fit(X,y)

print(clf.coef_)
```

    [[0.79684757 0.55286761]]


### Plotting the decision boundary:


```python
xx = np.linspace(-2, 3)
a = -w[0]/w[1]
yy = a*xx - b/w[1]

plt.scatter(ds[:,1], ds[:,2], c=ds[:,0].tolist())
plt.plot(xx, yy)
plt.grid()
plt.show()
```


![png](output_24_0.png)


### Plotting 3d surface of logistic regression:


```python
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import chart_studio.plotly
import plotly.graph_objs as go
import plotly.offline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
```


```python
#w, b = optimizer(LL, gradient_ascent_bias, gradient_ascent, X, y, bias=True)
def logistic_func(xx, yy, w, b):
    ll = []
    for i in range(len(xx)):
        ll.append(1/(1+np.e**(-(w[0]*xx[i]+w[1]*yy[i]+b))))
    
    return ll
```


```python
xx = np.arange(-3,3, 0.1)
yy = np.arange(-3,3, 0.1)

XX, YY = np.meshgrid(xx,yy)

LL = logistic_func(XX, YY, w, b)
LL = np.array(LL)
```


```python
fig = plt.figure(figsize = (10,6))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(XX, YY, LL, rstride=1, cstride=1, cmap=cm.RdBu, linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)
ax.scatter(ds[:,1], ds[:,2], y, c=ds[:,0].tolist())
plt.show()
```


![png](output_29_0.png)



```python
im = imshow(LL,cmap=cm.RdBu)
cset = contour(LL,arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2)
clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
colorbar(im) # adding the colobar on the right
title('Contour lines of Logisitic Regression surface')
show()
```


![png](output_30_0.png)






