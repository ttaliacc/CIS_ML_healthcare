#!/usr/bin/env python
# coding: utf-8

# In[14]:


get_ipython().system('pip install numpy pandas matplotlib seaborn wheel pandas_profiling jupyter notebook -i https://pypi.tuna.tsinghua.edu.cn/simple')


# In[15]:


get_ipython().system('pip install graphviz pydotplus ')


# In[16]:


get_ipython().system('pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple')


# In[17]:


get_ipython().system('pip install pdpbox eli5 -i https://pypi.tuna.tsinghua.edu.cn/simple')


# In[18]:


get_ipython().system('pip install shap -i https://pypi.tuna.tsinghua.edu.cn/simple')


# In[19]:


import pandas as pd


# In[183]:


df = pd.read_csv("data.csv")


# In[184]:


df.head(10)


# In[185]:


df.shape


# In[186]:


df.tail()


# In[187]:


df


# In[71]:


df.describe()


# In[72]:


import pandas_profiling


# In[73]:


profile = pandas_profiling.ProfileReport(df)


# In[74]:


profile


# In[1]:


import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# In[4]:


df.info()


# In[6]:


def randSplit(dataSet, rate=0.9):
    n = dataSet.shape[0]
    m = int(n*rate)
    train = dataSet.iloc[:m,:]
    test = dataSet.iloc[m:,:]
    test .index = range(test.shape[0]) 
    return train,test


# In[7]:


train, test = randSplit(df)


# In[8]:


train.shape


# In[9]:


train


# In[10]:


test.shape


# In[11]:


test


# In[ ]:


acc, df_stay = stayClass(train, test, 5)
df_stay.head()


# In[1]:


def randSplit(dataSet, rate=0.9):
    n = dataSet.shape[0]
    m = int(n*rate)
    train = dataSet.iloc[:m,:]
    test = dataSet.iloc[m:,:]
    test .index = range(test.shape[0]) 
    return train,test


# In[ ]:


df = pd.read_csv("dataset5.csv")


# In[ ]:


df.shape


# In[ ]:


import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# In[ ]:


df.corr()


# In[ ]:


df.info()


# In[ ]:


df = pd.read_csv('dataset5000.csv')


# In[ ]:


df.info()


# In[ ]:


df.describe


# In[ ]:


def randSplit(dataSet, rate=0.9):
    n = dataSet.shape[0]
    m = int(n*rate)
    train = dataSet.iloc[:m,:]
    test = dataSet.iloc[m:,:]
    test .index = range(test.shape[0]) 
    return train,test


# In[ ]:


train, test = randSplit(df)


# In[ ]:


train.shape


# In[ ]:


train


# In[ ]:


train.info()


# In[ ]:


import numpy as np
def stayClass(train, test, k):
    n = train. shape[1]-1   
    m = test. shape[0]   
    result = []              
    
    for i in range(m):
        train=pd.DataFrame(train,dtype=np.float)
        test=pd.DataFrame(test,dtype=np.float)
        dist = np.linalg.norm(train.iloc[:, :n]-test.iloc[i, :n],ord=1)
        dist_1 = pd.DataFrame({'dist':dist, 'labels':(train.iloc[:, n])})  
        dr = dist_1.sort_values(by = 'dist')[: k]  
        re = dr.loc[:, 'labels'].value_counts()   
        result.append(re.index[0])    
    result = pd.Series(result)
    res = test.copy()
    res.loc[:, 'predict'] = result
    acc = (res.iloc[:,-1]==res.iloc[:,-2]).mean()
    print('accuracy{}'.format(acc))
    return acc, res


# In[ ]:


acc, df_stay = stayClass(train, test, 30)
df_stay.head()
train=pd.DataFrame(train,dtype=np.float)
test=pd.DataFrame(test,dtype=np.float)


# In[ ]:


import matplotlib.pyplot as plt
score = []
krange = range(1,20)

for i in krange:
    acc,re = stayClass(train,test,i)
    score.append(acc)
plt.plot(krange,score);
bestK = krange[score.index(max(score))]
print(bestK)
print(max(score))


# In[ ]:


df = pd.read_csv('datasetone.csv')


# In[ ]:


def randSplit(dataSet, rate=0.9):
    n = dataSet.shape[0]
    m = int(n*rate)
    train = dataSet.iloc[:m,:]
    test = dataSet.iloc[m:,:]
    test .index = range(test.shape[0]) 
    return train,test


# In[ ]:


train, test = randSplit(df)


# In[ ]:


train.shape


# In[ ]:


train.info()


# #Manhattan distance

# In[ ]:


import numpy as np
train=pd.DataFrame(train,dtype=np.float)
test=pd.DataFrame(test,dtype=np.float)
def stayClass(train, test, k):
    n = train. shape[1]-1   
    m = test. shape[0]    
    result = []            
    
    for i in range(m):
        train=pd.DataFrame(train,dtype=np.float)
        test=pd.DataFrame(test,dtype=np.float)
        dist = np.linalg.norm(train.iloc[:, :n]-test.iloc[i, :n],ord=1)
        dist_1 = pd.DataFrame({'dist':dist, 'labels':(train.iloc[:, n])}) 
        dr = dist_1.sort_values(by = 'dist')[: k] 
        re = dr.loc[:, 'labels'].value_counts()  
        result.append(re.index[0])    
    result = pd.Series(result)
    res = test.copy()
    res.loc[:, 'predict'] = result 
    acc = (res.iloc[:,-1]==res.iloc[:,-2]).mean()  
    print('accuracy{}'.format(acc))
    return acc, res


# In[ ]:


acc, df_stay = stayClass(train, test, 40)
df_stay.head()


# In[ ]:


import matplotlib.pyplot as plt
score = []
krange = range(1,20)

for i in krange:
    acc,re = stayClass(train,test,i)
    score.append(acc)
plt.plot(krange,score);
bestK = krange[score.index(max(score))]
print(bestK)
print(max(score))


# In[ ]:


import numpy as np
def stayClass(train, test, k):
    train=pd.DataFrame(train,dtype=np.float)
    test=pd.DataFrame(test,dtype=np.float)
    n = train. shape[1]-1   
    m = test. shape[0]   
    result = []              
    for i in range(m):
        dist = list((((train.iloc[:, :n] - test.iloc[i, :n]) **2).sum(1))**.5)  
        dist_1 = pd.DataFrame({'dist':dist, 'labels':(train.iloc[:, n])}) 
        dr = dist_1.sort_values(by = 'dist')[: k]  
        re = dr.loc[:, 'labels'].value_counts()  
        result.append(re.index[0])    
    result = pd.Series(result)
    res = test.copy()
    res.loc[:, 'predict'] = result 
    acc = (res.iloc[:,-1]==res.iloc[:,-2]).mean()  
    print('accuracy{}'.format(acc))
    return acc, res


# In[ ]:


import matplotlib.pyplot as plt
score = []
krange = range(1,30)

for i in krange:
    acc,re = stayClass(train,test,i)
    score.append(acc)
plt.plot(krange,score);
bestK = krange[score.index(max(score))]
print(bestK)
print(max(score))


# # class features are chosen according to correlation2

# In[ ]:


df = pd.read_csv("dataset2.csv")


# In[ ]:


df.shape


# In[ ]:


import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# In[ ]:


df.corr()


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


def randSplit(dataSet, rate=0.9):
    n = dataSet.shape[0]
    m = int(n*rate)
    train = dataSet.iloc[:m,:]
    test = dataSet.iloc[m:,:]
    test .index = range(test.shape[0]) 
    return train,test


# In[ ]:


train, test = randSplit(df)


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


test


# In[ ]:


import numpy as np
def stayClass(train, test, k):
    train=pd.DataFrame(train,dtype=np.float)
    test=pd.DataFrame(test,dtype=np.float)
    n = train. shape[1]-1   
    m = test. shape[0]   
    result = []              
    for i in range(m):
        dist = list((((train.iloc[:, :n] - test.iloc[i, :n]) **2).sum(1))**.5)   
        dist_1 = pd.DataFrame({'dist':dist, 'labels':(train.iloc[:, n])})  
        dr = dist_1.sort_values(by = 'dist')[: k]   
        re = dr.loc[:, 'labels'].value_counts()   
        result.append(re.index[0])    
    result = pd.Series(result)
    res = test.copy()
    res.loc[:, 'predict'] = result  
    acc = (res.iloc[:,-1]==res.iloc[:,-2]).mean()   
    print('accuracy{}'.format(acc))
    return acc, res


# In[ ]:


acc, df_stay = stayClass(train, test, 21)
df_stay.head()


# In[ ]:


score = []
krange = range(1,30)

for i in krange:
    acc,re = stayClass(train,test,i)
    score.append(acc)
plt.plot(krange,score);
bestK = krange[score.index(max(score))]
print(bestK)
print(max(score))


# # features are chosen according to correlation3

# In[ ]:


def randSplit(dataSet, rate=0.9):
    n = dataSet.shape[0]
    m = int(n*rate)
    train = dataSet.iloc[:m,:]
    test = dataSet.iloc[m:,:]
    test .index = range(test.shape[0]) 
    return train,test


# In[ ]:


def randSplit(dataSet, rate=0.9):
    n = dataSet.shape[0]
    m = int(n*rate)
    train = dataSet.iloc[:m,:]
    test = dataSet.iloc[m:,:]
    test .index = range(test.shape[0]) 
    return train,test


# In[ ]:


train, test = randSplit(df)


# In[ ]:


train.shape


# In[ ]:


train


# In[ ]:


test.shape


# In[ ]:


test


# In[ ]:


import numpy as np
def stayClass(train, test, k):
    train=pd.DataFrame(train,dtype=np.float)
    test=pd.DataFrame(test,dtype=np.float)
    n = train. shape[1]-1   
    m = test. shape[0]     
    result = []                
    for i in range(m):
        dist = list((((train.iloc[:, :n] - test.iloc[i, :n]) **2).sum(1))**.5)    
        dist_1 = pd.DataFrame({'dist':dist, 'labels':(train.iloc[:, n])})   
        dr = dist_1.sort_values(by = 'dist')[: k] 
        re = dr.loc[:, 'labels'].value_counts()  
        result.append(re.index[0])    
    result = pd.Series(result)
    res = test.copy()
    res.loc[:, 'predict'] = result 
    acc = (res.iloc[:,-1]==res.iloc[:,-2]).mean()   
    print('accuracy{}'.format(acc))
    return acc, res


# In[ ]:


acc, df_stay = stayClass(train, test, 50)
df_stay.head()


# In[ ]:


import matplotlib.pyplot as plt
score = []
krange = range(1,30)

for i in krange:
    acc,re = stayClass(train,test,i)
    score.append(acc)
plt.plot(krange,score);
bestK = krange[score.index(max(score))]
print(bestK)
print(max(score))


# In[ ]:


import matplotlib.pyplot as plt
score = []
krange = range(1,30)

for i in krange:
    acc,re = stayClass(train,test,i)
    score.append(acc)
plt.plot(krange,score);
bestK = krange[score.index(max(score))]
print(bestK)
print(max(score))


# In[ ]:


df = pd.read_csv('dataset5000.csv')


# In[ ]:


df.info()


# In[ ]:


df.describe


# In[ ]:


def randSplit(dataSet, rate=0.9):
    n = dataSet.shape[0]
    m = int(n*rate)
    train = dataSet.iloc[:m,:]
    test = dataSet.iloc[m:,:]
    test .index = range(test.shape[0]) 
    return train,test


# In[ ]:


train, test = randSplit(df)


# In[ ]:


train.shape


# In[ ]:


train


# In[ ]:


train.info()


# In[ ]:


import numpy as np
def stayClass(train, test, k):
    n = train. shape[1]-1   
    m = test. shape[0]     
    result = []               
    
    for i in range(m):
        train=pd.DataFrame(train,dtype=np.float)
        test=pd.DataFrame(test,dtype=np.float)
        dist = np.linalg.norm(train.iloc[:, :n]-test.iloc[i, :n],ord=1)
        dist_1 = pd.DataFrame({'dist':dist, 'labels':(train.iloc[:, n])})  
        dr = dist_1.sort_values(by = 'dist')[: k]  
        re = dr.loc[:, 'labels'].value_counts()   
        result.append(re.index[0])    
    result = pd.Series(result)
    res = test.copy()
    res.loc[:, 'predict'] = result 
    acc = (res.iloc[:,-1]==res.iloc[:,-2]).mean()  
    print('accuracy{}'.format(acc))
    return acc, res


# In[ ]:


acc, df_stay = stayClass(train, test, 30)
df_stay.head()
train=pd.DataFrame(train,dtype=np.float)
test=pd.DataFrame(test,dtype=np.float)


# In[ ]:


import matplotlib.pyplot as plt
score = []
krange = range(1,20)

for i in krange:
    acc,re = stayClass(train,test,i)
    score.append(acc)
plt.plot(krange,score);
bestK = krange[score.index(max(score))]
print(bestK)
print(max(score))


# In[ ]:


df = pd.read_csv('datasetone.csv')


# In[ ]:


def randSplit(dataSet, rate=0.9):
    n = dataSet.shape[0]
    m = int(n*rate)
    train = dataSet.iloc[:m,:]
    test = dataSet.iloc[m:,:]
    test .index = range(test.shape[0]) 
    return train,test


# In[ ]:


train, test = randSplit(df)


# In[ ]:


train.shape


# In[ ]:


train.info()


# In[ ]:


#Manhattan distance:
import numpy as np
train=pd.DataFrame(train,dtype=np.float)
test=pd.DataFrame(test,dtype=np.float)
def stayClass(train, test, k):
    n = train. shape[1]-1  
    m = test. shape[0]   
    result = []           
    
    for i in range(m):
        train=pd.DataFrame(train,dtype=np.float)
        test=pd.DataFrame(test,dtype=np.float)
        dist = np.linalg.norm(train.iloc[:, :n]-test.iloc[i, :n],ord=1)
        dist_1 = pd.DataFrame({'dist':dist, 'labels':(train.iloc[:, n])}) 
        dr = dist_1.sort_values(by = 'dist')[: k]  
        re = dr.loc[:, 'labels'].value_counts()   
        result.append(re.index[0])    
    result = pd.Series(result)
    res = test.copy()
    res.loc[:, 'predict'] = result  
    acc = (res.iloc[:,-1]==res.iloc[:,-2]).mean()    
    print('accuracy{}'.format(acc))
    return acc, res


# In[ ]:


acc, df_stay = stayClass(train, test, 40)
df_stay.head()


# In[ ]:


import matplotlib.pyplot as plt
score = []
krange = range(1,20)

for i in krange:
    acc,re = stayClass(train,test,i)
    score.append(acc)
plt.plot(krange,score);
bestK = krange[score.index(max(score))]
print(bestK)
print(max(score))


# In[ ]:


import numpy as np
def stayClass(train, test, k):
    train=pd.DataFrame(train,dtype=np.float)
    test=pd.DataFrame(test,dtype=np.float)
    n = train. shape[1]-1  
    m = test. shape[0]    
    result = []               
    for i in range(m):
        dist = list((((train.iloc[:, :n] - test.iloc[i, :n]) **2).sum(1))**.5)    
        dist_1 = pd.DataFrame({'dist':dist, 'labels':(train.iloc[:, n])})   
        dr = dist_1.sort_values(by = 'dist')[: k]
        re = dr.loc[:, 'labels'].value_counts()    
        result.append(re.index[0])    
    result = pd.Series(result)
    res = test.copy()
    res.loc[:, 'predict'] = result  
    acc = (res.iloc[:,-1]==res.iloc[:,-2]).mean()  
    print('accuracy{}'.format(acc))
    return acc, res


# In[ ]:


import matplotlib.pyplot as plt
score = []
krange = range(1,30)

for i in krange:
    acc,re = stayClass(train,test,i)
    score.append(acc)
plt.plot(krange,score);
bestK = krange[score.index(max(score))]
print(bestK)
print(max(score))


# # features are chosen according to correlation3

# In[ ]:


def randSplit(dataSet, rate=0.9):
    n = dataSet.shape[0]
    m = int(n*rate)
    train = dataSet.iloc[:m,:]
    test = dataSet.iloc[m:,:]
    test .index = range(test.shape[0]) 
    return train,test


# In[ ]:


rain, test = randSplit(df)


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


test


# In[ ]:


import numpy as np
def stayClass(train, test, k):
    train=pd.DataFrame(train,dtype=np.float)
    test=pd.DataFrame(test,dtype=np.float)
    n = train. shape[1]-1   
    m = test. shape[0]     
    result = []                
    for i in range(m):
        dist = list((((train.iloc[:, :n] - test.iloc[i, :n]) **2).sum(1))**.5)    
        dist_1 = pd.DataFrame({'dist':dist, 'labels':(train.iloc[:, n])})   
        dr = dist_1.sort_values(by = 'dist')[: k] 
        re = dr.loc[:, 'labels'].value_counts()  
        result.append(re.index[0])    
    result = pd.Series(result)
    res = test.copy()
    res.loc[:, 'predict'] = result 
    acc = (res.iloc[:,-1]==res.iloc[:,-2]).mean()   
    print('accuracy{}'.format(acc))
    return acc, res


# In[ ]:


acc, df_stay = stayClass(train, test, 50)
df_stay.head()


# In[ ]:


import matplotlib.pyplot as plt
score = []
krange = range(1,30)

for i in krange:
    acc,re = stayClass(train,test,i)
    score.append(acc)
plt.plot(krange,score);
bestK = krange[score.index(max(score))]
print(bestK)
print(max(score))


# In[ ]:


df = pd.read_csv('datasetfull.csv')


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


def randSplit(dataSet, rate=0.9):
    n = dataSet.shape[0]
    m = int(n*rate)
    train = dataSet.iloc[:m,:]
    test = dataSet.iloc[m:,:]
    test .index = range(test.shape[0]) 
    return train,test


# In[ ]:


train, test = randSplit(df)


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


import numpy as np
def stayClass(train, test, k):
    train=pd.DataFrame(train,dtype=np.float)
    test=pd.DataFrame(test,dtype=np.float)
    n = train. shape[1]-1   
    m = test. shape[0]     
    result = []                 
    for i in range(m)
        dist = list((((train.iloc[:, :n] - test.iloc[i, :n]) **2).sum(1))**.5)   
        dist_1 = pd.DataFrame({'dist':dist, 'labels':(train.iloc[:, n])})   
        dr = dist_1.sort_values(by = 'dist')[: k] 
        re = dr.loc[:, 'labels'].value_counts()    
        result.append(re.index[0])    
    result = pd.Series(result)
    res = test.copy()
    res.loc[:, 'predict'] = result
    acc = (res.iloc[:,-1]==res.iloc[:,-2]).mean()   
    print('accuracy{}'.format(acc))
    return acc, res


# In[ ]:


acc, df_stay = stayClass(train, test, 40)
df_stay.head()


# In[ ]:


import matplotlib.pyplot as plt
score = []
krange = range(21,45)

for i in krange:
    acc,re = stayClass(train,test,i)
    score.append(acc)
plt.plot(krange,score);
bestK = krange[score.index(max(score))]
print(bestK)
print(max(score))


# In[ ]:


import matplotlib.pyplot as plt
score = []
krange = range(1,45)

for i in krange:
    acc,re = stayClass(train,test,i)
    score.append(acc)
plt.plot(krange,score);
bestK = krange[score.index(max(score))]
print(bestK)
print(max(score))


# In[ ]:


df = pd.read_csv('datasetfull.csv')


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


def randSplit(dataSet, rate=0.9):
    n = dataSet.shape[0]
    m = int(n*rate)
    train = dataSet.iloc[:m,:]
    test = dataSet.iloc[m:,:]
    test .index = range(test.shape[0]) 
    return train,test


# In[ ]:


train, test = randSplit(df)


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


import numpy as np
def stayClass(train, test, k):
    train=pd.DataFrame(train,dtype=np.float)
    test=pd.DataFrame(test,dtype=np.float)
    n = train. shape[1]-1   
    m = test. shape[0] 
    result = []             
    for i in range(m):
        dist = list((((train.iloc[:, :n] - test.iloc[i, :n]) **2).sum(1))**.5)  
        dist_1 = pd.DataFrame({'dist':dist, 'labels':(train.iloc[:, n])})  
        dr = dist_1.sort_values(by = 'dist')[: k] 
        re = dr.loc[:, 'labels'].value_counts()   
        result.append(re.index[0])    
    result = pd.Series(result)
    res = test.copy()
    res.loc[:, 'predict'] = result
    acc = (res.iloc[:,-1]==res.iloc[:,-2]).mean()  
    print('accuracy{}'.format(acc))
    return acc, res


# In[ ]:


acc, df_stay = stayClass(train, test, 40)
df_stay.head()


# In[ ]:


import matplotlib.pyplot as plt
score = []
krange = range(21,45)

for i in krange:
    acc,re = stayClass(train,test,i)
    score.append(acc)
plt.plot(krange,score);
bestK = krange[score.index(max(score))]
print(bestK)
print(max(score))


# In[ ]:


import matplotlib.pyplot as plt
score = []
krange = range(1,45)

for i in krange:
    acc,re = stayClass(train,test,i)
    score.append(acc)
plt.plot(krange,score);
bestK = krange[score.index(max(score))]
print(bestK)
print(max(score))

