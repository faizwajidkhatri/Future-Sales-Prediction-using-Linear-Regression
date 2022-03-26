#!/usr/bin/env python
# coding: utf-8

# ### Let’s start the task of future sales prediction with machine learning by importing the necessary Python libraries and the dataset:

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


data = pd.read_excel(r"advertising_data.xlsx")
print(data.head())


# ### Let’s have a look at whether this dataset contains any null values or not:

# In[3]:


print(data.isnull().sum())


# ### So this dataset doesn’t have any null values. Now let’s visualize the relationship between the amount spent on advertising on TV and units sold:

# In[4]:


get_ipython().system('pip install plotly')
import plotly.express as px
import plotly.graph_objects as go
figure = px.scatter(data_frame = data, x="Sales",
                    y="TV", size="TV", trendline="ols")
figure.show()


# ### Now let’s visualize the relationship between the amount spent on advertising on newspapers and units sold:

# In[5]:


figure = px.scatter(data_frame = data, x="Sales",
                    y="Newspaper", size="Newspaper", trendline="ols")
figure.show()


# ### Now let’s visualize the relationship between the amount spent on advertising on radio and units sold:

# In[6]:


figure = px.scatter(data_frame = data, x="Sales",
                    y="Radio", size="Radio", trendline="ols")
figure.show()


# ### Out of all the amount spent on advertising on various platforms, I can see that the amount spent on advertising the product on TV results in more sales of the product. Now let’s have a look at the correlation of all the columns with the sales column:

# In[7]:


correlation = data.corr()
print(correlation["Sales"].sort_values(ascending=False))


# ## Future Sales Prediction Model

# ### Now in this section, I will train a machine learning model to predict the future sales of a product. But before I train the model, let’s split the data into training and test sets:

# In[8]:


x = np.array(data.drop(["Sales"], 1))
y = np.array(data["Sales"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)


# ### Now let’s train the model to predict future sales:

# In[9]:


model = LinearRegression()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


# ### Now let’s input values into the model according to the features we have used to train it and predict how many units of the product can be sold based on the amount spent on its advertising on various platforms:

# In[10]:


#features = [[TV, Radio, Newspaper]]
features = np.array([[230.1, 37.8, 69.2]])
print(model.predict(features))


# ## Summary
# ### So this is how we can train a machine learning model to predict the future sales of a product. Predicting the future sales of a product helps a business manage the manufacturing and advertising cost of the product
