from google.colab import drive
drive.mount('/content/drive')

'''
1. Import the credit risk data (customer_data.csv) and (payment_data.csv) as pandas Dataframe into your notebook, 
from the source Dataset: https://www.kaggle.com/praveengovi/credit-risk-classification-dataset, name the customer data 
as "customer_data".
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataframe = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/customer_data.csv", sep=",")
print(dataframe)

'''
2. How many features are there in customer_data?
'''
headline = [ i for i in dataframe.iloc[0]]
features = len(headline) - 2
print(f"The number of features in this csv is {features}.")

'''
3. Find the number of customers whose label = 0. (identified as the potential defaulted customer.)
'''
customers = [j for j in dataframe.iloc[:,0]]
print(customers)
potential_defaulted_customer = 0
for k in customers:
  if k == 0:
   potential_defaulted_customer += 1
print(f"The total number of potential defaulted customer is {potential_defaulted_customer}.")

'''
4. using the sklearn to perform data imputation and data scaling for the feature data.
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB

X = dataframe.iloc[:,[2,4]]
y = dataframe.iloc[:,0]
X_train = np.array(X)
y_train = np.array(y)
print(X_train)
print(y_train)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='RdBu');
plt.show()

