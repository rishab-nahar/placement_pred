
# THERE WERE 4 QUESTIONS WITH THE DATASET
#   Which factor influenced a candidate in salary determination?
#   Does percentage matters for one to get placed?
#   Which degree specialization is much demanded by corporate?
#  Play with the data conducting all statistical tests.



import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

place_rec = pd.read_csv("placement.csv")
features = place_rec.iloc[:, 6:12].values
placed = place_rec.iloc[:, -2].values
salary = place_rec.iloc[:, -1].values
deg_placed = []
spec_placed = []
for i in range(len(placed)):
    if placed[i] == "Placed":
        deg_placed.append(features[i][2])
        spec_placed.append(features[i][-1])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer([("encoder", OneHotEncoder(), [0])], remainder="passthrough")
features = ct.fit_transform(features)
features = features[:, 1:]
ct = ColumnTransformer([("encoder", OneHotEncoder(), [3])], remainder="passthrough")
features = ct.fit_transform(features)
features = features[:, 1:]
ct = ColumnTransformer([("encoder", OneHotEncoder(), [5])], remainder="passthrough")
features = ct.fit_transform(features)
features = features[:, 1:]
ct = ColumnTransformer([("encoder", OneHotEncoder(), [7])], remainder="passthrough")
features = ct.fit_transform(features)
features = features[:, 1:]
ct = ColumnTransformer([("encoder", OneHotEncoder(), [0])], remainder="passthrough")
placed = ct.fit_transform(placed.reshape(-1, 1))
placed = placed[:, 1:]
df = pd.DataFrame(features)

# finding clusters for placedstatus and etest scores

X = np.append(arr=features[:, 7].reshape(-1, 1), values=placed.reshape(-1, 1), axis=1)
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 5):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 5), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# from the elbow diagram i infer that 2 clusters is best suited


kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
print("""ans 2. Students who got very good grates in etest mostly
          mostly got placed but many students with low grades
          alseo passed""")
plt.hist(deg_placed)
plt.hist(spec_placed)

print("""Ans 3. from the histogram it is visible that out of
      149 placed students nearly took comm&mgmt and 40 took Sci&Tech
      and 90 were from Mkt&Fin , 60 were from Mkt&HR """)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features, placed)

# traing for classificatiom
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(features, placed)
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print("on the small testing data i got an accuracy of nearly 80% ")

import statsmodels.api as sm

features = np.append(arr=np.ones((215, 1)).astype(int), values=features, axis=1)

from sklearn.impute import SimpleImputer

imp = SimpleImputer(verbose=1)
salary = imp.fit_transform(salary.reshape(-1, 1))


def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(salary, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
        else:
            break
    return x


x_opt = np.array(features[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]], dtype=float)
x_model = backwardElimination(x_opt, 0.05)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_model, salary, test_size=0.1, random_state=0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

print("""Ans 1.By creating the backward elimination model for
      feature extractions it is found that features like
      degree and etest results mainly determine the salary""")
