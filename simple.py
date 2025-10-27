import numpy as np
from  sklearn.model_selection import train_test_split #for splitting data into training and testing sets
from sklearn.linear_model import LogisticRegression #for logistic regression model
from sklearn.metrics import accuracy_score #for evaluating model accuracy

# importing dataset

x=np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]) #x=no. of hours studied - data/features
y=np.array([[0],[0],[0],[1],[1],[1],[1],[1],[1], [1]]) #y= whether the student passed(1) or failed(0) - label

#splitting dataset into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20, random_state=42) #random_state for preserving the order of the data same split every time
#rpint(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_test.shape)
print(y_test)


#creating and training the model
model=LogisticRegression() # 0 or 1 output and model is the object of LogisticRegression 
model.fit(x_train, y_train.ravel()) #ravel() to convert y_train to 1D array

#making predictions
y_pred=model.predict(x_test)
print(y_pred)
print(accuracy_score(y_test, y_pred)) #comparing predicted values with actual values to get accuracy

#predicting for a new data point
hours=np.array([[7.5], [3.9], [50], [1]]) #new data point - hours studied
result=model.predict(hours)
print("predictions for unseen data", result)
for i in range(len(hours)):
    print("if you study for", hours[i][0], "hours you will", "pass" if result[i]==1 else "fail") 