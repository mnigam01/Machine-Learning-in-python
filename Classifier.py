from sklearn import linear_model

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

#making the logistic regression model and predicting it's result
lin_clf = linear_model.LogisticRegression(random_state=42)
lin_clf.fit(X,Y)
print("The result of Logistic Regression is:",lin_clf.predict([[190, 70, 43]]))


#making the SGD classifier and printing it's result
sgd_clf = linear_model.SGDClassifier(random_state=42)
sgd_clf.fit(X,Y)
print('The result of SGD is:',sgd_clf.predict([[190, 70, 43]]))

#making the naive bayes classifier and printing it's result
from sklearn import naive_bayes
nai_clf = naive_bayes.GaussianNB()
nai_clf.fit(X,Y)
print('The result of Naive Bayes is:',nai_clf.predict([[190, 70, 43]]))

#making the Support Vector Machine and printing it's result
from sklearn import svm
svc_clf = svm.SVC(kernel='rbf')
svc_clf.fit(X,Y)
print('The result of SVM is:',svc_clf.predict([[190, 70, 43]]))

#making the KNN and printing it's result
from sklearn import neighbors
knn_clf = neighbors.KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X,Y)
print('The result of KNN is:',knn_clf.predict([[190, 70, 43]]))

print() # just to leave one line
from sklearn import metrics
model_name = ['Logistic','Stochastic Gradient Descent','Naive Bayes',
              'Support Vector Machine','K Nearest Neighbours']
accuracy = []
for model in [lin_clf,sgd_clf,nai_clf,svc_clf,knn_clf]:
    
    accuracy.append(metrics.accuracy_score(Y,model.predict(X)))

import pandas as pd
df = pd.DataFrame(accuracy,index=model_name,columns=['Accuracy'])
print('The accuracy of each model used\n\n',df)



#AS PER THE RESULTS THE BEST MODEL FOR THIS SET IS LOGISTIC REGRSSION, KNN AND NAIVE BAYES 
