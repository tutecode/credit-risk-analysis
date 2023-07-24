#### Normalization, Impute, Encoding

1. Standarization using pipelines

reload(data_utils)
reload(config)
reload(evaluation)
app_train_cop
reload(data_utils)
data_utils.manage_data(app_train_cop)

# Open the file in binary mode
with open('features_target.pkl', 'rb') as file:
      
    # Call load method to deserialze
    features_target_vars = pickle.load(file)
#### Logistic Regression
clf = LogisticRegression(max_iter=200)
classifier_predictions = manage_models.model_fit_predict(clf,features_target_vars)
accuracy, precision, recall, f1_score, report = evaluation.get_performance(classifier_predictions, features_target_vars["y_test"])
evaluation.print_performance(accuracy, precision, recall, f1_score, report)
evaluation.plot_metrics(clf, features_target_vars["y_test"], features_target_vars["X_test"],classifier_predictions)
reload(manage_models)
reload(evaluation)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=200,random_state=0)
distributions = dict(C=uniform(loc=0, scale=4),penalty=['l2', 'l1'])
clf = RandomizedSearchCV(logistic, distributions,n_iter=2)
clf.fit(X_train, y_train.ravel())
sorted(clf.cv_results_.keys())
res_logistic = pd.DataFrame(clf.cv_results_)
res_logistic
# logistic regression
clf = LogisticRegression(solver='saga',C=2,tol=1e-2, max_iter=200).fit(X_train, y_train.ravel())
print("model score: %.3f" % clf.score(X_val, y_val))
clf_preds = clf.predict(X_test)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
#class_weight={0:1, 1:3}
clfr = RandomForestClassifier(class_weight={0:1, 1:3})
distributions = dict(n_estimators=list(range(100,150,10)),max_depth=list(range(1,100,10)))
clfr = RandomizedSearchCV(clfr, distributions,cv=3,n_iter=9)
clfr.fit(X_train, y_train.ravel())
print("model score: %.3f" % clfr.score(X_val, y_val))
res_logistic = pd.DataFrame(clfr.cv_results_)
res_logistic
clfr.best_score_ #train
clfr.score(X_val, y_val)
clfr.best_estimator_.get_params
clfr_preds = clfr.predict(X_test)