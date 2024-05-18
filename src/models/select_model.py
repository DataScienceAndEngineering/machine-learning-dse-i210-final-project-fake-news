
def model_selection(X_train,X_test,y_train,y_test):
    # calling all the ML classification algorithms imported above
    # importing the classification algorithms
    import itertools
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Perceptron, LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC, LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier,GradientBoostingClassifier
    import xgboost as xgb
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.naive_bayes import MultinomialNB

    ppn = Perceptron(eta0=0.1, random_state=1)
    lr_solver1 = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr')
    lr_solver2 = LogisticRegression(C=100.0, solver='liblinear', multi_class='ovr')
    knn = KNeighborsClassifier(n_neighbors=5, p=2)
    svm_linear = SVC(kernel='linear', C=1.0, random_state=1)
    svm_rbf = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
    linear_svc = LinearSVC(dual="auto", random_state=0, tol=1e-5)
    tree_gini = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
    tree_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=1)
    abc = AdaBoostClassifier(algorithm='SAMME', n_estimators=100, learning_rate=0.1, random_state=1)
    RF = RandomForestClassifier(n_estimators=20, random_state=1, n_jobs=2)
    bag = BaggingClassifier(n_estimators=100, max_samples=1.0, max_features=1.0, bootstrap=True,
                            bootstrap_features=False, n_jobs=1, random_state=1)
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.01, max_depth=4, random_state=1,
                                  use_label_encoder=False)
    gb_clf = GradientBoostingClassifier()
    nb_classifier = MultinomialNB()
    tfidf = TfidfVectorizer(strip_accents=None,
                            lowercase=False,
                            preprocessor=None)

    # defining a dictionary containing all the algorithms and their names

    clf_dict = {'perceptron': ppn, 'Log Reg lbfgs': lr_solver1, 'Log Reg liblinear': lr_solver2, 'KNN': knn,
                'Linear kernel svm': svm_linear, 'RBF kernel svm': svm_rbf, 'Linear SVC': linear_svc,
                'Decision Tree gini': tree_gini, 'Decision Tree entropy': tree_entropy, 'AdaBoost': abc,
                'RandomForest': RF, 'Bagging Clf': bag, 'xgb': xgb_model, 'gradient boosting': gb_clf,
                'multinomial': nb_classifier}

    accuracy_scores = {}
    for clf_name, clf in clf_dict.items():
        clf_tfidf = Pipeline([
        ('vect',tfidf),
        ('clf',clf)])
        clf_tfidf.fit(X_train,y_train)
        accuracy_scores[clf_name]= clf_tfidf.score(X_test, y_test)
        print(f'Test Accuracy for {clf_name}: {clf_tfidf.score(X_test, y_test):.3f}')
        print('----------------------------------------------')

if __name__ == "__main__":
    model_selection
    