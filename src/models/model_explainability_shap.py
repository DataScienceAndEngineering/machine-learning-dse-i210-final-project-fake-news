import shap
shap.initjs()
import numpy as np
'''
 This python file builds python functions that uses shap to explain the Machine Learning classifier performance. 
'''

def get_shap_values(model, X_train_vec, X_test_vec, feature_names):
    """
    Parameters:
    model (object): The trained model.
    X_train_vec (np.ndarray): The training data transformed by the vectorizer.
    X_test_vec (np.ndarray): The test data transformed by the vectorizer.
    feature_names (np.ndarray): The feature names from the vectorizer.

    Returns:
    SHAP values for the test data
    """
    feature_names = vectorizer.get_feature_names_out()
    explainer = shap.Explainer(model, X_train_vec, feature_names=feature_names)
    shap_values = explainer(X_test_vec)
    return shap_values

def plot_waterfall(shap_values, index, X_test):
    """
    creating a SHAP waterfall plot.

    Parameters:
    shap_values: SHAP values for the test data.
    index (int): Index of the test data to plot.
    X_test (np.ndarray): The test data.
    """
    shap.initjs()
    print(X_test[index])
    shap.plots.waterfall(shap_values[index, :, 1])

def plot_summary(shap_values, X_test_vec, feature_names):
    """
    creating a SHAP summary plot.

    Parameters:
    shap_values (shap.Explanation): SHAP values for the test data.
    X_test_vec (np.ndarray): The test data transformed by the vectorizer.
    feature_names (np.ndarray): The feature names from the vectorizer.
    """
    shap.initjs()
    shap.summary_plot(shap_values[:, :, 1], X_test_vec, feature_names=feature_names)
