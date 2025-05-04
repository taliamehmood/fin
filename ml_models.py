import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

def train_linear_regression(X_train, y_train):
    """
    Train a linear regression model
    
    Parameters:
    ----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training features
    y_train : pandas.Series or numpy.ndarray
        Training target values
    
    Returns:
    -------
    tuple
        (model, feature_importance)
        model: trained LinearRegression model
        feature_importance: DataFrame with feature names and their coefficients
    """
    # Initialize and train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Get feature importance (coefficients)
    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns
    else:
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    })
    
    feature_importance = feature_importance.sort_values('Coefficient', key=abs, ascending=False)
    
    return model, feature_importance

def evaluate_linear_regression(model, X_test, y_test):
    """
    Evaluate a linear regression model
    
    Parameters:
    ----------
    model : sklearn.linear_model.LinearRegression
        Trained linear regression model
    X_test : pandas.DataFrame or numpy.ndarray
        Test features
    y_test : pandas.Series or numpy.ndarray
        Test target values
    
    Returns:
    -------
    dict
        Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Mean absolute percentage error (MAPE)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    return {
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }

def train_logistic_regression(X_train, y_train, random_state=42):
    """
    Train a logistic regression model
    
    Parameters:
    ----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training features
    y_train : pandas.Series or numpy.ndarray
        Training target values (binary)
    random_state : int, optional (default=42)
        Random seed for reproducibility
    
    Returns:
    -------
    tuple
        (model, feature_importance)
        model: trained LogisticRegression model
        feature_importance: DataFrame with feature names and their coefficients
    """
    # Initialize and train model
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Get feature importance (coefficients)
    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns
    else:
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0]
    })
    
    feature_importance = feature_importance.sort_values('Coefficient', key=abs, ascending=False)
    
    return model, feature_importance

def evaluate_logistic_regression(model, X_test, y_test):
    """
    Evaluate a logistic regression model
    
    Parameters:
    ----------
    model : sklearn.linear_model.LogisticRegression
        Trained logistic regression model
    X_test : pandas.DataFrame or numpy.ndarray
        Test features
    y_test : pandas.Series or numpy.ndarray
        Test target values (binary)
    
    Returns:
    -------
    dict
        Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_kmeans_clustering(X, n_clusters=3, random_state=42):
    """
    Train a K-means clustering model
    
    Parameters:
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Features to cluster
    n_clusters : int, optional (default=3)
        Number of clusters
    random_state : int, optional (default=42)
        Random seed for reproducibility
    
    Returns:
    -------
    tuple
        (model, labels, centroids)
        model: trained KMeans model
        labels: cluster assignments
        centroids: cluster centers
    """
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and train model
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    model.fit(X_scaled)
    
    # Get cluster labels and centroids
    labels = model.labels_
    centroids = model.cluster_centers_
    
    return model, labels, centroids

def evaluate_kmeans_clustering(model, X):
    """
    Evaluate a K-means clustering model
    
    Parameters:
    ----------
    model : sklearn.cluster.KMeans
        Trained K-means clustering model
    X : pandas.DataFrame or numpy.ndarray
        Features used for clustering
    
    Returns:
    -------
    dict
        Dictionary containing evaluation metrics
    """
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate metrics
    inertia = model.inertia_  # Sum of squared distances to closest centroid
    
    # Silhouette score would require additional import, so we'll skip it for simplicity
    
    return {
        'inertia': inertia,
        'n_clusters': model.n_clusters
    }