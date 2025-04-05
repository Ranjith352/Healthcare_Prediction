import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/ranja/OneDrive/Desktop/Ranjith/Semester 4/Predictive Analytics Lab/Project/Dataset.csv")

st.set_page_config(page_title="Predictive Analytics Tool", layout="wide")

st.sidebar.title("Navigation")
analysis_option = st.sidebar.radio("Choose Analysis", [
    "Home",
    "Hierarchical Clustering",
    "K-means Clustering",
    "PCA",
    "Factor Analysis",
    "Multiple Linear Regression",
    "F-test"
])

if analysis_option == "Home":
    st.title("Predictive Analytics Tool")
    st.write("Welcome to the Predictive Analytics Tool. Use the sidebar to select different analyses.")
    st.image("C:/Users/ranja/OneDrive/Desktop/Ranjith/Semester 4/Predictive Analytics Lab/Project/Healthcare(1).jpg", use_column_width=True)

def hierarchical_clustering(X, n_clusters=2):
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = clustering.fit_predict(X)
    return clusters

def kmeans_clustering(X, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(X)
    return clusters

def apply_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca

def apply_factor_analysis(X, n_components=2):
    fa = FactorAnalysis(n_components=n_components)
    X_fa = fa.fit_transform(X)
    return X_fa

def multiple_linear_regression(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    mse = mean_squared_error(y, y_pred)
    return mse, y_pred

def perform_f_test(X, y):
    f_values, p_values = f_regression(X, y)
    return f_values, p_values

# Analysis Pages
def show_clustering_results(X, clusters, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis')
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label='Cluster')
    st.pyplot(plt)

def show_pca_results(X_pca, y, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label='Target')
    st.pyplot(plt)

def show_linear_regression_results(y, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred)
    plt.title("Multiple Linear Regression")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    st.pyplot(plt)

def show_f_test_results(f_values, p_values):
    plt.figure(figsize=(8, 6))
    plt.plot(f_values, p_values, 'o')
    plt.title("F-test Results")
    plt.xlabel("F-values")
    plt.ylabel("p-values")
    st.pyplot(plt)

if analysis_option == "Hierarchical Clustering":
    st.title("Hierarchical Clustering")
    n_clusters = st.slider("Number of clusters", 2, 10, 2)
    X = data.drop(columns=['PoorCare'])
    clusters = hierarchical_clustering(X, n_clusters)
    show_clustering_results(X, clusters, "Hierarchical Clustering")

elif analysis_option == "K-means Clustering":
    st.title("K-means Clustering")
    n_clusters = st.slider("Number of clusters", 2, 10, 2)
    X = data.drop(columns=['PoorCare'])
    clusters = kmeans_clustering(X, n_clusters)
    show_clustering_results(X, clusters, "K-means Clustering")

elif analysis_option == "PCA":
    st.title("PCA Analysis")
    X = data.drop(columns=['PoorCare'])
    y = data['PoorCare']
    X_pca = apply_pca(X)
    show_pca_results(X_pca, y, "PCA Analysis")

elif analysis_option == "Factor Analysis":
    st.title("Factor Analysis")
    X = data.drop(columns=['PoorCare'])
    y = data['PoorCare']
    X_fa = apply_factor_analysis(X)
    show_pca_results(X_fa, y, "Factor Analysis")

elif analysis_option == "Multiple Linear Regression":
    st.title("Multiple Linear Regression")
    X = data.drop(columns=['PoorCare'])
    y = data['PoorCare']
    mse, y_pred = multiple_linear_regression(X, y)
    st.write(f"Mean Squared Error: {mse}")
    show_linear_regression_results(y, y_pred)

elif analysis_option == "F-test":
    st.title("F-test Analysis")
    X = data.drop(columns=['PoorCare'])
    y = data['PoorCare']
    f_values, p_values = perform_f_test(X, y)
    st.write("F-values:", f_values)
    st.write("p-values:", p_values)
    show_f_test_results(f_values, p_values)
