import tkinter as tk
from tkinter import messagebox, Toplevel
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageSequence

data = pd.read_csv("C:/Users/ranja/OneDrive/Desktop/Ranjith/Semester 4/Predictive Analytics Lab/Project/Dataset.csv")

root = tk.Tk()
root.title("Predictive Analytics Tool")
root.geometry("800x600")

def set_background_gif():
    global background_label
    gif_path = "C:/Users/ranja/OneDrive/Desktop/Ranjith/Semester 4/Predictive Analytics Lab/Project/Blue White Modern Medical Healthcare Presentation(1)(1)(1).gif"
    gif = Image.open(gif_path)
    frames = [ImageTk.PhotoImage(frame) for frame in ImageSequence.Iterator(gif)]
    gif.close()

    background_label = tk.Label(root)
    background_label.pack(expand=True, fill="both")

    def update(frame_number):
        frame = frames[frame_number]
        background_label.configure(image=frame)
        background_label.image = frame

    def animate():
        for i in range(len(frames)):
            root.after(i * 100, update, i % len(frames))
        root.after(len(frames) * 100, animate)

    animate()

set_background_gif()

def add_background_image(window, image_path):
    img = Image.open(image_path)
    img = img.resize((800, 600), Image.Resampling.LANCZOS)
    background_img = ImageTk.PhotoImage(img)

    background_label = tk.Label(window, image=background_img)
    background_label.image = background_img
    background_label.place(relwidth=1, relheight=1)

def create_analysis_window(title, image_path, plot_command, analyze_command, inference_command):
    window = Toplevel(root)
    window.title(title)
    window.geometry("800x600")

    add_background_image(window, image_path)

    graph_button = tk.Button(window, text="Graph", bg='yellow', font=('Helvetica', 14), command=plot_command)
    graph_button.pack(pady=10, padx=20)

    analyze_button = tk.Button(window, text="Analyze", bg='yellow', font=('Helvetica', 14), command=analyze_command)
    analyze_button.pack(pady=10, padx=20)

    inference_button = tk.Button(window, text="Inference", bg='yellow', font=('Helvetica', 14), command=inference_command)
    inference_button.pack(pady=10, padx=20)

def hierarchical_clustering_analysis():
    create_analysis_window("Hierarchical Clustering", "C:/Users/ranja/OneDrive/Desktop/Ranjith/Semester 4/Predictive Analytics Lab/Project/Healthcare(1).jpg",
                           lambda: plot_data(data.drop(columns=['PoorCare']), data['PoorCare'], "Hierarchical Clustering"),
                           lambda: analyze_data(data.drop(columns=['PoorCare']), data['PoorCare'], "Hierarchical Clustering"),
                           lambda: show_inference("Hierarchical Clustering"))

def kmeans_clustering_analysis():
    create_analysis_window("K-means Clustering", "C:/Users/ranja/OneDrive/Desktop/Ranjith/Semester 4/Predictive Analytics Lab/Project/Healthcare(1).jpg",
                           lambda: plot_data(data.drop(columns=['PoorCare']), data['PoorCare'], "K-means Clustering"),
                           lambda: analyze_data(data.drop(columns=['PoorCare']), data['PoorCare'], "K-means Clustering"),
                           lambda: show_inference("K-means Clustering"))

def pca_analysis():
    create_analysis_window("PCA Analysis", "C:/Users/ranja/OneDrive/Desktop/Ranjith/Semester 4/Predictive Analytics Lab/Project/Healthcare(1).jpg",
                           lambda: plot_data(data.drop(columns=['PoorCare']), data['PoorCare'], "PCA"),
                           lambda: analyze_data(data.drop(columns=['PoorCare']), data['PoorCare'], "PCA"),
                           lambda: show_inference("PCA"))

def factor_analysis():
    create_analysis_window("Factor Analysis", "C:/Users/ranja/OneDrive/Desktop/Ranjith/Semester 4/Predictive Analytics Lab/Project/Healthcare(1).jpg",
                           lambda: plot_data(data.drop(columns=['PoorCare']), data['PoorCare'], "Factor Analysis"),
                           lambda: analyze_data(data.drop(columns=['PoorCare']), data['PoorCare'], "Factor Analysis"),
                           lambda: show_inference("Factor Analysis"))

def multiple_linear_regression_analysis():
    create_analysis_window("Multiple Linear Regression", "C:/Users/ranja/OneDrive/Desktop/Ranjith/Semester 4/Predictive Analytics Lab/Project/Healthcare(1).jpg",
                           lambda: plot_data(data.drop(columns=['PoorCare']), data['PoorCare'], "Multiple Linear Regression"),
                           lambda: analyze_data(data.drop(columns=['PoorCare']), data['PoorCare'], "Multiple Linear Regression"),
                           lambda: show_inference("Multiple Linear Regression"))

def f_test_analysis():
    create_analysis_window("F-test Analysis", "C:/Users/ranja/OneDrive/Desktop/Ranjith/Semester 4/Predictive Analytics Lab/Project/Healthcare(1).jpg",
                           lambda: plot_data(data.drop(columns=['PoorCare']), data['PoorCare'], "F-test"),
                           lambda: analyze_data(data.drop(columns=['PoorCare']), data['PoorCare'], "F-test"),
                           lambda: show_inference("F-test"))

def hierarchical_clustering(X, n_clusters):
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = clustering.fit_predict(X)
    return clusters

def kmeans_clustering(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(X)
    return clusters

def apply_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    return kmeans.labels()

def apply_pca(X, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca

def apply_factor_analysis(X, n_components):
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

def analyze_data(X, y, method):
    if method == "PCA":
        pca = PCA()
        X_pca = pca.fit_transform(X)
        messagebox.showinfo("PCA Analysis", f"Principal Component Analysis transformed data shape: {X_pca.shape}", parent=root)
    elif method == "Factor Analysis":
        fa = FactorAnalysis()
        X_fa = fa.fit_transform(X)
        messagebox.showinfo("Factor Analysis", f"Factor Analysis transformed data shape: {X_fa.shape}", parent=root)
    elif method == "Multiple Linear Regression":
        mse, y_pred = multiple_linear_regression(X, y)
        messagebox.showinfo("Multiple Linear Regression", f"Mean Squared Error (Multiple Linear Regression): {mse}", parent=root)
    elif method == "F-test":
        f_values, p_values = perform_f_test(X, y)
        messagebox.showinfo("F-test Results", f"F-values: {f_values}, p-values: {p_values}", parent=root)
    elif method == "Hierarchical Clustering":
        n_clusters = 2
        clusters = hierarchical_clustering(X, n_clusters)
        messagebox.showinfo("Hierarchical Clustering", f"Clusters: {clusters}", parent=root)
    elif method == "K-means Clustering":
        n_clusters = 2
        clusters = kmeans_clustering(X, n_clusters)
        messagebox.showinfo("K-means Clustering", f"Clusters: {clusters}", parent=root)
    else:
        messagebox.showinfo("Analysis", "Analysis not available for the selected method.", parent=root)

def plot_data(X, y, method):
    if method == "Hierarchical Clustering":
        n_clusters = 2
        clusters = hierarchical_clustering(X, n_clusters)
        plt.figure(figsize=(8, 6))
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis')
        plt.title("Hierarchical Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.colorbar(label='Cluster')
        plt.show()
    elif method == "K-means Clustering":
        n_clusters = 2
        clusters = kmeans_clustering(X, n_clusters)
        plt.figure(figsize=(8, 6))
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='plasma')
        plt.title("K-means Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.colorbar(label='Cluster')
        plt.show()
    elif method == "PCA":
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
        plt.title("PCA Analysis")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(label='Target')
        plt.show()
    elif method == "Factor Analysis":
        fa = FactorAnalysis(n_components=2)
        X_fa = fa.fit_transform(X)
        plt.figure(figsize=(8, 6))
        plt.scatter(X_fa[:, 0], X_fa[:, 1], c=y, cmap='viridis')
        plt.title("Factor Analysis")
        plt.xlabel("Factor 1")
        plt.ylabel("Factor 2")
        plt.colorbar(label='Target')
        plt.show()
    elif method == "Multiple Linear Regression":
        mse, y_pred = multiple_linear_regression(X, y)
        plt.figure(figsize=(8, 6))
        plt.scatter(y, y_pred)
        plt.title("Multiple Linear Regression")
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.show()
    elif method == "F-test":
        f_values, p_values = perform_f_test(X, y)
        plt.figure(figsize=(8, 6))
        plt.plot(f_values, p_values, 'o')
        plt.title("F-test Results")
        plt.xlabel("F-values")
        plt.ylabel("p-values")
        plt.show()
    else:
        messagebox.showinfo("Plotting", "Plotting is not available for the selected method.", parent=root)

def show_inference(method):
    inferences = {
        "Hierarchical Clustering": "Hierarchical Clustering groups data based on hierarchy and can show nested clusters. It's useful for discovering groupings in your data.",
        "K-means Clustering": "K-means Clustering partitions data into k clusters, where each data point belongs to the cluster with the nearest mean. It's effective for flat clustering.",
        "PCA": "Principal Component Analysis reduces the dimensionality of data, transforming it into new variables (principal components) that retain most of the variance.",
        "Factor Analysis": "Factor Analysis identifies underlying relationships between variables, reducing them to a smaller set of factors. It helps in understanding data structure.",
        "Multiple Linear Regression": "Multiple Linear Regression models the relationship between a dependent variable and multiple independent variables. It's useful for prediction and understanding impact.",
        "F-test": "F-test assesses the significance of predictors in a model, comparing the model with and without the predictors. It helps in feature selection and model evaluation."
    }
    messagebox.showinfo("Inference", inferences.get(method, "Inference not available for the selected method."), parent=root)

hierarchical_button = tk.Button(root, text="Hierarchical Clustering", bg='yellow', font=('Helvetica', 14), command=hierarchical_clustering_analysis)
hierarchical_button.place(relx=0.5, rely=0.4, anchor='center')

kmeans_button = tk.Button(root, text="K-means Clustering", bg='yellow', font=('Helvetica', 14), command=kmeans_clustering_analysis)
kmeans_button.place(relx=0.5, rely=0.5, anchor='center')

pca_button = tk.Button(root, text="PCA", bg='yellow', font=('Helvetica', 14), command=pca_analysis)
pca_button.place(relx=0.5, rely=0.6, anchor='center')

factor_button = tk.Button(root, text="Factor Analysis", bg='yellow', font=('Helvetica', 14), command=factor_analysis)
factor_button.place(relx=0.5, rely=0.7, anchor='center')

linear_regression_button = tk.Button(root, text="Multiple Linear Regression", bg='yellow', font=('Helvetica', 14), command=multiple_linear_regression_analysis)
linear_regression_button.place(relx=0.5, rely=0.8, anchor='center')

f_test_button = tk.Button(root, text="F-test", bg='yellow', font=('Helvetica', 14), command=f_test_analysis)
f_test_button.place(relx=0.5, rely=0.9, anchor='center')

root.mainloop()
