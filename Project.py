
import tkinter as tk
from tkinter import ttk, messagebox, Toplevel
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

button_style = ttk.Style()
button_style.configure('Custom.TButton', font=('Helvetica', 14), width=20, padding=10, background='yellow')

def hierarchical_clustering_analysis():
    hierarchical_window = Toplevel(root)
    hierarchical_window.title("Hierarchical Clustering")
    hierarchical_window.geometry("400x300")

    graph_button = ttk.Button(hierarchical_window, text="Graph", style='Custom.TButton', command=lambda: plot_data(data.drop(columns=['PoorCare']), data['PoorCare'], "Hierarchical Clustering"))
    graph_button.pack(pady=10, padx=20)

    analyze_button = ttk.Button(hierarchical_window, text="Analyze", style='Custom.TButton', command=lambda: analyze_data(data.drop(columns=['PoorCare']), data['PoorCare'], "Hierarchical Clustering"))
    analyze_button.pack(pady=10, padx=20)

def kmeans_clustering_analysis():
    kmeans_window = Toplevel(root)
    kmeans_window.title("K-means Clustering")
    kmeans_window.geometry("400x300")

    graph_button = ttk.Button(kmeans_window, text="Graph", style='Custom.TButton', command=lambda: plot_data(data.drop(columns=['PoorCare']), data['PoorCare'], "K-means Clustering"))
    graph_button.pack(pady=10, padx=20)

    analyze_button = ttk.Button(kmeans_window, text="Analyze", style='Custom.TButton', command=lambda: analyze_data(data.drop(columns=['PoorCare']), data['PoorCare'], "K-means Clustering"))
    analyze_button.pack(pady=10, padx=20)

def pca_analysis():
    pca_window = Toplevel(root)
    pca_window.title("PCA Analysis")
    pca_window.geometry("400x300")

    graph_button = ttk.Button(pca_window, text="Graph", style='Custom.TButton', command=lambda: plot_data(data.drop(columns=['PoorCare']), data['PoorCare'], "PCA"))
    graph_button.pack(pady=10, padx=20)

    analyze_button = ttk.Button(pca_window, text="Analyze", style='Custom.TButton', command=lambda: analyze_data(data.drop(columns=['PoorCare']), data['PoorCare'], "PCA"))
    analyze_button.pack(pady=10, padx=20)

def factor_analysis():
    factor_window = Toplevel(root)
    factor_window.title("Factor Analysis")
    factor_window.geometry("400x300")

    graph_button = ttk.Button(factor_window, text="Graph", style='Custom.TButton', command=lambda: plot_data(data.drop(columns=['PoorCare']), data['PoorCare'], "Factor Analysis"))
    graph_button.pack(pady=10, padx=20)

    analyze_button = ttk.Button(factor_window, text="Analyze", style='Custom.TButton', command=lambda: analyze_data(data.drop(columns=['PoorCare']), data['PoorCare'], "Factor Analysis"))
    analyze_button.pack(pady=10, padx=20)

def multiple_linear_regression_analysis():
    regression_window = Toplevel(root)
    regression_window.title("Multiple Linear Regression")
    regression_window.geometry("400x300")

    graph_button = ttk.Button(regression_window, text="Graph", style='Custom.TButton', command=lambda: plot_data(data.drop(columns=['PoorCare']), data['PoorCare'], "Multiple Linear Regression"))
    graph_button.pack(pady=10, padx=20)

    analyze_button = ttk.Button(regression_window, text="Analyze", style='Custom.TButton', command=lambda: analyze_data(data.drop(columns=['PoorCare']), data['PoorCare'], "Multiple Linear Regression"))
    analyze_button.pack(pady=10, padx=20)

def f_test_analysis():
    f_test_window = Toplevel(root)
    f_test_window.title("F-test Analysis")
    f_test_window.geometry("400x300")

    graph_button = ttk.Button(f_test_window, text="Graph", style='Custom.TButton', command=lambda: plot_data(data.drop(columns=['PoorCare']), data['PoorCare'], "F-test"))
    graph_button.pack(pady=10, padx=20)

    analyze_button = ttk.Button(f_test_window, text="Analyze", style='Custom.TButton', command=lambda: analyze_data(data.drop(columns=['PoorCare']), data['PoorCare'], "F-test"))
    analyze_button.pack(pady=10, padx=20)

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
    return kmeans.labels_

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

hierarchical_button = ttk.Button(root, text="Hierarchical Clustering", style='Custom.TButton', command=hierarchical_clustering_analysis)
hierarchical_button.place(relx=0.5, rely=0.4, anchor='center')

kmeans_button = ttk.Button(root, text="K-means Clustering", style='Custom.TButton', command=kmeans_clustering_analysis)
kmeans_button.place(relx=0.5, rely=0.5, anchor='center')

pca_button = ttk.Button(root, text="PCA", style='Custom.TButton', command=pca_analysis)
pca_button.place(relx=0.5, rely=0.6, anchor='center')

factor_button = ttk.Button(root, text="Factor Analysis", style='Custom.TButton', command=factor_analysis)
factor_button.place(relx=0.5, rely=0.7, anchor='center')

linear_regression_button = ttk.Button(root, text="Multiple Linear Regression", style='Custom.TButton', command=multiple_linear_regression_analysis)
linear_regression_button.place(relx=0.5, rely=0.8, anchor='center')

f_test_button = ttk.Button(root, text="F-test", style='Custom.TButton', command=f_test_analysis)
f_test_button.place(relx=0.5, rely=0.9, anchor='center')

root.mainloop()
