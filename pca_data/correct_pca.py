import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split


class PCA:

    def __init__(self):

        self.filename = os.path.join(
            os.path.dirname(__file__),
            'data',
            'healthcare_synthetic_data.csv'
        )

    # Read FULL dataset
    def read(self):

        self.df_full = pd.read_csv(self.filename)


    # Fit PCA on TRAIN DATA
    def fit(self, X_train, k):

        # Save mean & std
        self.mean_ = X_train.mean()
        self.std_ = X_train.std(ddof=1)

        # Standardize
        X_scaled = (X_train - self.mean_) / self.std_

        # Covariance matrix
        cov = np.cov(X_scaled, rowvar=False)

        # Eigen decomposition
        eig_val, eig_vec = np.linalg.eigh(cov)

        # Sort descending
        idxs = np.argsort(eig_val)[::-1]

        self.eig_val = eig_val[idxs]
        self.eig_vec = eig_vec[:, idxs]

        # Projection matrix
        self.W = self.eig_vec[:, :k]


    # Apply PCA
    def transform(self, X):

        X_scaled = (X - self.mean_) / self.std_

        return np.dot(X_scaled, self.W)


    # PCA Scatter Plot
    def visual1(self, X_pca):

        plt.figure(figsize=(8, 6))

        plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            edgecolors='k'
        )

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA Projection')

        plt.show()


    # Scree Plot
    def visual2(self):

        explained_var = self.eig_val / np.sum(self.eig_val)

        plt.figure(figsize=(8, 6))

        plt.plot(
            range(1, len(explained_var) + 1),
            explained_var,
            marker='o'
        )

        plt.xlabel("Principal Component")
        plt.ylabel("Variance Explained")
        plt.title("Scree Plot")

        plt.show()


# =============================
# MAIN
# =============================

pca = PCA()


# Columns used for PCA
features = [
    'Age', 'Height_cm', 'Weight_kg', 'BMI',
    'Systolic_BP', 'Diastolic_BP',
    'Cholesterol_Total', 'Cholesterol_HDL',
    'Cholesterol_LDL', 'Fasting_Blood_Sugar'
]


# Load full data
pca.read()

df = pca.df_full


# Split FULL dataset
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)


# Reset index (IMPORTANT)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# Select only PCA features
X_train = train_df[features]
X_test = test_df[features]


# Fit PCA on TRAIN
pca.fit(X_train, k=2)


# Transform both sets
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


# Visualize TRAIN
pca.visual1(X_train_pca)
pca.visual2()


# Convert PCA results to DataFrames
train_pca_df = pd.DataFrame(
    X_train_pca,
    columns=['PC1', 'PC2']
)

test_pca_df = pd.DataFrame(
    X_test_pca,
    columns=['PC1', 'PC2']
)


# Merge PCA with FULL data
train_with_pca = pd.concat([train_df, train_pca_df], axis=1).drop('Patient_ID', axis=1)
test_with_pca = pd.concat([test_df, test_pca_df], axis=1).drop('Patient_ID', axis=1)


# Save
train_with_pca.to_csv("train_pca.csv", index=False)
test_with_pca.to_csv("test_pca.csv", index=False)


# Check
print("Train Data with PCA:")
print(train_with_pca.head())

print("\nTest Data with PCA:")
print(test_with_pca.head())
