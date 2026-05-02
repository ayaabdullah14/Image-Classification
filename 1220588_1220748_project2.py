import os
import cv2
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# Configuration constants
DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset2")
IMG_SIZE = (32, 32)
TEST_RATIO = 0.2
RANDOM_SEED = 42

def get_data(path):
    """
    Load images from the folder structure in `path`.
    Assumes structure: path/class_label/*.jpg/png
    Returns:
        X: np.array of shape (n_samples, IMG_SIZE[0]*IMG_SIZE[1]*channels)
        y: np.array of labels (int indices)
        labels: list of class names matching label indices
    """
    X, y, labels = [], [], []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset directory '{path}' does not exist.")
    
    classes = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    if not classes:
        raise RuntimeError(f"No class subdirectories found in '{path}'.")
    
    labels = classes
    for idx, cls in enumerate(classes):
        cls_path = os.path.join(path, cls)
        for fname in os.listdir(cls_path):
            img_path = os.path.join(cls_path, fname)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image '{img_path}', skipping.")
                continue
            img = cv2.resize(img, IMG_SIZE)
            X.append(img.flatten())
            y.append(idx)
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    if X.size == 0:
        raise RuntimeError(f"No images loaded from '{path}'. Check data and file formats.")
    return X, y, labels

def naive_bayes_classifier(X_train, y_train, X_test, y_test, labels):
    """
    Train and evaluate a Gaussian Naive Bayes classifier using simple features:
    mean and variance of pixel intensities.
    """
    def extract_features(X):
        """
        Compute mean and variance of pixel intensities per sample.
        X: shape (n_samples, n_features)
        Returns: shape (n_samples, 2)
        """
        means = np.mean(X, axis=1).reshape(-1, 1)
        variances = np.var(X, axis=1).reshape(-1, 1)
        return np.hstack([means, variances])

    X_train_feat = extract_features(X_train)
    X_test_feat = extract_features(X_test)

    clf = GaussianNB()
    clf.fit(X_train_feat, y_train)
    y_pred = clf.predict(X_test_feat)

    print("\n=== Naive Bayes Classifier (Mean & Variance Features) ===")
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def decision_tree_classifier(X_train, y_train, X_test, y_test, labels):
    """
    Train and evaluate a Decision Tree classifier on raw pixel values.
    Visualize the decision tree.
    Returns:
        clf: trained DecisionTreeClassifier instance
    """
    clf = DecisionTreeClassifier(max_depth=15, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n=== Decision Tree Classifier ===")
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature names as pixel indices
    feature_names = [f"pixel_{i}" for i in range(X_train.shape[1])]
    visualize_decision_tree(clf, feature_names, labels)
    return clf

def visualize_decision_tree(tree, feature_names, class_labels):
    """
    Plot the decision tree up to 3 levels using matplotlib.
    """
    plt.figure(figsize=(20, 10))
    plot_tree(
        tree,
        max_depth=3,
        filled=True,
        feature_names=feature_names,
        class_names=class_labels,
        rounded=True,
        fontsize=8
    )
    plt.title("Decision Tree Visualization (First 3 Levels)")
    plt.show()

def mlp_classifier(X_train, y_train, X_test, y_test, labels):
    """
    Train and evaluate MLP classifiers with different architectures.
    Normalizes pixel intensities and standardizes features.
    """
    # Normalize pixel values to [0, 1]
    X_train_norm = X_train / 255.0
    X_test_norm = X_test / 255.0

    # Standardize features for better convergence
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_norm)
    X_test_scaled = scaler.transform(X_test_norm)

    configs = [
        ((100,), "1 hidden layer (100 neurons)"),
        ((100, 50), "2 hidden layers (100, 50 neurons)")
    ]

    for hidden_layers, description in configs:
        print(f"\n=== MLP Classifier: {description} ===")
        mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=1000, random_state=RANDOM_SEED)
        mlp.fit(X_train_scaled, y_train)
        y_pred = mlp.predict(X_test_scaled)

        print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

def main():
    try:
        # Load dataset
        X, y, labels = get_data(DATA_DIR)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, stratify=y, random_state=RANDOM_SEED
    )

    # Print dataset statistics
    train_counts = Counter(y_train)
    test_counts = Counter(y_test)
    print(f"Total samples:    {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print("\nSamples per class:")
    for idx, label in enumerate(labels):
        print(f"  {label:>10}: train={train_counts.get(idx, 0):4d}  test={test_counts.get(idx, 0):4d}")

    # Evaluate models
    naive_bayes_classifier(X_train, y_train, X_test, y_test, labels)
    decision_tree_classifier(X_train, y_train, X_test, y_test, labels)
    mlp_classifier(X_train, y_train, X_test, y_test, labels)

if __name__ == "__main__":
    main()