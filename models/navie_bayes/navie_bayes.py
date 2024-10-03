from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class CustomNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def fit(self, X, y):
        # Tính toán các xác suất tiên nghiệm P(y)
        self.classes_, class_counts = np.unique(y, return_counts=True)
        self.class_priors_ = class_counts / y.shape[0]

        # Tính toán các xác suất có điều kiện P(X|y)
        self.feature_probs_ = {}
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]  # Chọn điểm dữ liệu thuộc lớp c
            self.feature_probs_[c] = (np.sum(X_c, axis=0) + self.alpha) / (X_c.shape[0] + self.alpha * X.shape[1])
        
        return self

    def predict(self, X):
        # Dự đoán xác suất posterior cho từng lớp
        posteriors = []
        for x in X:
            class_posteriors = []
            for idx, c in enumerate(self.classes_):
                prior = np.log(self.class_priors_[idx])
                likelihood = np.sum(np.log(self.feature_probs_[c]) * x)
                class_posteriors.append(prior + likelihood)
            posteriors.append(self.classes_[np.argmax(class_posteriors)])
        
        return np.array(posteriors)

    def predict_proba(self, X):
        # Trả về xác suất cho từng lớp
        proba = []
        for x in X:
            class_proba = []
            for idx, c in enumerate(self.classes_):
                prior = np.log(self.class_priors_[idx])
                likelihood = np.sum(np.log(self.feature_probs_[c]) * x)
                class_proba.append(np.exp(prior + likelihood))
            proba.append(class_proba / np.sum(class_proba))
        
        return np.array(proba)