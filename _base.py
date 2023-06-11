import numpy as np
import pandas as pd

class LinearRegression:

    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept
        self.coef_ = []
        self.intercept_ = 0

    def fit(self, X:np.ndarray | pd.DataFrame, y: np.ndarray) -> None:
        """
        Fit data training input (X) dan output/label (y)
        X: dataframe/array, fitur input
        y: dataframe/array, output/label
        Return
            None
        """
        X = np.array(X)
        y = np.array(y)
        if self.fit_intercept:
            X = np.column_stack((X, np.ones(X.shape[0])))
        weights = np.linalg.inv(X.T @ X) @ X.T @ y
        if self.fit_intercept:
            self.coef_ = weights[:-1]
            self.intercept_ = weights[-1]
        else:
            self.coef_ = weights
            self.intercept_ = 0


    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Prediksi nilai y berdasarkan fitur input X
        X: dataframe/array, fitur input
        Return
            y_pred: nilai y hasil prediksi
        """
        X = np.array(X)
        y_pred = np.dot(self.coef_, X) + self.intercept_
        return y_pred
    
    @staticmethod
    def calc_mse(y_pred: np.ndarray, y_obs: np.ndarray) -> np.ndarray:
        """
        Menghitung mean-squared-error
        y_pred: array, nilai y hasil prediksi
        y_obs: array, nilai y observasi
        Return
            nilai MSE
        """
        return np.mean((y_pred - y_obs)**2)
