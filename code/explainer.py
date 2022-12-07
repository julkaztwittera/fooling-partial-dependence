import numpy as np
import tensorflow as tf
from copy import deepcopy
import pandas as pd
import warnings


class Explainer:
    def __init__(self, model, data, predict_function=None, constrain=False):
        self.model = model

        self.original_data = data
        data_copy = deepcopy(data)
        self.constrain = constrain
        if constrain:
            self.normalizator = [
                lambda x, c=c: (x - self.original_data[c].min())
                / (self.original_data[c].max() - self.original_data[c].min())
                for c in self.original_data.columns
            ]
            # print(self.normalizator[0](29))
            # assert False

            self.unnormalizator = [
                lambda x, c=c: x
                * (self.original_data[c].max() - self.original_data[c].min())
                + self.original_data[c].min()
                for c in self.original_data.columns
            ]

            for i, column in enumerate(data_copy.columns):
                data_copy[column] = self.normalizator[i](data_copy[column])

                data_copy.loc[data_copy[column] > 0.999, column] = 1.0 - 1e-9
                data_copy.loc[data_copy[column] < 0.001, column] = 1e-9

                data_copy[column] = logit(data_copy[column])

        if isinstance(data_copy, pd.DataFrame):
            self.data = data_copy
        elif isinstance(data_copy, np.ndarray):
            warnings.warn("`data` is a numpy.ndarray -> coercing to pandas.DataFrame.")
            self.data = pd.DataFrame(data_copy)
        else:
            raise TypeError(
                "`data` is a "
                + str(type(data_copy))
                + ", and it should be a pandas.DataFrame."
            )

        if predict_function:
            self.predict_function = predict_function
        else:
            # scikit-learn extraction
            if hasattr(model, "_estimator_type"):
                if model._estimator_type == "classifier":
                    self.predict_function = lambda m, d: m.predict_proba(d)[:, 1]
                elif model._estimator_type == "regressor":
                    self.predict_function = lambda m, d: m.predict(d)
                else:
                    raise ValueError(
                        "Unknown estimator type: " + str(model._estimator_type) + "."
                    )
            # tensorflow extraction
            elif str(type(model)).startswith("<class 'tensorflow.python.keras.engine"):
                if model.output_shape[1] == 1:
                    self.predict_function = lambda m, d: m.predict(np.array(d)).reshape(
                        -1,
                    )
                elif model.output_shape[1] == 2:
                    self.predict_function = lambda m, d: m.predict(np.array(d))[:, 1]
                else:
                    warnings.warn(
                        "`model` predict output has shape greater than 2, predicting column 1."
                    )
            # default extraction
            else:
                if hasattr(model, "predict_proba"):
                    self.predict_function = lambda m, d: m.predict_proba(d)[:, 1]
                elif hasattr(model, "predict"):
                    self.predict_function = lambda m, d: m.predict(d)
                else:
                    raise ValueError(
                        "`predict_function` can't be extracted from the model. \n"
                        + "Pass `predict_function` to the Explainer, e.g. "
                        + "lambda m, d: m.predict(d), which returns a (1d) numpy.ndarray."
                    )

        try:
            pred = self.predict_normalized(data_copy.values)
        except:
            raise ValueError("`predict_function(model, data)` returns an error.")
        if not isinstance(pred, np.ndarray):
            raise TypeError(
                "`predict_function(model, data)` returns an object of type "
                + str(type(pred))
                + ", and it must return a (1d) numpy.ndarray."
            )
        if len(pred.shape) != 1:
            raise ValueError(
                "`predict_function(model, data` returns an object of shape "
                + str(pred.shape)
                + ", and it must return a (1d) numpy.ndarray."
            )

    def predict_unnormalized(self, data):
        return self.predict_function(self.model, data)

    def predict_normalized(self, data):
        data_copy = deepcopy(data)
        for i in range(data_copy.shape[1]):
            data_copy[:, i] = sigmoid(data_copy[:, i])
            data_copy[:, i] = self.unnormalizator[i](data_copy[:, i])
        return self.predict_function(self.model, data_copy)

    # ************* pd *************** #

    def pd(self, X, idv, grid):
        """
        numpy implementation of pd calculation for 1 variable

        takes:
        X - np.ndarray (2d), data
        idv - int, index of variable to calculate profile

        returns:
        y - np.ndarray (1d), vector of pd profile values
        """

        grid_points = len(grid)
        # take grid_points of each observation in X
        X_long = np.repeat(X, grid_points, axis=0)
        # take grid for each observation
        grid_long = np.tile(grid.reshape((-1, 1)), (X.shape[0], 1))
        # merge X and grid in long format
        X_long[:, [idv]] = grid_long
        # calculate ceteris paribus
        y_long = self.predict_normalized(X_long)
        # calculate partial dependence
        y = y_long.reshape(X.shape[0], grid_points).mean(axis=0)

        return y

    def ale(self, X, idv, grid):
        """
        numpy implementation of pd calculation for 1 variable

        takes:
        X - np.ndarray (2d), data
        idv - int, index of variable to calculate profile

        returns:
        y - np.ndarray (1d), vector of pd profile values
        """

        grid_copy = deepcopy(grid)
        X_copy = deepcopy(X)
        idv_copy = deepcopy(idv)

        grid_copy[0] -= 0.0001
        grid_copy[-1] += 0.0001

        bins = grid_copy

        X_copy_2 = deepcopy(X_copy)
        
        b = np.digitize(X[:, idv_copy], bins, right=True)

        X_copy[:, idv_copy] = grid_copy[b - 1]
        X_copy_2[:, idv_copy] = grid_copy[b]
        diff = self.predict_normalized(X_copy_2) - self.predict_normalized(X_copy)

        local_effects = np.nan_to_num(
            np.bincount(b - 1, weights=diff, minlength=20)
            / np.bincount(b - 1, minlength=20)
        )
        y = np.cumsum(local_effects)

        c = np.dot(np.bincount(b - 1), y) / X_copy.shape[0]
        z = deepcopy(y)
        z = np.append(z, [z[-1]]) - c

        # print("X", X_copy[:,3:])
        # print("b", b)
        # print("bins", bins)
        return z


def logit(x):
    """Computes the logit function, i.e. the logistic sigmoid inverse."""
    return -tf.math.log(1.0 / x - 1.0)


def sigmoid(x):
    return tf.math.sigmoid(x)
