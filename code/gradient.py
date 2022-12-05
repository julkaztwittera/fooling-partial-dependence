import numpy as np
import pandas as pd
import tqdm

from . import algorithm
from . import loss
from . import utils

from scipy import stats

try:
    import tensorflow as tf
except:
    import warnings

    warnings.warn("`import tensorflow as tf` returns an error: gradient.py won't work.")


class GradientAlgorithm(algorithm.Algorithm):
    def __init__(
        self,
        explainer,
        variable,
        constant=None,
        n_grid_points=21,
        learning_rate=1e-2,
        **kwargs
    ):
        super().__init__(
            explainer=explainer,
            variable=variable,
            constant=constant,
            n_grid_points=n_grid_points,
        )

        params = dict(
            epsilon=1e-5,
            stop_iter=10,
            learning_rate=learning_rate,
            optimizer=utils.AdamOptimizer(),
        )

        for k, v in kwargs.items():
            params[k] = v

        self.params = params

    def fool(
        self,
        grid=None,
        max_iter=50,
        random_state=None,
        save_iter=False,
        verbose=True,
        aim=False,
        center=None,
    ):

        self._aim = aim
        self._center = not aim if center is None else center
        if aim is False:
            super().fool(grid=grid, random_state=random_state)

        # init algorithm
        self._initialize()

        for j, (explanation_name, result_explanation) in enumerate(
            zip(self.result_explanations.keys(), self.result_explanations.values())
        ):
            explanation_func = getattr(self.explainer, explanation_name)

            result_explanation["changed"] = explanation_func(
                self._X_changed, self._idv, result_explanation["grid"]
            )
            if j > 0:
                self.append_losses(explanation_name)
            else:
                self.append_losses(explanation_name, i=0)
            if save_iter:
                self.append_explanations(explanation_name, i=0)

            pbar = tqdm.tqdm(range(1, max_iter + 1), disable=not verbose)
            for i in pbar:
                # gradient of output w.r.t input
                _ = self._calculate_gradient(self._X_changed)
                d_output_input_long = self._calculate_gradient_long(
                    result_explanation, self._X_changed
                )
                result_explanation["changed"] = explanation_func(
                    self._X_changed, self._idv, result_explanation["grid"]
                )
                d_loss = self._calculate_gradient_loss(
                    result_explanation, d_output_input_long
                )
                step = self.params["optimizer"].calculate_step(d_loss)
                self._X_changed -= self.params["learning_rate"] * step

                if j > 0:
                    self.append_losses(explanation_name)
                else:
                    self.append_losses(explanation_name, i=i)
                if save_iter:
                    self.append_explanations(explanation_name, i=i)
                pbar.set_description(
                    "Iter: %s || Loss: %s"
                    % (i, self.iter_losses["loss"][explanation_name][-1])
                )
                if utils.check_early_stopping(
                    self.iter_losses, self.params["epsilon"], self.params["stop_iter"]
                ):
                    break

            result_explanation["changed"] = explanation_func(
                X=self._X_changed, idv=self._idv, grid=result_explanation["grid"]
            )

        _data_changed = pd.DataFrame(
            self._X_changed, columns=self.explainer.data.columns
        )
        self.result_data = (
            pd.concat((self.explainer.data, _data_changed))
            .reset_index(drop=True)
            .rename(index={"0": "original", "1": "changed"})
            .assign(
                dataset=pd.Series(["original", "changed"])
                .repeat(self._n)
                .reset_index(drop=True)
            )
        )

    def fool_aim(
        self,
        target="auto",
        grid=None,
        max_iter=50,
        random_state=None,
        save_iter=False,
        verbose=True,
    ):
        super().fool_aim(target=target, grid=grid, random_state=random_state)
        self.fool(
            grid=None,
            max_iter=max_iter,
            random_state=random_state,
            save_iter=save_iter,
            verbose=verbose,
            aim=True,
        )

    #:# inside

    def _calculate_gradient(self, data):
        # gradient of output w.r.t input
        input = tf.convert_to_tensor(data)
        with tf.GradientTape() as t:
            t.watch(input)
            output = self.explainer.model(input)
            d_output_input = t.gradient(output, input).numpy()
        return d_output_input

    def _calculate_gradient_long(self, result_explanation, data):
        # gradient of output w.r.t input with changed idv to splits
        data_long = np.repeat(data, self._n_grid_points, axis=0)
        # take splits for each observation
        grid_long = np.tile(result_explanation["grid"], self._n)
        data_long[:, self._idv] = grid_long
        # merge X and splits in long format
        d_output_input_long = self._calculate_gradient(data_long)
        return d_output_input_long

    def _calculate_gradient_loss(self, result_explanation, d):
        # d = d_output_input_long
        d = d.reshape(self._n, self._n_grid_points, self._p)
        if self._aim:
            d_loss = (
                d
                * (
                    result_explanation["changed"] - result_explanation["target"]
                ).reshape(1, -1, 1)
            ).mean(axis=1)
        else:
            if self._center:
                d_loss = -(
                    (d - d.mean(axis=1).reshape(self._n, 1, self._p))
                    * (
                        (
                            result_explanation["changed"]
                            - result_explanation["changed"].mean()
                        )
                        - (
                            result_explanation["original"]
                            - result_explanation["original"].mean()
                        )
                    ).reshape(1, -1, 1)
                ).mean(axis=1)
            else:
                d_loss = -(
                    d
                    * (
                        result_explanation["changed"] - result_explanation["original"]
                    ).reshape(1, -1, 1)
                ).mean(axis=1)
        d_loss = d_loss / self._n
        d_loss[:, self._idv] = 0
        if self._idc is not None:
            d_loss[:, self._idc] = 0
        return d_loss

    def _initialize(self):
        _X_std = self._X.std(axis=0) * 1 / 9
        _X_std[self._idv] = 0
        if self._idc is not None:
            for c in self._idc:
                _X_std[c] = 0
        _theta = np.random.normal(loc=0, scale=_X_std, size=self._X.shape)
        self._X_changed = self._X + _theta

    #:# helper

    def append_losses(self, explanation_name, i=None):
        _loss = loss.loss(
            original=self.result_explanations[explanation_name]["target"]
            if self._aim
            else self.result_explanations[explanation_name]["original"],
            changed=self.result_explanations[explanation_name]["changed"],
            aim=self._aim,
            center=self._center,
        )
        if i is not None:
            self.iter_losses["iter"].append(i)
        self.iter_losses["loss"][explanation_name].append(_loss)

    def append_explanations(self, explanation_name, i=0):
        self.iter_explanations[explanation_name][i] = self.result_explanations[
            explanation_name
        ]["changed"]

    
    def get_metrics(self, save_path=None):
        output_str = ""
        for explanation_name in self.result_explanations.keys():
            _loss = loss.loss(
            original=self.result_explanations[explanation_name]["original"],
            changed=self.result_explanations[explanation_name]["changed"],
            aim=self._aim,
            center=self._center,
            )


            output_str += (f"{explanation_name} L2: {_loss}\n")

            l1 = np.abs(self.result_explanations[explanation_name]["original"] - self.result_explanations[explanation_name]["changed"]).mean()
            output_str += (f"{explanation_name} L1: {l1}\n")

            spearman_r, _ = stats.spearmanr(self.result_explanations[explanation_name]["original"],
                        self.result_explanations[explanation_name]["changed"])
            output_str += (f"{explanation_name} Spearman R: {spearman_r}\n")

        print(output_str)
        if save_path:
            with open(save_path, "w") as text_file:
                text_file.write(output_str)





        
