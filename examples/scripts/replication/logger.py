import numpy as _np
import pandas as _pd
import os as _os

class Logger:
    #TODOC
    def __init__(
            self,
            dataset_name, 
            evaluable_datasets = [
                "pretrain", "clustering"
                ],
            metric_names = ["uACC", "NMI", "ARS"],
            save_path="./results/"
            ):
        
        self.__dataset_name = dataset_name
        self.__evaluable_datasets = evaluable_datasets
        self.__metric_names = metric_names
        self.__save_path = save_path
        self.__make_dataframes()

    def add(self, models, metrics, datasets):
        self.__models = models
        self.__metrics = metrics
        self.__datasets = datasets

    def __make_dataframes(self):
        #TODOC
        self.__dfs = {}
        for dataset in self.__evaluable_datasets:
            self.__dfs[dataset] = _pd.DataFrame(
                    columns=[
                        "iteration",
                        *self.__metric_names
                        ]
                    )
            self.__dfs[dataset].set_index("iteration", inplace=True)

    def __compute_preds(
            self, model,
            dataset, steps
            ):
        y_true = []; y_pred = []
        for X, y in dataset.take(steps):
            y_pred.append(model.predict(X))
            y_true.append(y.numpy().flatten())

        return _np.concatenate(y_true), _np.concatenate(y_pred)

    
    def __call__(
            self, model_name, 
            current_dataset,
            iteration,
            ):
        #TODOC
        model = self.__models[model_name]
        dataset = self.__datasets["test"]

        y_true, y_pred = self.__compute_preds(
                model, dataset,
                self.__datasets["test_steps"]
                )

        scores = [
                self.__metrics[metric](
                    y_true, y_pred 
                    ) for metric in\
                        self.__dfs[current_dataset].columns
                ]

        self.__dfs[current_dataset].loc[iteration] = scores

    def save(self):
        #TODOC
        results_path = _os.path.join(
                self.__save_path,
                self.__dataset_name
                )

        if not _os.path.exists(results_path):
            _os.makedirs(results_path)

        for df_name, df in self.__dfs.items():
            df.loc["mean"] = df.mean()
            df.loc["std"] = df.std()

            df.to_csv(
                    _os.path.join(
                        results_path, df_name + ".csv"
                        )
                    )
