import pandas as _pd
import os as _os

class Logger:
    def __init__(
            self, models, metrics, 
            datasets, dataset_name, 
            trainable_models = [
                "autoencoder", "full_model"
                ],
            save_path="./results/"
            ):
        
        self.__models = models
        self.__metrics = metrics
        self.__datasets = datasets
        self.__dataset_name = dataset_name
        self.__trainable_models = trainable_models
        self.__save_path = save_path
        self.__make_dataframes()

    def __make_dataframes(self):
        self.__dfs = {}
        for model in self.__trainable_models:
            metric_names = tuple(self.__metrics.keys())
            self.__dfs[model] = _pd.DataFrame(
                    columns=[
                        "iteration",
                        *metric_names
                        ]
                    )
            self.__dfs[model].set_index("iteration", inplace=True)
    
    def __call__(self, model_name, dataset_name, iteration):
        model = self.__models[model_name]
        dataset = self.__datasets[dataset_name]
        y_true = self.__datasets["labels"]
        preds = model.predict(dataset)

        scores = [
                self.__metrics[metric](
                    y_true, preds
                    ) for metric in\
                        self.__dfs[dataset_name].columns
                ]

        self.__dfs[dataset_name].loc[iteration] = scores

    def save(self):
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
