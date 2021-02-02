import pandas as _pd
import os as _os

class Logger:
    #TODOC
    def __init__(
            self, models, metrics, 
            datasets, dataset_name, 
            evaluable_datasets = [
                "pretrain", "clustering"
                ],
            save_path="./results/"
            ):
        
        self.__models = models
        self.__metrics = metrics
        self.__datasets = datasets
        self.__dataset_name = dataset_name
        self.__evaluable_datasets = evaluable_datasets
        self.__save_path = save_path
        self.__make_dataframes()

    def __make_dataframes(self):
        #TODOC
        self.__dfs = {}
        for dataset in self.__evaluable_datasets:
            metric_names = tuple(self.__metrics.keys())
            self.__dfs[dataset] = _pd.DataFrame(
                    columns=[
                        "iteration",
                        *metric_names
                        ]
                    )
            self.__dfs[dataset].set_index("iteration", inplace=True)
    
    def __call__(
            self, model_name, 
            current_dataset,
            iteration,
            ):
        #TODOC
        model = self.__models[model_name]
        dataset = self.__datasets["test"]
        dataset.reset_gen()
        dataset = dataset()

        y_true = self.__datasets["labels"]
        preds = model.predict(
                dataset,
                steps=self.__datasets["steps"]
                )

        scores = [
                self.__metrics[metric](
                    y_true, preds
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
