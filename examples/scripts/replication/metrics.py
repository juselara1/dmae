import numpy as _np

from dmae.metrics import unsupervised_classification_accuracy as _uacc
from sklearn.metrics import normalized_mutual_info_score as _nmi, adjusted_rand_score as _ars

_metrics = {
        "uACC": lambda y_true, y_pred:\
                _uacc(
                    y_true, _np.argmax(
                        y_pred, axis=1
                        )
                    ),
        "NMI": lambda y_true, y_pred:\
                _nmi(
                    y_true, _np.argmax(
                        y_pred, axis=1
                        )
                    ),
        "ARS": lambda y_true, y_pred:\
                _ars(
                    y_true, _np.argmax(
                        y_pred, axis=1
                        )
                    )
        }

def make_metrics():
    return _metrics
