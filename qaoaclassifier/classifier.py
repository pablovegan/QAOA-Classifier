import pennylane as qml
from pennylane.optimize import AdamOptimizer
from pennylane import numpy as np
from sklearn.metrics import roc_auc_score


class QuantumClassifier:
    """
    Quantum circuit with one wire used as a classifier.

    Attributes
    ----------
    X_train (np.ndarray): training features
    Y_train (np.ndarray): training classes
    X_test (np.ndarray): test features
    Y_test (np.ndarray): test classes
    learning_rate (float):
    epochs (int): iterations of our optimizer
    batch_size (int): sizes of the mini-batches used in each step of the gradient
    """

    def __init__(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        learning_rate: float = 0.6,
        epochs: int = 20,
        batch_size: int = 100,
    ) -> None:
        """Initialize instance attributes for our quantum optimizer class."""
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def cost(self, params: np.ndarray) -> float:
        """ "
        Using the predictions expectation values given by our circuit, we can build
        a cost function in such a way that if the signs of the guess and actual label
        coincide, the cost function is smaller, having a minimium when the guess equals
        the actual label. State |0> has <Z>=1 and |1> has <Z> = -1, so we transform the
        labels 0 and 1 of Y_train into 1 and -1.

        Parameters
        ----------
        params (30 arraylike): variational parameters of our circuit

        Returns
        -------
        loss (float): error in the approximation

        """
        Yhat = np.array(QuantumClassifier.qaoa_circ(self.X_train.T, params))
        loss = 1 - Yhat * (1 - 2 * self.Y_train)
        return loss.sum() / len(self.Y_train)

    def iterate_minibatches(self, batch_size: int):
        """
        A generator for batches of the training data.

        Parameters
        ----------
        X_train (array[float]): features data
        Y_train (array[float]): labels

        Returns
        -------
        features (array[float]): one batch of features data of length `batch_size`
        labels (array[float]): one batch of labels of length `batch_size`

        """
        for start_idx in range(0, self.X_train.shape[0] - batch_size + 1, batch_size):
            idxs = slice(start_idx, start_idx + batch_size)
            yield self.X_train[idxs], self.Y_train[idxs]

    @staticmethod
    def metric(Y_test: np.ndarray, Yhat: np.ndarray):
        """
        Returns area under the curve of the Receiver operating characteristic.

        Parameters
        ----------
        Y_test (array[float]): exact test labels
        Yhat (array[float]): predicted test labels

        """
        return roc_auc_score(Y_test, Yhat)

    def optimize_circ(self):
        """
        Train the classifier using Adam optimizer by minimizing the cost function.

        Returns
        -------
        params (np.ndarray): array of optimal parameters

        """
        opt = AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999)

        params = np.random.uniform(QuantumClassifier.depth, requires_grad=True)

        for t in range(self.epochs):
            # for X_batch, Y_batch in self.iterate_minibatches(self.batch_size):
            # def intermediate_cost(params):
            # return Q_classifier.cost(params, X_batch, Y_batch)
            # params = opt.step(intermediate_cost, params)
            params = opt.step(self.cost, params, grad_fn=None)

            Yhat_train = QuantumClassifier.qaoa_circ(self.X_train.T, params)
            Yhat_test = QuantumClassifier.qaoa_circ(self.X_test.T, params)
            loss = self.cost(params)
            res = [
                t + 1,
                loss,
                QuantumClassifier.metric(self.Y_train, Yhat_train),
                QuantumClassifier.metric(self.Y_test, Yhat_test),
            ]
            print(f"Epoch: {res[0]} | Loss: f{res[1]} | Train AUROC: {res[2]} | Test AUROC: {res[3]}")

        return params
