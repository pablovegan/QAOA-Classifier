from pennylane.optimize import AdamOptimizer
from pennylane import numpy as np
from sklearn.metrics import roc_auc_score

from .circuit import Circuit


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
        self, circuit: Circuit, learning_rate: float = 0.6, epochs: int = 20, batch_size: int = 100
    ) -> None:
        """Initialize instance attributes for our quantum optimizer class."""
        self.circuit = circuit
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray):
        """
        Train the classifier using Adam optimizer by minimizing the cost function.

        Returns
        -------
        params (np.ndarray): array of optimal parameters
        """
        assert X_train.shape[1] % 2 == 0, "Number of columns of the data must be even."

        opt = AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999)

        params = np.random.uniform(2 * self.circuit.layers, requires_grad=True)

        for t in range(self.epochs):
            # for X_batch, Y_batch in self.iterate_minibatches(X_train, Y_train, self.batch_size):
            # def intermediate_cost(params):
            # return Q_classifier.cost(params, X_batch, Y_batch)
            # params = opt.step(intermediate_cost, params)

            params = opt.step(self.cost, params, grad_fn=None, X_train=X_train, Y_train=Y_train)

            loss, Yhat_train = self.cost(params, X_train=X_train, Y_train=Y_train, return_Yhat=True)

            Yhat_test = []
            for i in range(X_test.shape[0]):
                enc_data = X_test[i, :] * params
                Yhat_test.append(self.circuit.expval(enc_data.reshape(-1, 2)))
            Yhat_train = np.array(Yhat_train)
            Yhat_test = np.array(Yhat_test)

            try:
                train_score = roc_auc_score(Y_train, Yhat_train)
                test_score = roc_auc_score(Y_test, Yhat_test)
            except ValueError:
                pass

            print(
                f"Epoch: {t} | Loss: f{loss} | Train AUROC: {train_score} | Test AUROC: {test_score}"
            )

        return params

    def iterate_minibatches(self, X_train: np.ndarray, Y_train: np.ndarray, batch_size: int):
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
        for start_idx in range(0, X_train.shape[0] - batch_size + 1, batch_size):
            idxs = slice(start_idx, start_idx + batch_size)
            yield X_train[idxs], Y_train[idxs]

    def cost(
        self,
        params: np.ndarray,
        X_train: np.ndarray = None,
        Y_train: np.ndarray = None,
        return_Yhat=False,
    ) -> float:
        """
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
        loss = 0
        Yhat_train = []
        for i in range(X_train.shape[0]):
            enc_data = X_train[i, :] * params
            Yhat = self.circuit.expval(enc_data.reshape(-1, 2))
            Yhat_train.append(Yhat)
            loss = 1 - Yhat * (1 - 2 * Y_train[i])
            loss = loss / X_train.shape[0]
        return loss if return_Yhat is False else loss, Yhat_train
