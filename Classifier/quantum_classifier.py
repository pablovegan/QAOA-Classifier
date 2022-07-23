import pennylane as qml
from pennylane.optimize import AdamOptimizer
from pennylane import numpy as np
from sklearn.metrics import roc_auc_score


class Q_classifier(object):
    """
    Quantum circuit with one wire used as a classifier.

    Class Attributes
    ----------------
    dev (quantum device): Simulator for our circuit
    Hc (Hamiltonian): Cost Hamiltonian of QAOA circuit
    Hm (Hamiltonian): Mixer Hamiltonian of QAOA circuit
    depth (int): number of layers in QAOA circuit

    Instance Attributes
    -------------------
    X_train (np.ndarray): training features
    Y_train (np.ndarray): training classes
    X_test (np.ndarray): test features
    Y_test (np.ndarray): test classes
    learning_rate (float): 
    epochs (int): iterations of our optimizer
    batch_size (int): sizes of the mini-batches used in each step of the gradient

    Methods
    -------
    qaoa_layer: individual layer of QAOA circuit   
    qaoa_circ: creates qaoa circuit and returns expected value of Hc
    cost: returns the error of our classifier
    iterate_minibatches: returns an iterator of mini-batches
    optimize: minimizes cost function
    metric: AUROC metric to check the performance of our optimizer

    """
    dev = qml.device("qulacs.simulator", wires=1) # qulacs is faster for simulating circuits
    Hc = qml.Hamiltonian([1.], [qml.PauliZ(0)])
    Hm = qml.Hamiltonian([1.], [qml.PauliX(0)])
    depth = 30

    def __init__(self,
                X_train: np.ndarray,
                Y_train: np.ndarray,
                X_test: np.ndarray,
                Y_test: np.ndarray,
                learning_rate: float = 0.6,
                epochs: int = 20,
                batch_size: int = 100) -> None:
        '''Initialize instance attributes for our quantum optimizer class.'''
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
    
    @staticmethod
    def qaoa_layer(t_Hc: float, t_Hm: float) -> None:
        """"
        Each QAOA circuit layer consists on the exp of Hc and the exp of Hm.
        
        Parmeters
        ---------
        t_Hc (float): Hc evolution time
        t_Hm (float): Hm evolution time

        """   
        qml.ApproxTimeEvolution(Q_classifier.Hc, t_Hc, 1)
        qml.ApproxTimeEvolution(Q_classifier.Hm, t_Hm, 1)

    
    @staticmethod
    @qml.qnode(dev)  # ,diff_method = 'adjoint'
    def qaoa_circ(Xi_train: np.ndarray, params: np.ndarray) -> float:
        """
        We use the QAOA ansatz as our circuit. The circuit deppends on a set of parameters
        (params), which will be the time we evolve the mixer hamiltonian Hm in each layer.
        The time we evolve Hc will be given by the features of our model (Xi_train). Thus,
        there will be 30 layers, since our data has 30 features. Whereas exp(i*Hc*t) changes
        the phase of our state, exp(i*Hm*t) changes the probabilities of measuring |0> and |1>.

        Parmeters
        ---------
        Xi_data (30 arraylike): data to train our QAOA circuit
        params (30 arraylike): parameters for our QAOA circuit

        Returns
        -------
        expval (float): expected value of the Hamiltonian Hc

        References
        ----------
        [1] Farhi, Goldstone, Gutmann, "A Quantum Approximate
            Optimization Algorithm" (2014) arXiv:1411.4028

        """
        qml.Hadamard(wires = 0)  # Initial state is a |+> state
        qml.layer(Q_classifier.qaoa_layer, Q_classifier.depth, Xi_train, params)
        return qml.expval(qml.PauliZ(0))

    def cost(self, params: np.ndarray) -> float:
        """"
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
        Yhat = np.array(Q_classifier.qaoa_circ(self.X_train.T, params))
        loss = 1-Yhat*(1-2*self.Y_train)
        return loss.sum()/len(self.Y_train)

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
        """ Train the classifier using Adam optimizer by minimizing the cost function."""
        opt = AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999)
        
        params = np.random.uniform(Q_classifier.depth, requires_grad=True)

        for t in range(self.epochs):
            for X_batch, Y_batch in self.iterate_minibatches(self.batch_size):
                # def intermediate_cost(params):
                    # return Q_classifier.cost(params, X_batch, Y_batch)
                # params = opt.step(intermediate_cost, params)
                params = opt.step(self.cost, params, grad_fn=None)
                
            Yhat_train = Q_classifier.qaoa_circ(self.X_train.T, params)
            Yhat_test= Q_classifier.qaoa_circ(self.X_test.T, params)
            loss = self.cost(params)
            res = [t + 1, loss, Q_classifier.metric(self.Y_train, Yhat_train),
                    Q_classifier.metric(self.Y_test, Yhat_test)]
            print("Epoch: {:2d} | Loss: {:3f} | Train AUROC: {:3f} | Test AUROC: {:3f}".format(*res))

