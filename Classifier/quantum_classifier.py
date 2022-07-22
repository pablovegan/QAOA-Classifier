import pennylane as qml
from pennylane.optimize import AdamOptimizer
from pennylane import numpy as np
from sklearn.metrics import roc_auc_score


class Q_classifier(object):
    """
    Quantum circuit with one wire used as a classifier.

    Attributes
    ----------
    dev (quantum device): Simulator for our circuit
    Hc (Hamiltonian): Cost Hamiltonian of QAOA circuit
    Hm (Hamiltonian): Mixer Hamiltonian of QAOA circuit
    depth (int): number of layers in QAOA circuit

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

    def __init__(self, learning_rate: float = 0.6,
                        epochs: int = 20,
                        batch_size: int = 100) -> None:
        '''
        Initialize variables for our quantum optimizer class.
        
        Parameters
        ----------
        learning_rate (float): 
        epochs (int): iterations of our optimizer
        batch_size (int): sizes of the mini-batches used in each step of the gradient

        '''
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
    
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
        We use the QAOA ansatz as our circuit.

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

    def cost(self, params: np.ndarray, X_train: np.ndarray, Y_train: np.ndarray) -> float:
        """"
        Cost function deppends on a set of parameters, which will be the time
        we evolve the mixer hamiltonian Hm in each layer layer. The time we evolve
        Hc will be given by the features of our model. Thus, there will be 30 layers,
        since our data has 30 features.  Whereas exp(i*Hc*t) changes the phase of our
        state, exp(i*Hm*t) changes the probabilities of measuring |0> and |1>.

        Parameters
        ----------
        params (30 arraylike): variational parameters of our circuit
        X_data (rowsx30 arraylike): data to train our QAOA circuit

        Returns
        -------
        loss (float): error in the approximation

        """
        Yhat = np.array(self.qaoa_circ(X_train.T, params))
        loss = (1 - Yhat*(2*Y_train-1))
        return loss.sum()/len(Y_train)

    @staticmethod
    def iterate_minibatches(X_train, Y_train, batch_size):
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

    def optimize(self, X_train, Y_train):
        """
        Train the classifier using Adam optimizer by minimizing the cost function.
        
        Parameters
        ----------
        X_train (array[float]): features data
        Y_train (array[float]): labels

        """
        opt = AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999)
        params = np.random.uniform(Q_classifier.depth, requires_grad=True)
        for it in range(self.epochs):
            for X_batch, Y_batch in Q_classifier.iterate_minibatches(X_train, Y_train, batch_size=self.batch_size):
                def intermediate_cost(params):
                    return self.cost(params, X_batch, Y_batch)
                params = opt.step(intermediate_cost, params)
                # params, _, _ = opt.step(self.cost, params, X_batch, Y_batch)

    def metric(Y_test, Yhat):
        '''Returns area under the curve of the Receiver operating characteristic.'''
        return roc_auc_score(Y_test, Yhat)


