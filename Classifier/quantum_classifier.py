import pennylane as qml
from pennylane.optimize import AdamOptimizer
from pennylane import numpy as np

class Q_classifier(object):
    """
    Quantum circuit with one wire used as a classifier.

    Attributes
    ----------
    dev (quantum device): Simulator for our circuit
    Hc (Hamiltonian): Cost Hamiltonian of our QAOA circuit
    Hm (Hamiltonian): Mixer Hamiltonian of our QAOA circuit

    Methods
    -------
    método1 : returns whatever
    """
    dev = qml.device("default.qubit", wires=1)
    Hc = qml.Hamiltonian([1.], [qml.PauliZ(0)])
    Hm = qml.Hamiltonian([1.], [qml.PauliX(0)])
    depth = 30

    def __init__(self) -> None:
        pass
    
    @staticmethod
    def qaoa_layer(t_Hc, t_Hm):
        """"
        Each QAOA circuit layer consists on the exp of Hc and the exp of Hm.
        
        Parmeters
        ---------
        gamma (float): Hc evolution time
        alpha (float): Hm evolution time
        """   
        qml.ApproxTimeEvolution(Q_classifier.Hc, t_Hc, 1)
        qml.ApproxTimeEvolution(Q_classifier.Hm, t_Hm, 1)

    @qml.qnode(dev)
    def qaoa_circ(self, Xi_train, params):
        """
        We use the QAOA ansatz as our circuit

        Parmeters
        ---------
        params (30 arraylike): parameters for our QAOA circuit
        Xi_data (30 arraylike): data to train our QAOA circuit

        Returns
        -------

        References
        ----------
        [1] Farhi, Goldstone, Gutmann, "A Quantum Approximate
            Optimization Algorithm" (2014) arXiv:1411.4028
        """
        qml.Hadamard(wires = 0)  # Initial state is a |+> state
        qml.layer(Q_classifier.qaoa_layer, Q_classifier.depth, Xi_train, params)
        return qml.expval(qml.PauliZ(0))

    def cost(self, params, X_train, Y_train):
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
        loss = 0.0
        Yhat = np.zeros(len(Y_train))
        for i in range(len(Y_train)):
            Yhat[i] = self.qaoa_circ(X_train[i], params)
            # If sign of label == sign of predicted -> smaller loss function
            loss = loss + (1 - Yhat[i]*Y_train[i])
        return loss / len(Y_train)

    def optimize():
        """Train the classifier using Adam optimizer."""
        num_layers = 3
        learning_rate = 0.6
        epochs = 10
        batch_size = 32

        opt = AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999)
        params = np.random.uniform(size=(num_layers, 3), requires_grad=True)

        params, _, _, _ = opt.step(self.cost, params, Xbatch, ybatch, state_labels)






    def iterate_minibatches(inputs, targets, batch_size):
        """
        A generator for batches of the input data

        Args:
            inputs (array[float]): input data
            targets (array[float]): targets

        Returns:
            inputs (array[float]): one batch of input data of length `batch_size`
            targets (array[float]): one batch of targets of length `batch_size`
        """
        for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
            idxs = slice(start_idx, start_idx + batch_size)
            yield inputs[idxs], targets[idxs]

