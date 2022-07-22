import pennylane as qml


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
 
    def __init__(self, depth) -> None:
        self.depth = depth

    def encoding(self):
        pass
    
    @staticmethod
    def qaoa_layer(gamma, alpha):   
        qml.ApproxTimeEvolution(Q_classifier.Hc, gamma, 1)
        qml.ApproxTimeEvolution(Q_classifier.Hm, alpha, 1)

    
    def qaoa_circ(self, params):
        """
        We use the QAOA ansatz as our circuit

        Parmeters
        ---------
        params (2xdepth arraylike): parameters for our QAOA circuit

        Returns
        -------

        References
        ----------
        [1] Farhi, Goldstone, Gutmann, "A Quantum Approximate
            Optimization Algorithm" (2014) arXiv:1411.4028
        """
        # Initial state is a |+> state
        qml.Hadamard(wires = 0)
        qml.layer(Q_classifier.qaoa_layer, self.depth, params[0], params[1])


    @qml.qnode(dev)
    def cost(self, params, labels):
        loss = 0.0
        for i in range(len(labels)):
            f = self.qaoa_circ(params, x[i], y[i])
            # If sign of label == sign of predicted -> smaller loss function
            loss = loss + (1 - f*labels[i])
        return loss / len(labels)


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

