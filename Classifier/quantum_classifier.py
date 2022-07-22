import pennylane as qml


class Q_classifier(object):
    """
    Quantum circuit with one wire used as a classifier.

    Attributes
    ----------
    dev (quantum device): Simulator for our circuit

    Methods
    -------
    método1 : returns whatever
    """
    dev = qml.device("default.qubit", wires=1)
 
    def __init__(self) -> None:
        pass

    def encoding(self):
        pass

    @qml.qnode(dev)
    def qaoa_circ(self):
        """
        We use the QAOA ansatz as our circuit

        Parmeters
        ---------
        
        Returns
        -------

        References
        ----------
        [1] Farhi, Goldstone, Gutmann, "A Quantum Approximate
            Optimization Algorithm" (2014) arXiv:1411.4028
        """
        # Initial state is a |+> state
        qml.Hadamard(wires = 0)

        # ESTRUCTURE OF QAOA CIRCUIT

        return qml.expval(qml.PauliZ(0))

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

