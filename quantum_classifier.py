import pennylane as qml


class Q_classifier(object):
    """
    Quantum circuit with one wire used as a classifier.

    Attributes
    ----------
    H : operator
        Hamiltonian (Z Pauli gate)
    dev: quantum device
        Simulator for our circuit

    Methods
    -------
    método1 : returns whatever
        descripción
    """
    H = qml.PauliZ(0)
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
    
    def compute_expectation(self):
        pass

    def cost(self, params, labels):
        loss = 0.0
        for i in range(len(labels)):
            f = self.qaoa_circ(params, x[i], y[i])
            # If sign of label == sign of predicted -> smaller loss function
            loss = loss + (1 - f*labels[i])
        return loss / len(labels)

    def metric(self):
        """"
        As suggested in the description of the dataset,
        we use AUROC as a metric for performance.

        References
        ----------
        https://glassboxmedicine.com/2019/02/23/measuring-performance-auc-auroc/
        """
        pass