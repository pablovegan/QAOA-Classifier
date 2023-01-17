from abc import ABC, abstractmethod

import pennylane as qml
from numpy import ndarray


class Circuit(ABC):
    """
    Create our ansatz circuit.

    Attributes
    ----------
    wires: int
            the number of wires in our circuit
    device_name: str
        the name of the quantum device to use as backend
    """

    __slots__ = ("wires", "dev")  # faster memory access to the attributes than using __dict__

    def __init__(self, wires: int, device_name: str = "default.qubit"):
        """
        Initialize the Circuit class.

        Parameters
        ----------
        wires : int
            The number of wires in our circuit.
        device_name : srt
            The name of the quantum device to use as backend.
        """
        self.wires = wires
        self.dev = qml.device(device_name, wires=wires)

    @abstractmethod
    def circuit(self, params: ndarray) -> None:
        """Add the parametrized gates necessary to create the circuit."""
        ...

    def expval(self, params: ndarray) -> float:
        """
        Get the expectation value of Z.

        Parameters
        ----------
        params : ndarray
            The parameters of the variational circuit.

        Returns
        -------
        float
            Expectation value of PauliZ for the first (and only) qubit.
        """

        @qml.qnode(self.dev)  # ,diff_method = 'adjoint'
        def _circuit():
            self.circuit(params)
            return qml.expval(qml.PauliZ(0))

        return _circuit()

    def draw(self, params: ndarray) -> None:
        """Draw the circuit."""

        @qml.qnode(self.dev)
        def _circuit():
            self.circuit(params)
            return qml.state()

        fig, ax = qml.draw_mpl(_circuit)()
        fig.show()


class QaoaCircuit(Circuit):
    """
    Quantum circuit using only RY gates, which gives a real state vector.

    Attributes
    ----------
    layers: int
        the number of layers in our circuit

    Methods
    -------
    circuit

    References
    ----------
    The ansatz can be found in [1]_.

    .. [1] https://qiskit.org/documentation/stubs/qiskit.circuit.library.RealAmplitudes.html
    """

    __slots__ = ("layers",)  # faster memory access to the attributes than using __dict__

    def __init__(self, layers: int, wires: int, device_name: str = "qulacs.simulator"):
        """
        Initialize the Circuit class.

        Parameters
        ----------
        wires: int
            the number of wires in our circuit
        layers: int
            the number of layers in our circuit
        device_name: srt
            the name of the quantum device to use as backend
        """
        super().__init__(wires, device_name)
        self.layers = layers

    @staticmethod
    def _qaoa_layer(params: ndarray) -> None:
        """Each QAOA circuit layer consists on the exp of Hc and the exp of Hm."""
        Hc = qml.Hamiltonian([1.0], [qml.PauliZ(0)])
        Hm = qml.Hamiltonian([1.0], [qml.PauliX(0)])
        qml.ApproxTimeEvolution(Hc, params[0], n=1)  # ? change parameter n?
        qml.ApproxTimeEvolution(Hm, params[1], n=1)

    def circuit(self, params: ndarray) -> None:
        """
        We use the QAOA ansatz as our circuit. The circuit deppends on a set of parameters
        (params), which will be the time we evolve the mixer hamiltonian Hm in each layer.
        The time we evolve Hc will be given by the features of our model (Xi_train). Thus,
        there will be 30 layers, since our data has 30 features. Whereas exp(i*Hc*t) changes
        the phase of our state, exp(i*Hm*t) changes the probabilities of measuring |0> and |1>.

        Parmeters
        ---------
        params : (layers, 2) arraylike
            The parameters for our QAOA circuit.

        Returns
        -------
        expval (float): expected value of the Hamiltonian Hc

        References
        ----------
        [1] Farhi, Goldstone, Gutmann, "A Quantum Approximate
            Optimization Algorithm" (2014) arXiv:1411.4028
        """
        qml.Hadamard(wires=0)  # Initial state is a |+> state
        qml.layer(self._qaoa_layer, self.layers, params)
