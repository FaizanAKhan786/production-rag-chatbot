# Quantum Algorithms and NISQ Era Physical Limitations

## Abstract
Noisy Intermediate-Scale Quantum (NISQ) devices feature 50 to a few hundred physical qubits without fault-tolerant quantum error correction. This paper benchmarks fundamental quantum algorithms including Shor's integer factorization and Grover's search algorithm under realistic decoherence and gate infidelities.

## 1. Introduction
Quantum computing leverages quantum mechanical superposition and entanglement to execute non-classical computation. Unlike classical binary bits which are strictly 0 or 1, a qubit represents a normalized state vector in a two-dimensional Hilbert space: $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$.

## 2. Fundamental Quantum Algorithms
Shor's algorithm provides an exponential speedup over the best known classical factoring algorithm, factoring an integer $N$ in polynomial time $O((\log N)^3)$ using quantum phase estimation.
Grover's algorithm searches an unsorted database of $N$ elements in $O(\sqrt{N})$ queries, achieving a quadratic speedup using amplitude amplification.

## 3. Physical Hardware Constraints in the NISQ Era
Current superconducting and trapped-ion qubits suffer from finite coherence times $T_1$ (longitudinal relaxation) and $T_2$ (dephasing). Two-qubit gate fidelities (such as Controlled-NOT and CZ gates) remain around 99.0% to 99.8%, limiting circuit depth before the quantum state decays into maximum entropy thermal noise.

## 4. Conclusion
While fault-tolerant quantum computation requires millions of physical qubits with surface code syndromes, NISQ-era variational quantum eigensolvers (VQE) demonstrate promise in molecular electronic structure simulations.
