# Algorithm 1: Four-query Clifford tester
https://www.arxiv.org/abs/2510.07164 pg. 21

## Structure

- `lib/` contains various utility functions, which come together into `lib.clifford_tester.clifford_tester`
- `algorithm-1.ipynb` contains some demo usage of the Clifford tester
- `qi-testing.ipynb` contains demos of running the Clifford tester on Quantum Inspire's hardware
- `scripts/expected_acceptance_probability.py` calculates the theoretical acceptance probability for a given unitary (in the demo script, a TOFFOLI) using the formula from the paper above

## Algorithm breakdown

- Input: Black-box access to an $n$-qubit unitary $U$.

1. $x \leftarrow \text{Uniform}(\mathbb{F}_2^{2n})$

> $\mathbb{F}_2$ represents a 2-element field (a set with well-defined addition and multiplication operators)
>
> For this context, we can take an element $x \in \mathbb{F}_2$ to be a binary digit $x \in \{0, 1\}$
>
> Then $\mathbb{F}_2^{2n}$ is the set of $2n$ bit strings $\{0, 1\}^{2n}$

So we want a uniformly random $2n$-bit binary string called $x$

2. Prepare two independent copies of $U^{\otimes 2} | P_x \rangle \rangle$

> $P$ represents an $n \times n$ Weyl operator; a matrix with $Z$ and $X$ gates on each qubit, with each toggleable on and off by an input $(a, b)$ (where $a$ and $b$ are of length $n$)
> 
> $$ P_{a, b} = i^{\langle a , b \rangle } Z^{a_1} \otimes \cdots \otimes Z^{a_n} X^{b_1} \otimes \cdots \otimes X^{b_n} $$
> 
> Any $n \times n$ unitary can be expressed as a linear combination of Weyl operators
> 
> The Choi state of a matrix $M \in \mathbb{C}^{n \times n}$ is a way of mapping $M$ to a quantum state
> 
> $$ | M \rangle \rangle = \frac{1}{\sqrt{n}} \sum_{i, j = 1}^n M_{ij} | i \rangle | j \rangle $$
> 
> We can produce this with a quantum circuit by first making the maximally entangled state (tensor product of Bell pairs)
> 
> $$ | \psi \rangle = \frac{1}{\sqrt{n}} \sum_i | i \rangle | i \rangle $$
> 
> Then apply
> 
> $$ (M \otimes I) | \psi \rangle $$

So we want to use $x$ to generate a Weyl operator, and then apply it to the $2n$-qubit maximally entangled state to produce $| P_x \rangle \rangle$, then apply $U^{\otimes 2}$ to it, twice.

3. Measure each copy in the Bell basis $\{ | P_y \rangle \rangle \langle \langle P_y | \}_y$ to obtain outcomes $y$ and $y'$

> $| P_y \rangle \rangle \langle \langle P_y |$ is the **projector** onto the Choi state $| P_y \rangle \rangle$
>
> The set $\{ | P_y \rangle \rangle \}_y$ for all $y \in \mathbb{F}_2^{2n}$ forms an orthonormal basis for the $2n$-qubit Hilbert space (there are $4^n$ such states, matching the dimension $2^{2n}$)
>
> So $\{ | P_y \rangle \rangle \langle \langle P_y | \}_y$ is a complete set of projectors — one for each basis state — defining a projective measurement
>
> When we measure, we get outcome $y$ with probability $| \langle \langle P_y | \psi \rangle \rangle |^2$

To measure in this basis with a circuit: undo the Bell state preparation (CNOT then H on each pair), then measure in the computational basis. The classical outcome directly gives $y$.

4. if $y = y'$ then return _Accept_
5. else return _Reject_
