# Error Bounds for Stochastic User Equilibrium Traffic Assignment

This repository contains the implementation of my MS thesis:  

**"Error Bounds for Stochastic User Equilibrium Traffic Assignment"**  

**Author:** Debojjal Bagchi, The University of Texas at Austin  

**Supervising Committee:**  
- Dr. Stephen D. Boyles, Supervisor (@sboyles)
- Dr. Zhaomiao Guo, Reader

---

## Conferences

This work was (or will be) presented at the following conferences:

- **Bagchi, D.** and Boyles, S. D. (2025). *Error bounds for stochastic user equilibrium traffic assignment.* *Accepted in* _12th Triennial Symposium on Transportation Analysis conference (TRISTAN XII)_, Okinawa, Japan.

- **Bagchi, D.**, and Boyles, S. D. (2024, October). *Error Bounds for Stochastic User Equilibrium Traffic Assignment.* *In* _Institute for Operations Research and the Management Sciences (INFORMS) Annual Meeting 2024_, Seattle, USA. (TSL Invited session)

The full MS thesis will by May 30th, 2025. 

---

## Abstract

In stochastic user equilibrium traffic assignment, we develop bounds on the distance between a given feasible solution and the equilibrium solution. The intent is to provide guidance on termination criteria to reduce run times, which is important because traffic assignment problem is often a subproblem to a more complex bilevel optimization.  These mathematical bounds complement existing rules-of-thumb drawn empirically from numerical case studies.  We demonstrate that proposed path flow, link flow and travel cost bounds are tight and cannot be further improved without additional restrictions on the network topology or problem instance. Empirical results show that the bound on the distance between any feasible path flow and the equilibrium flow follow a linear rate of convergence near equilibrium. Results on city-level networks indicate that the bounds are reasonably tight. Finally, we illustrate the practical benefits of our bounds through their application to a simple network design problem involving tolling.

---

## Repository Structure

```text
.
├── main.py                 # Runs the SUE assignment and computes bounds  
├── readNetwork.py          # Loads input network and demand data from DAT files  
├── SUEPath.py              # The SUE and MSA-based logit path flow computation  
├── helper.py               # Miscellaneous helper functions  
├── genPlots.py             # Plot generation for results and thesis figures  
├── networkDesign.py        # Implements the toll-based network design case study  
├── Plots/                  # Contains all generated plots  
│   └── paperPlots/         # Plots used directly in the thesis/paper  
├── testNetworks/           # Input data files  
├── optimalSolutions/       # Optimal path flow solutions for a given tolerance  
└── derivationChecks/       # Symbolic derivations and proof verifications (Jupyter notebooks)
```


---

## How to Run

To execute the full SUE traffic assignment along with bound computations, first set all the parameters in `main.py` file and then run `main.py`.

To skip bound computation, disable it using the `calBound` flag in `main.py`.

---

## Outputs

- **All plots** are saved under the `Plots/` directory.  
- **Thesis figures** are available in `Plots/paperPlots/`, all generated from a single  main.py  run.  
- **Input networks** are stored in `testNetworks/`. *(Please fix citation source)*  
- **Reference optimal solutions** (for convergence gap of 1) are saved in `optimalSolutions/`.  
- **Proofs and derivation verifications** are in `derivationChecks/`, implemented using symbolic solvers (e.g., SymPy).  

---

## Requirements

Install dependencies using:

    pip install -r requirements.txt

Tested with Python 3.12.5

---
## Acknowledgements

- The input data files are sourced from [TransportationNetworks](https://github.com/bstabler/TransportationNetworks).
- The code to run Stochastic User Equilibrium (SUE) was adapted from [Traffic-Assignment](https://github.com/prameshk/Traffic-Assignment).

---

## Contact

For questions, collaborations, or further discussions, feel free to reach out.
I am availabale at debojjalb@utexas.edu or https://debojjalb.github.io