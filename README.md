# fullyIncrementalPH: A Python Library To Compute "Fully Incremental" Persistent Homology

The first model for the computation of persistent homology on streaming data was introduced by [this paper](https://ieeexplore.ieee.org/document/9346556). Another model was later developed that increased the 'incrementalism' of the previous model. This model was developed as part of a [C++ library](https://github.com/wilseypa/lhf) for fast computation of persistent homology. However, the matrix reduction step in the second model was still performed in an 'offline' fashion. The goal of the current project is to develop a fully incremental computational model for persistent homology that performs all the computations in an online manner. The theoretical foundation for the online boundary matrix reduction is established [here](https://github.com/Anindya-Moitra/fullyIncrementalPH/blob/master/theoreticalFoundation/main.pdf).

---
