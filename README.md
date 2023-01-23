
# PerturbPaths

  

Implementation and extension of [PATHPERTURB](https://arxiv.org/pdf/2107.03347.pdf), but with a slightly modified goal. Given a weighted graph and a set of pairs of nodes within that graph, this code is able to find the minimum weight added to each edge that will increase the pariwise distances between those nodes to prespecified target distances.
  

The main algorithm is contained in [general_attack.py](./general_attack.py)

In each iteration, it will:

- Select path(s) to add to the set being considered (code in [selector_classes.py](./selector_classes.py))

- Calculate (or approximate) the smallest perturbation of edge weights that will make all paths at least as long as some predefined goal (code in [perturbation_classes.py](./perturbation_classes.py))

- Check if this local optimum is also a global solution (i.e. a global optimum)

  

A series of experiments can be performed using [run_experiments.py](./run_experiments.py)

  

## References

Miller, Benjamin & Shafi, Zohair & Ruml, Wheeler & Vorobeychik, Yevgeniy & Eliassi-Rad, Tina & Alfeld, Scott. (2021). Optimal Edge Weight Perturbations to Attack Shortest Paths.
