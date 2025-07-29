# Atom

Atom is a parallel and distributed Binary Particle Swarm Optimization (BPSO) algorithm to EEG classification. The bioinspired procedure follows a master-worker scheme where the particles are distributed among the nodes of a cluster. Inside of each node, the application is able to parallelize the evaluation of the particles using all the CPU threads simultaneously, which provides 2 levels of parallelism.

## Version

1.0

## Author

Juan José Escobar ([jjescobar@ugr.es](mailto:jjescobar@ugr.es)) and Jesús López Rodríguez ([jlopezpeque@gmail.com](mailto:jlopezpeque@gmail.com))

## Requirements

Atom requires a GCC compiler. It also depends on the following APIs and libraries:

* [OpenMPI](https://www.open-mpi.org/doc/current/).

## Usage

To build the application, a Makefile and Dockerfile are provided.

* Compile with `make -j N_FEATURES=3600`.
* Run with `mpirun --bind-to none --map-by node --host [LIST_OF_HOSTS] ./pso [ARGS]`.
* ARGS:
  1. `nP`: Number of particles in the swarm.
  2. `nH`: Number of CPU threads to compute.
  1. `nI`: Number of iterations of the BPSO algorithm.
  1. `cC`: Value of the cognitive component.
  1. `cI`: Value of the inertia component.
  1. `cS`: Value of the social component.
  1. `k`: Number of Neighbors of the KNN algorithm.
  1. `sI`: Enable the use of OpenMP SIMD directive.
  1. `pC`: Probability that the component of the position vector takes value 1.

## Output

* Average and best classification rate.
* Best feature subset.
* Execution time.

## Publications

#### Conferences

1. J.J. Escobar, J. López-Rodríguez, D. García-Gil, R. Morcillo-Jiménez, B. Prieto, A. Ortiz and D. Kimovski. *Analysis of a Parallel and Distributed BPSO Algorithm for EEG Classification: Impact on Energy, Time and Accuracy*. In: **11th International Conference on Bioinformatics and Biomedical Engineering. IWBBIO'2024**. Gran Canaria, Spain: Springer, July 2024, pp. 77-90. DOI: [10.1007/978-3-031-64629-4_6](https://doi.org/10.1007/978-3-031-64629-4_6).

## Funding

This work has been funded by:

* University of Granada under grant number PPJIA-2023-25.
* Spanish *Ministerio de Ciencia, Innovación y Universidades* under grants number PID2022-137461NB-C32 and PID2020-119478GB-I00.
* Spanish *Ministerio de Universidades* as part of the program of mobility stays for professors and researchers in foreign higher education and research centers under grant number CAS22/00332.
* *European Regional Development Fund (ERDF)*.

<div style="text-align: right">
  <img src="https://raw.githubusercontent.com/rotty11/Atom/main/docs/logos/miciu.jpg" height="60">
  <img src="https://raw.githubusercontent.com/rotty11/Atom/main/docs/logos/erdf.png" height="60">
  <img src="https://raw.githubusercontent.com/rotty11/Atom/main/docs/logos/ugr.jpg" height="60">
</div>

## License

[GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.md).

## Copyright

Atom © 2024 [University of Granada](https://www.ugr.es/).

If you use this software, please cite it.
