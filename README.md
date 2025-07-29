# Finder

Finder is a collection of different wrapper approaches to EEG classification. These methods combine optimization methods for feature selection (GA, PSO, ABC, SA and ACO) and classification algorithms. The application includes a script whose objective is to analyze the appoaches to find those that present better trade-off between execution time, energy consumption and accuracy rate.

## Version

1.0

## Author

Juan José Escobar ([jjescobar@ugr.es](mailto:jjescobar@ugr.es)) and Manuel Sánchez Jiménez.

## Requirements

EEGxplorer requires Python 3.0.9 or superior.

## Usage

The document `user_guide.pdf` contains the instructions necessary to run the script. Here the available algorithms:

* Bioinspired methods:
  1. Genetic Algorithm (GA).
  1. Particle Swarm Optimization (PSO).
* Run with `mpirun --bind-to none --map-by node --host [LIST_OF_HOSTS] ./pso [ARGS]`.
* ARGS:
  1. `nP`: Number of particles in the swarm.
  2. `nH`: Number of CPU threads to compute.
  1. `nI`: Number of iterations of the BPSO algorithm.

## Output

* Classification rate.
* Energy consumption.
* Execution time.

## Publications

#### Conferences

1. J.J. Escobar, D. Aquino-Brítez, B. Prieto, R.J. Martínez, G.M. Ortiz and A. Ortiz. *Balancing Accuracy and Energy Efficiency in EEG Classification: An Evaluation of Wrapper-based Approaches*. In: **To be determined**. 2025.

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

Finder © 2025 [University of Granada](https://www.ugr.es/).

If you use this software, please cite it.
