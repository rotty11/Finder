# Finder

Finder is a collection of wrapper approaches to EEG classification which combine optimization methods for feature selection and classification algorithms. The application includes a script whose objective is to analyze the appoaches to find those that present better trade-off between execution time, energy consumption and accuracy rate.

## Version

1.0

## Author

Juan José Escobar ([jjescobar@ugr.es](mailto:jjescobar@ugr.es)) and Manuel Sánchez Jiménez.

## Requirements

Finder requires Python 3.0.9 or superior.

## Usage

The document `docs/user_guide.pdf` contains the instructions necessary to run the script. Here the available algorithms:

* Bioinspired methods:
  1. Genetic Algorithm (GA).
  1. Particle Swarm Optimization (PSO).
  1. Simulated Annealing (SA).
  1. Artificial Bee Colony (ABC).
  1. Ant Colony Optimization (ACO).

* Classification algorithms:
  1. K-Nearest Neighbors (KNN).
  1. Support Vector Machine (SVM).
  1. Extreme Gradient Boosting (XGBM).
  1. LightGBM.
  1. Multilayer Perceptron (MLP).

## Output

* Classification rate.
* Energy consumption.
* Execution time.

## Publications

#### Conferences

1. J.J. Escobar, D. Aquino-Brítez, B. Prieto, R.J. Martínez, G.M. Ortiz and A. Ortiz. *Balancing Accuracy and Energy Efficiency in EEG Classification: An Evaluation of Wrapper-based Approaches*. In: **To be determined**. 2025.

## Funding

This work has been funded by:

* Andalusian *Consejería de Universidad, Investigación e Innovación* under grant number DGP_PIDI_2024_01887.
* Spanish *Ministerio de Ciencia, Innovación y Universidades* under grants number PID2022-137461NB-C32 and PID2022-137461NB-C31.
* Paraguayan *National Council of Science and Technology (CONACYT) under grant BINV04-68*.
* *European Regional Development Fund (ERDF)*.

<div style="text-align: right">
  <img src="https://raw.githubusercontent.com/rotty11/Finder/main/docs/logos/junta.png" height="60">
  <img src="https://raw.githubusercontent.com/rotty11/Finder/main/docs/logos/miciu.jpg" height="60">
  <img src="https://raw.githubusercontent.com/rotty11/Finder/main/docs/logos/erdf.png" height="60">
  <img src="https://raw.githubusercontent.com/rotty11/Finder/main/docs/logos/conacyt.jpg" height="60">
</div>

## License

[GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.md).

## Copyright

Finder © 2025 [University of Granada](https://www.ugr.es/).

If you use this software, please cite it.
