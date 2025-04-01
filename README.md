# QML-Research

Repository for QML research on the quantum entanglement detection problem.

## Project Structure

The repository is organized as follows:

- **`dataset/`**: Contains datasets used for experiments.
  - **`3x3/`**, **`4x4/`**, **`5x5/`**, **`7x7/`**: Datasets for different grid sizes.
  - **`efficiency_study/`**: Datasets for sizestest-related experiments, that is, different training dataset size.
  - **`toy/`**: Toy datasets for quick initial testing with different ppt ratio.

- **`environment/`**: Contains environment configuration files.
  - **`penny-env.yml`**: YAML file for setting up the Python environment with dependencies.

- **`profiling/`**: Stores profiling results for performance analysis.
  - Examples: `3x3.prof`, `svm_16mp.prof`, `svm_8mp.prof`.

- **`report_graphs/`**: Contains graphs and visualizations for reports.
  - Subdirectories like **`3x3/`**, **`4x4/`**, etc., organize graphs by dataset size.

- **`results/`**: Stores results from experiments and analyses. Both graphs and efficiency results (`efficiency_results_*.csv`), which are the results for different sizes of the training set. (Right, the word efficiency is not the best choice for what I mean)

- **`src/`**: Source code for the project, including implementations of algorithms, models, and utilities.

*Disclaimer: there is a lot a lot a lot of code repetition that I am not proud of.*

- **`svm_specs_n100/`**: Likely contains specifications or results related to SVM experiments with `n=100`.

## Getting Started

### Setting Up the Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/QML-Research.git
   cd QML-Research
   ```
2. - Required dependencies are listed in [`environment/penny-env.yml`](environment/penny-env.yml) to create a conda environment.


