# 3D Bin Packing Problem (3D-BPP) Optimization System

This project implements a comprehensive optimization system for the 3D Bin Packing Problem (3D-BPP) using various evolutionary and swarm intelligence algorithms. It supports loading data from Mendeley datasets and Pickle (`.pkl`) files, offering visualization and detailed configuration options.

## Features

*   **Algorithms:**
    *   **GA (Genetic Algorithm)**: Classic approach.
    *   **BRKGA (Biased Random-Key Genetic Algorithm)**: State-of-the-art for BPP.
    *   **DE (Differential Evolution)**: Uses SHADE adaptive mechanism.
    *   **ACO (Ant Colony Optimization)**: Pheromone-based sequencing.
    *   **PSO (Particle Swarm Optimization)**: Swarm-based search.
    *   **CMA-ES (Covariance Matrix Adaptation ES)**: Advanced continuous optimization.
    *   **MFEA (Multifactorial Evolutionary Algorithm)**: Multi-tasking capability.
    *   **NSGA-II**: Multi-objective optimization (Implementation pending).
*   **Encodings:** Random Key Encoding (Sequence, Rotation, Heuristic).
*   **Packing Heuristics:** Maximal-Space Representation for efficient space management.
*   **Input Formats:** text files (Mendeley format) and Python Pickle (`.pkl`) DataFrames.
*   **Visualization:** Interactive 3D visualization using Plotly.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd 3D-BPP
    ```

2.  **Install dependencies:**
    The project requires Python 3.8+. Install core libraries:
    ```bash
    pip install numpy pandas pyyaml plotly scipy
    ```

## Usage

The system is run via `main.py`. You can specify the algorithm, data file, and configuration.

### Basic Command
```bash
python main.py --algo <ALGORITHM_NAME> --data <PATH_TO_DATA> [--viz]
```

### Examples

**1. Run Differential Evolution (Default) on Pickle Data:**
```bash
python main.py --algo de --data "data/instances/test-data/Input/Elhedhli_dataset/products.pkl" --viz
```

**2. Run BRKGA (Biased Random-Key GA):**
```bash
python main.py --algo brkga --data "data/instances/test-data/Input/Elhedhli_dataset/products.pkl" --viz
```

**3. Run Ant Colony Optimization (ACO):**
```bash
python main.py --algo aco --data "data/instances/test-data/Input/Elhedhli_dataset/products.pkl" --viz
```

**4. Run Particle Swarm Optimization (PSO):**
```bash
python main.py --algo pso --data "data/instances/test-data/Input/Elhedhli_dataset/products.pkl" --viz
```

**5. Run MFEA (Multifactorial):**
```bash
python main.py --algo mfea --data "data/instances/test-data/Input/Elhedhli_dataset/products.pkl" --viz
```

### Configuration

Configuration files are located in `config/`. The system automatically loads the specific config for the chosen algorithm (e.g., `config/brkga_v2.yaml` for `--algo brkga`).

You can override settings by providing a custom config file:
```bash
python main.py --algo de --config my_custom_config.yaml
```

## Project Structure

*   `main.py`: Entry point.
*   `src/core/`: Core classes (Item, Bin, Population).
*   `src/algorithms/`: Implementations of DE, BRKGA, ACO, etc.
*   `src/evaluation/`: Packing simulator and Fitness function.
*   `src/utils/`: Parsers (Mendeley, Pickle), Visualization, ConfigLoader.
*   `config/`: YAML configuration files.
*   `data/`: Input datasets.