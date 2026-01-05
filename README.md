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

The system is run via `main.py`. You can specify the algorithm, data file, number of generations, and enable visualization.

### Basic Command
```bash
python main.py --algo <ALGORITHM_NAME> --data <PATH_TO_DATA> [--gen <GENERATIONS>] [--viz] [--limit <N_ITEMS>]
```

### Command Line Arguments

- `--algo`: Algorithm to run. Choices: `ga`, `es`, `aco`, `pso` (default: `ga`)
- `--data`: Path to data file (`.txt` or `.pkl` format)
- `--gen`: Number of generations to run (default: 100)
- `--viz`: Enable visualization (generates convergence plots and 3D packing visualizations)
- `--limit`: Limit number of items to load from large datasets (useful for `.pkl` files)

### How to Run

#### 1. Run on Text Files (Mendeley Format)
```bash
# Run Genetic Algorithm on a text instance
python main.py --algo ga --data "data/instances/test-data/Input/3dBPP_1.txt" --gen 100

# Run with visualization
python main.py --algo ga --data "data/instances/test-data/Input/3dBPP_1.txt" --gen 100 --viz
```

#### 2. Run on Pickle Files (.pkl)
```bash
# Run on full dataset
python main.py --algo ga --data "data/instances/test-data/Input/Elhedhli_dataset/products.pkl" --gen 50

# Run on limited number of items (recommended for large datasets)
python main.py --algo ga --data "data/instances/test-data/Input/Elhedhli_dataset/products.pkl" --limit 100 --gen 50 --viz
```

#### 3. Run Different Algorithms
```bash
# Genetic Algorithm (GA)
python main.py --algo ga --data "data/instances/test-data/Input/3dBPP_1.txt" --gen 100 --viz

# Evolution Strategy (ES)
python main.py --algo es --data "data/instances/test-data/Input/3dBPP_1.txt" --gen 100 --viz

# Ant Colony Optimization (ACO)
python main.py --algo aco --data "data/instances/test-data/Input/3dBPP_1.txt" --gen 100 --viz

# Particle Swarm Optimization (PSO)
python main.py --algo pso --data "data/instances/test-data/Input/3dBPP_1.txt" --gen 100 --viz
```

### How to Visualize Results

#### Enable Visualization
Add the `--viz` flag to any command to generate visualizations:
```bash
python main.py --algo ga --data "data/instances/test-data/Input/3dBPP_1.txt" --gen 100 --viz
```

#### What Gets Generated
When `--viz` is enabled, the following files are created in `results/{algorithm}_{instance_name}/`:

1. **Convergence Plot** (`convergence_plot_{algo}.png`)
   - Shows fitness improvement over generations
   - Helps analyze algorithm performance

2. **3D Packing Visualization** (`result_visualization_{algo}.html`)
   - Interactive 3D visualization of all bins
   - Color-coded items (same item ID = same color)
   - Rotatable and zoomable view
   - Hover tooltips with item details

3. **Individual Bin Views** (`result_visualization_bin_{idx}_{algo}.html`)
   - Detailed view of first 3 bins
   - Useful for analyzing packing quality

4. **Solution Summary** (`solution_summary.txt`)
   - Detailed text output with all item positions
   - Bin assignments and orientations

#### Viewing Visualizations
- **HTML files**: Open in any web browser (Chrome, Firefox, Edge, etc.)
- **PNG files**: Open with any image viewer
- **Text files**: Open with any text editor

### How to Run on New Dataset (.pkl Files)

#### Step 1: Investigate Your Dataset
Before running, investigate your dataset structure:
```bash
python investigate_dataset.py --file "path/to/your/dataset.pkl" --limit 100
```

This will show you:
- Dataset structure and statistics
- Item dimensions and weight ranges
- Theoretical bin requirements
- Memory usage

#### Step 2: Prepare Your Dataset
Your `.pkl` file should be a pandas DataFrame with columns:
- `width` (or `w`): Item width
- `depth` (or `d`): Item depth  
- `height` (or `h`): Item height
- `weight`: Item weight

Example:
```python
import pandas as pd
data = pd.DataFrame({
    'width': [100, 200, 150],
    'depth': [80, 120, 100],
    'height': [60, 90, 70],
    'weight': [10, 25, 15]
})
pd.to_pickle(data, 'my_dataset.pkl')
```

#### Step 3: Run with Appropriate Limits
For large datasets, use `--limit` to test with a subset first:
```bash
# Test with 100 items
python main.py --algo ga --data "my_dataset.pkl" --limit 100 --gen 50 --viz

# Run with 1000 items
python main.py --algo ga --data "my_dataset.pkl" --limit 1000 --gen 100 --viz

# Run on full dataset (may take long time)
python main.py --algo ga --data "my_dataset.pkl" --gen 100 --viz
```

#### Step 4: Check Results
Results are saved in `results/{algorithm}_{dataset_name}/`:
- Check convergence plots to see if algorithm converged
- View 3D visualizations to verify packing quality
- Compare different algorithms on the same dataset

### Configuration

Configuration is managed via `config.yaml`. You can modify:
- **Problem settings**: Bin dimensions, max weight
- **Algorithm parameters**: Population size, generations, mutation rates, etc.

Example `config.yaml`:
```yaml
problem:
  default_bin_dims: [1200, 1200, 1200]
  default_max_weight: 10000

ga:
  pop_size: 100
  generations: 100
  pc: 0.9
  pm_factor: 1.0
  eta_c: 20
  eta_m: 20
```

## Project Structure

*   `main.py`: Entry point for running algorithms
*   `investigate_dataset.py`: Tool to analyze dataset structure and statistics
*   `config.yaml`: Configuration file for algorithms and problem settings
*   `src/`: Source code directory
  *   `solver_ga.py`: Genetic Algorithm implementation
  *   `solver_es.py`: Evolution Strategy implementation
  *   `solver_aco.py`: Ant Colony Optimization implementation
  *   `solver_pso.py`: Particle Swarm Optimization implementation
  *   `solver_base.py`: Base solver class
  *   `decoder.py`: Heuristic decoder (chromosome → packing solution)
  *   `domain.py`: Item and Bin classes
  *   `constraints.py`: Constraint checking for valid placements
  *   `visualization.py`: Visualization functions (convergence plots, 3D packing)
*   `data/`: Input datasets
*   `results/`: Output directory (created automatically)
  *   `{algorithm}_{instance}/`: Results for each run
    *   `convergence_plot_{algo}.png`: Convergence curve
    *   `result_visualization_{algo}.html`: 3D visualization of all bins
    *   `result_visualization_bin_{idx}_{algo}.html`: Individual bin views
    *   `solution_summary.txt`: Detailed packing solution

## Tips

- **For large datasets**: Always use `--limit` to test with a subset first
- **Generations**: Start with 50-100 generations, increase if solution quality needs improvement
- **Visualization**: Use `--viz` to verify packing quality and compare algorithms
- **Multiple algorithms**: Run all 4 algorithms on the same dataset to compare performance
- **Results folder**: All outputs are organized by algorithm and instance name for easy comparison