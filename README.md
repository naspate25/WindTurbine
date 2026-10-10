# Wind Turbine Fault Detection

Machine learning analysis of wind turbine sensor observations to detect drivetrain faults. The project includes exploratory analysis, tree-based classifiers, XGBoost, and artificial neural networks.

## Project files

- `AIML_Project_1_Full_Code_Notebook.ipynb`: main analysis and modeling notebook.
- `wind_turbine_detection.csv`: complete dataset, included in the repository (131,760 observations and 35 columns).
- `EDA_Analysis.md` and `EDA_findings.csv`: exploratory analysis and findings.
- PDF, PowerPoint, Word, and PNG files: supporting reports, presentation, and figures.

## Run the notebook

The notebook records Python 3.11.9 in its metadata. From the project directory, create and activate a Python environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter notebook AIML_Project_1_Full_Code_Notebook.ipynb
```

Keep the CSV beside the notebook: the notebook loads it using a relative path. Run the cells in order. Neural network training can take time and may benefit from a GPU.

## Dataset

Data is stored as a CSV file; this project does not currently use a database server or a separate database file. Columns include timestamps, turbine identifiers, operating conditions, sensor measurements, maintenance history, and the binary `failure` target (`0` = normal, `1` = fault). The dataset contains 127,807 normal observations and 3,953 fault observations.

Dependency versions are not pinned; the requirements list reflects notebook imports rather than a verified reproducible environment. Local virtual environments, notebook drafts, and secrets are excluded from Git.
