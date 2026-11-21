# Iris Flower Classification Using MLOps

A beginner-friendly local MLOps project that classifies Iris flowers with a
Random Forest. The dataset comes only from `sklearn.datasets.load_iris()`; no
dataset is downloaded or generated.

## What this project does

- Uses all 150 built-in Iris observations, four measurements, and three species
  (50 observations per species).
- Makes a reproducible, stratified 80/20 split (120 training and 30 testing rows).
- Trains and evaluates a Random Forest, saving a Joblib model and JSON metrics.
- Records model parameters, model, and evaluation metrics in a local MLflow
  experiment.
- Uses DVC to reproduce the training/evaluation pipeline and track generated
  artifacts.
- Tests the data, split, model, predictions, and saved-model behavior with pytest.
- Serves predictions from FastAPI and includes a Dockerfile and GitHub Actions CI.

## Project layout

```text
iris-mlops/
├── api/                 # FastAPI app
├── data/                # DVC-managed split generated from scikit-learn data
├── models/              # DVC-managed Joblib model
├── reports/             # Evaluation metrics
├── src/                 # Data, training, evaluation, and prediction code
├── tests/               # pytest tests
├── .github/workflows/   # GitHub Actions CI
├── dvc.yaml             # Reproducible pipeline
├── params.yaml          # Model and split settings
├── requirements.txt     # App/runtime dependencies
└── requirements-dev.txt # Runtime + DVC and test dependencies
```

`src/__init__.py` and `api/__init__.py` use Python's conventional double-
underscore package filenames.

## 1. Python environment and installation

Use Python 3.13 for this project. In the VS Code PowerShell terminal, from the
project folder:

```powershell
py -3.13 --version
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

If PowerShell blocks activation, allow it for the current terminal only, then
activate the environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Select `.venv\Scripts\python.exe` as the interpreter in VS Code with **Python:
Select Interpreter**. Python 3.15 is installed on some systems, but scientific
Python packages can lag behind new Python releases; use Python 3.13 if package
installation fails under 3.15.

## 2. Inspect the built-in dataset

```powershell
python -c "from src.data_loader import display_dataset_info; display_dataset_info()"
```

This prints the complete scikit-learn dataset description, first five rows,
shape, feature and species names, missing-value count, and class counts. The
data is loaded directly from scikit-learn, so there is no raw external dataset
file to version with DVC.

## 3. Train and evaluate

```powershell
python -m src.train
python -m src.evaluate
```

Training reads `params.yaml`, prints the dataset details, makes the stratified
120/30 split, trains the Random Forest, and writes:

- `models/iris_model.joblib` — trained model.
- `data/processed/iris_split.npz` — saved split used by evaluation.

Evaluation prints accuracy, weighted precision, recall and F1-score, the
classification report, and confusion matrix. It writes `reports/metrics.json`.
Model settings and the model are logged during training; the four evaluation
metrics and JSON report are logged during evaluation.

## 4. MLflow experiment tracking

The local tracking store is `mlruns/`. MLflow 3 requires the opt-in file-store
setting used by this project. After training and evaluation, open
another terminal in the project folder, activate `.venv`, and run:

```powershell
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open <http://127.0.0.1:5000> to inspect the **Iris Flower Classification**
experiment, training parameters, logged model, and evaluation metrics. Stop
the dashboard with Ctrl+C.

## 5. Git and DVC

Initialize Git and DVC once (DVC expects a Git repository):

```powershell
git init
dvc init
```

Track project code and DVC metadata with Git; DVC stores the generated model
and split in its local cache. The built-in dataset itself is reproducible from
the installed scikit-learn version and is not copied into the repository.

```powershell
git add .gitignore .dockerignore .github api src tests README.md Dockerfile requirements.txt requirements-dev.txt params.yaml dvc.yaml .dvc .dvcignore
git commit -m "Create Iris MLOps project"
dvc repro
dvc status
dvc metrics show
```

`dvc repro` runs the training stage and then evaluation when code, parameters,
or dependencies change. `dvc status` reports changed pipeline inputs.
`dvc metrics show` displays the saved evaluation scores. Change a value in
`params.yaml`, then run `dvc repro` again to reproduce that experiment.

The DVC cache is local until a remote is configured. To keep artifacts on this
computer, no remote is required. To share them later, configure a DVC remote
and run `dvc push`; do not commit generated model binaries to Git.

## 6. Run the tests

```powershell
python -m pytest -q
```

The tests verify that the built-in dataset has 150 rows and four features, the
classes each have 50 rows, the stratified split has 120/30 rows, training and
predictions work, predictions are valid Iris species, and a Joblib model can be
saved, loaded, and used.

## 7. Run the FastAPI service

Train the model first, then start the API from the project root:

```powershell
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/docs> for Swagger UI. The `/health` endpoint checks
the API process. Use **POST /predict** with:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

The response has the form `{"prediction": "Setosa"}`. Measurements must be
finite positive numbers. The API returns HTTP 503 with an actionable message
if the model has not yet been trained.

## 8. Build and run with Docker

Docker must be installed and running. Train the model before building so the
model file is present:

```powershell
python -m src.train
docker build -t iris-mlops .
docker run --rm -p 8000:8000 iris-mlops
```

Open <http://127.0.0.1:8000/docs>. The image installs runtime dependencies and
copies the trained model into the container.

## 9. GitHub Actions and CI/CD

The workflow at `.github/workflows/ci.yml` installs Python 3.13 and the
development dependencies, then runs pytest for each push and pull request.
Continuous Integration (CI) automatically checks that changes still pass the
tests. Continuous Delivery/Deployment (CD) automates preparing or releasing
those changes; this starter workflow does CI only and does not publish or
deploy the project.

To publish this local project to GitHub, create an empty GitHub repository,
then use its URL:

```powershell
git remote add origin https://github.com/<your-username>/<your-repository>.git
git branch -M main
git push -u origin main
```

## Troubleshooting

- **`git` is not recognized:** install Git for Windows, reopen VS Code, and
  confirm `git --version` works.
- **Wrong Python after activation:** select `.venv\Scripts\python.exe` in VS
  Code and open a fresh terminal.
- **Model file missing:** run `python -m src.train` before using the API.
- **DVC is not recognized:** activate `.venv` and install
  `python -m pip install -r requirements-dev.txt`.
- **Docker command is not recognized:** install and start Docker Desktop.
