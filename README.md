# Iris MLflow Serve Testing Framework  — Local Testing Guide

![MLflow](https://img.shields.io/badge/MLflow-2.17.0-orange) ![Python](https://img.shields.io/badge/Python-≥3.11-blue)

A comprehensive testing suite for validating MLflow model serving endpoints (`mlflow serve`) in iris classification scenarios. Ensures endpoint reliability and prediction consistency.

This approach is ideal for lightweight applications or for testing your model locally before moving it to a staging or production environment.

## Authors

- **Mehmet Acikgoz** 
- **Maria Vechtomova** 


## 🚀 Quick Start

### Prerequisites
- Python ≥3.11
- MLflow model artifact (`pyfunc-lg-pipeline-model` directory)

### Setup & Execution
1. **Install dependencies**
   ```
   uv sync --all-extras
   ```

2. **Launch MLflow Server**
   ```
   uv run mlflow models serve -m pyfunc-lg-pipeline-model --port 5088
   ```

3. **Run Test Suite** (in new terminal)
   ```
   uv run pytest tests/
   ```

## 🌟 Key Features

### Core Validation Suite
| Feature | Verification Points |
|---------|---------------------|
| **Service Health** | ✅ 200 status on `/health` endpoint |
| **Server Responsiveness** | ✅ Successful `/ping` responses |
| **Version Integrity** | ✅ MLflow 2.17.0 confirmation |
| **Prediction Formats** | ✅ Supports both `dataframe_split` and `dataframe_records` |

### Advanced Testing Capabilities
- **Class Validation**: Ensures predictions {setosa, versicolor, virginica}
- **Null Handling**: Robust null value processing
- **Batch Testing**: CSV dataset validation (`test_data/test_set.csv`)
- **Diagnostics**: Detailed logging with `loguru`

## 🧪 Test Architecture

### Endpoint Specifications
| Endpoint | Method | Success Criteria |
|----------|--------|------------------|
| `/health` | GET | HTTP 200 |
| `/ping` | GET | HTTP 200 |
| `/version` | GET | Returns "2.17.0" |
| `/invocations` | POST | Valid prediction JSON |

### Test Types
1. **Single Prediction**
   - Validates response structure
   - Tests both DataFrame serialization formats

2. **Batch Validation**
   - Processes full CSV dataset
   - Verifies class membership for all predictions

## 📂 Project Structure

```
.
├── pyfunc-lg-pipeline-model/  # MLflow model artifacts
│   ├── MLmodel                # Model metadata
│   ├── code/                  # Custom code
│   ├── python_model.pkl       # Serialized model
│   └── ...                    # Additional model files
├── tests/
│   ├── test_data/             # Test datasets
│   │   └── test_set.csv       # Validation samples
│   └── test_mlflow_serve.py   # Core test suite
├── pyproject.toml             # Build configuration
└── README.md                  # This documentation
```

## 📦 Dependencies

### Core Requirements
```
mlflow==2.17.0
virtualenv>=20.30.0
```

### Testing Suite
```
loguru>=0.7.3
pandas>=2.2.3
pytest>=8.3.5
requests>=2.32.3
```

## 🔍 Troubleshooting
- **Port Conflicts**: Ensure port 5088 is available
- **Model Path**: Verify `pyfunc-lg-pipeline-model` directory exists
- **Dependency Issues**: Use `uv sync --all-extras` for clean installs
