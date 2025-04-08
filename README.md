# SHL Backend

This repository contains the backend code for the SHL Assessment Chatbot. The Flask application leverages a recommendation engine to recommend assessments based on user queries. It uses data from a CSV file and utilizes various Python libraries for data processing and natural language processing.
# Project Overview
+ Flask App (app.py):<br><br>
This is the main file that runs the Flask web server. It sets up the `/recommend` endpoint, handles incoming JSON requests, and returns the recommendations. I’ve also added Flask-CORS to make sure that requests from the frontend can reach the server without issues.<br><br>
+ SHL_RECOMMENDER_Copy1.py:<br><br>
This is the module that loads the assessment data from `SHL_DATASET_TOKENIZED.csv`, preprocesses it, and uses TF-IDF vectorization along with cosine similarity to rank the assessments. This module is also responsible for calling Google Generative AI (Gemini) to generate a friendly conversational response.<br><br>
+ SHL_DATASET_TOKENIZED.csv:<br><br>
  This CSV file contains the assessment data that the recommender reads and processes.<br><br>

  ## Requirements <br>
  The code runs on my local environment with version : Python 3.13.2 <br>
  Lower versions might give issue to run this project.

# Backend Deployment Issues on Render

## Deployment Failure Details

The backend deployment on Render has failed due to the following technical issues:

```
Preparing metadata (pyproject.toml): finished with status 'error'
error: subprocess-exited-with-error

Caused by: `cargo metadata` exited with an error:
Error running maturin: Command '['maturin', 'pep517', 'write-dist-info', '--metadata-directory', '/tmp/pip-modern-metadata-x1u3tnop', '--interpreter', '/opt/render/project/src/.venv/bin/python']' returned non-zero exit status 1.

e: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed
Encountered error while generating package metadata.
```

## Python Version Compatibility Issues

In addition to the deployment errors above, significant Python version compatibility issues were encountered:

- The Render environment uses a different Python version than our development environment
- Certain dependencies require Python 3.8+ but Render was configured with an older version
- Version-specific syntax in the codebase caused compatibility errors during deployment
- Package dependencies have version constraints that conflict with the Python version on Render

These errors have prevented the successful deployment of the backend service. The deployment process is failing during the package installation phase due to issues with Rust dependencies, build configuration, and Python version incompatibilities in the Render environment.

As a result of these technical deployment issues, the backend service is currently unable to be deployed to production.
