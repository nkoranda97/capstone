# Running YOLO on LIDC-IDRI dataset

The project is split into 2 main categories: notebooks and scripts.

#### notebooks
Notebooks contains the high level code used to work with our functions, objects, and packages. There are five notebooks for data processing, data exploration, train, results, and supplemental. The notebooks often rely on data or models generated in notebooks earlier in the pipeline. 

#### scripts
Scripts contains user written functions and objects that are used by the notebooks. The primary reason for this formatting is to keep the notebooks clean, organized, and easily navigatable. Tinkering with the data pipeline often involves interacting with both the scripts and notebooks. The other reason for this formatting is to design the code so that it could one day be packaged and used independently. While the code is far from being ready or useful for that purpose, this was a basic protyping. 

#### Other
UV was used as the package mananger. This allows for this project to be easily reproduced. the pyproject.toml, .python-version, and uv.lock files contain this information. With UV installed and the CWD being this directory, the project environment can be installed with the following command:

```bash
uv sync
```

Automatic formatting by ruff for consistency

#### AI usage
Google Gemini 3 was used for coding assistance, particularly for displaying images. Github copilot was used for refactoring large functions into smaller parts.