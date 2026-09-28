### installation

python -m venv .venv
.\.venv\Scripts\activate # source .venv/bin/activate  <- on MacOS/Linux
pip install -r requirements.txt

### Running the pipeline

01_data_cleaning_pipeline.ipynb contains links the execute the cleaning and most preprocessing, it will replicate the cleaned data from the input sources. Please use it before the exploratory and predictive notebooks if the data is not already present.