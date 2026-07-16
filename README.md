# Insurance Fraud Detector Pipeline

An end-to-end data pipeline that generates a live, streaming feed of synthetic insurance claims, lands it in MongoDB, replicates it into Snowflake via Airbyte's ELT service, and trains a fraud-detection classifier directly inside Snowflake using Snowpark ML.

The project demonstrates a full "NoSQL source → cloud data warehouse → in-warehouse ML" pattern: instead of pulling data out to a separate ML environment, the model is trained and scored where the data already lives.

## Architecture

```
generator.py                Airbyte (MongoDB → Snowflake)         Analysis_Notebook.ipynb         snowpark_model.py
 (synthetic claims   ──────►   CDC/ELT replication         ──────►  (EDA in Pandas)         ──────►  (Snowpark ML pipeline:
  + customers)                 into Snowflake CLAIMS table                                            train, predict, evaluate)
```

- **Data generation** — a Python/Faker script continuously inserts synthetic customer and claim records into MongoDB Atlas, simulating a live claims intake system.
- **Ingestion** — Airbyte replicates the MongoDB `claims` collection into a Snowflake `CLAIMS` table (CDC metadata columns in the schema confirm this integration).
- **Exploratory analysis** — a Jupyter notebook pulls the replicated data into Pandas for cleaning and visual exploration of fraud patterns.
- **Modeling** — a Snowpark ML pipeline (`snowpark_model.py`) trains and evaluates a logistic regression fraud classifier entirely inside Snowflake's compute, without exporting data to an external environment.

## Repository Structure

```
data_generator/
└── generator.py                  # Synthetic customer/claim stream generator (Faker → MongoDB Atlas)
scripts/
├── Analysis_Notebook.ipynb       # EDA & visualization of claims data pulled from Snowflake
└── snowpark_model.py             # Snowpark ML pipeline: preprocessing, logistic regression, evaluation
```

## How It Works

### 1. Synthetic data generation (`data_generator/generator.py`)
Continuously generates and inserts synthetic records into two MongoDB collections:
- `customers` — customer ID, name, US state, and policy type (Auto / Home / Health)
- `claims` — claim ID, linked customer ID, claim date (within the past year), claim amount, claim type (Accident / Theft / Fire), and an `is_fraud` flag set true ~10% of the time

The loop runs continuously up to a configurable `MAX_RECORDS` cap, printing each inserted claim — simulating a live claims-intake feed rather than a one-off batch load.

### 2. Ingestion into Snowflake (Airbyte)
The `CLAIMS` table columns referenced later in the pipeline (`_AIRBYTE_RAW_ID`, `_AIRBYTE_META`, `_AIRBYTE_GENERATION_ID`, `_AB_CDC_CURSOR`, `_AB_CDC_DELETED_AT`, `_AB_CDC_UPDATED_AT`, `_AIRBYTE_EXTRACTED_AT`) confirm the MongoDB → Snowflake replication is handled via **Airbyte**, using change-data-capture (CDC) syncing so new and updated claims flow into Snowflake automatically.

### 3. Exploratory analysis (`scripts/Analysis_Notebook.ipynb`)
Connects to Snowflake directly (`snowflake.connector`) and pulls the full `CLAIMS` table into Pandas to:
- Drop Airbyte's internal CDC/metadata columns and rename `_ID` → `ID`
- Normalize `IS_FRAUD` from string/boolean variants (`'true'`, `'True'`, etc.) to a clean 0/1 integer
- Check class balance, null counts, and the overall fraud rate
- Visualize claim type distribution, fraud rate by claim type, claim amount distribution (log-scaled), claim amount vs. fraud status, fraud incidents over time, and the customers with the most claims and the most flagged-fraudulent claims

This notebook is the exploratory step that informed which features (claim type, amount) were carried into the modeling pipeline.

### 4. Fraud classification model (`scripts/snowpark_model.py`)
Trains entirely inside Snowflake using **Snowpark ML**, so no data leaves the warehouse:
- Loads the `CLAIMS` table as a Snowpark DataFrame, renames `_ID` → `ID`, and normalizes `IS_FRAUD` to 0/1 using `iff()`
- Drops the same Airbyte metadata columns as the notebook
- Splits into a 70/30 train/test set (seeded for reproducibility)
- Builds a `Pipeline` with:
  - `StandardScaler` on the numeric `AMOUNT` feature
  - `OneHotEncoder` on the claim-type feature
  - `LogisticRegression` (max 1000 iterations) as the classifier
- Fits on the training split, predicts on the test split, casts predictions to match the label's integer type
- Evaluates with **accuracy, precision, recall, and a confusion matrix**, all computed using Snowflake ML's native metrics functions

## Tech Stack
- **MongoDB Atlas** — source data store for the synthetic claims stream
- **Airbyte** — CDC-based ELT replication from MongoDB into Snowflake
- **Snowflake** — cloud data warehouse and query engine
- **Snowpark ML** — in-warehouse feature preprocessing, model training, and evaluation
- **Python** — Faker (synthetic data), Pandas/Matplotlib/Seaborn (EDA), PyMongo and the Snowflake Connector

## Getting Started
1. Run `data_generator/generator.py` against your own MongoDB Atlas cluster to start generating the synthetic claims stream (update the connection string and credentials for your own cluster — do not reuse any credentials committed in this repo).
2. Configure an Airbyte connection replicating the MongoDB `claims` (and `customers`) collections into a Snowflake `CLAIMS` table.
3. Use `scripts/Analysis_Notebook.ipynb` to explore the replicated data (fill in your own Snowflake `<USERNAME>`, `<PASSWORD>`, `<ACCOUNT_ID>`, `<WAREHOUSE_ID>`, `<DATABASE_ID>`, `<SCHEMA_ID>`, and `<TARGET_TABLE>` placeholders).
4. Run `scripts/snowpark_model.py` (with your own Snowflake `connection_params` filled in) to train and evaluate the fraud classifier.

> **Security note:** `snowpark_model.py` and the notebook use placeholder credentials (`<USERNAME>`, `<PASSWORD>`, etc.) — always supply your own via environment variables or a secrets manager rather than hardcoding them. The current `data_generator/generator.py` script contains a live-looking MongoDB Atlas connection string; replace it with your own credentials (ideally loaded from an environment variable) before running or sharing this code further.

## Known Limitations
- **Synthetic fraud signal is artificial.** The `is_fraud` flag is assigned via a flat ~10% random draw, independent of the other generated fields, so the "patterns" the model learns are limited to whatever incidental correlations arise from the generation logic (e.g. claim type or amount) rather than genuine fraud behavior.
- **Class imbalance is not explicitly addressed.** With fraud at ~10% of records, the modeling pipeline does not currently apply class weighting or resampling, so accuracy alone can be a misleading metric — precision/recall/confusion matrix (already included) are the more informative numbers to watch.
- **No feature beyond claim type and amount** is currently fed into the classifier; customer-level features (state, policy type, prior claim history) from the `customers` collection are not yet joined in.
