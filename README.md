# insurance_fraud_detector_pipeline

Faker → MongoDB → Snowflake (Airbyte ELT) → Snowpark ML: </br>
<p>A practical end-to-end pipeline that creates synthetic data with Python + Faker, stores it in MongoDB, uses Airbyte to move it to Snowflake, and trains a Logistic Regression model inside Snowflake using Snowpark.</p>

What’s in this repo:

Generate realistic fake data (customizable size and fields).

Save raw data to MongoDB using PyMongo.

Use Airbyte to copy data from MongoDB into Snowflake (ELT).

Do transforms and train a Logistic Regression model inside Snowflake with Snowpark.

Store model and metrics back in Snowflake so you can report or serve later.

Why this is useful

Reproducible demo pipeline you can run locally or on cloud.

Shows hands-on skills: data generation, NoSQL, ELT, Snowflake, and in-database ML.

Handy for interviews, portfolio pieces, or learning Snowpark ML.

Quick architecture
Data generator (Python) → MongoDB → Airbyte (ELT) → Snowflake (raw & curated tables) → Snowpark training → model + metrics in Snowflake

Tech I used

Python 3.9+ (generator + Snowpark scripts)

Faker for synthetic records

PyMongo for writing to MongoDB

Airbyte for ELT (MongoDB → Snowflake)

Snowflake + Snowpark for transforms and model training

Docker / docker-compose for local setup

Optional: dbt or SQL scripts for extra transforms

Repo layout

README.md (this file)

docs/ — setup notes and architecture image

src/

generator/ — Faker-based generator

mongodb/ — PyMongo writer

airbyte/ — connector configs + docker-compose (optional)

snowflake/ — SQL and Snowpark training scripts

tests/ — unit tests

examples/ — sample output

scripts/ — handy run/deploy scripts

.env.example, requirements.txt, LICENSE

Quickstart (fast path)

Clone:
git clone <repo-url> && cd <repo-folder>

Copy and edit env:
cp .env.example .env
Fill in MongoDB and Snowflake details.

Optional: start local services
Use the provided docker-compose to run MongoDB + Airbyte locally:
docker-compose -f src/airbyte/docker-compose.yml up -d

Install Python stuff:
pip install -r requirements.txt

Generate data and push to MongoDB:
python src/generator/generate_data.py --count 5000 --batch-size 1000

Set up Airbyte:
Use the UI or provided configs to create a MongoDB source and Snowflake destination, then sync.

Train the model in Snowflake:
Run the Snowpark training script:
python src/snowflake/snowpark_train.py
It reads the data in Snowflake, trains a Logistic Regression model, saves the model and writes metrics to a results table.

What the scripts do

generate_data.py: makes fake users/transactions, saves as JSON documents.

write_to_mongo.py: bulk inserts documents into MongoDB with retries.

Airbyte configs: sample setup to sync MongoDB → Snowflake.

snowpark_train.py: feature prep, train/test split, train Logistic Regression via Snowpark, save model and metrics.
