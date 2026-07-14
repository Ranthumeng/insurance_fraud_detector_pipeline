from snowflake.snowpark import Session
from snowflake.snowpark.functions import col, iff
from snowflake.ml.modeling.preprocessing import StandardScaler, OneHotEncoder
from snowflake.ml.modeling.linear_model import LogisticRegression
from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.modeling.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Build a Snowpark session
connection_params = {
    "user": "<USERNAME>",
    "password": "<PASSWORD>",
    "account": "<ACCOUNT_ID>",
    "warehouse": "<WAREHOUSE>",
    "database": "<DATABASE_NAME>",
    "schema": "<SCHEMA_NAME>",
}
session = Session.builder.configs(connection_params).create()

target_table = "CLAIMS"
df = session.table(target_table)
print(f"Initial row count: {df.count()}")

df = df.with_column_renamed("_ID", "ID")

# Normalize IS_FRAUD to 0/1
df = df.with_column(
    "IS_FRAUD",
    iff(col("IS_FRAUD").in_(["true", "True", True]), 1, 0)
)

# Drop Airbyte metadata columns
airbyte_cols = [
    '_AIRBYTE_RAW_ID', '_AIRBYTE_META', '_AIRBYTE_GENERATION_ID',
    '_AB_CDC_CURSOR', '_AB_CDC_DELETED_AT', '_AB_CDC_UPDATED_AT', '_AIRBYTE_EXTRACTED_AT'
]
df = df.drop(*airbyte_cols)

# Split data before processing
train_df, test_df = df.random_split(weights=[0.7, 0.3], seed=42)


ohe = OneHotEncoder(input_cols=["IS_TYPE"], output_cols=["IS_TYPE_OHE"], drop_input_cols=True)
ohe.fit(train_df)

# Retrieve the actual dynamic columns created by the encoder
ohe_features = ohe.get_output_cols() 
numeric_features = ["AMOUNT"]
label_col = ["IS_FRAUD"]

# Build pipeline with explicit feature lists
pipe = Pipeline(steps=[
    ("scaler", StandardScaler(input_cols=numeric_features, output_cols=numeric_features)),
    ("ohe", ohe),
    ("classifier", LogisticRegression(
        input_cols=numeric_features + ohe_features,
        label_cols=label_col,
        output_cols=["PREDICTION"],
        max_iter=1000,
    )),
])

# Fit and predict
pipe.fit(train_df)
result_df = pipe.predict(test_df)

# Ensure PREDICTION is cast to matches the label_col data type exactly
result_df = result_df.with_column("PREDICTION", col("PREDICTION").cast("integer"))

# Evaluate metrics
print("Accuracy:", accuracy_score(df=result_df, y_true_col_names=label_col, y_pred_col_names=["PREDICTION"]))
print("Precision:", precision_score(df=result_df, y_true_col_names=label_col, y_pred_col_names=["PREDICTION"]))
print("Recall:", recall_score(df=result_df, y_true_col_names=label_col, y_pred_col_names=["PREDICTION"]))
print(confusion_matrix(df=result_df, y_true_col_names=label_col, y_pred_col_names=["PREDICTION"]))
