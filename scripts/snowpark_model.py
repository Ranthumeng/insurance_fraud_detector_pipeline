from snowflake.snowpark import Session
from snowflake.snowpark.functions import col, iff
from snowflake.ml.modeling.preprocessing import StandardScaler, OneHotEncoder
from snowflake.ml.modeling.linear_model import LogisticRegression
from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.modeling.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Build a Snowpark session (NOT snowflake.connector + pd.read_sql)
connection_params = {
    "user": "<USERNAME>",
    "password": "<PASSWORD>",
    "account": "<ACCOUNT_ID>",
    "warehouse": "<WAREHOUSE>",
    "database": "<DATABASE_NAME>",
    "schema": "<SCHEMA_NAME>",
}
session = Session.builder.configs(connection_params).create()

target_table = "<TARGET_TABLE>"

# This returns a Snowpark DataFrame, lazily evaluated — nothing pulled locally yet
df = session.table(target_table)
print(f"Initial row count: {df.count()}")

# 1. Rename — now this is genuinely a Snowpark DataFrame, so this method exists
df = df.rename(col("_ID"), "ID")

# 2. Normalize IS_FRAUD to 0/1
df = df.with_column(
    "IS_FRAUD",
    iff(col("IS_FRAUD").in_(["true", "True", True]), 1, 0)
)

# 3. Drop Airbyte metadata columns — the line that went missing
airbyte_cols = [
    '_AIRBYTE_RAW_ID', '_AIRBYTE_META', '_AIRBYTE_GENERATION_ID',
    '_AB_CDC_CURSOR', '_AB_CDC_DELETED_AT', '_AB_CDC_UPDATED_AT', '_AIRBYTE_EXTRACTED_AT'
]
df = df.drop(*airbyte_cols)

categorical = ["IS_TYPE"]
categorical_ohe = ["IS_TYPE_OHE"]
numeric = ["AMOUNT"]
label_col = ["IS_FRAUD"]

# Now random_split works, because df is a Snowpark DataFrame
train_df, test_df = df.random_split(weights=[0.7, 0.3], seed=42)

pipe = Pipeline(steps=[
    ("scaler", StandardScaler(input_cols=numeric, output_cols=numeric)),
    ("ohe", OneHotEncoder(input_cols=categorical, output_cols=categorical_ohe, drop_input_cols=True)),
    ("classifier", LogisticRegression(
        input_cols=numeric + categorical_ohe,
        label_cols=label_col,
        output_cols=["PREDICTION"],
        max_iter=1000,
    )),
])

pipe.fit(train_df)
result_df = pipe.predict(test_df)

print("Accuracy:", accuracy_score(df=result_df, y_true_col_names=label_col, y_pred_col_names=["PREDICTION"]))
print("Precision:", precision_score(df=result_df, y_true_col_names=label_col, y_pred_col_names=["PREDICTION"]))
print("Recall:", recall_score(df=result_df, y_true_col_names=label_col, y_pred_col_names=["PREDICTION"]))
print(confusion_matrix(df=result_df, y_true_col_names=label_col, y_pred_col_names=["PREDICTION"]))
