import snowflake.connector

# Connecting to data source
conn = snowflake.connector.connect(
    user='<USERNAME>',
    password='<PASSWORD>',
    account='<ACCOUNT_ID>', 
    warehouse='<WAREHOUSE>',
    database='<DATABASE_NAME',
    schema='<SCHEMA_NAME>'
)

# Define the specific table you want to stream
target_table = "<TARGET_TABLE>"

# Fetch data into a Pandas DataFrame
query = f"SELECT * FROM {target_table}"
df = pd.read_sql(query, con=conn) 
print(f"Initial load of {len(df)} rows completed.")

from snowflake.snowpark.functions import col, iff
from snowflake.ml.modeling.preprocessing import StandardScaler, OneHotEncoder
from snowflake.ml.modeling.linear_model import LogisticRegression
from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.modeling.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


df = df.rename(col("_ID"), "ID")

df = df.with_column(
    "IS_FRAUD",
    iff(col("IS_FRAUD").in_(["true", "True", True]), 1, 0)
)

airbyte_cols = [
    '_AIRBYTE_RAW_ID', '_AIRBYTE_META', '_AIRBYTE_GENERATION_ID',
    '_AB_CDC_CURSOR', '_AB_CDC_DELETED_AT', '_AB_CDC_UPDATED_AT', '_AIRBYTE_EXTRACTED_AT'
]
categorical = ["IS_TYPE"]
categorical_ohe = ["IS_TYPE_OHE"]
numeric = ["AMOUNT"]
label_col = ["IS_FRAUD"]

# Snowpark DataFrame has a native random_split — no need to leave the warehouse
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
