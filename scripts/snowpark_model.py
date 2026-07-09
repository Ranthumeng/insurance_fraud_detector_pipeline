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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

print(df.head())
print(df.columns)

#renaming column
df = df.rename({'_ID': 'ID'})

# 2. Replace values and overwrite the column using with_column
df = df.with_column(
    "IS_FRAUD", 
    df["IS_FRAUD"].replace({'true': 1, 'false': 0, 'True': 1, 'False': 0, True: 1, False: 0})
)

# 3. Drop Airbyte metadata columns (Pass names directly as arguments)
airbyte_cols = [
    '_AIRBYTE_RAW_ID', '_AIRBYTE_META', '_AIRBYTE_GENERATION_ID',
    '_AB_CDC_CURSOR', '_AB_CDC_DELETED_AT', '_AB_CDC_UPDATED_AT', '_AIRBYTE_EXTRACTED_AT'
]
df = df.drop(airbyte_cols)
df.show()


categorical = ["IS_TYPE"]
numeric = ["AMOUNT"]


#train/test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, stratify=y)

preprocessor = ColumnTransformer(
    transformers = [
        ("num",StandardScaler(),numeric),
        ("cat",OneHotEncoder(drop="first"),categorical)
    ],
    remainder = "drop"
)

preprocessor = ColumnTransformer(
    transformers = [
        ("num",StandardScaler(),numeric),
        ("cat",OneHotEncoder(drop="first"),categorical)
    ],
    remainder = "drop"
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print(classification_report(y_test,y_pred))

confusion_matrix(y_test,y_pred)

