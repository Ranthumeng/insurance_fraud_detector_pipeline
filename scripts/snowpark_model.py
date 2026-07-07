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
