from faker import Faker
import random
import time
import pymongo



fake = Faker()

# Connect to MongoDB Atlas
client = pymongo.MongoClient("mongodb+srv://ranthumeng_db_user:ranthumeng@cluster0.apusyhz.mongodb.net/?appName=Cluster0")
db = client["insurance_stream"]
customers_col = db["customers"]
claims_col = db["claims"]

MAX_RECORDS = 500000
records_inserted = 15000

while MAX_RECORDS is None or records_inserted < MAX_RECORDS:
    customer = {
        "customer_id": f"CUST-{random.randint(100,75000)}",
        "name": fake.name(),
        "state": fake.state_abbr(),
        "policy_type": random.choice(["Auto", "Home", "Health"])
    }

    claim = {
        "claim_id": f"CLM-{random.randint(1000,9999)}",
        "customer_id": customer["customer_id"],
        "date": fake.date_between(start_date='-1y', end_date='today').isoformat(),
        "amount": round(random.uniform(100, 20000), 2),
        "claim_type": random.choice(["Accident", "Theft", "Fire"]),
        "is_fraud": random.random() < 0.1
    }

    # Insert into MongoDB
    customers_col.insert_one(customer)
    claims_col.insert_one(claim)

    print(f"Inserted claim {claim['claim_id']} for customer {customer['customer_id']}")

    
