import pandas as pd
from sqlalchemy import create_engine

# Load dataset
df = pd.read_csv('creditcard.csv')

# Connect to PostgreSQL database
engine = create_engine('postgresql://postgres:kelly%402003@localhost:5432/credi_card')

# Load the data into the database
df.to_sql('creditcard_transactions', engine, if_exists='replace', index=False)
