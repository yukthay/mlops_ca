# Load existing data
import pandas as pd
import numpy as np
existing_df = pd.read_csv('C:/Users/yukth/OneDrive/Desktop/mlops_ca/customer_data.csv')

# Generate new data
new_data = {
    'Customer_ID': range(51, 71),
    'Contract_Type': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], size=20),
    'Monthly_Charges': np.random.uniform(20, 100, size=20).round(2),
    'Tenure': np.random.randint(1, 73, size=20),  # Tenure in months
    'Churn_Flag': np.random.choice([0, 1], size=20)  # 0: Not Churned, 1: Churned
}

new_df = pd.DataFrame(new_data)

# Append new records to existing dataset
updated_df = pd.concat([existing_df, new_df], ignore_index=True)
updated_df.to_csv('customer_data.csv', index=False)

print("New records added.")
