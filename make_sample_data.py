"""
Makes a SMALL FAKE dataset with the same columns as the real PaySim dataset.
Use it only to test that the project runs when you don't have the real CSV yet.

Run:  python make_sample_data.py
Creates: data/sample_data.csv
"""

import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 60_000

types = rng.choice(
    ["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"],
    size=N,
    p=[0.34, 0.22, 0.35, 0.08, 0.01],
)
amount = rng.lognormal(mean=9, sigma=1.2, size=N)
old_org = rng.lognormal(mean=9.5, sigma=1.5, size=N)
old_dest = rng.lognormal(mean=9.5, sigma=1.5, size=N)

df = pd.DataFrame({
    "step": rng.integers(1, 744, size=N),
    "type": types,
    "amount": amount,
    "nameOrig": ["C" + str(i) for i in rng.integers(1e8, 9e8, size=N)],
    "oldbalanceOrg": old_org,
    "newbalanceOrig": np.maximum(old_org - amount, 0),
    "nameDest": ["C" + str(i) for i in rng.integers(1e8, 9e8, size=N)],
    "oldbalanceDest": old_dest,
    "newbalanceDest": old_dest + amount,
    "isFraud": 0,
    "isFlaggedFraud": 0,
})

# In PaySim, fraud only happens in TRANSFER and CASH_OUT and often empties the account.
risky = df["type"].isin(["TRANSFER", "CASH_OUT"])
fraud_mask = risky & (rng.random(N) < 0.03)
df.loc[fraud_mask, "isFraud"] = 1
df.loc[fraud_mask, "amount"] = df.loc[fraud_mask, "oldbalanceOrg"]
df.loc[fraud_mask, "newbalanceOrig"] = 0.0
df.loc[fraud_mask, "oldbalanceDest"] = 0.0
df.loc[fraud_mask, "newbalanceDest"] = 0.0

os.makedirs("data", exist_ok=True)
df.to_csv("data/sample_data.csv", index=False)
print(f"Saved data/sample_data.csv  ({len(df)} rows, {int(df.isFraud.sum())} fraud)")
