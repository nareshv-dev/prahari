"""Small in-memory feature store with as-of retrieval used by Sentinel and Advocate."""
from __future__ import annotations
import pandas as pd

class FeatureStore:
    def __init__(self, transactions: pd.DataFrame | None = None):
        self.transactions = transactions.copy() if transactions is not None else pd.DataFrame()
    def as_of(self, transaction_id: str, at_time=None) -> dict:
        if self.transactions.empty: return {"transaction_id": transaction_id}
        rows = self.transactions[self.transactions["transaction_id"].astype(str) == str(transaction_id)]
        if rows.empty: return {"transaction_id": transaction_id}
        row = rows.iloc[0]
        if at_time is not None and "TransactionDT" in self.transactions:
            row = row if row["TransactionDT"] <= at_time else rows.iloc[0]
        return row.where(pd.notna(row), None).to_dict()
    def history(self, transaction_id: str) -> pd.DataFrame:
        record = self.as_of(transaction_id)
        if not record or "TransactionDT" not in record or self.transactions.empty: return self.transactions.iloc[0:0]
        return self.transactions[self.transactions["TransactionDT"] <= record["TransactionDT"]]
