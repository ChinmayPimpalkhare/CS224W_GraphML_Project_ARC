import pandas as pd

SPLIT_PATH = "data/processed/ml1m/splits/ratings_split.csv"

s = pd.read_csv(SPLIT_PATH).sort_values(["user_id","timestamp"])

print("Splits:", s["split"].value_counts())

# Per-user counts: expect exactly 1 val, 1 test
per_user = s.groupby("user_id")["split"].value_counts().unstack(fill_value=0)
assert (per_user["val"] == 1).all() and (per_user["test"] == 1).all()
print("Users with 1 val & 1 test each:", len(per_user))

# Vectorized leakage check: max(train_ts) <= val_ts <= test_ts
train_max = (s[s["split"]=="train"].groupby("user_id")["timestamp"].max())
val_ts  = (s[s["split"]=="val"].set_index("user_id")["timestamp"])
test_ts = (s[s["split"]=="test"].set_index("user_id")["timestamp"])

cmp = pd.concat([train_max.rename("train_max"),
                 val_ts.rename("val"),
                 test_ts.rename("test")], axis=1)

violations = ((cmp["train_max"] > cmp["val"]) | (cmp["val"] > cmp["test"])).sum()
print("Leakage violations:", int(violations))