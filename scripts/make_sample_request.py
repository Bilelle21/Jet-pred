import json

from jetpred.data import load_data, split_data

X, y = load_data()
_, _, X_test, _, _, y_test = split_data(X, y)

sample = X_test.head(3).astype(float)
with open("tests/fixtures/sample_request.json", "w") as f:
    json.dump({"jets": sample.to_dict(orient="records")}, f, indent=2)

print("True classes:", y_test.head(3).tolist())