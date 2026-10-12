import random
import time

import httpx

from jetpred.data import load_data, split_data

URL = "http://localhost:8000/predict"
DURATION_SECONDS = 300

X, y = load_data()
_, _, X_test, _, _, _ = split_data(X, y)
records = X_test.astype(float).to_dict(orient="records")

end = time.time() + DURATION_SECONDS
sent = 0
with httpx.Client(timeout=10) as client:
    while time.time() < end:
        batch = random.sample(records, random.choice([1, 5, 20, 100]))
        if random.random() < 0.05:
            batch = [{"not_a_feature": 1.0}]
        client.post(URL, json={"jets": batch})
        sent += 1
        time.sleep(random.uniform(0.02, 0.2))

print(f"Sent {sent} requests")