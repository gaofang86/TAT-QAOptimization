import os
import json
from collections import Counter

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
data_path = os.path.join(BASE_DIR, "dataset_raw", "tatqa_dataset_dev.json")
with open(data_path, "r") as f:
    data = json.load(f)

# traverse
# answer_type collection
types = []
for entry in data:          # entry = financial statement
    for q in entry['questions']:   # q = a question in the statement
        types.append(q['answer_type'])

# Counter convert list into dictionary
print(Counter(types))
print(f"Total questions: {len(types)}")
