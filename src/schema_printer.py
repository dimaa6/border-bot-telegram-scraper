import json
from llm_extractor import BorderSentimentExtraction

# Generate and print the exact raw JSON schema layout
print(json.dumps(BorderSentimentExtraction.model_json_schema(), indent=2))
