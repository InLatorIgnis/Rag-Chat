#!/home/martin/rag-venv/bin/python3.11
import enums

print(enums.modelNames.HUGGINGFACE_MODEL)          # "openai/gpt-oss-20b"
print(enums.modelNames.HUGGINGFACE_MODEL == "openai/gpt-oss-20b")  # True
