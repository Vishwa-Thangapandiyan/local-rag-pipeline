# Eval set
import json
import sys

from rag import load_pdfs, chunk_text, embed_chunks, collection

with open("eval_set.json","r") as f:
    eval_set = json.load(f)

print(f"Loaded {len(eval_set)} eval questions")

sys.path.append(".")
#--------------------------------------------------------------------

