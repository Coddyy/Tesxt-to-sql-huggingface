import torch, datetime
from db import get_schema
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("chatdb/natural-sql-7b")


snow = datetime.datetime.now()

model = AutoModelForCausalLM.from_pretrained(
    "chatdb/natural-sql-7b",
    device_map="auto",
    torch_dtype=torch.float16,
)

question = 'Add one customer with fullname Abhra Sarkar'

prompt = f"""
### Task 

Generate a `SELECT` SQL query to answer the following question: `{question}` 

The query will run on a database with the following schema: 
```
{get_schema()}
```

### Answer 
Here is the SQL query that answers the question: `{question}` 
```sql
"""

print ("Question: " + question)
print ("SQL: ")

inputs = tokenizer(prompt, return_tensors="pt").to("cuda") # use 'cpu' for cpu & 'cuda' for gpu

generated_ids = model.generate(
    **inputs,
    num_return_sequences=1,
    eos_token_id=100001,
    pad_token_id=100001,
    max_new_tokens=100,
    do_sample=False,
    num_beams=1,

)

outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(outputs)
print(outputs[0].split("```sql")[-1])

print("Start Time", snow)
now = datetime.datetime.now()
print("End Time", now)

fmt = '%Y-%m-%d %H:%M:%S'
tstamp1 = snow
tstamp2 = now
if tstamp1 > tstamp2:
    td = tstamp1 - tstamp2
else:
    td = tstamp2 - tstamp1
td_mins = int(round(td.total_seconds() / 60))

print('The difference is approx. %s minutes' % td_mins)



