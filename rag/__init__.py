"""Research-assistant RAG.

Read the modules in this order:

1. load.py      PDF pages become text
2. chunk.py     long text becomes overlapping passages
3. embed.py     a passage becomes a vector of 384 numbers
4. index.py     vectors are saved as files you can open
5. retrieve.py  a question finds the nearest passages
6. answer.py    a language model writes an answer from those passages only

config.py holds the knobs you are meant to change.
"""
