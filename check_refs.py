#!/usr/bin/env python3
import re

with open("paper/main.tex") as f:
    text = f.read()
with open("paper/references.bib") as f:
    bib = f.read()

# Extract all citation keys from main.tex
cites = set()
for m in re.findall(r'\\cite[pt]?\{([^}]+)\}', text):
    for k in m.split(','):
        cites.add(k.strip())

# Extract all bib keys
bib_keys = set(re.findall(r'@\w+\{(\w+)', bib))

print(f'Citations in main.tex: {len(cites)}')
print(f'Entries in references.bib: {len(bib_keys)}')
missing = cites - bib_keys
if missing:
    print(f'MISSING from bib: {missing}')
else:
    print('All citations have matching bib entries!')
unused = bib_keys - cites
if unused:
    print(f'Unused bib entries ({len(unused)}): {unused}')
