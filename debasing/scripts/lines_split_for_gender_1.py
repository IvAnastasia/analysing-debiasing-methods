with open('gender_corpora_aligned.txt') as file:
    texts = file.readlines()

import re
lines_clean = []
reg = re.compile(r'[А-Яа-яёË][.?!][А-Яа-яёË]')

for i in range(len(texts)-1):
   text = texts[i]
   to_find = texts[i+1]
   to_find = to_find[:5]
   if len(re.findall(r'\d. ', text)) > 0:
     text = re.sub(r'\d. ', '', text)
   if to_find in text.strip():
    idx = text.index(to_find)
    line1 = text[:idx]
       # print(line1)
    line2 = text[idx:]
       # print(line2.strip())
    lines_clean.append(line1.strip())
    lines_clean.append(line2.strip())
   elif to_find[:3] in text.strip():
    idx = text.index(to_find[:3])
    line1 = text[:idx]
       # print(line1)
    line2 = text[idx:]
       # print(line2.strip())
    lines_clean.append(line1.strip())
    lines_clean.append(line2.strip())
   elif len(re.findall(reg, text)) > 0:
      #print(text, texts[i-1])
      idx = re.search(reg, text).end() - 1
      #print(idx)
      line1 = text[:idx]
       # print(line1)
      line2 = text[idx:]
       # print(line2.strip())
      #print(line1)
      #print(line2)
      lines_clean.append(line1.strip())
      lines_clean.append(line2.strip())
   else:
     print(text)