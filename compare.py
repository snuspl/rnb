import os
import sys
job1 = sys.argv[1]
job2 = sys.argv[2]

dict1 = {}
dict2 = {}

for d, j in zip([dict1, dict2], [job1, job2]):
  for filename in os.listdir(j):
    if filename[-3:] != 'csv': continue
    with open(j + '/' + filename, 'r') as f:
      for line in f:
        line = line.strip().split(' ')
        d[line[0]] = line[1]

same = True
for k, v in dict1.items():
  if k not in dict2: continue
  if v != dict2[k]:
    same = False

print('Are the results the same?', same)
