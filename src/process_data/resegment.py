# re-segment the test set into conversations after generation

import sys

test_set = "../../data/samsum/test-orig.txt"

to_fix = sys.argv[1]

canonical_convos = []
temp = []
with open(test_set) as f:
    for line in f:
        if line.strip() != "" and line.strip() != "-":
            temp.append(line)
        elif temp != []:
            canonical_convos.append(len(temp))
            temp = []
print(len(canonical_convos))

convos = []
index = 0 
curr = []
with open(to_fix) as f:
    for line in f:
        if canonical_convos[index] > len(curr):
            curr.append(line)
        elif canonical_convos[index] == len(curr):
            convos.extend(curr)
            convos.append("\n\n")
            curr = [line]
            index += 1
    convos.extend(curr) 


with open(to_fix, 'w') as f:
    f.writelines(convos)
print(len(convos))
