import sys
import json

from load_data import load_each_group

input_file = sys.argv[1]
output_file = sys.argv[2] 

converted = []
examples = load_each_group(input_file)
for ex in examples:
    lines = [l.strip() for l in ex.split("\n")]
    for l in lines:
        if l.strip() == "": 
            continue
        speaker = l.split(":")[0]
        l = l.replace("I ", speaker + " ")
        l = l.replace("I'm ", speaker + " is ")
        l = l.replace("i'm ", speaker + " is ")
        l = l.replace("I've ", speaker + " has ")
        l = l.replace("i've ", speaker + " has ")
        l = l.replace("I'd ", speaker + " would ") 
        l = l.replace("i'd ", speaker + " would ") 
        l = l.replace(" me ", " " + speaker + " ")
        l = l.replace(" my ", " " + speaker + "'s ")
        l = l.strip() 
        if l[-1] != "!" and l[-1] != "?" and l[-1] != ".":
            l += "."
        converted.append(speaker + " says" + ":".join(l.split(":")[1:]) + "\n")
    converted.append("\n\n")
    #lines = [l.split(":")[0] + " says" + ":".join(l.split(":")[1:]) for l in lines]
    #print(lines)

with open(output_file, 'w') as f:
    f.writelines(converted)
"""For each message, we prepend the speaker’s name
and the word “says” to the utterance. We replace
each instance of the pronoun “I” in the message 
with the speaker’s name. After observing that most 
messages are not well-punctuated, we also append 
a period to the end of each utterance."""


