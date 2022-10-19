"""
>>> client = Client(Config())
>>> 
>>> inputs = [{
...     "source": "Hello, my world",
...     "references": ["Hello, world", "Hello my world"],
...     "hypothesis": "Hi, my world"
... }]
>>> metrics = ["rouge1", "bleu", "chrf"]
>>> 
>>> score_dic = client.score(inputs, metrics=metrics)

>>> 
>>> score_dic
{'scores': [{'corpus': 0.6666666666666666, 'sample': [0.6666666666666666]}, {'corpus': 0.35355339059327373, 'sample': [0.35355339059327373]}, {'corpus': 0.4900623006253688, 'sample': [0.4900623006253688]}]}
"""
import sys

from eaas import Config, Client

def load_each_group(filename):
    # load each blank-line-separated group as a new data point
    data = []
    temp = ""
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line == "":
                if temp != "":
                     data.append(temp)
                     temp = ""
            else:
                temp += line + "\n"
        if temp != "":
            data.append(temp)
    return data



client = Client(Config())
orig = "../../data/ps-final/test-orig.txt"
gold = "../../data/ps-final/test-ps.txt"
gen = sys.argv[1]

orig_convos = load_each_group(orig)
gold_convos = load_each_group(gold)
gen_convos = load_each_group(gen)

print(len(orig_convos))
print(len(gold_convos))
print(len(gen_convos))

inputs = list(map(lambda x,y,z: {"source": x, "references": [y], "hypothesis": z}, orig_convos, gold_convos, gen_convos))
metrics = ["rouge1", "rouge2", "rougeL", "bart_score_en_ref"]
#print(inputs[0]["source"])
#print(inputs[0]["references"][0])
#print(inputs[0]["hypothesis"])
score = client.score(inputs, metrics=metrics)
print(score)
with open("nochg-tmp.txt", 'w') as f:
    f.write(str(score))

# read in orig, ps, genPS and build a different dict for each convo 
