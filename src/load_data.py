from typing import Text

folder = "../data/ps-final/"
samsum_folder = "../data/samsum/"
gyafc_folder = "../data/gyafc/"
summary_suffix = "-summaries.txt"
ps_suffix = "-ps.txt"
orig_suffix = "-orig.txt"

method_choices = ["lbl", "convo", "mask", "nocontext", "heuristic"]

seed = 72

def load_data(method: Text, split: Text = "PStrain"):
    if method == "gyafc":
        return load_gyafc()

    train_ps = load_each_line(folder + "train" + ps_suffix)
    val_ps = load_each_line(folder + "val" + ps_suffix)
    test_ps = load_each_line(folder + "test" + ps_suffix)

    if method == "lbl":
        train_orig = data_extraction_lbl(folder + "train" + orig_suffix)
        val_orig = data_extraction_lbl(folder + "val" + orig_suffix)
        test_orig = data_extraction_lbl(folder + "test" + orig_suffix)
    elif method == "convo":
        train_orig = load_each_group(folder + "train" + orig_suffix)
        val_orig = load_each_group(folder + "val" + orig_suffix)
        test_orig = load_each_group(folder + "test" + orig_suffix)

        train_ps = load_each_group(folder + "train" + ps_suffix)
        val_ps = load_each_group(folder + "val" + ps_suffix)
        test_ps = load_each_group(folder + "test" + ps_suffix)
    elif method == "mask":
        train_orig = data_extraction_masking(folder + "train" + orig_suffix)
        val_orig = data_extraction_masking(folder + "val" + orig_suffix)
        test_orig = data_extraction_masking(folder + "test" + orig_suffix)
    elif method == "nocontext":
        train_orig = load_each_line(folder + "train" + orig_suffix)
        val_orig = load_each_line(folder + "val" + orig_suffix)
        test_orig = load_each_line(folder + "test" + orig_suffix)
    elif method == "heuristic":
        train_orig = load_each_line(folder + "train" + orig_suffix)
        val_orig = load_each_line(folder + "val" + orig_suffix)
        test_orig = load_each_line(folder + "heuristic-test.txt")
    else:
        raise ValueError("Method \"" + method + "\" is not recognized!")

    if split == "PStrain":
        # use most of the val data as train (80% of the total train/val data, or ~63% of val and all of train)
        split_point = int(len(val_orig) * 0.63)
        train_orig.extend(val_orig[:split_point])
        train_ps.extend(val_ps[:split_point])

        val_orig = val_orig[split_point:]
        val_ps = val_ps[split_point:]
    elif split == "GENtest":
        test_orig = data_extraction_masking(samsum_folder + "test" + orig_suffix)
        test_ps = test_orig
    elif split == "GENval":
        # new validation data
        val_orig = test_orig
        val_ps = test_ps

        # new test data
        test_orig = data_extraction_masking(samsum_folder + "val" + orig_suffix)
        test_ps = test_orig
    elif split == "GENtrain":
        # new train data
        train_orig = val_orig
        train_ps = val_ps

        # new validation data
        val_orig = test_orig
        val_ps = test_ps

        # new test data
        test_orig = data_extraction_masking(samsum_folder + "train" + orig_suffix)
        test_ps = test_orig
        print(len(test_ps))
    else:
        raise ValueError("dataset split unexpected")         

    # sanity checks
    assert len(train_orig) == len(train_ps)
    assert len(val_orig) == len(val_ps)
    assert len(test_orig) == len(test_ps)

    train = {"original": train_orig, "shifted": train_ps}
    val = {"original": val_orig, "shifted": val_ps}
    test = {"original": test_orig, "shifted": test_ps}
    return train, val, test 

def load_gyafc():
    train_inf = load_each_line(gyafc_folder + "train-informal.txt")
    train_form = load_each_line(gyafc_folder + "train-formal.txt")
    val_inf = load_each_line(gyafc_folder + "val-informal.txt")
    val_form = load_each_line(gyafc_folder + "val-formal.txt")
    
    train = {"original": train_inf, "shifted": train_form}
    val = {"original": val_inf, "shifted": val_form}
    test = {}
    return train, val, test

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

def load_each_line(filename):
    # load each nonempty line as a new data point; suitable for either PS or summ
    data = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line != "" and line != "-":
                data.append(line)
    return data

def data_extraction_lbl(filename):
    orig = []
    with open(filename) as f:
        prev = ""
        for line in f:
            line = line.strip()
            if line != "" and line != "-":
                orig.append(prev + "<SEP> " + line)
                prev += line + "\n"
            elif prev != "": # when there's more than 1 blank line between datapoints
                prev = "" 

    return orig


def data_extraction_masking(datafile):
    orig = []
    curr_convo = []
    with open(datafile) as f:
       for line in f:
           line = line.strip()
           if line != "-" and line != "":
               curr_convo.append(line)
           elif curr_convo != []:
               for i, message in enumerate(curr_convo):
                   this_x = '\n'.join(curr_convo[:i]) + "\n<SEP> " + message + " <SEP>\n"
                   if i < len(curr_convo) - 1:
                       this_x += '\n'.join(curr_convo[i + 1:])
                   orig.append(this_x)
               curr_convo = []
       if curr_convo != []:
           for i, message in enumerate(curr_convo):
               this_x = '\n'.join(curr_convo[:i]) + "\n<SEP> " + message + " <SEP>\n"
               if i < len(curr_convo) - 1:
                   this_x += '\n'.join(curr_convo[i + 1:])
               orig.append(this_x)
           curr_convo = []

    return orig


#load_data("mask", "GENtrain")
