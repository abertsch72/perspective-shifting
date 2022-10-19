from typing import Text

orig_files_ts1 = ["learning_curve_data/amanda-orig-ts1.txt", "learning_curve_data/mturk-orig-all.txt", "learning_curve_data/upwork-orig-all.txt"]
ps_files_ts1 = ["learning_curve_data/amanda-ps-ts1.txt", "learning_curve_data/mturk-ps-all.txt", "learning_curve_data/upwork-ps-all.txt"]

orig_files_ts2 = ["learning_curve_data/amanda-orig-ts2.txt", "learning_curve_data/mturk-orig-ts2.txt", "learning_curve_data/upwork-orig-ts2.txt"]
ps_files_ts2 = ["learning_curve_data/amanda-ps-ts2.txt", "learning_curve_data/mturk-ps-ts2.txt", "learning_curve_data/upwork-ps-ts2.txt"]


orig_files_all = ["learning_curve_data/amanda-orig-all.txt", "learning_curve_data/upwork-orig-all.txt", "learning_curve_data/mturk-orig-all.txt"]
ps_files_all = ["learning_curve_data/amanda-ps-all.txt", "learning_curve_data/upwork-ps-all.txt", "learning_curve_data/mturk-ps-all.txt"]

testset1 = ["learning_curve_data/testset1-orig.txt", "learning_curve_data/testset1-ps.txt"]
testset2 = ["learning_curve_data/testset2-orig.txt", "learning_curve_data/testset2-ps.txt"]

def gen_dataset_2():
    import random
    random.seed(42)
    test_count = 0
    train_count = 0
    line_count = 0
    test_docs = []
    for file in orig_files_all:
        current = []
        train_docs = []
        with open(file) as f:
            for line in f:
                line = line.strip()
                if line == "-" or line == "":
                    if current != []:
                        prob = random.random()
                        line_count += 1
                        if prob > 0.9:
                            test_docs.extend(current)
                            test_docs.append("\n")
                            test_count += len(current)
                        else:
                            train_docs.extend(current)
                            train_docs.append("\n")
                            train_count += len(current)
                    current = []
                else:
                    current.append(line + "\n")
        if current != []:
            prob = random.random()
            line_count += 1
            if prob > 0.9:
                test_docs.extend(current)
                test_docs.append("\n")
                test_count += len(current)
            else:
                train_docs.extend(current)
                train_docs.append("\n")
                train_count += len(current)
        current = []
        with open(file.split("-all")[0] + "-ts2.txt", 'w') as f:
            f.writelines(train_docs)

    with open("testset2-orig.txt", 'w') as f:
        f.writelines(test_docs)

    print(train_count)
    print(test_count)
    print(line_count)


def get_train_test_split(dataset: Text, testset: int, method: Text, max_data: int = -1):
    datafiles_ps = []
    datafiles_orig = []
    testfile = ""
    if testset == 1:
        datafiles_ps = ps_files_ts1
        datafiles_orig = orig_files_ts1
        testfile = testset1
    elif testset == 2:
        datafiles_ps = ps_files_ts2
        datafiles_orig = orig_files_ts2
        testfile = testset2
    else:
        raise ValueError("Invalid testset number")

    if dataset == "all":
        pass
    elif dataset == "amanda_only":
        datafiles_ps = [datafiles_ps[0]]
        datafiles_orig = [datafiles_orig[0]]
        assert len(datafiles_ps) == 1
        assert len(datafiles_orig) == 1
    elif dataset == "amanda_mturk":
        datafiles_ps = datafiles_ps[0:2]
        datafiles_orig = datafiles_orig[0:2]
        assert len(datafiles_ps) == 2
        assert len(datafiles_orig) == 2
    elif dataset == "amanda_upwork":
        datafiles_ps = [datafiles_ps[0]].append(datafiles_ps[2])
        datafiles_orig = [datafiles_orig[0]].append(datafiles_orig[2])
        assert len(datafiles_ps) == 2
        assert len(datafiles_orig) == 2
    else:
        raise ValueError("Invalid dataset name!")

    if method == "aqa":
        orig, orig_test = data_extraction_aqa(datafiles_orig, testfile)
    elif method == "mask":
        orig, orig_test = data_extraction_masking(datafiles_orig, testfile)
    else:
        raise ValueError("invalid method name!")

    shifted = []
    for file in datafiles_ps:
        with open(file) as f:
            for line in f:
                line = line.strip()
                if line != "" and line != "-":
                    shifted.append(line)

    shifted_test = []
    with open(testfile[1]) as f:
        for line in f:
            line = line.strip()
            if line != "" and line != "-":
                shifted_test.append(line)


    if max_data > 0:
        if max_data > len(orig):
            raise ValueError("Max data is larger than dataset; there are " + str(len(orig)) + " datapoints!")
        orig = orig[:max_data]
        shifted = shifted[:max_data]

    train = {"original": orig, "shifted": shifted}
    test = {"original": orig_test, "shifted": shifted_test}

    return train, test

def data_extraction_aqa(datafiles_orig, testfile):
    orig = []
    for file in datafiles_orig:
        with open(file) as f:
            prev = ""
            for line in f:
                line = line.strip()
                if line != "" and line != "-":
                    orig.append(prev + " <SEP> " + line)
                    prev += line + " "
                elif prev != "":
                    prev = ""

    orig_test = []
    with open(testfile[0]) as f:
        prev = ""
        for line in f:
            line = line.strip()
            if line != "" and line != "-":
                orig_test.append(prev + " <SEP> " + line)
                prev += line + " "
            elif prev != "":
                prev = ""
    return orig, orig_test

def data_extraction_masking(datafiles_orig, testfile):
    orig = []
    for file in datafiles_orig:
        curr_convo = []
        with open(file) as f:
            for line in f:
                line = line.strip()
                if line != "-" and line != "":
                    curr_convo.append(line)
                elif curr_convo != []:
                    for i, message in enumerate(curr_convo):
                        this_x = ' '.join(curr_convo[:i]) + " <SEP> " + message + " <SEP> "
                        if i < len(curr_convo) - 1:
                            this_x += ' '.join(curr_convo[i + 1:])
                        orig.append(this_x)
                    curr_convo = []
            if curr_convo != []:
                for i, message in enumerate(curr_convo):
                    this_x = ' '.join(curr_convo[:i]) + " <SEP> " + message + " <SEP> "
                    if i < len(curr_convo) - 1:
                        this_x += '\n'.join(curr_convo[i + 1:])
                    orig.append(this_x)
                curr_convo = []

    orig_test = []
    curr_convo = []
    with open(testfile[0]) as f:
        for line in f:
            line = line.strip()
            if line != "-" and line != "":
                curr_convo.append(line)
            elif curr_convo != []:
                for i, message in enumerate(curr_convo):
                    this_x = ' '.join(curr_convo[:i]) + " <SEP> " + message + " <SEP> "
                    if i < len(curr_convo) - 1:
                        this_x += '\n'.join(curr_convo[i + 1:])

                    orig_test.append(this_x)
                curr_convo = []
        if curr_convo != []:
            for i, message in enumerate(curr_convo):
                this_x = ' '.join(curr_convo[:i]) + " <SEP> " + message + " <SEP> "
                if i < len(curr_convo) - 1:
                    this_x += '\n'.join(curr_convo[i + 1:])

                orig_test.append(this_x)
            curr_convo = []


    return orig, orig_test

get_train_test_split("amanda_only", 1, "mask")