original_count = 373

def add_speaker_sep(train, test):

    test["original"] = sep_list(test["original"])
    test["shifted"] = sep_list(test["shifted"])

    train["original"] = sep_list(train["original"])
    train["shifted"] = sep_list(train["shifted"])

    return train, test

def sep_list(convolist):
    seplist = []
    for convo in convolist:
        sep_convo = ""
        for line in convo.split('\n'):
            sep_convo += "<SPEAKERSTART> " + line + " <SPEAKEREND> "
        seplist.append(sep_convo)
    return seplist

def data_extraction(para = False):
    import random

    file_sum = "raw_data/shifted.txt"
    file_raw = "raw_data/original.txt"
    random.seed(42)

    with open(file_sum) as f:
        sums = []
        test_sums = []
        count = 0
        test_count = 0
        total = 0
        temp = ""
        for line in f:
            line = line.strip()
            if line != "" and line != "-":
                total += 1
                temp += line + "\n"
                if not para:
                    prob = random.random()
                    if prob <= 0.9 or total > original_count:
                        sums.append(line)
                        count += 1
                    else:
                        test_sums.append(line)
                        test_count +=1
                temp = ""
            elif temp != "" and para:
                prob = random.random()
                if prob <= 0.9:
                    sums.append(temp )
                    count += 1
                else:
                    test_sums.append(temp )
                    test_count +=1
                temp = ""

    print(test_sums)

    random.seed(42)

    with open(file_raw) as f:
        orig = []
        test_orig = []
        count = 0
        test_count = 0
        temp = ""
        total = 0
        for line in f:
            line = line.strip()
            if line != "" and line != "-":
                total += 1
                line = ": ".join(line.split("\t"))
                temp += line + "\n"
                if not para:
                    prob = random.random()
                    if prob <= 0.9 or total > original_count:
                        orig.append(line)
                        count += 1
                    else:
                        test_orig.append(line)
                        test_count +=1
                temp = ""
            elif temp != "" and para:
                prob = random.random()
                if prob <= 0.9:
                    orig.append(temp )
                    count += 1
                else:
                    test_orig.append(temp )
                    test_count +=1
                temp = ""

    print(test_count)
    print(test_orig)


    with open("data_in/train_original.txt", 'w') as f:
        f.writelines(orig)

    with open("data_in/test_original.txt", 'w') as f:
        f.writelines(test_orig)

    train = {}
    train["original"] = orig
    train["shifted"] = sums
    test = {}
    test["original"] = test_orig
    test["shifted"] = test_sums
    data = {}
    data['train'] = train
    data['test'] = test
    return train, test

def data_extraction_aqa():
    import random

    file_sum = "raw_data/shifted.txt"
    file_raw = "raw_data/original.txt"
    random.seed(42)

    with open(file_sum) as f:
        sums = []
        test_sums = []
        count = 0
        test_count = 0
        prev = ""
        total = 0
        for line in f:
            line = line.strip()
            if line != "" and line != "-":
                total += 1
                prob = random.random()

                if prob <= 0.9 or total > original_count:
                    sums.append(line)
                    count += 1
                else:
                    test_sums.append(line)
                    test_count +=1

                prev += line + "\n"

            elif prev != "":
                prev = ""


    random.seed(42)

    with open(file_raw) as f:
        orig = []
        test_orig = []
        count = 0
        test_count = 0
        prev = ""
        total = 0
        for line in f:
            line = line.strip()
            if line != "" and line != "-":
                total += 1
                line = ": ".join(line.split("\t"))
                temp = prev + " <SEP> " + line
                prev += line + "\n"
                prob = random.random()
                if prob <= 0.9 or total > original_count:
                    orig.append(temp)
                    count += 1
                else:
                    test_orig.append(temp)
                    test_count += 1
            elif prev != "":

                prev = ""


    with open("data_in/train_original.txt", 'w') as f:
        f.writelines(orig)

    with open("data_in/test_original.txt", 'w') as f:
        f.writelines(test_orig)

    train = {}
    train["original"] = orig
    train["shifted"] = sums
    test = {}
    test["original"] = test_orig
    test["shifted"] = test_sums
    data = {}
    data['train'] = train
    data['test'] = test

    return train, test


def data_extraction_aqa2():
    import random

    file_sum = "raw_data/shifted.txt"
    file_raw = "raw_data/original.txt"
    random.seed(42)

    with open(file_sum) as f:
        sums = []
        test_sums = []
        count = 0
        test_count = 0
        prev = ""
        total = 0
        for line in f:
            line = line.strip()
            if line != "" and line != "-":
                total += 1
                prob = random.random()

                if prob <= 0.9 or total > original_count:
                    sums.append(line)
                    count += 1
                else:
                    test_sums.append(line)
                    test_count +=1

                prev += line + "\n"

            elif prev != "":
                prev = ""


    random.seed(42)

    with open(file_raw) as f:
        orig = []
        orig_q = []
        test_orig = []
        test_orig_q = []
        count = 0
        test_count = 0
        prev = ""
        total = 0
        for line in f:
            line = line.strip()
            if line != "" and line != "-":
                total += 1
                line = ": ".join(line.split("\t"))
                temp = prev + " <SEP> " + line + " <SEP> "
                prev += line + "\n"
                prob = random.random()
                if prob <= 0.9 or total > original_count:
                    orig_q.append([line, temp])
                    count += 1
                else:
                    test_orig_q.append([line, temp])
                    test_count += 1
            elif prev != "":
                for item in orig_q:
                    leftover = prev.split(item[0])
                    if len(leftover) > 1:
                        leftover = leftover[1]
                    else:
                        leftover = ""
                    orig.append(item[1] + leftover)

                for item in test_orig_q:
                    leftover = prev.split(item[0])
                    if len(leftover) > 1:
                        leftover = leftover[1]
                    else:
                        leftover = ""
                test_orig.append(item[1] + leftover)

                prev = ""

    with open("data_in/train_original.txt", 'w') as f:
        f.writelines(orig)

    with open("data_in/test_original.txt", 'w') as f:
        f.writelines(test_orig)

    train = {}
    train["original"] = orig
    train["shifted"] = sums
    test = {}
    test["original"] = test_orig
    test["shifted"] = test_sums
    data = {}
    data['train'] = train
    data['test'] = test

    return train, test

def data_extraction_masking():
    import random

    file_sum = "raw_data/shifted-aug.txt"
    file_raw = "raw_data/original-aug.txt"
    random.seed(42)

    with open(file_sum) as f:
        sums = []
        test_sums = []
        count = 0
        test_count = 0
        prev = ""
        total = 0
        for line in f:
            line = line.strip()
            if line != "" and line != "-":
                total += 1
                prob = random.random()

                if prob <= 0.9 or total > original_count:
                    sums.append(line)
                    count += 1
                else:
                    test_sums.append(line)
                    test_count += 1

                prev += line + "\n"

            elif prev != "":
                prev = ""

    random.seed(42)

    with open(file_raw) as f:
        orig = []
        orig_q = []
        test_orig = []
        test_orig_q = []
        count = 0
        test_count = 0
        prev = ""
        total = 0
        curr_convo = []
        for line in f:
            line = line.strip()
            # collect a full dialogue
            if line != "" and line != "-":
                total += 1
                line = ": ".join(line.split("\t"))
                curr_convo.append(line)
            elif len(curr_convo) != 0:
                # for each line in dialogue:
                for i, message in enumerate(curr_convo):
                    this_x = ' '.join(curr_convo[:i]) + " <SEP> " + message + " <SEP>"
                    if i < len(curr_convo) - 1:
                        this_x += '\n'.join(curr_convo[i+1:])
                    # calculate which set it should be in
                    # add <context> + <SEP> + line + <SEP> + <context> to relevant set
                    prob = random.random()
                    if prob <= 0.9 or total > original_count:
                        orig.append(this_x)
                        count += 1
                    else:
                        test_orig.append(this_x)
                        test_count += 1
                curr_convo = []


    with open("data_in/train_original.txt", 'w') as f:
        f.writelines(orig)

    with open("data_in/train_shifted.txt", 'w') as f:
        f.writelines(sums)

    print(len(orig))
    print(len(test_orig))
    print(len(sums))
    print(len(test_sums))
    train = {}
    train["original"] = orig
    train["shifted"] = sums
    test = {}
    test["original"] = test_orig
    test["shifted"] = test_sums
    data = {}
    data['train'] = train
    data['test'] = test

    return train, test

print(add_speaker_sep(*data_extraction_aqa()))