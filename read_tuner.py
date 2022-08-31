import json
import os
import numpy as np

# data_file = open("./tuner/dense_net2_tune/trial_13/trial.json", "r")
# data = json.load(data_file)
# metrics = data["hyperparameters"]["values"]
# print(metrics.keys())
# print(json.dumps(data, indent=4))


def read_tuner(tuner_dir):
    """docstring\n
    :parameter
        - read_tunerarg:\n
          description\n
    :return
        - read_tunerreturn\n
          description\n"""

    scores = []
    settings = []
    paths = []
    for subdirs, dirs, files in os.walk(tuner_dir):
        path = os.path.join(subdirs, "trial.json")
        if "trial_" in path:
            sd_file = open(path, "r")
            data = json.load(sd_file)
            try:
                path_score = float(
                    data["metrics"]["metrics"]["val_mae"]["observations"][0]["value"][0]
                )
                scores += [path_score]
                settings += [data["hyperparameters"]["values"]]
                paths += [path]
            except KeyError:
                print(os.path.split(os.path.split(path)[0])[-1], "in progress")

    score = np.asarray(scores, dtype=float)
    settings = np.asarray(settings)
    paths = np.asarray(paths)
    order = np.argsort(score)
    return score, settings, paths, order


runs = [50, 100, 250, 500, 1000, 6000]
runs = [50]
display_num = 5

num = input("settings number:")
sc, se, pa, o = read_tuner("./tuner/dense_net2_tune_{}/".format(num))   

for i in range(display_num):
    si = se[o][i]
    print(pa[o][i])
    print(sc[o][i])
    for key, value in si.items():
        print(key+":", value)
    print()

scs = []
ses = []
pas = []
oss = []
for i in runs:
    sc, se, pa, o = read_tuner(
        "./tuner/dense_net2_tune_{}/".format(i)
    )
    scs += [sc]
    ses += [se]
    pas += [pa]
    oss += [o]

best_settings = []
for i in range(len(runs)):
    best_settings += [[]]
for i in range(display_num):
    for s in range(len(runs)):
        best_settings[s] += [list(ses[s][oss[s]][i].values())]

all_settings = []
for i in range(len(runs)):
    all_settings += [[]]
for i in range(len(runs)):
    for k in ses[i]:
        all_settings[i] += [list(k.values())]

for i in range(len(runs)):
    u, ui = np.unique(all_settings[i], axis=0, return_index=True)
    i_score = scs[i][ui]
    i_order = np.argsort(i_score)
    all_settings[i] = u[i_order][:10]
    print(i_score[i_order])

all_settings = np.asarray(all_settings)
print(all_settings)
