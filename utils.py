import numpy as np
import matplotlib.pyplot as plt
import itertools, os, csv

def pretty(value, htchar = "\t", lfchar = "\n", indent = 0):
    nlch = lfchar + htchar * (indent + 1)
    if type(value) is dict:
        items = [
            nlch + repr(key) + ": " + pretty(value[key], htchar, lfchar, indent + 1)
            for key in value
        ]
        return "{%s}" % (",".join(items) + lfchar + htchar * indent)
    elif type(value) is list:
        items = [
            nlch + pretty(item, htchar, lfchar, indent + 1)
            for item in value
        ]
        return "[%s]" % (",".join(items) + lfchar + htchar * indent)
    elif type(value) is tuple:
        items = [
            nlch + pretty(item, htchar, lfchar, indent + 1)
            for item in value
        ]
        return "(%s)" % (",".join(items) + lfchar + htchar * indent)
    else:
        return repr(value)

def write_params_file(savepath, input_dict):
    with open(os.path.join(savepath, "params.py"), "w+") as f:
        f.write("params = dict(\n")
        for k in input_dict.keys():
            if type(input_dict[k]) == str:
                txt = "'" + input_dict[k] + "'"
            else:
                txt = str(input_dict[k])
            f.write("    " + str(k) + " = " + txt + ",\n")
        f.write(")")

def product_dict(inp):
    for k in inp.keys():
        if type(inp[k]) is not list:
            inp[k] = [inp[k]]
    return list((dict(zip(inp.keys(), values)) for values in itertools.product(*inp.values())))

def plot_times(savepath):
    filepath = os.path.join(savepath, "times.csv")
    f, axs = plt.subplots(nrows = 3, ncols = 1)
    color_count = 0
    start = 0
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        labels = []
        for i, row in enumerate(reader):
            if i != 0:
                h = len(row[1:]) // 3
                labels.append([row[0] + r" $t=$" + str(t + 1) for t in range(h)])
                axs[0].bar(range(start, start + h), [float(y) for y in row[1:h + 1]], color = "C" + str(color_count))
                axs[1].bar(range(start, start + h), [float(y) for y in row[h + 1:2 * h + 1]], color = "C" + str(color_count))
                axs[2].bar(range(start, start + h), [float(y) for y in row[2 * h + 1:3 * h + 1]], color = "C" + str(color_count))
                color_count += 1
                start = start + h
    labels = sum(labels, [])
    axs[0].set_xticks([])
    axs[1].set_xticks([])
    axs[2].set_xticks(range(len(labels)), labels, rotation = 45, horizontalalignment = "right")
    axs[0].set_ylabel("Mean Time")
    axs[1].set_ylabel("Total Time")
    axs[2].set_ylabel("Calls")
    plt.savefig(os.path.join(savepath, "Times.pdf"), bbox_inches = "tight")
    plt.close()

def plot_training(savepath, all_algorithms, problem):
    if len(all_algorithms) > 1:
        savepath = os.path.join(savepath, problem["problem"], "horizon_" + str(problem["horizon"]))
    f, axs = plt.subplots(nrows = 2, ncols = 1)
    for c, algorithm in enumerate(all_algorithms):
        dirpath = os.path.join(savepath, algorithm if len(all_algorithms) > 1 else "")
        subdirs = [dirpath]
        for root, dirs, _ in os.walk(dirpath):
            for d in dirs:
                subdirs.append(os.path.join(root, d))
        points_approx = []
        points_value = []
        for subdirpath in subdirs:
            filepath = os.path.join(subdirpath, "training.csv")
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    reader = csv.reader(f)
                    points_approx.append([])
                    points_value.append([])
                    for i, row in enumerate(reader):
                        if i != 0:
                            points_approx[-1].append(float(row[1]))
                            points_value[-1].append(float(row[2]))
        axs[0].plot(np.arange(len(points_approx[0])), np.mean(points_approx, axis = 0), color = "C" + str(c), label = algorithm)
        axs[1].plot(np.arange(len(points_value[0])), np.mean(points_value, axis = 0), color = "C" + str(c))
        if len(subdirs) > 1:
            axs[0].fill_between(np.arange(len(points_approx[0])), np.mean(points_approx, axis = 0) - np.std(points_approx, axis = 0), np.mean(points_approx, axis = 0) + np.std(points_approx, axis = 0), color = "C" + str(c), alpha = 0.2)
            axs[1].fill_between(np.arange(len(points_value[0])), np.mean(points_value, axis = 0) - np.std(points_value, axis = 0), np.mean(points_value, axis = 0) + np.std(points_value, axis = 0), color = "C" + str(c), alpha = 0.2)
    axs[0].legend()
    axs[1].set_xlabel("Episode")
    axs[0].set_ylabel(r"$Q$-Values Approximation")
    axs[1].set_ylabel("Plan Value")
    axs[0].set_title(problem["problem"] + r" $h=$" + str(problem["horizon"]))
    plt.savefig(os.path.join(savepath, "Training.pdf"), bbox_inches = "tight")
    plt.close()

def plot_general(savepath, all_algorithms, problem):
    savepath = os.path.join(savepath, problem["problem"], "horizon_" + str(problem["horizon"]))
    f, axs = plt.subplots(nrows = 3, ncols = 1)
    for c, algorithm in enumerate(all_algorithms):
        dirpath = os.path.join(savepath, algorithm if len(all_algorithms) > 1 else "")
        subdirs = [dirpath]
        for root, dirs, _ in os.walk(dirpath):
            for d in dirs:
                subdirs.append(os.path.join(root, d))
        points_approx = []
        points_value = []
        points_times = []
        for subdirpath in subdirs:
            filepath = os.path.join(subdirpath, "results.csv")
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    reader = csv.reader(f)
                    for i, row in enumerate(reader):
                        if i != 0:
                            points_approx.append(float(row[0]))
                            points_value.append(float(row[1]))
                            points_times.append(float(row[2]))
        if len(subdirs) > 1:
            axs[0].bar(c, np.mean(points_approx), yerr = np.std(points_approx), color = "C" + str(c), ecolor = "black", capsize = 10)
            axs[1].bar(c, np.mean(points_value), yerr = np.std(points_value), color = "C" + str(c), ecolor = "black", capsize = 10)
            axs[2].bar(c, np.mean(points_times), yerr = np.std(points_times), color = "C" + str(c), ecolor = "black", capsize = 10)
        else:
            axs[0].bar(c, np.mean(points_approx), color = "C" + str(c))
            axs[1].bar(c, np.mean(points_value), color = "C" + str(c))
            axs[2].bar(c, np.mean(points_times), color = "C" + str(c))
    axs[0].set_xticks([])
    axs[1].set_xticks([])
    axs[2].set_xticks(range(len(all_algorithms)), all_algorithms, rotation = 45, horizontalalignment = "right")
    axs[0].set_ylabel("Approximation")
    axs[1].set_ylabel("Value")
    axs[2].set_ylabel("Time")
    axs[0].set_title(problem["problem"] + r" $h=$" + str(problem["horizon"]))
    plt.savefig(os.path.join(savepath, "Results.pdf"), bbox_inches = "tight")
    plt.close()