
import run_gan_classifier as gan

from matplotlib import pyplot as plt

import time
import shutil
import numpy as np
import json


def test_different_dimension():
    """Test method to try out different dimensions of the GAN model.
    Run the gan_classifier several times, each time with different depths of Encoder, Decoder and Disciminator

    Save the prediction MCC for each run in a plot

    Old method (not currently used)
    """
    f_start = "input_text/"
    f_dst = "input_data/"
    old_f = ""
    for f in ["liar_and_buzzfeed_sentence_150dim.csv", "liar_and_buzzfeed_sentence_250dim.csv", "liar_and_buzzfeed_sentence_350dim.csv"]:
        if not old_f == "": # remove old one
            shutil.move(src=f_dst + old_f, dst=f_start + f)

        shutil.move(src=f_start + f, dst=f_dst + f)
        test_runs_start = 10
        test_runs_end = 14
        results = []
        for i in range(test_runs_start, test_runs_end):
            res = gan.main("train", i)
            res = gan.main("predict", i)
            if res != None:
                results.append(res)
        plt.plot([x for x in range(test_runs_start, test_runs_end)], [x["MCC"] for x in results])
        plt.savefig(str(f) + "_" + str(test_runs_end - test_runs_start) + "runs.png")

        old_f = f

def test_n_times(n, gen_dim=10, method="classical"):
    """Run the gan_classifier n times and save the results in json files
    Saves the results after prediction in a json file and the train_hist after training (including all losses) in a json file as well.

    Args:
        n (int): amount of runs of the gan_classifer
        gen_dim (int, optional): Set the depth of the generator models Encoder and Decoder. Not currently implemented. Defaults to 10.
        method (str, optional): classical or quantum. Only needed for setting the file path for saving the results. Defaults to "classical".
    """
    results = []
    train_hists = []
    for _ in range(n):
        train_hist = gan.main("train")
        if train_hist == None:
            print("Train history was None. Check log.log file")
        else:
            train_hists.append(train_hist)
        res = gan.main("predict")
        if res != None:
            results.append(res)

    new_results = []
    for res in results: # FK: to avoid some json and int64 errors
        tmp = dict()
        for k, v in res.items():
            if isinstance(v, list):
                tmp[k] = int(v[0])
            else:
                tmp[k] = float(v)
        new_results.append(tmp)
    json.dump(new_results, open(str(method) + "_results_" + str(n) + "times.json", 'w', encoding="utf-8"))

    new_hists = []
    for hist in train_hists:
        tmp = dict()
        for k, vs in hist.items():
            if isinstance(vs, list):
                tmp[k] = []
                for v in vs:
                    if isinstance(v, float):
                        tmp[k].append(float(v))
                    elif isinstance(v, int):
                        tmp[k].append(float(v))
                    else:
                        tmp[k].append(str(v))
            else:
                tmp[k] = vs
        new_hists.append(tmp)
    json.dump(new_hists, open(str(method) + "_train_hists_" + str(n) + "times.json", 'w', encoding="utf-8"))


def display_results(n=35, method="classical", save_plots=True):
    """Load the saved results from a test run and display them in plots.
    Display MCC and threshold after prediction and the losses after training.

    Args:
        n (int, optional): amount of runs of the gan_classifier. Needed for reading the correct file path. Defaults to 35.
        method (str, optional): classical or quantum. Needed for reading the correct file path. Defaults to "classical".
        save_plots (bool, optional): Create and save plot of MCC, threshold and losses if True. Defaults to True.
    """
    results = json.load(open(str(method) + "_results_" + str(n) + "times.json", 'r', encoding="utf-8"))

    print("MCC: " + str(np.mean([x["MCC"] for x in results])) + " (mean), " + str(np.median([x["MCC"] for x in results])) + " (median), " + str(np.std([x["MCC"] for x in results])) + " (st dev)")
    print("threshold: " + str(np.mean([x["threshold"] for x in results])) + " (mean), " + str(np.median([x["threshold"] for x in results])) + " (median), " + str(np.std([x["threshold"] for x in results])) + " (st dev)")

    if save_plots:
        ##### MCC
        plt.scatter([i for i in range(n)], [x["MCC"] for x in results])
        plt.ylim(0, 1)
        plt.title("MCC after prediction")
        plt.ylabel("MCC")
        plt.xlabel("runs")
        plt.savefig(str(method) + "_MCC_" + str(n) + "times.png", bbox_inches="tight")
        plt.cla()
        plt.clf()

        plt.boxplot([x["MCC"] for x in results])
        plt.ylabel("MCC")
        plt.savefig(str(method) + "_MCC_boxplot_" + str(n) + "times.png", bbox_inches="tight")
        plt.cla()
        plt.clf()

        ##### Threshold
        plt.scatter([i for i in range(n)], [x["threshold"] for x in results])
        plt.title("Threshold after prediction")
        plt.xlabel("runs")
        plt.ylabel("threshold")
        plt.savefig(str(method) + "_threshold_" + str(n) + "times.png", bbox_inches="tight")
        plt.cla()
        plt.clf()

    
    train_hists = json.load(open(str(method) + "_train_hists_" + str(n) + "times.json", 'r', encoding="utf-8"))
    



if __name__ == "__main__":
    tic = time.perf_counter()

    test_n_times(n=35, method="classical")
    display_results(n=35, method="classical", save_plots=False)

    toc = time.perf_counter()
    print("Total runtime: ", toc-tic)