
import doc2vec_FK as d2v
import run_gan_classifier as gan

from matplotlib import pyplot as plt

import time
import shutil
import numpy as np
import json
import os


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

def test_n_times(n, gen_dim=10, method="classical", calc_embeddings=True, save_results=True):
    """Run the gan_classifier n times and save the results in json files
    Saves the results after prediction in a json file and the train_hist after training (including all losses) in a json file as well.

    Args:
        n (int): amount of runs of the gan_classifer
        gen_dim (int, optional): Set the depth of the generator models Encoder and Decoder. Not currently implemented. Defaults to 10.
        method (str, optional): classical or quantum. Only needed for setting the file path for saving the results. Defaults to "classical".
        calc_embeddings (bool, optional): Set to caclulate new embeddings each time you run the classifier. Defaults to True.
        save_results (bool, optional): Save the prediction results and the training histories in seperate json files. Defaults to True.

    Returns:
        list, list: all prediction results, all training histories
    """

    results = []
    train_hists = []
    for _ in range(n):
        # first check if new embeddings have to be calculated
        if calc_embeddings:
            # move all existing input files
            for f in os.listdir("input_data"):
                if os.path.isfile(f):
                    shutil.move("input_data/" + f, "input_text/" + f)

            d2v.main(True, False, False, 150, "dbow", "input_data/")
        
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
    if save_results:
        with open(str(method) + "_results_" + str(n) + "times.json", 'w', encoding="utf-8") as res_fd: # FK: I run in errors because this file was not closed later on? Therefore: with open() as
            json.dump(new_results, res_fd)

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
    
    if save_results:
        with open(str(method) + "_train_hists_" + str(n) + "times.json", 'w', encoding="utf-8") as hist_fd: # FK: to avoid errors with still open files
            json.dump(new_hists, hist_fd)
    
    return new_results, new_hists


def display_results(n=35, method="classical", save_plots=True):
    """Load the saved results from a test run and display them in plots.
    Display MCC and threshold after prediction and the losses after training.

    Args:
        n (int, optional): amount of runs of the gan_classifier. Needed for reading the correct file path. Defaults to 35.
        method (str, optional): classical or quantum. Needed for reading the correct file path. Defaults to "classical".
        save_plots (bool, optional): Create and save plot of MCC, threshold and losses if True. Defaults to True.
    """
    with open(str(method) + "_results_" + str(n) + "times.json", 'r', encoding="utf-8") as res_fd:
        results = json.load(res_fd)

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

        ##### MCC with threshold
        plt.plot([i for i in range(n)], [x["MCC"] for x in results], color="green", label="MCC")
        plt.plot([i for i in range(n)], [x["threshold"] for x in results], color="blue", label="optimized anomaly threshold")
        plt.ylim(0, 1)
        plt.title("MCC and optimized anomaly threshold after prediciton")
        plt.legend()
        plt.xlabel("runs")
        plt.savefig(str(method) + "_MCC_threshold_" + str(n) + "times.png", bbox_inches="tight")
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

    with open(str(method) + "_train_hists_" + str(n) + "times.json", 'r', encoding="utf-8") as hist_fd:
        train_hists = json.load(hist_fd)
    if save_plots:
        ##### all losses separately
        for loss in ["contextual_loss", "adversarial_loss", "encoder_loss", "generator_loss", "discriminator_loss"]:
            for i in range(len(train_hists)):
                plt.plot(train_hists[i]["step_number"], [i for i in train_hists[i][loss]]) # if i < 1000 else 0
            plt.title(loss)
            plt.xlabel("runs")
            plt.savefig(str(method) + '_' + str(loss) + '_' + str(n) + "times.png", bbox_inches="tight")
            plt.cla()
            plt.clf()
        
        ##### all losses in one
        for loss, color in zip(["contextual_loss", "adversarial_loss", "encoder_loss", "generator_loss", "discriminator_loss"], ["green", "red", "blue", "black", "purple"]):
            for i in range(len(train_hists)):
                line, = plt.plot(train_hists[i]["step_number"], [i for i in train_hists[i][loss]], color=color) # if i < 1000 else 0 
            line.set_label(loss) # FK: only add the label once
        plt.title("All five losses over all runs")
        plt.legend()
        plt.xlabel("runs")
        plt.savefig(str(method) + "_all_losses_" + str(n) + "times.png", bbox_inches="tight")
        plt.cla()
        plt.clf()


def test_latent_dimensions(latent_dim_range, latent_dim_steps, each_run_n, method):
    """_summary_

    Args:
        latent_dim_range (tupel): (start of range, end of range)
        latent_dim_steps (int): amount of steps
        each_run_n (_type_): _description_
        method (_type_): _description_
    """
    all_MCC_means = []
    for dim in [int(x) for x in np.linspace(latent_dim_range[0], latent_dim_range[1], latent_dim_steps)]:
        print(dim)
        test_n_times(n=each_run_n, method=method, latent_dims=dim)
        pred_results, train_hists = display_results(n=each_run_n, method=method, save_plots=False)
        all_MCC_means.append(np.mean([x["MCC"] for x in pred_results]))
    
    plt.plot(np.linspace(latent_dim_range[0], latent_dim_range[1], latent_dim_steps), all_MCC_means)
    plt.title("mean MCCs for different latent dimensions")
    plt.ylabel("mean MCC")
    plt.xlabel("latent dimensions")
    plt.savefig(f"{method}_meanMC_latDim{latent_dim_range[0]}_{latent_dim_range[1]}_{latent_dim_steps}steps.png", bbox_inches="tight")
    plt.cla()
    plt.clf()

    return all_MCC_means


if __name__ == "__main__":
    tic = time.perf_counter()

    n = 5
    method = "quantum"
    #file_path = "input_text/liar_buzzfeed_amtCeleb_sents_150dim_dbowMethod.csv"

    test_n_times(n=n, method=method, calc_embeddings=False)
    display_results(n=n, method=method, save_plots=True)
    # test_latent_dimensions((10, 150), latent_dim_steps=15, each_run_n=15, method="classical")

    toc = time.perf_counter()
    print("Total runtime: ", toc-tic)
