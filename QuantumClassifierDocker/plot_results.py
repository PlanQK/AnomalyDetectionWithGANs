
import matplotlib.pyplot as plt
import numpy as np
import json

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

    print("MCC class: " + str(np.mean([x["MCC"] for x in results["classical"]])) + " (mean), " + str(np.median([x["MCC"] for x in results["classical"]])) + " (median), " + str(np.std([x["MCC"] for x in results["classical"]])) + " (st dev)")
    print("threshold class: " + str(np.mean([x["threshold"] for x in results["classical"]])) + " (mean), " + str(np.median([x["threshold"] for x in results["classical"]])) + " (median), " + str(np.std([x["threshold"] for x in results["classical"]])) + " (st dev)")

    print("MCC quan: " + str(np.mean([x["MCC"] for x in results["quantum"]])) + " (mean), " + str(np.median([x["MCC"] for x in results["quantum"]])) + " (median), " + str(np.std([x["MCC"] for x in results["quantum"]])) + " (st dev)")
    print("threshold quan: " + str(np.mean([x["threshold"] for x in results["quantum"]])) + " (mean), " + str(np.median([x["threshold"] for x in results["quantum"]])) + " (median), " + str(np.std([x["threshold"] for x in results["quantum"]])) + " (st dev)")

    if save_plots:
        ##### MCC
        for meth, color in zip(["classical", "quantum"], ["blue", "black"]):
            plt.scatter([i for i in range(n)], [x["MCC"] for x in results[meth]],
                        c=[color for _ in range(n)], label=meth)
        plt.ylim(0, 1)
        plt.title("MCC after prediction")
        plt.ylabel("MCC")
        plt.xlabel("runs")
        plt.legend()
        plt.savefig(str(method) + "_MCC_" + str(n) + "times.png", bbox_inches="tight")
        plt.cla()
        plt.clf()

        plt.boxplot([[x["MCC"] for x in results["classical"]], [x["MCC"] for x in results["quantum"]]], showmeans=True, labels=["classical", "quantum"])
        plt.ylabel("MCC")
        plt.ylim(0, 1)
        plt.savefig(str(method) + "_MCC_boxplot_" + str(n) + "times.png", bbox_inches="tight")
        plt.cla()
        plt.clf()

        ##### MCC with threshold
        for meth, color in zip(["classical", "quantum"], ["blue", "black"]):
            plt.plot([i for i in range(n)], [x["MCC"] for x in results[meth]], color=color, label=f"MCC-{meth}")
            plt.plot([i for i in range(n)], [x["threshold"] for x in results[meth]], color=color,
                     linestyle="dashed", label=f"optimized anomaly threshold-{meth}")
        plt.ylim(0, 1)
        plt.title("MCC and optimized anomaly threshold after prediciton")
        plt.legend()
        plt.xlabel("runs")
        plt.savefig(str(method) + "_MCC_threshold_" + str(n) + "times.png", bbox_inches="tight")
        plt.cla()
        plt.clf()

        ##### Threshold
        for meth, color in zip(["classical", "quantum"], ["blue", "black"]):
            plt.scatter([i for i in range(n)], [x["threshold"] for x in results[meth]],
                        c=[color for _ in range(n)], label=meth)
        plt.title("Threshold after prediction")
        plt.xlabel("runs")
        plt.ylabel("threshold")
        plt.legend()
        plt.savefig(str(method) + "_threshold_" + str(n) + "times.png", bbox_inches="tight")
        plt.cla()
        plt.clf()

    with open(str(method) + "_train_hists_" + str(n) + "times.json", 'r', encoding="utf-8") as hist_fd:
        train_hists = json.load(hist_fd)
    if save_plots and False:
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


def plot_grouped_boxplot():
    ticks = ["100steps", "200steps", "600steps"]
    # with open("saved_results/multi_input_qubits/100_steps/both_results_5times.json")
    bpl = plt.boxplot(data_a, positions=np.array(xrange(len(data_a)))*2.0-0.4, sym='', widths=0.6)
    bpr = plt.boxplot(data_b, positions=np.array(xrange(len(data_b)))*2.0+0.4, sym='', widths=0.6)

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#D7191C', label='Apples')
    plt.plot([], c='#2C7BB6', label='Oranges')
    plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('boxcompare.png')


if __name__ == "__main__":
    n = 5
    method = "both"
    display_results(n=n, method=method, save_plots=True)