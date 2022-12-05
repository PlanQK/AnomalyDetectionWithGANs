import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    """
    Class holding all the plot methods that should be executed after the training or after testing.
    The plots used are executed as defined in the plot-function.
    """

    def __init__(self, train_hist, fp="", pix_num_one_side=0, validation=True):
        self.train_hist = train_hist
        self.fp = fp
        self.pix_num_one_side = pix_num_one_side
        self.validation = validation

    def plot(self):
        if self.validation:
            self.loss_plot()
            self.threshold_and_MCC_plot()
            self.confusion_matrix()
            self.show_pixXpix_samples()
        else:
            self.confusion_matrix()

    def loss_plot(self):
        plt.plot(
            self.train_hist["step_number"],
            np.asarray(self.train_hist["generator_loss"])
            / self.train_hist["generator_loss"][0],
            label="gen loss",
        )
        plt.plot(
            self.train_hist["step_number"],
            np.asarray(self.train_hist["discriminator_loss"])
            / self.train_hist["discriminator_loss"][0],
            label="disc loss",
        )
        plt.plot(
            self.train_hist["step_number"],
            np.asarray(self.train_hist["adversarial_loss"])
            / self.train_hist["adversarial_loss"][0],
            label="adversarial_loss",
        )
        plt.plot(
            self.train_hist["step_number"],
            np.asarray(self.train_hist["contextual_loss"])
            / self.train_hist["contextual_loss"][0],
            label="contextual_loss",
        )
        plt.plot(
            self.train_hist["step_number"],
            np.asarray(self.train_hist["encoder_loss"])
            / self.train_hist["encoder_loss"][0],
            label="encoder_loss",
        )
        plt.yscale("log")
        plt.title("Normalized Generator and discriminator_loss")
        plt.xlabel("Steps")
        plt.ylabel("Normalized Loss")
        plt.legend()
        plt.savefig("model/train_history/loss_plot.png")
        plt.close()
        return None

    def threshold_and_MCC_plot(self):
        fig, ax = plt.subplots()
        plt.title("Anomaly-score threshold and MCC")
        plt.xlabel("Steps")
        plt.plot(
            self.train_hist["step_number"],
            self.train_hist["MCC"],
            "y",
            label="MCC",
        )
        plt.legend(loc="lower left")
        plt.ylabel("MCC")
        plt.ylim(bottom=0)

        ax2 = ax.twinx()
        plt.plot(
            self.train_hist["step_number"],
            self.train_hist["threshold"],
            "b",
            label="threshold",
        )
        plt.legend(loc="lower right")
        plt.ylabel("threshold")

        plt.savefig("model/train_history/threshold_and_MCC_plot.png")
        plt.close()
        return None

    def confusion_matrix(self):
        # determine indices/steps of best MCC values
        best_MCC = -2
        indizes = []
        if self.validation:
            for i in range(len(self.train_hist["MCC"])):
                if self.train_hist["MCC"][i] > best_MCC:
                    best_MCC = self.train_hist["MCC"][i]
                    indizes.append(i)
        else:
            indizes = [0]

        # output plots
        for i in indizes:
            x_labels = ["Predicted Normal", "Predicted Anomaly"]
            y_labels = ["True Normal", "True Anomaly"]
            total_num_true_normals = (
                self.train_hist["TP"][i] + self.train_hist["FN"][i]
            )
            total_num_true_anomalies = (
                self.train_hist["FP"][i] + self.train_hist["TN"][i]
            )
            conf_matrix = np.array(
                [
                    [
                        self.train_hist["TP"][i]
                        / total_num_true_anomalies
                        * 100,
                        self.train_hist["FN"][i]
                        / total_num_true_anomalies
                        * 100,
                    ],
                    [
                        self.train_hist["FP"][i]
                        / total_num_true_normals
                        * 100,
                        self.train_hist["TN"][i]
                        / total_num_true_anomalies
                        * 100,
                    ],
                ]
            )
            conf_matrix = np.around(conf_matrix, decimals=2)
            fig, ax = plt.subplots()
            im = ax.imshow(conf_matrix, cmap="brg", vmin=0, vmax=100)

            # Create colorbar
            cbar = ax.figure.colorbar(im, ax=ax, cmap="brg")
            cbar.ax.set_ylabel("percentage", rotation=-90, va="bottom")

            # Show all ticks and label them with the respective list entries
            ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
            ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)

            # Rotate the tick labels and set their alignment.
            plt.setp(
                ax.get_xticklabels(),
                rotation=45,
                ha="right",
                rotation_mode="anchor",
            )

            # Loop over data dimensions and create text annotations.
            for k in range(len(y_labels)):
                for j in range(len(x_labels)):
                    text = ax.text(
                        j,
                        k,
                        conf_matrix[k, j],
                        ha="center",
                        va="center",
                        color="w",
                    )

            if self.validation:
                ax.set_title(
                    f"Rowwise normalized confusion matrix @ step {self.train_hist['step_number'][i]}"
                )
            else:
                ax.set_title(
                    f"Rowwise normalized confusion matrix on test set"
                )
            fig.tight_layout()
            if self.fp:
                plt.savefig(self.fp + "/confusion_matrix.png")
            else:
                plt.savefig(
                    f"model/train_history/confusion_matrix_step_{self.train_hist['step_number'][i]}_MCC_{self.train_hist['MCC'][i]:.2f}.png"
                )
            plt.close()

        return None

    def show_pixXpix_samples(self):
        # determine indices/steps of best MCC values
        best_MCC = -2
        indizes = []
        for i in range(len(self.train_hist["MCC"])):
            if self.train_hist["MCC"][i] > best_MCC:
                best_MCC = self.train_hist["MCC"][i]
                indizes.append(i)

        x_normal_samples = np.reshape(
            self.train_hist["x_normal_samples"],
            (
                len(self.train_hist["x_normal_samples"]),
                len(self.train_hist["x_normal_samples"][0]),
                self.pix_num_one_side,
                self.pix_num_one_side,
            ),
        )
        x_hat_normal_samples = np.reshape(
            self.train_hist["x_hat_normal_samples"],
            (
                len(self.train_hist["x_hat_normal_samples"]),
                len(self.train_hist["x_hat_normal_samples"][0]),
                self.pix_num_one_side,
                self.pix_num_one_side,
            ),
        )
        x_unnormal_samples = np.reshape(
            self.train_hist["x_unnormal_samples"],
            (
                len(self.train_hist["x_unnormal_samples"]),
                len(self.train_hist["x_unnormal_samples"][0]),
                self.pix_num_one_side,
                self.pix_num_one_side,
            ),
        )
        x_hat_unnormal_samples = np.reshape(
            self.train_hist["x_hat_unnormal_samples"],
            (
                len(self.train_hist["x_hat_unnormal_samples"]),
                len(self.train_hist["x_hat_unnormal_samples"][0]),
                self.pix_num_one_side,
                self.pix_num_one_side,
            ),
        )

        for i in indizes:
            fig, axs = plt.subplots(2, 2)
            im = axs[0, 0].imshow(x_normal_samples[i][0])
            axs[0, 0].set_title(
                f"x_normal - step {self.train_hist['step_number'][i]}"
            )
            plt.colorbar(im, ax=axs[0, 0])
            im = axs[1, 0].imshow(x_hat_normal_samples[i][0])
            axs[1, 0].set_title(
                f"x_hat_normal - step {self.train_hist['step_number'][i]}"
            )
            plt.colorbar(im, ax=axs[1, 0])
            im = axs[0, 1].imshow(x_unnormal_samples[i][0])
            axs[0, 1].set_title(
                f"x_anomaly - step {self.train_hist['step_number'][i]}"
            )
            plt.colorbar(im, ax=axs[0, 1])
            im = axs[1, 1].imshow(x_hat_unnormal_samples[i][0])
            axs[1, 1].set_title(
                f"x_hat_anomaly - step {self.train_hist['step_number'][i]}"
            )
            plt.colorbar(im, ax=axs[1, 1])
            for ax in axs.flat:
                ax.label_outer()
            for k in range(2):
                for j in range(2):
                    axs[k, j].get_xaxis().set_visible(False)
                    axs[k, j].get_yaxis().set_visible(False)
            plt.savefig(
                f"model/train_history/samples_step_{self.train_hist['step_number'][i]}_MCC_{self.train_hist['MCC'][i]:.2f}.png"
            )
            plt.close()

        return None


class QuantumDecoderPlotter(Plotter):
    """
    Class holding additionally methods and attributes to conduct the training or testing procedure.
    """

    def __init__(self, train_hist, fp="", pix_num_one_side=0, validation=True):
        super().__init__(train_hist, fp, pix_num_one_side, validation)

    def plot(self):
        if self.validation:
            self.loss_plot()
            self.threshold_and_MCC_plot()
            self.confusion_matrix()
            self.show_pixXpix_samples()
        else:
            self.confusion_matrix()
