
import doc2vec_FK as d2v
import run_gan_classifier as gan

import time
import shutil
import json
import os, sys


def test_n_times(n: int, gen_dim=10, method="classical", calc_embeddings=True, save_results=True, par_number=-1):
    """Run the gan_classifier n times and save the results in json files
    Saves the results after prediction in a json file and the train_hist after training (including all losses) in a json file as well.

    Args:
        n (int): amount of runs of the gan_classifer
        gen_dim (int, optional): Set the depth of the generator models Encoder and Decoder. Not currently implemented. Defaults to 10.
        method (str, optional): classical or quantum. Only needed for setting the file path for saving the results. Defaults to "classical".
        calc_embeddings (bool, optional): Set to caclulate new embeddings each time you run the classifier. Defaults to True.
        save_results (bool, optional): Save the prediction results and the training histories in seperate json files. Defaults to True.
        par_number (int, optional): Save the results to a specific "part" file of the complete result. Defaults to -1.

    Returns:
        list, list: all prediction results, all training histories
    """

    results = {
        "classical": [],
        "quantum": []
    }
    train_hists = {
        "classical": [],
        "quantum": []
    }
    for _ in range(n):
        # first check if new embeddings have to be calculated
        if calc_embeddings:
            # move all existing input files
            for f in os.listdir("input_data"):
                if os.path.isfile(f):
                    shutil.move("input_data/" + f, "input_text/" + f)

            d2v.main(True, False, False, 150, "dm", "input_data/")
        
        for meth in train_hists.keys():
            train_hist = gan.main("train", meth)
            if train_hist == None:
                print("Train history was None. Check log.log file")
            else:
                train_hists[meth].append(train_hist)

            res = gan.main("predict", meth)
            if res != None:
                results[meth].append(res)


    new_results = {
        "classical": [],
        "quantum": []
    }
    for meth, ress in results.items(): # FK: to avoid some json and int64 errors
        for res in ress:
            tmp = dict()
            for k, v in res.items():
                if isinstance(v, list):
                    tmp[k] = int(v[0])
                else:
                    tmp[k] = float(v)
            new_results[meth].append(tmp)
    if save_results:
        with open(str(method) + "_results_" + str(n) + "times" + (str("_" + str(par_number)) if int(par_number)!=-1 else "") + ".json", 'w',
                  encoding="utf-8") as res_fd: 
            json.dump(new_results, res_fd)

    new_hists = {
        "classical": [],
        "quantum": []
    }
    for meth, hists in train_hists.items():
        for hist in hists:
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
            new_hists[meth].append(tmp)
    if save_results:
        with open(str(method) + "_train_hists_" + str(n) + "times" + (str("_" + str(par_number)) if int(par_number)!=-1 else "") + ".json", 'w',
                  encoding="utf-8") as hist_fd:
            json.dump(new_hists, hist_fd)

    return new_results, new_hists


def merge_par_results(n=35, method="classical", file_path='./'):
    """Merge the part results of parallel run test runs.
    Save them all together in one json file

    Args:
        n (int, optional): amount of runs of the gan_classifer. Defaults to 35.
        method (str, optional): classical or quantum. Only needed for setting the file path for saving the results. Defaults to "classical".
        file_path (str, optional): the location in which the separated runs are located. Defaults to './'.

    Returns:
        int: the amount of test runs
    """
    for res_type in ["results", "train_hists"]:
        amount = 0
        print(res_type)
        all_classical = []
        all_quantum = []
        for file in os.listdir(file_path):
            if method in file and res_type in file: # and file.endswith(".json")
                with open(file_path + file, 'r', encoding="utf-8") as js_fd:
                    one_result = json.load(js_fd)
                    [all_classical.append(cla) for cla in one_result["classical"]]
                    [all_quantum.append(cla) for cla in one_result["quantum"]]
                amount += 1
        complete_results = {
            "classical": all_classical,
            "quantum": all_quantum
        }
        with open(str(method) + '_' + str(res_type) + '_' + str(int(amount)) + "times.json", 'w', encoding="utf-8") as js_fd:
            json.dump(complete_results, js_fd)
    
    return amount


if __name__ == "__main__":
    tic = time.perf_counter()

    parallel_number = -1
    if len(sys.argv) > 1:
        parallel_number = sys.argv[1]
    
    n = 1
    method = "both"

    test_n_times(n=n, method=method, calc_embeddings=False, par_number=parallel_number)
    n = merge_par_results(n=n, method=method, file_path="saved_results/multi_input_qubits/200steps/")

    toc = time.perf_counter()
    print("Total runtime in seconds: ", toc-tic)