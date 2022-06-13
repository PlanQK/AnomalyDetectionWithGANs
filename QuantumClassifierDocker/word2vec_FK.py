
from nltk.tokenize import word_tokenize

from gensim.models import Word2Vec, FastText

import numpy as np
import sys
import os

from xml.etree import ElementTree

import random as rnd

from matplotlib import pyplot as plt

xml_path = "input_text/BuzzFeed/articles/articles/"

window_length = 1
min_word_occ = 1


class News():
    """Class to represent one news with the category, the news itself as a list of words, and the file path in which the news is located
    """
    
    def __init__(self, category, words, file_path):
        self.category = category
        self.words = words
        self.file_path = file_path


def get_value_range(word2occ, word2emb):
    """Print the minimum and maximum value of the given embeddings

    Args:
        word2occ (dict): mapping the word string to the amount of occurences. Needed to only include the embeddings of words with a occurence of x and higher. x is the global variable min_word_occ
        word2emb (dict): mapping the word string to the corresponding embedding
    """
    min_val = sys.maxsize
    max_val = -sys.maxsize - 1
    for w in [w for w, occ in word2occ.items() if occ >= min_word_occ]:
        emb = word2emb[w]
        min_val = min(min(emb), min_val)
        max_val = max(max(emb), max_val)
    print("min_val", min_val)
    print("max_val", max_val)


def normalize_data(word2emb):
    """Normalize given embeddings to a value range of [0, 1]

    Args:
        word2emb (dict): embeddings to normalize

    Returns:
        dict: normalized embeddings
    """
    return {w:( (data-np.min(data)) / (np.max(data)-np.min(data)) ) for w, data in word2emb.items()}


def norm_text(text):
    """Normalize given text.
    Currently not implemented

    Args:
        text (list): text to normalize

    Returns:
        list: normalized text
    """
    if text == None:
        return None
    return text


def analyze_dataset(dataset_name, current_cat2news, current_word2occ, plot=False):
    """Provide some meta information about a dataset.
    Print amount of data points and the distributions of true and false news.
    Also plot these distributions

    Args:
        dataset_name (str): name of the corpus
        current_cat2news (dict): mapping the category (true or false) to the corresponding news
        current_word2occ (dict): mapping the words to their amount of occurrences
    """
    if plot:
        plot_path = str(dataset_name) + "_allLengths.png"
        print("--------------------------------")
        print("The " + str(dataset_name) + " dataset")
        print("\tvocabulary (= amount of distinct words) = " + str(len(current_word2occ)))
        print("\tamount of false news = " + str(len(current_cat2news["false"])))
        print("\tamount of true news = " + str(len(current_cat2news["true"])))
        all_lengths = [len(v.words) for key, vs in current_cat2news.items() if key in ["true", "false"] for v in vs]
        true_lengths = [len(v.words) for key, vs in current_cat2news.items() if key in ["true"] for v in vs]
        false_lengths = [len(v.words) for key, vs in current_cat2news.items() if key in ["false"] for v in vs]
        print("\tlength of all news = " + str(np.mean(all_lengths)) + " (mean), " + str(np.median(all_lengths)) + " (median), " + str(np.min(all_lengths)) + " (shortest), " + str(np.max(all_lengths)) + " (longest)")
        print("\tlength of true news = " + str(np.mean(true_lengths)) + " (mean), " + str(np.median(true_lengths)) + " (median), " + str(np.min(true_lengths)) + " (shortest), " + str(np.max(true_lengths)) + " (longest)")
        print("\tlength of false news = " + str(np.mean(false_lengths)) + " (mean), " + str(np.median(false_lengths)) + " (median), " + str(np.min(false_lengths)) + " (shortest), " + str(np.max(false_lengths)) + " (longest)")
        print("\tthe distribution of the lengths can be found in the created plot: " + str(plot_path))
        print()
        plt.hist([true_lengths, false_lengths], label=["true", "false"], bins=20)
        plt.title(dataset_name + " corpus")
        plt.xlabel("words per news")
        plt.ylim(0, 1100)
        plt.legend()
        plt.savefig(plot_path, bbox_inches="tight")

        plt.cla()
        plt.clf()


def add_words_to_word2occ(word2occ, new_text):
    for i in [w for w in word_tokenize(new_text) if w != ' ']:
        if not i.lower() in word2occ.keys():
            word2occ[i.lower()] = 0
        word2occ[i.lower()] += 1
    return word2occ

def add_news_to_cat2news(cat2news, cat, new_news, file_path):
    if not cat in cat2news.keys():
        cat2news[cat] = []

    cat2news[cat].append(News(cat, [i.lower() for i in word_tokenize(new_news) if i != ' '], file_path))

    return cat2news

def _word2vec4Liar(cat2news, word2occ):
    """Add the news and words of the liar dataset to the two given dict structures

    Args:
        cat2news (dict): mapping category (true or false) to the corresponding news. In here, the new news of the liar dataset are added
        word2occ (dict): mapping words to their occurrence. Update this structure with the new words of the liar dataset

    Returns:
        dict, dict: the updated cat2news and word2occ
    """
    current_cat2news = dict()
    current_word2occ = dict()
    for f in ["input_text/liar_dataset/train.tsv", "input_text/liar_dataset/valid.tsv", "input_text/liar_dataset/test.tsv"]:
        with open(f, 'r', encoding="utf-8") as train_fd:
            for line in train_fd.readlines():
                _, news_class, news, *_ = line.strip('\n').split('\t')

                if news_class == "false" or news_class == "true":
                    pass
                elif news_class == "mostly-true":
                    news_class = "true"
                else:
                    news_class = "misc"

                cat2news = add_news_to_cat2news(cat2news, news_class, news, f)
                current_cat2news = add_news_to_cat2news(current_cat2news, news_class, news, f)

                word2occ = add_words_to_word2occ(word2occ, news)
                current_word2occ = add_words_to_word2occ(current_word2occ, news)

    analyze_dataset("liar", current_cat2news, current_word2occ)
    
    return cat2news, word2occ

def _word2vec4Buzzfeed(cat2news, word2occ):
    """Add the news and words of the buzzfeed dataset to the two given dict structures

    Args:
        cat2news (dict): mapping category (true or false) to the corresponding news. In here, the new news of the buzzfeed dataset are added
        word2occ (dict): mapping words to their occurrence. Update this structure with the new words of the buzzfeed dataset

    Returns:
        dict, dict: the updated cat2news and word2occ
    """
    current_cat2news = dict()
    current_word2occ = dict()

    with open("input_text/BuzzFeed/overview.csv", 'r', encoding="utf-8") as overview_fd:
        first_line = False
        for line in overview_fd.readlines():
            if not first_line:
                first_line = True
                continue
            xml_file, _, _, category, _ = line.split(",")
            if category in ["no factual content", "mostly false"]:
                category = "false"
            elif category in ["mostly true"]:
                category = "true"
            else:
                category = "misc"

            with open(xml_path + xml_file, "r", encoding="utf-8") as xml_fd:
                e = ElementTree.fromstring(xml_fd.read())
                for ch in e.getchildren():
                    if ch.tag == "mainText":
                        text = norm_text(ch.text)
                        if text != None:
                            cat2news = add_news_to_cat2news(cat2news, category, text, xml_path + xml_file)
                            current_cat2news = add_news_to_cat2news(current_cat2news, category, text, xml_path + xml_file)
                            
                            word2occ = add_words_to_word2occ(word2occ, text)
                            current_word2occ = add_words_to_word2occ(current_word2occ, text)
    
    analyze_dataset("buzzfeed", current_cat2news, current_word2occ)
    
    return cat2news, word2occ

def _word2vec4AMTandCeleb(cat2news, word2occ):
    """Add the news and words of the AmtCeleb dataset to the two given dict structures

    Args:
        cat2news (dict): mapping category (true or false) to the corresponding news. In here, the new news of the AmtCeleb dataset are added
        word2occ (dict): mapping words to their occurrence. Update this structure with the new words of the AmtCeleb dataset

    Returns:
        dict, dict: the updated cat2news and word2occ
    """
    current_cat2news = dict()
    current_word2occ = dict()

    file_path = "input_text/AMT_and_Celebrity/archive/overall/overall/"

    for dir_path in [file_path + "fake", file_path + "real", file_path + "celebrityDataset/fake", file_path + "celebrityDataset/legit"]:
        cat = "true"
        if "fake" in dir_path:
            cat = "false"
        
        for f in os.listdir(dir_path):
            file_path = dir_path + '/' + f
            content = open(file_path, 'r', encoding="utf-8").read().strip()
                            
            word2occ = add_words_to_word2occ(word2occ, content)
            current_word2occ = add_words_to_word2occ(current_word2occ, content)

            cat2news = add_news_to_cat2news(cat2news, cat, content, file_path)
            current_cat2news = add_news_to_cat2news(current_cat2news, cat, content, file_path)
    
    analyze_dataset("amtCeleb", current_cat2news, current_word2occ)

    return cat2news, word2occ


def create_word2vec(cat2news, word2occ, liar_dataset, buzzfeed_dataset, amtAndCelebrity):
    """Create a gensim Word2Vec model

    Args:
        cat2news (dict): the news are saved corresponding to their category [true, false]
        word2occ (dict): the words with their amount of occurences are saved in here

    Returns:
        Word2Vec, dict, dict: the created gensim model, cat2news, word2occ
    """

    if liar_dataset:
        cat2news, word2occ = _word2vec4Liar(cat2news, word2occ)

    if buzzfeed_dataset:
        cat2news, word2occ = _word2vec4Buzzfeed(cat2news, word2occ)

    if amtAndCelebrity:
        cat2news, word2occ = _word2vec4AMTandCeleb(cat2news, word2occ)

    if "misc" in cat2news.keys():
        print("There are " + str(len(cat2news["true"])) + " true news, " + str(len(cat2news["false"])) + " false news and " + str(len(cat2news["misc"])) + " miscellaneous.")
    else:
        print("There are " + str(len(cat2news["true"])) + " true news and " + str(len(cat2news["false"])) + " false news.")
    print("There are " + str(len(word2occ.keys())) + " distinct words in total.")

    model = Word2Vec(sentences=[w.words for ws in list(cat2news.values()) for w in ws], min_count=min_word_occ, vector_size=300)

    return model, cat2news, word2occ


def save_embeddings_in_file(cat2embeddings, filepath2save="input_text/liar_and_buzzfeed.csv", used_percentage_of_dataset=1.0):
    """Save the embeddings to a given file path in the following format:
    V1,V2,...,Vx,Class
    ...
    ...

    NB: Currently, the used_percentage_of_dataset does not ensure, that there are more non-anomalies than anomalies.

    Args:
        cat2embeddings (dict): dictionary mapping the category [true, false] to the actual embeddings.
        filepath2save (str, optional): The filepath to which the embddings shall be saved. Defaults to "input_text/liar_and_buzzfeed.csv".
        used_percentage_of_dataset (float, optional): Only use this fraction of the whole dataset. Defaults to 1.0.
    """

    with open(filepath2save, 'w', encoding="utf-8") as train_fd:
        train_fd.write(",".join(["V"+str(i) for i in range(1, len(cat2embeddings["true"][0])+1)]) + ",Class\n") # header
        all_lines = []
        for news_class, all_embds in cat2embeddings.items():
            if news_class in ["true", "false"]:
                class_id = -1
                if news_class in ["true"]:
                    class_id = 1
                elif news_class in ["false"]:
                    class_id = 0
                for emb in all_embds:
                    all_lines.append(",".join([str(e) for e in emb]) + "," + str(class_id) + '\n')
        rnd.shuffle(all_lines)
        for line in rnd.sample(all_lines, int(used_percentage_of_dataset * len(all_lines))):
            train_fd.write(line)


if __name__ == "__main__":
    model, cat2news, word2occ = create_word2vec(dict(), dict())

    model.save("input_text/word2vec.model")