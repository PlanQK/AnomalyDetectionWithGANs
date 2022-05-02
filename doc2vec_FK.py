
import sys
import numpy as np

from gensim.models import doc2vec

from word2vec_FK import create_word2vec, save_embeddings_in_file

from matplotlib import pyplot as plt



def analyze_data(cat2news):
    """Don't know, some old method. Maybe not needed anymore

    Args:
        cat2news (_type_): _description_
    """
    plt.hist([len(v) for vs in cat2news.values() for v in vs], range= (0, 600))
    plt.savefig("tmp.png", bbox_inches="tight")

    # print(np.mean([len(v) for vs in cat2news.values() for v in vs]))


def get_value_range(cat2embeddings):
    """Print the minimum and maximum value of the given embeddings

    Args:
        cat2embeddings (dict): mapping the categories (true or false) to embeddings
    """
    min_val = sys.maxsize
    max_val = -sys.maxsize - 1

    for vs in cat2embeddings.values():
        for v in vs:
            min_val = min(min_val, min(v))
            max_val = max(max_val, max(v))
    
    print("min_val", min_val)
    print("max_val", max_val)


def normalize_embeddings(cat2sent_embds):
    """Normalize given embeddings to a value range of [0, 1]

    Args:
        cat2sent_embds (dict): embeddings to normalize

    Returns:
        dict: normalized embeddings
    """
    normed = dict()
    for cat, embds in cat2sent_embds.items():
        normed[cat] = ( (embds - np.min(embds)) / (np.max(embds) - np.min(embds)) )

    # get_value_range(normed)
    return normed



if __name__ == "__main__":
    liar=False
    buzzfeed=False
    amtAndCelebrity=True
    dimensions = 250
    used_percentage = 1

    _, cat2news, word2occ = create_word2vec(dict(), dict(), liar_dataset=liar, buzzfeed_dataset=buzzfeed, amtAndCelebrity=amtAndCelebrity)

    documents = [doc2vec.TaggedDocument(sent, [i]) for i, sent in enumerate([v.words for vs in cat2news.values() for v in vs])]
    doc_model = doc2vec.Doc2Vec(documents, vector_size=dimensions, min_count=1, epochs=60)

    cat2sent_embds = dict()
    for cat, news in cat2news.items():
        if not cat in cat2sent_embds.keys():
            cat2sent_embds[cat] = []
        for new in news:
            cat2sent_embds[cat].append(doc_model.infer_vector(new.words))

    save_path = "input_text/"
    appends = []
    if liar:
        appends.append("liar")
    if buzzfeed:
        appends.append("buzzfeed")
    if amtAndCelebrity:
        appends.append("amtCeleb")
    save_path = save_path + '_'.join(appends) + "_sents_" + str(dimensions) + "dim_" + str(int(used_percentage*100)) + "perc.csv"

    save_embeddings_in_file(normalize_embeddings(cat2sent_embds), filepath2save=save_path, used_percentage_of_dataset=used_percentage)
