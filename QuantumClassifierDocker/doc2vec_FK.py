
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


def main(liar, buzzfeed, amtAndCeleb, dims, dm_or_dbow, path2save="input_text/"):
    _, cat2news, _ = create_word2vec(dict(), dict(), liar_dataset=liar, buzzfeed_dataset=buzzfeed, amtAndCelebrity=amtAndCeleb)

    documents = [doc2vec.TaggedDocument(sent, [i]) for i, sent in enumerate([v.words for vs in cat2news.values() for v in vs])]
    doc_model = doc2vec.Doc2Vec(documents, vector_size=dims, min_count=1, epochs=60, dm=1 if dm_or_dbow=="dm" else 0)

    cat2sent_embds = dict()
    for cat, news in cat2news.items():
        if not cat in cat2sent_embds.keys():
            cat2sent_embds[cat] = []
        for new in news:
            cat2sent_embds[cat].append(doc_model.infer_vector(new.words))

    appends = []
    if liar:
        appends.append("liar")
    if buzzfeed:
        appends.append("buzzfeed")
    if amtAndCeleb:
        appends.append("amtCeleb")
    save_path = path2save + '_'.join(appends) + "_sents_" + str(dims) + "dim_" + str(dm_or_dbow) + "Method.csv"

    save_embeddings_in_file(normalize_embeddings(cat2sent_embds), filepath2save=save_path)

if __name__ == "__main__":
    liar=True
    buzzfeed=False
    amtAndCelebrity=False
    dimensions = 250
    dm_or_dbow = "dbow"
    
    main(liar, buzzfeed, amtAndCelebrity, dimensions, dm_or_dbow)
