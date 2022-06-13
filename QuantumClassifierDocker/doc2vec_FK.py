
import sys
import numpy as np

from gensim.models import doc2vec

from word2vec_FK import create_word2vec, save_embeddings_in_file


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

    return normed


def main(dims, dm_or_dbow, liar=True, buzzfeed=False, amtAndCeleb=False, path2save="input_text/"):
    """Create a Doc2Vec neural model based on `gensim` and the following article: http://arxiv.org/abs/1405.4053v2. The training material for the model are all news of the parameter-specified data sets.
    Infer the news of all parameter-specified data sets and save them in a json file.

    Args:
        dims (int): the amount of dimensions for the numeric paragraph embeddings
        dm_or_dbow (str): "dm" or "dbow". Defines the method of the creation of the paragraph embeddings. Check the article in the above description for reference.
        liar (bool, optional): IF True, the liar dataset will be included in the Doc2Vec model. Defaults to True.
        buzzfeed (bool, optional): If True, the buzzfeed dataset will be included in the Doc2Vec model. Defaults to True.
        amtAndCeleb (bool, optional): If True, the AMTandCelebrity dataset will be included in the Doc2Vev model. Defaults to True.
        path2save (str, optional): Defines the prefix of the file path in which to save the new embeddings. Defaults to "input_text/".
    """
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
    dimensions = 150
    dm_or_dbow = "dm"
    
    main(dimensions, dm_or_dbow)
