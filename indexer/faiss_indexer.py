import faiss



def get_faiss_indexer(dimension_size):
    '''


    Return
    ------
    '''
    indexer = faiss.IndexFlatL2(dimension_size)

    return indexer