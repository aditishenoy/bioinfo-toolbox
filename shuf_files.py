import shutil

def shuf_files(from_dirr, ids, to_dirr):
    """Move/Copy files in certain directory which fullfils certain conditions to other dirr..

    Parameters
    ----------
    dirr: str // Path to src dirr

    ids : numpy matrix // List of ids to satify criteria

    Returns
    -------
    None : Output is that the required files are moved/copied
    """

    for filename in os.listdir(from_dirr):
        print (filename[:-6])
        if (filename[:-6]) in ids:
            """if 'test' in filename:"""
            path = os.path.join(from_dirr, filename)
            print (path)
            shutil.copy(path, to_dirr+"{}".format(filename))

def main():

    from_dirr = "/home/ashenoy/workspace/SubPred/data/EXP-5-20-05-03/FastaFiles25389"
    to_dirr = "/home/ashenoy/workspace/SubPred/data/EXP-5.1-20-05-06-embeddings/BLOSUM/testing/"

    ids = []
    file_ids = open("/home/ashenoy/workspace/SubPred/data/EXP-5.1-20-05-06-embeddings/BLOSUM/testing/testing_ids", "r")

    for i in file_ids:
        ids.append(i.strip())

    data = shuf_files(from_dirr, ids, to_dirr)


if __name__ == "__main__":
    main()
