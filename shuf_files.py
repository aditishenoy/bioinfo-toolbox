import shutil
<<<<<<< HEAD
import os
import argparse
=======
>>>>>>> 46fca9b4cd460b946d19f7725d3a85da84a2653a

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
<<<<<<< HEAD
        if filename.endswith('.fa'):
      
           print (filename[18:-4])
           if (filename[18:-4]) in ids:
                   path = os.path.join(from_dirr, filename)
                   print (path)
                   shutil.move(path, to_dirr+"{}".format(filename[18:-4]+'npz'))

def main():

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--from_dirr", type=str)
    parser.add_argument('--to_dirr', type=str)
    parser.add_argument('--ids', type=str)

    args = parser.parse_args()

    from_dirr = args.from_dirr 
    to_dirr = args.to_dirr

    ids = []
    file_ids = open(args.ids, 'r')
=======
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
>>>>>>> 46fca9b4cd460b946d19f7725d3a85da84a2653a

    for i in file_ids:
        ids.append(i.strip())

    data = shuf_files(from_dirr, ids, to_dirr)


if __name__ == "__main__":
    main()
