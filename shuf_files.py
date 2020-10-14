import shutil
import os
import argparse

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

    for i in file_ids:
        ids.append(i.strip())

    data = shuf_files(from_dirr, ids, to_dirr)


if __name__ == "__main__":
    main()
