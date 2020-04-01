#!/usr/bin/env python3

import os

def get_fasta_file_path(folder, pdb_code):
    """
    ___function to generate file paths___
    parameters:
    ----------
    folder : str (path to fasta folder)
    pdb_code : str (single pdb_id)
    returns:
    ----------
    paths : dict 
            (Dictionary of {(pdb_id) : path to all the fasta files})
    """

    paths = {}
    dirr = (folder+(pdb_code.lower()))
    for filename in os.listdir(dirr):
        if filename.endswith(".fa"):
             path = (os.path.join(dirr, filename))
        else:
             continue
        paths[filename[5]] = path
    return paths



def get_pdb_file_path(folder):
    """
    ___function to generate file paths___
    parameters:
    ----------
    folder : str (path to pdb folder)
    pdb_code : str (single pdb_id)
    returns:
    ----------
    paths : dict 
            (Dictionary of {(pdb_id) : path to all the pdb files})
    """

    paths = {}
    for filename in os.listdir(folder):
        if filename.endswith(".pdb"):
            path = (os.path.join(folder, filename))
        else:
           continue
        paths[filename[0:4]] = path
    return paths
