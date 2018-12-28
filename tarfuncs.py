import sys, os, shutil
import tarfile

def tarFileAndRemove(fname):
    if os.path.isdir(fname):
        if os.path.isfile(fname+".gz"):
            os.remove(fname+".gz")
        with tarfile.open(fname+".gz", "w:gz") as tar_handle:
            for root, dirs, file in os.walk(fname):
                for f in file:
                    tar_handle.add(os.path.join(root, f), arcname=f)
        shutil.rmtree(fname)
    else:
        print("Directory does not exist!")

def untar(fname):
    if os.path.isdir(fname):
        print("Something's already extracted there!")
    elif (os.path.isfile(fname+".gz")):
        tar = tarfile.open(fname+".gz")
        tar.extractall(path=fname)
        tar.close()
    else:
        print("Not a .gz file!")