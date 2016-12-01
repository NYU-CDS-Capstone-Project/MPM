import errno
import os

def mkdir_all(dirName,n_iter,n_exp):
    try:
        for i in range(n_iter):
            path=dirName+"/n_iter"+str(i)+"/exp_exp_init"
            os.makedirs(path)
            for j in range(n_exp):
                path=dirName+"/n_iter"+str(i)+"/exp_exp_"+str(j)
                os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
