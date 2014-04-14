"""
Given a raw co-occurrence matrix this program has the following functionality.

1) Read the matrix in libsvm sparse format and compute positive pointwise
    mutual information (PPMI) for the elements in this matrix and save the 
    result to a text file in the libsvm sparse format.
    
2) Read a matrix (this can be any matrix not limiting to the PPMI matrix in (1))
    and compute its singular value decomposition.
    
3) Read the singular value decomposition result (ut, s, vt) and compute the
    dimension reduced matrix ut*s.
    
4) Read the singular value decomposition result (ut, s, vt) and parameters
    k (number of singular vector to use for the approximation),
    p (power to which we must raise the diagonal matrix that contain
    the singular values),
    and compute the approximation u_k * s_k^p * vt_k.

5) Given two sparse matrices, learn a PLSR model. Use the learnt PLSR model 
    to predict the distributions for a given matrix.
    
Danushka Bollegala
"""


import numpy as np
import scipy.sparse as sp
from scipy.io import mmread, mmwrite

import getopt, sys, time, pickle

from sparsesvd import sparsesvd
from svmlight_loader import load_svmlight_file, dump_svmlight_file
from sklearn.cross_decomposition import PLSRegression

DTYPE = np.float64

def generateTestData():
    """
    Generate a sparse random matrix and save it to a file.
    We will test this code using this matrix.
    """
    n = 1000
    m = 1000
    # generate a random matrix of (n,m) dimensions.
    M = np.random.rand(n, m)
    # sparsify the matrix.
    for i in range(0, n):
        for j in range(0, m):
            if M[i,j] < 0.1:
                M[i,j] = 0
    # write M to a text file in libsvm format.
    zeroCount = 0
    F = open("../work/testMatrix", "w")
    for i in range(0,n):
        F.write("%d " % (i + 1))
        for j in range(0, m):
            if M[i,j] != 0:
                F.write("%d:%f " % (j + 1, M[i,j]))
            else:
                zeroCount += 1
        F.write("\n")
    F.close()
    print "No. of zero elements =", zeroCount
    pass   


def loadSVDLIBCmatrix(fname):
    """
    Returns ndarray from the matrix in fname.
    """
    F = open(fname)
    p = F.readline().split()
    n = int(p[0])
    m = int(p[1])
    A = sp.lil_matrix((n,m), dtype=np.float64)
    line = F.readline()
    i = 0
    while line:
        p = line.strip().split()
        for j in range(0,m):
            A[i,j] = float(p[j])
        i += 1
        line = F.readline()
    F.close()
    return A


def loadSVDLIBCmatrixDiagonal(fname):
    """
    Load a diagonal matrix.
    """
    F = open(fname)
    n = int(F.readline())
    A = sp.lil_matrix((n,n), dtype=np.float64)
    line = F.readline()
    i = 0
    while line:
        A[i,i] =  float(line.strip())
        i += 1
        line = F.readline()
    F.close()
    return A
    

def compressSVDLIBCresult():
    """
    Perform SVD2 approximation using the result produced by SVDLIBC.
    """
    k = 1000
    sys.stdout.write("Loading matrices...")
    Ut = loadSVDLIBCmatrix("../work/ppmi_svd-Ut")
    Vt = loadSVDLIBCmatrix("../work/ppmi_svd-Vt")
    S = loadSVDLIBCmatrixDiagonal("../work/ppmi_svd-S")
    sys.stdout.write("\nDone")
    sys.stdout.write("Multiplying...")
    X = Ut.T[:,0:k].tocsr() * S[0:k,0:k].tocsr() * Vt[0:k,:].tocsr()
    mmwrite("../work/X", X)
    sys.stdout.write("Done")
    pass

def loadMatrix(matrixFileName):
    """
    Load the sparse matrix in the libsvm format from the given file.
    Returns the csr matrix and an index to row ids.
    e.g. rowids is a list of row ids that precede each line.
    """
    return load_svmlight_file(matrixFileName)


def saveMatrix(mat, rowIndex, matrixFileName, zero_based=True):
    """
    Write the matrix and the row index to external text files.
    """
    return dump_svmlight_file(mat, rowIndex, F, zero_based)
    pass


def convertPPMI_original(mat):
    """
    Compute the PPMI values for the raw co-occurrence matrix.
    PPMI values will be written to mat and it will get overwritten.
    """    
    (nrows, ncols) = mat.shape
    colTotals = np.zeros(ncols, dtype=DTYPE)
    for j in range(0, ncols):
        colTotals[j] = np.sum(mat[:,j].data)
    print colTotals
    N = np.sum(colTotals)
    for i in range(0, nrows):
        row = mat[i,:]
        rowTotal = np.sum(row.data)
        for j in row.indices:
            val = np.log((mat[i,j] * N) / (rowTotal * colTotals[j]))
            mat[i,j] = max(0, val)
    return mat


def convertPPMI(mat):
    """
     Compute the PPMI values for the raw co-occurrence matrix.
     PPMI values will be written to mat and it will get overwritten.
     """    
    (nrows, ncols) = mat.shape
    print "no. of rows =", nrows
    print "no. of cols =", ncols
    colTotals = mat.sum(axis=0)
    rowTotals = mat.sum(axis=1).T
    N = np.sum(rowTotals)
    rowMat = np.ones((nrows, ncols), dtype=np.float)
    for i in range(nrows):
        rowMat[i,:] = 0 if rowTotals[0,i] == 0 else rowMat[i,:] * (1.0 / rowTotals[0,i])
    colMat = np.ones((nrows, ncols), dtype=np.float) 
    for j in range(ncols):
        colMat[:,j] = 0 if colTotals[0,j] == 0 else (1.0 / colTotals[0,j])
    P = N * mat.toarray() * rowMat * colMat
    P = np.fmax(np.zeros((nrows,ncols), dtype=np.float64), np.log(P))
    return sp.csr_matrix(P)


def allclose(x, y, rtol=1.e-5, atol=1.e-8):
    return np.all(np.less_equal(np.abs(x-y), atol + rtol * np.abs(y)))

def process_SVD1(inputFileName, outputFileName, n, p):
    """
    Peform SVD1.
    """
    mat, rowids = loadMatrix(inputFileName)
    X = mat.tocsc()
    ut, s, vt = sparsesvd(X, n)
    A = np.dot(ut.T, np.diag(s ** p))
    saveMatrix(A, rowids, outputFileName)
    mmwrite("%s.ut" % inputFileName, ut)
    np.savetxt("%s.s" % inputFileName, s)
    mmwrite("%s.vt" % inputFileName, vt)
    pass


def process_SVD2(inputFileName, outputFileName, n, p, showError):
    """
    Perform SVD2.
    """
    mat, rowids = loadMatrix(inputFileName)
    X = mat.tocsc()
    ut, s, vt = sparsesvd(X, n)
    A = np.dot(np.dot(ut.T, np.diag(s ** p)), vt)
    saveMatrix(A, rowids, outputFileName)
    mmwrite("%s.ut" % inputFileName, ut)
    np.savetxt("%s.s" % inputFileName, s)
    mmwrite("%s.vt" % inputFileName, vt) 
    
    if showError:
        Xnorm = np.linalg.norm(X.todense(), ord='fro')
        Error = np.linalg.norm((X - A), ord='fro')
        rate = (100 * Error) / Xnorm
        print "Approximation Error Percentage = %f%%" % rate
        print "Frobenius norm of the original matrix =", Xnorm
        print "Frobenius norm of the error matrix =", Error            
    pass


def train_PLSR(x_filename, y_filename, model_filename, n):
    """
    Train a PLSR model and save it to the model_filename.
    X and Y matrices are read from x_filename and y_filename.
    The no. of PLSR components is given by n. 
    """
    X = loadMatrix(x_filename)[0].todense()
    Y = loadMatrix(y_filename)[0].todense()
    if X.shape[0] != Y.shape[0]:
        sys.stderr.write("X and Y must have equal number of rows!\n")
        raise ValueError
    sys.stderr.write("Learning PLSR...")
    startTime = time.time()
    pls2 = PLSRegression(copy=True, max_iter=10000, n_components=n, scale=True, tol=1e-06)
    pls2.fit(X, Y)  
    model = open(model_filename, 'w') 
    pickle.dump(pls2, model, 1)
    model.close()
    endTime = time.time()
    sys.stderr.write(" took %ss\n" % str(round(endTime-startTime, 2)))  
    pass


def predict_PLSR(x_filename, y_filename, model_filename, showError):
    """
    Read the PLSR model from the model_fname and read the X matrix from 
    x_filename. Write the predicted output to the y_filename.
    """
    sys.stderr.write("Predicting PLSR...")
    startTime = time.time()
    X = loadMatrix(x_filename)[0].todense()
    model = open(model_filename)
    pls2 = pickle.load(model)
    model.close()
    Y = pls2.predict(X)
    n = X.shape[0]
    dump_svmlight_file(X, np.arange(1, n+1), y_filename, zero_based=True)
    endTime = time.time()
    sys.stderr.write(" took %ss\n" % str(round(endTime-startTime, 2)))

    if showError:
        Xnorm = np.linalg.norm(X, ord='fro')
        Error = np.linalg.norm((X - Y), ord='fro')
        rate = (100 * Error) / Xnorm
        print "Approximation Error Percentage = %f%%" % rate
        print "Frobenius norm of the original matrix =", Xnorm
        print "Frobenius norm of the error matrix =", Error  
    pass
    

def usage():
    """
    Display help.
    """
    print """Usage: python svdmi.py -m SVD1 | SVD2 | PMI | PLSR.train | PLSR.pred [SVD1 = U * S^p, SVD2 = U * S^p * V\T]
                                    -i input_file_name or model_file_name for PLSR
                                    -o output_file_name
                                    -n no. of dimensions for SVD or PLSR (int)
                                    -p power of the diagonal matrix
                                    -x x matrix file name for PLSR.
                                    -y y matrix file name for PLSR.
                                    -v if this option is set, we will print approximation error.

        In the case of SVD mode we will write three addional files
        input_file_name.ut ut matrix
        input_file_name.vt vt matrix
        input_file_name.s s array of eigenvalues
        """
    pass


def commandLine():
    """
    Check the command line arguments and perform PPMI and SVD.
    """
    try:
        opts,args = getopt.getopt(sys.argv[1:],
                                  "m:i:o:n:p:x:y:hv",
                                  ["mode=", "input=", "output=", "num=", "pow=", "xfile=",
                                  "yfile", "help", "error"])
    except getopt.GetoptError, err:
        print err
        usage()
        sys.exit(1)
    
    mode = inputFileName = outputFileName = n = p = x_filename = y_filename = None
    showError = False
    
    for opt, val in opts:
        if opt == "-h":
            usage()
            sys.exit(1)
        elif opt == "-m":
            mode = val.lower()
        elif opt == "-i":
            inputFileName = val
        elif opt == "-o":
            outputFileName = val
        elif opt == "-n":
            n = int(val)
        elif opt == "-p":
            p = float(val)
        elif opt == "-x":
            x_filename = val
        elif opt == "-y":
            y_filename = val
        if opt == "-v":
            showError = True
    
    if mode == "pmi" and inputFileName and outputFileName:
        mat, rowids = loadMatrix(inputFileName)
        mat = convertPPMI(mat)
        saveMatrix(mat, rowids, outputFileName)
        
    elif mode == "svd1" and inputFileName and outputFileName and n and p:
        process_SVD1(inputFileName, outputFileName, n, p)
        
    elif mode == "svd2" and inputFileName and outputFileName and n and p:
        process_SVD1(inputFileName, outputFileName, n, p, showError)    

    elif mode == "plsr.train" and x_filename and y_filename and n and inputFileName:
        train_PLSR(x_filename, y_filename, inputFileName, n)

    elif mode == "plsr.pred" and x_filename and y_filename and inputFileName:
        predict_PLSR(x_filename, y_filename, inputFileName, showError)

    else:
        usage()
        sys.exit(1)        
    
    pass


def debug():
    """
    Test the various functions implemented in this module.
    """
    mat, rowids = loadMatrix("../work/testMatrix")
    #convertPPMI(mat)
    #saveMatrix(mat, rowids, "../work/pmiMatrix")
    X = mat.tocsc()
    ut, s, vt = sparsesvd(X, 50)
    #print allclose(X, np.dot(ut.T, np.dot(np.diag(s), vt)))    
    A = np.dot(ut.T, np.diag(s))
    saveMatrix(A, rowids, "../work/featMatrix")
    pass

def generate_random_matrix_PLSR():
    """
    Generate random X and Y matrices to test PLSR.
    """
    n = 1000
    m = 500
    X = np.random.randn(n, m)
    dump_svmlight_file(X, np.arange(1, n+1), "../work/Y")
    pass


if __name__ == "__main__":
    commandLine()
    #generateTestData()
    #generate_random_matrix_PLSR()
    pass    
