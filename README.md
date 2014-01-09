# svdmi


This Python script contains several common pre-processing in NLP such as PPMI computation, SVD-based dimensionality reduction, and PLSR-based distribution prediction.

## Dependencies

The following packages are required.

* Python 2.7 (not tested with Python 3)
* [numpy]([http://www.numpy.org/) 
* [scipy](http://scipy.org/) 
* [sparsesvd](https://pypi.python.org/pypi/sparsesvd/) 
* [sklearn](http://scikit-learn.org/stable/) 
* [svmlight-loader](https://github.com/mblondel/svmlight-loader)

## Installation

There is no specific installation for svdmi. Once you have all the dependencies installed, you can run svmi as described in the usage section.

## Usage


### PPMI
  Positive Pointwise Mutual Information
  
  ```
  $ python svdmi.py -m PPMI -i raw_co-occurrences_matrix_file_name -o ppmi_matrix_file_name
  ```

### SVD
  Singular Value Decomposition-based dimensionality reduction (SVD1) and matrix smoothing (SVD2).
  
  For SVD1 mode
  ```
  $ python svdmi.py -m SVD1 -i matrix_file_name -o dimensionality_reduced_matrix_file_name -n svd_dimensions -p power_to_raise_singular_values
  ```
   
  For SVD2 mode
  ```
  $ python svdmi.py -m SVD2 -i raw_co-occurrences_matrix_file_name -o smoothed_matrix_file_name -n svd_dimensions -p power_to_raise_singular_values
  ```
  Use -v option to print the reproduction error (Frobenious norm)

### PLSR
  Partial Least Square Regression-based distribution prediction.
  
  Training a PLSR model
  ```
  $ python svdmi.py -m  -m PLSR.train -x x_matrix_file_name -y y_matrix_file_name -n PLSR_components -i model_file_name
  ```
  Use -v option to print the reproduction error (Frobenious norm)
  
  Predicting using the trained PLSR model.
  ```
  $ python svdmi.py -m PLSR.pred -x x_matrix_file_name -y predicted_y_matrix_file_name -i model_file_name
  ```
  
  
## License
  Simple BSD
  
## Author
  Danushka Bollegala



