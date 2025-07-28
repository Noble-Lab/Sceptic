# Sceptic

[**Installation**](#installation)
| [**Enviroment**](#enviroment)
| [**Example**](#example)
| [**Input**](#input)
| [**Output**](#output)
| [**Parameter**](#parameter)
| [**Citation**](#citation)
| [**Contact**](#contact)

Sceptic can perform pseudotime analysis on various types of single-cell/single-nucleus data. The model takes as input a collection of single-cell/single-nucleus data and then learns the relationship between the observed data and the associated time stamps, and finally uses the trained model to assign to each cell a real-valued pseudotime. Ideally, the pseudotimes assigned by Sceptic reflect each cell's progression along a notion of time---developmental, cell cycle, disease progression, aging---that is appropriate to the given data. Ideally, the pseudotimes assigned by Sceptic reflect each cell's progression along a notion of time---developmental, cell cycle, disease progression, aging---that is appropriate to the given data.

![Sceptic schematic](https://raw.githubusercontent.com/Noble-Lab/Sceptic/main/sceptic-schematic.jpg)


## Installation<a id="installation"></a>
Sceptic software is available on the Python package index (PyPI), latest version 0.3.3. To install it using pip, simply type:
```bash
$ pip install sceptic
```

## Enviroment<a id="enviroment"></a>
Sceptic is associated with the following packages.
- python >= 3.7.7
- numpy >= 1.19.5   
- pandas >= 1.3.5
- sklearn >= 1.0.2 

## Example (python script) <a id="example"></a>
We downloaded the processed [scGEM](https://github.com/caokai1073/UnionCom/tree/master/scGEM) dataset from UnionCom’s GitHub page.

```bash
$ python test/scGEM/scGEM.py 
```
The script will generate 4 outputs from Sceptic described in the section above and save it at: test/scGEM/.

## Parameters of ```Sceptic``` <a id="parameter"></a>

The list of parameters is given below:
> + ```eFold```: # of folds for external cross-validation (default=3).
> + ```iFold```: # of folds for internal cross-validation (default=4).

For SVM implementation:
> + ```kernel```: The kernel function for sceptic SVM classfier (default=('linear', 'rbf')). Sklearn supports four kinds of [kernels](https://scikit-learn.org/stable/modules/svm.html#kernel-functions): linear, polynomial, rbf, sigmoid.  
> + ```C```: The C parameter for rbf kernel (default=[0.1, 1, 10]). The C parameter trades off correct classification of training examples against maximization of the decision function’s margin. See more details [here](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py).

For XGboost implementation:
> + ```max_depth``` : Maximum depth of a tree (default=[3, 5]). Increasing this value will make the model more complex and more likely to overfit. 0 indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree. See more details [here](https://xgboost.readthedocs.io/en/stable/parameter.html).
> + ```learning_rate``` : Step size shrinkage used in update to prevent overfitting (default=[0.1, 0.3]). After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative. See more details [here](https://xgboost.readthedocs.io/en/stable/parameter.html).

## Input<a id="input"></a>

In case the user is providing the input data:
- `data_concat`: the input cell by measurement matrix. (# of cells by # of measurements)
- `label`: processed cell time label. (# of cells by 1)
- `label_list`: unique list of possible cell time labels. (# of time points by 1)
- `parameters`: Sceptic parameter dictionary. (SVM default={'kernel': ('linear', 'rbf'), 'C': [0.1, 1, 10]}; XGboost default= {"max_depth": [3, 5], "learning_rate": [0.1, 0.3],
        "n_estimators": [100], "subsample": [0.8]})
- `method`: "svm" or "xgboost" implementation. For large dataset, we recommend "xgboost" implementation.
- `use_gpu`: Only applies if method="xgboost".


## Output<a id="output"></a>

When one uses sceptic.run_sceptic_and_evaluate function, several outputs are generated:
- `cm`: the confusion matrix for Sceptic's nested cross-validation. (# of time_points by # of time points)
- `label_predicted`: Sceptic's predicted discrete label for each cell. (# of cells by 1)
- `pseudotime`: Sceptic's predicted pseudotime (continuous) for each cell. (# of cells by 1)
- `sceptic_prob`: the class-proabilities for each cell. (# of cells by # of time points)


## Contact<a id="contact"></a>
In case you have questions, reach out to `gangliuw@uw.edu`.


## Citation<a id="citation"></a>
[Pseudotime analysis for time-series single-cell sequencing and imaging data](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03679-3)

If you have found our work useful, please consider citing us:

Li, G., Kim, HJ., Pendyala, S. et al. Sceptic: pseudotime analysis for time-series single-cell sequencing and imaging data. Genome Biol 26, 209 (2025). https://doi.org/10.1186/s13059-025-03679-3

```
@article{li2025sceptic,
  title={Sceptic: pseudotime analysis for time-series single-cell sequencing and imaging data},
  author={Li, Gang and Kim, Hyeon-Jin and Pendyala, Sriram and Zhang, Ran and Vert, Jean-Philippe and Disteche, Christine M and Deng, Xinxian and Fowler, Douglas M and Noble, William Stafford},
  journal={Genome Biology},
  volume={26},
  pages={209},
  year={2025}
}
```
