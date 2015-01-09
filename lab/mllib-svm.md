# MLlib Basics (in Spark/Scala)

## Support Vector Machine (SVM)

Just like Logistic Regression, (linear) SVM provdes a model to do binary classification, i.e., for each sample, it classify it as either positive (1) or negative (0).

### Example

The class `SVMWithSGD` trains an SVM with stochastic gradient descent (SGD). Its usage is very much similar to `LogisticRegressionWithSGD`. Try to modify the code of last lab to build an SVM from the dataset in `data/mllib/sample_libsvm_data.txt`.

## Input Transformation and Manipulation
