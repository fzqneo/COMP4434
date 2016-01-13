# MLlib Basics (in Spark/Scala)

## Support Vector Machine (SVM) 

Just like Logistic Regression, (linear) SVM provdes a model to do binary classification, i.e., for each sample, it classify it as either positive (1) or negative (0).

### Exercise

The class `SVMWithSGD` trains an SVM with stochastic gradient descent (SGD). Its usage is very much similar to `LogisticRegressionWithSGD`. Try to modify the code of last lab to build an SVM from the dataset in `data/mllib/sample_libsvm_data.txt`. See https://spark.apache.org/docs/latest/mllib-linear-methods.html#linear-support-vector-machines-svms for hints.

Hint: the minimal effort requires only a few search and replace.

## Input Transformation

#### Standard Scaler

The source code in this section is in the file **InputScaler.scala** under package **example**.

If two features are at very different scales, they may have imbalanced influence on the machine learning model. For example, a feature *x1* may range from 0.0001 to 0.001 and another feature *x2* may range from 1 billion to 10 billion. To improve training efficiency, people sometimes scale the features. A typical approach is to scale all features to have a mean of 0 and a standard deviation of 1, such that each feature falls in roughly the same scale.

The class `StandardScaler` in **MLlib** provides convenient facilities to do that:

```scala
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils

	// Load and parse data in LIBSVM format.
	val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
	
	/**
	  * Set up a StandardScaler called scaler1.
	  * This scaler will scale the features against the standard deviation 
	  * but not the mean (default behavior).
	  * The fit() function scans all data once, 
	  * calculates the global information (mean and stddev),
	  * and stores them in internal states.
	  */
	val scaler1 = new StandardScaler().fit(data.map(labeledPoint => labeledPoint.features))
	
	/**
	  * Set up a StandardScaler called scaler2.
	  * This scaler will scale the features against both the standard deviation and the mean.
	  */
	val scaler2 = new StandardScaler(withMean = true, withStd = true).fit(data.map(labeledPoint => labeledPoint.features))
	
	// Use scaler1 to scale features of data
	// and store the result in data1.
	// The transform() function turns an unscaled feature vector into a scaled one.
	// data1's features will have unit variance. (stddev = 1)
	val data1 = data.map(labeledPoint => new LabeledPoint(labeledPoint.label, scaler1.transform(labeledPoint.features)))
	
	// Use scaler2 to scale features of data
	// and store the result in data2.
	// Without converting the features into dense vectors, transformation with zero mean will raise
	// exception on sparse vector.
	// data2's features will have unit variance and zero mean.
	val data2 = data.map(labeledPoint => LabeledPoint(labeledPoint.label, scaler2.transform(Vectors.dense(labeledPoint.features.toArray))))
			
```
Reference: https://spark.apache.org/docs/latest/mllib-feature-extraction.html#standardscaler

Explanation:
+ First, we parse the data file. Remember: `data` is an `RDD[LabeledPoint]`.
+ Before a scaler can scale a feature, it must first scan through **all** training data to calculate the global means and variances. If the feature vectors have *d* dimensions, it would compute *d* means and *d* variances. The `fit()` function does this job.
+ The means and variances of features **have nothing to do with the labels**. So the `StandardScaler.fit()` function only needs the feature vectors. We need to extract the feature vectors from `data` using `map()`.
+ Here, `data1` and `data2` are both  of type \_\_\_\_\_\_\_\_\_\_. 
+ Note how we use the `map()` function to scale a dataset: for each labeled point in the original `data`, we keep its label unchanged, but use a transformed (scaled) version of its features.
+ We can then create ML models based on `data1` or `data2`. Note that **a model trained on data1 cannot be used on data2**, because their features are scaled differently.

### Exercise
Create and train an SVM model `model` based on the original unscaled `data`. Create and train an SVM model `model1` based on `data1` and another model `model2` based on `data2`. Then use the **Area under ROC** metrics to compare the quality of these three models.
