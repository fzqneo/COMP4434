# MLlib Basics (in Spark/Scala)

## Support Vector Machine (SVM) 

Just like Logistic Regression, (linear) SVM provdes a model to do binary classification, i.e., for each sample, it classify it as either positive (1) or negative (0).

### Exercise

The class `SVMWithSGD` trains an SVM with stochastic gradient descent (SGD). Its usage is very much similar to `LogisticRegressionWithSGD`. Try to modify the code of last lab to build an SVM from the dataset in `data/mllib/sample_libsvm_data.txt`. See https://spark.apache.org/docs/latest/mllib-linear-methods.html#linear-support-vector-machines-svms for hints.

## Input Transformation and Manipulation

#### Standard Scale
If two features are at very different scales, they may have imbalanced influence on the machine learning model. For example, a feature x<sub>1</sub> may range from 0.0001 to 0.001 and another feature x<sub>2</sub> may range from 1 billion to 10 billion. To improve training efficiency, people sometimes transform/normalize/scale the features. A typical approach is to scale the features to have a mean of 0 and a variance of 1, such that each feature falls in roughly the same scale.

The class `StandardScaler` in **MLlib** provides convenient facilities to do that:

```scala
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils

val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")

val scaler1 = new StandardScaler().fit(data.map(x => x.features))
val scaler2 = new StandardScaler(withMean = true, withStd = true).fit(data.map(x => x.features))

// data1 will be unit variance.
val data1 = data.map(x => (x.label, scaler1.transform(x.features)))

// Without converting the features into dense vectors, transformation with zero mean will raise
// exception on sparse vector.
// data2 will be unit variance and zero mean.
val data2 = data.map(x => (x.label, scaler2.transform(Vectors.dense(x.features.toArray))))
```
Reference: https://spark.apache.org/docs/latest/mllib-feature-extraction.html#standardscaler

Explanation:

+ The class `StandardScaler` provides the facility to scale the features with mean and variance optionally.
+ Before it can scale a feature, it must first scan through all training data to calculate the mean and variance, and store them in internal states. The `fit()` method does this job.
+ The `transform()` method scales a feature vector according to the values calculated in last step.

### Adding Higher Degree Terms
Sometimes, we can fit non-linear data with a linear model by explicitly adding higher degree terms of existing features.

For example, assuming the original data file contains two features (x<sub>1</sub>, x<sub>2</sub>), we can extend it by adding square terms: (x<sub>1</sub>, x<sub>2</sub>, x<sub>1</sub><sup>2</sup>, x<sub>2</sub><sup>2</sup>, x<sub>1</sub>x<sub>2</sub>)

```scala
	val extendedData = parsedData.map { point =>
	  val label = point.label
	  val original = point.features.toArray
	  val squared = original.map(x => x*x)
	  val extendedFeatures = Vectors.dense(original ++ squared)
	  LabeledPoint(label, extendedFeatures)
	}
```
	
