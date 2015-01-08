# MLlib Basics (in Spark/Scala)

## Logistic Regression

Logsitic Regression is a model that learns **binary classification**. That is, for each sample, it tries to classify it as positive (1) or negative (0).

### Example

Reference: https://spark.apache.org/docs/latest/mllib-linear-methods.html#logistic-regression

Step 1. Download `MLexample.zip` from blackboard, unzip and import it into scala-elipse.

Step 2. Study the code in `src/example/LogisticReg.scala`

```scala
    // Load training data in LIBSVM format.
    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
	
	// Split data into training (60%) and test (40%).
	val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
	val training = splits(0).cache()
	val test = splits(1)
	
	// Run training algorithm to build the model
	val numIterations = 100
	val model = LogisticRegressionWithSGD.train(training, numIterations)

	// Compute raw scores on the test set. 
	val scoreAndLabels = test.map { point =>
	  val score = model.predict(point.features)
	  (score, point.label)
	}
	
	// Get evaluation metrics.
	val metrics = new BinaryClassificationMetrics(scoreAndLabels)
	val auROC = metrics.areaUnderROC()
	
	println("Area under ROC = " + auROC)
  }
  ```
  
  Explanation:
  
  + ddf


