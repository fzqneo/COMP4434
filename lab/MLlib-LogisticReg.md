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
  
  + The input file is in the `LIBSVM` format. The utility object `MLUtils` helps to parse LIBSVM files as input.
  + The code splits the input data into two sets: a training set and a test set. We train the model with the training set and evaulate the model with the test set. (Why?)
  + Similarly to `LinearRegressionWithSGD`, `LogisticRegressionWithSGD` is the name of both a class and a singleton. The singleton `LogisticRegressionWithSGD` trains a model on input data using default settings.
  + **Area under ROC** is a metric that measures how *good* the model is. Generally it is a value between 0.5 and 1. A larger area under ROC means a better model.

Step 3. Export the project to a `jar` file.

Step 4. Copy the jar file to the shared folder of the virtual machine.

Step 5. In the virtual machine, submit the job to Spark. (Assume you already have Spark started.) Note that you need to specify the --class that contains the main function. You may also need administrator privilege to access the shared folder via sudo.

