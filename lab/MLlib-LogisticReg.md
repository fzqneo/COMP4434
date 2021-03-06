# MLlib Basics (in Spark/Scala)

## Logistic Regression and Split of Dataset

Logsitic Regression is a model that learns **binary classification**. That is, for each point, it tries to classify it as either positive (1) or negative (0).

### Example

Reference: https://spark.apache.org/docs/latest/mllib-linear-methods.html#logistic-regression

Step 1. Download [MLexample.zip](../5-MLlib/MLexample.zip). The source code used in this section is in the file **LogisticReg.scala** under package **example**.

Step 2. Study the code in `src/example/LogisticReg.scala`

```scala
    // Load and parse training data in LIBSVM format.
    // Remember what's the type of 'data' below?
	val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
	
	// Split data into training set (70%) and test set (30%).
	val splits = data.randomSplit(Array(0.7, 0.3), seed = System.currentTimeMillis)
	val trainingSet = splits(0).cache()
	val testSet = splits(1)
	
	// Train the model
	val numIterations = 100
	/** 
	  * Similarly to LinearRegressionModel,
	  * here LogisticRegressionModel is a built-in object with default settings.
	  * It provides a train() function that returns a trained LogisticRegressionModel model.
	  */
	val trainedModel = LogisticRegressionWithSGD.train(trainingSet, numIterations)
	
	// Collect actual and predicted labels on the test set 
	val actualAndPredictedLabels = testSet.map { labeledPoint =>
	  // Similarly to LinearRegressionModel,
	  // the LogisticRegressionModel provides a predict() function
	  // that receives a feature vector and outputs a predicted label.
	  val prediction = trainedModel.predict(labeledPoint.features)
	  (prediction, labeledPoint.label)
	}
	
	/*
	 *  BinaryClassificationMetrics is a class hat helps you
	 *  calculate some quality measurements for a binary classifier.
	 *  Here we use the "area under ROC curve" measurement.
	 */
	val metrics = new BinaryClassificationMetrics(actualAndPredictedLabels)
	val auROC = metrics.areaUnderROC()
	
	println("Area under ROC = " + auROC)  
  ```
  
  Explanation:
  
  + As we have learnt, the input file is in the `LIBSVM` format. The utility object `MLUtils` helps to parse LIBSVM files as input and return an **RDD** of `LabeledPoint`.
  + The second code segment splits the input data into two sets: a training set (70%) and a test set (20%). We should train the model with the training set and evaulate the quality of that model with the test set. (**Why?**)
  + **Area under ROC curve** is a metric that measures *how well the model fits the given data* . Generally it is a value between 0.5 and 1. A larger area under ROC means the model fits the given data more closely. For deeper understanding, see http://en.wikipedia.org/wiki/Receiver_operating_characteristic

Step 3. Export the project to a `jar` file.

Step 4. Start Spark with `start-all.sh`

Step 5. Submit the job to Spark. Note that you need to specify the `--class` that contains the main function.

```scala
bigdata@bigdata-VirtualBox:~/Programs/spark$ bin/spark-submit --class "example.LogisticReg" --master spark://localhost:7077 ~/MLexample.jar
...
Area under ROC = 0.98        
```

## Adding Polynomial Terms
The source code in this section is in the file **AddPolynomial.scala** under package **example**.

Sometimes, we can fit non-linear data with a linear model by explicitly adding polynomial terms of existing features. Doing so may helps us learn more complicated hypothesis.

For example, assuming the original data contains only two features *(x1, x2)*, we can extend it by adding 2-degree terms:
*(x1, x2, x1\*x1, x2\*x2, x1\*x2)*.

```scala
    // Let's cook a toy example:
    // A two dimensional dataset with only two samples
    val point1 = new LabeledPoint(1.0, Vectors.dense(2.0, 3.0))
    val point2 = new LabeledPoint(0.0, Vectors.dense(40.0, 15.0))
    val data = sc.parallelize(Array(point1, point2))
    
    // Let's see it
    data.collect.foreach {point =>
      println(point.label + "," + point.features.toArray.mkString(" "))
    }
    println("-"*50)
    
    // Prepare a function that receives a Vector and returns a new Vector    
    def addTerms (inVec: Vector) = {
      val x1 = inVec.toArray(0)
      val x2 = inVec.toArray(1)
      Vectors.dense(x1, x2, x1*x1, x2*x2, x1*x2)
    }
    
    // Add polynomial terms to each point in the dataset
    val extendedData = data.map { labeledPoint =>
      new LabeledPoint(labeledPoint.label, addTerms(labeledPoint.features))
    }
    
    // Let's see it
    extendedData.collect.foreach {point =>
      println(point.label + "," + point.features.toArray.mkString(" "))
    }
```
Notes:

+ Recall how to define and call a *function* in Scala. (Recall previous lab on Scala).
+ Pay attention to the usage of `map()` above.

### Exercise
The default settings of `LogisticRegressionWithSGD` do not add the **intercept**, which may cause \_\_\_\_\_\_\_\_\_\_\_\_\_\_. Try to modify the code above to add the intercept to the model. Observe how it would improve the quality of the model. 
1. When you want to train a model with non-default settings, you need to explicitly create a new model object like `new LogisticRegressionWithSGD()`. Refer to: https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.mllib.classification.LogisticRegressionWithSGD
2. Alternatively, you can add the intercept '1' to the features manually by yourself, in a similar way to the AddPolynomial example above --- this is **not** the standard way of doing it, but you can practise your Scala skills.