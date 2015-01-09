# MLlib Basics (in Spark/Scala)

## Linear Regression and Mean Squared Error

### Example

Reference: https://spark.apache.org/docs/latest/mllib-linear-methods.html#linear-least-squares-lasso-and-ridge-regression

Step 1. Download `MLexample.zip` from blackboard, unzip and import it into scala-elipse

Step 2. Study the code in `src/example/LinearReg.scala`

```scala
    // Load and parse the data
	val data = sc.textFile("data/mllib/ridge-data/lpsa.data")
	val parsedData = data.map { line =>
	  val parts = line.split(',')
	  LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
	}.cache()
	
	// Building the model
	val numIterations = 100
	val model = LinearRegressionWithSGD.train(parsedData, numIterations)
	
	// Evaluate model on training examples and compute training error
	val valuesAndPreds = parsedData.map { point =>
	  val prediction = model.predict(point.features)
	  (point.label, prediction)
	}
	val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
	println("training Mean Squared Error = " + MSE)
  }
  ```
  Explanation:
  + The spark installation comes along with some sample data of machine learning under `data/mllib`
  + The class `LinearRegressionWithSGD` provides a linear regression model *without* regularization.
  + It trains the model using **stochastic gradient descent (SGD)**.
  + The singleton object `LinearRegressionWithSGD` (same name) provides a convenience way to obtain a trained model using default settings.
  + If you want to train a model with custom settings, you need to create a new object with `new LinearRegressionWithSGD()`. See https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.mllib.regression.LinearRegressionWithSGD 
  + The **mean squared error (MSE)** measures how well the model fits the training data. A smaller MSE means the model fits the training data better.
 
Step 3. Export the project as a `jar` file. (c.f. previous Spark lab)

Step 4. Copy the jar file to the shared folder of the virtual machine.

Step 5. In the virtual machine, submit the job to **Spark**. (Assume you already have Spark started.) Note that you need to specify the `--class` that contains the `main` function. You may also need administrator privilege to access the shared folder via `sudo`.

```bash
bigdata@bigdata-VirtualBox:~$ cd Programs/spark-1.2.0-bin-hadoop1/
bigdata@bigdata-VirtualBox:~/Programs/spark-1.2.0-bin-hadoop1$ sudo bin/spark-submit --class "example.LinearReg" --master spark://localhost:7077 /media/sf_vmshared/MLexample.jar
[sudo] password for bigdata: 
Spark assembly has been built with Hive, including Datanucleus jars on classpath
training Mean Squared Error = 6.206807793307759
```

### Exercise
1. Try to increase the number of training iterations. Observe how it would affect the MSE on training data.
2. *MSE on training data* is actually not a good measure of the quality of the trained model. Why?
3. The class `LinearRegressionWithSGD` uses no regularization, which may lead to the problem of \_\_\_\_\_\_\_\_\_\_\_. 
MLlib also provides linear regression models that use L1 or L2 regularization. Read the short section at the following link and try to modify the example code to train a model with L2 regularization.
https://spark.apache.org/docs/latest/mllib-linear-methods.html#linear-least-squares-lasso-and-ridge-regression