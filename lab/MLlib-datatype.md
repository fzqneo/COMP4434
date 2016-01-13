# MLlib Basics (in Spark/Scala)

## Data Types

Download [MLexample.zip](MLexample.zip) and import it into Eclipse. You need to copy the `spark-assembly-1.6.0-hadoop2.6.0.jar` file into the `lib` folder under the project. Create the folder if it doesn't exist yet. The source code used in this lab is in the file **DataType.scala** under package **example**.


### Vector

Remember what is a **feature vector** in machine learning?
In **MLlib**, a feature vector is represented as a `Vector` object, which is **a vector of Double's**.

*NOTE: Scala language has its native `Vector` class. But **this Vector is not that Vector**. When doing machine learning, we need to use the `Vector` class provided by MLlib. So you must explicitly import `org.apache.spark.mllib.linalg.Vector`.*

A vector can be represented in two forms: **dense** and **sparse**. We can use the factory class `Vectors` to create the vectors we need. (Note the **s** in `Vectors` here.)

```scala
import org.apache.spark.mllib.linalg.{Vector, Vectors}

    /* 
     * Use the Vectors factory to create a vector.
     * Let's create the vector (56.0, 0.0, 78.0, 0.0, 0.0) in three different ways.
     */
    
    /* Way 1: create a dense vector. Specify the value of each element. */
	val dv: Vector = Vectors.dense(56.0, 0.0, 78.0, 0.0, 0.0)
	
	/* 
	 * Way 2: create a sparse vector.
	 * 5 is the length of the vector.
	 * Array(0,2) is the indexes of non-zero elements
	 * Array(56.0, 78.0) is the values of non-zero elements
	 */
	val sv1: Vector = Vectors.sparse(5, Array(0, 2), Array(56.0, 78.0))
	
	/*
	 * Way 3: create a sparse vector (alternative)
	 * 5 is the length of the vector.
	 * Seq() contains a sequence of (index, value) pair of non-zero elements.
	 */
	val sv2: Vector = Vectors.sparse(5, Seq((0, 56.0), (2, 78.0)))
```
Explanation:
1. `val dv: Vector` declares a value `dv` of type `Vector`. Unlike in C/C++ and Java, in Scala, the **type declaration** is placed **after** the variable and can often be omitted, if Scala can infer it automatically. In the above example, you can also write `val dv = Vectors.dense(...)` by dropping the `:Vector` type declaration. Ditto for `sv1` and `sv2`.

See the documentation of `Vectors` at https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.mllib.linalg.Vectors$

### Exercise
Try to create the following vector in the three differnt ways described above:
```scala
// (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 3.0, 0.0)
```
What's the pros and cons of the different ways? Which one do you prefer?

### LabeledPoint
Remember what is a **sample** in machine learning?
In **MLlib**, each sample is also called a **labeled point**. The class `LabeledPoint` has two members:
1. `label: Double`: The label of this point. In the case of binary classification, it is either 1 or 0. In multi-class classification, it is natual numbers 0, 1, 2, 3, ... In regression problems, it is usually a real number.
2. `features: Vector`: The feature vector. Each feature is a Double.

For example, you can create a labeled point with label "1" (a positive point) and a feature vector (0.2, 3, -0.1):  
```scala
val point1:LabeledPoint = new LabeledPoint(1.0, Vectors.dense(0.23, 31.0, -0.89))
```
Question:  
1. `point1.label` is \_\_\_\_\_\_\_\_\_\_
2. `point1.features` is \_\_\_\_\_\_\_\_\_\_

See the class documentation of `LabeledPoint` at https://spark.apache.org/docs/latest/api/scala/ndex.html#org.apache.spark.mllib.regression.LabeledPoint

## Loading Data Files

### LIBSVM File

`LIBSVM` is a file format. It stores a collection of labeled points (or say, a training set). **Each line represents a labeled point**. It starts with its label followed by its feature vector **in sparse form**.

`label index1:value1 index2:value2 ...`

For example, let's look at the first line of the file `sample_libsvm_data.txt` under direcotory *~/Programs/spark/data/mllib*:

```bash
bigdata@bigdata-VirtualBox:~/Programs/spark$ head -1 data/mllib/sample_libsvm_data.txt 
0 128:51 129:159 130:253 131:159 132:50 155:48 156:238  ...
```

We can see the first training sample has a label '0'. And the data set has at least several hundred features. The 128-th feature has value 51, 
the 130-th feature has value 253, etc.

The `MLUtils` utility class helps us to load the training set from a `LIBSVM` file into an `RDD[LabeledPoint]`. At this moment, we just think `RDD` as a collection type.

```scala
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

val examples: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
```

### Other formats
We can also load the samples from a dense-formed text file. Let's look at the first 2 lines of file `data/mllib/ridge-data/lpsa.data `:  
```bash
bigdata@bigdata-VirtualBox:~/Programs/spark$ head -2 data/mllib/ridge-data/lpsa.data 
-0.4307829,-1.63735562648104 -2.00621178480549 -1.86242597251066 -1.02470580167082 -0.522940888712441 -0.863171185425945 -1.04215728919298 -0.864466507337306
-0.1625189,-1.98898046126935 -0.722008756122123 -0.787896192088153 -1.02470580167082 -0.522940888712441 -0.863171185425945 -1.04215728919298 -0.864466507337306
```
Again, each line is a labeled point. Each line is in the format:  
``label,feature1 feature2 feature3 ...``

For example, the first point has label "-0.4307829", and has 8 real-valued features.

We can parse this file into an `RDD[LabeledPoint]`  as follows:
```scala
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

	// sc.textFile() parses the files into an RDD of strings
	val data: RDD[String] = sc.textFile("data/mllib/ridge-data/lpsa.data")
	// Pay attention to the usage of .map() here.
	val parsedData: RDD[LabeledPoint] = data.map { line =>
		val parts = line.split(',')
		LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
	}.cache()
```
Explanation:  
1. `sc.textFile()` reads an ordinary text file and return an `RDD[String]` --- for now, we just consider it as a collection of String. Each line in the file is represented as a String.


#### The **map()** function
The `map()` function is a very important function in Scala and Spark.
You can call this function on a variety of classes.
Roughly speaking, these classes are usually a *collection* of things,
e.g., an array or an RDD.
A `map()` function transforms a collection of something to a collection of something else. 
That *something else* depends on the parameter you give to `map()`.
Specifically, the parameter you give to `map()` stipulates how an element in the input is **mapped** (transformed) to an element in the output.

In the above example, there are two uses of `map()`:
1. `data.map {...}`
2. `parts(1).split(' ').map(_.toDouble)`

In (1), remember `data` is an `RDD[String]`. So, the `data.map()` maps each line (a String) to something else.
The whole body inside **{}** is the argument passed to `map()`.
The inside of **{}** itself is a function.
It says that, for a given String called `line` (the variable name can actually be arbitrary),
it will return a `LabeledPoint` calculated from `line`.
Hence, `data.map {...}` returns an `RDD[LabeledPoint]`.

In (2), the `map()` function transforms an array of String to an array of Double.
`_.toDouble` is a shorthand for writing `x => x.toDouble()`.
Consider the following example run in the scala shell (see last lab for how to run the scala shell).
```scala
scala> val line="1,23 56 89"
line: String = 1,23 56 89

scala> val parts = line.split(',')
parts: Array[String] = Array(1, 23 56 89)

scala> parts(0)
res0: String = 1

scala> parts(1)
res1: String = 23 56 89

scala> parts(0).toDouble
res2: Double = 1.0

scala> parts(1).split(' ')
res3: Array[String] = Array(23, 56, 89)

scala> parts(1).split(' ').map(_.toDouble)
res4: Array[Double] = Array(23.0, 56.0, 89.0)
```

### Running the Examples in Spark
All of the above example code can be found in **DataType** under package **example**. You can submit it to Spark for execution (also refer to previous lab on Spark):

1. Export the project as  `MLExample.jar`. Save it under your home directory `~`.
2. In the virtual machine, start Spark with `start-all.sh`
3. Submit the job to Spark. Note that you need to specify the `--class` that contains the `main` function. You may also need administrator privilege to access the shared folder via `sudo`.

```bash
bigdata@bigdata-VirtualBox:~/Programs/spark$ bin/spark-submit --class "example.DataType" --master spark://localhost:7077 ~/MLExample.jar
[sudo] password for bigdata: 
```

### Excercise
Very often we want to extract only the features from the data and drop the labels. We can use the `map()` function to do it.  
Fill in the ???'s in below:  
```scala
val onlyFeatures: ??? = parsedData.map({point => ???????})
```
