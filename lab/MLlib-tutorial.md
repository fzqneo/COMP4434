# MLlib Basics (in Scala)

## Data Types

### Vector

In **MLlib**, a feature vector is represented as *a Vector of Doubles*. `Vector` is a class.

A vector can be represented in two forms: **dense** and **sparse**. We can use the utility class `Vectors` to create the vectors we need. (Note the difference between `Vector` and `Vectors`.)

```scala
import org.apache.spark.mllib.linalg.{Vector, Vectors}

// Create a dense vector (1.0, 0.0, 3.0).
val dv: Vector = Vectors.dense(1.0, 0.0, 3.0)
// Create a sparse vector (1.0, 0.0, 3.0) by specifying its indices and values corresponding to nonzero entries.
val sv1: Vector = Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0))
// Create a sparse vector (1.0, 0.0, 3.0) by specifying its nonzero entries.
val sv2: Vector = Vectors.sparse(3, Seq((0, 1.0), (2, 3.0)))
```

See the documentation of `Vectors` at https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.mllib.linalg.Vectors$

### LabeledPoint
In **MLlib**, each trainning sample is also called a **labeled point**. The class `LabeledPoint` have two members:
1. `label: Double`: The label of this point. In the case of binary classification, it is either 1 or 0. In multi-class classification, it is 0, 1, 2, 3, ... In regression problems, it is the actual output from the hypothesis.
2. `features: Vector`: The feature vector.

See the class documentation of `LabeledPoint` at https://spark.apache.org/docs/latest/api/scala/ndex.html#org.apache.spark.mllib.regression.LabeledPoint

The `LIBSVM` file format stores a collection of labeled points (or say, a training set) in the sparse form:

`label index1:value1 index2:value2 ...`

For example, let's check out the first line of `sample_libsvm_data.txt`:

```bash
bigdata@bigdata-VirtualBox:~$ cd Programs/spark-1.2.0-bin-hadoop1/
bigdata@bigdata-VirtualBox:~/Programs/spark-1.2.0-bin-hadoop1$ head -1 data/mllib/sample_libsvm_data.txt 
0 128:51 129:159 130:253 131:159 132:50 155:48 156:238 157:252 158:252 159:252 160:237 182:54 183:227 184:253 185:252 186:239 187:233 188:252 189:57 190:6 208:10 209:60 210:224 211:252 212:253 213:252 214:202 215:84 216:252 217:253 218:122 236:163 237:252 ...
```

The `MLUtils` utility object helps us to load the training set from a `LIBSVM` file into an object of type `RDD[LabeledPoint]`.

```scala
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

val examples: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
```

We can also load the samples from a dense-formed text file.

```bash
bigdata@bigdata-VirtualBox:~/Programs/spark-1.2.0-bin-hadoop1$ head -2 data/mllib/ridge-data/lpsa.data 
-0.4307829,-1.63735562648104 -2.00621178480549 -1.86242597251066 -1.02470580167082 -0.522940888712441 -0.863171185425945 -1.04215728919298 -0.864466507337306
-0.1625189,-1.98898046126935 -0.722008756122123 -0.787896192088153 -1.02470580167082 -0.522940888712441 -0.863171185425945 -1.04215728919298 -0.864466507337306
```

```scala
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

val data = sc.textFile("data/mllib/ridge-data/lpsa.data")
val parsedData = data.map { line =>
  val parts = line.split(',')
  LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
}.cache()
```

Extract only the features of the dataset (remove the labels):

```scala
val onlyFeatures: RDD[Vector] = parsedData.map(x => x.features)
```
