# MLlib Basics

## Data Types

### Vector

In **MLlib**, a feature vector is represented as *a Vector of Doubles*. `Vector` is a class.

A vector can be represented in two forms: **dense** and **sparse**. We can use the utility class `Vectors`

###LabeledPoint
In **MLlib**, each trainning sample is also called a **labeled point**. The class `LabeledPoint` have two members:
1. `label`: The label of this point. In the case of binary classification, it is either 1 or 0.
2. `features`: The feature vector.

See the class documentation of `LabeledPoint` at https://spark.apache.org/docs/latest/api/scala/ndex.html#org.apache.spark.mllib.regression.LabeledPoint


```bash
$ head -1 sample_libsvm_data.txt
$ ls
```

