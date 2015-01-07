# MLlib Basics

## Data Types
In **MLlib**, each trainning sample is also called a **labeled point**. The class `LabeledPoint` have two members:
1. `label`: The label of this point. In the case of binary classification, it is either 1 or 0.
2. `features`: The feature vector.

See the class documentation of `LabeledPoint` at https://spark.apache.org/docs/latest/api/scala/ndex.html#org.apache.spark.mllib.regression.LabeledPoint


```bash
$ head -1 sample_libsvm_data.txt
$ ls
```

