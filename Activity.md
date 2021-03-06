Background of the study
-----------------------

> There is an ever increasing availability of data on physical exercises. The usual analytics are focussed on the amount of exercise which is carried out, but not so much on the effectiveness of the exercises. 


The goal of the project
-----------------------

The goal of this project is to predict in what way an exercise was being done.  The participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways (A,B,C,D and E).


The training of the model
-------------------------

In the following steps were executed in order to train the predictive model.

### Read the data

First, the `.csv` file contain the training data is read into R, in which unavailable values are set as `NA`.


```r
pmlTrain <- read.csv2("pml-training.csv", na.strings = c("NA", ""))
```

### Reduce the dataset

In the next step the proportion of missing values (`NA`s) is checked in the columns.


```r
propNAs <- colMeans(is.na(pmlTrain))
table(propNAs)
```

```
## propNAs
##                 0 0.979308938946081 
##                60               100
```

There are 100 columns in which almost all values (97.93%) are missing. If a column contains a large number of `NA`s, it will be removed. Only the columns without any `NA`s will be kept.


```r
# index of columns with NA values
idx <- !propNAs
# check
sum(idx)
```

```
## [1] 60
```

```r
# remove these columns  
pmlTrainReduced <- pmlTrain[idx]
# check
ncol(pmlTrainReduced)
```

```
## [1] 60
```

There are further unnecessary columns that can be removed. The column `ï..Column1` contains the row numbers. The column `user_name` contains the name of the user. Both variables cannot be predictors for the type of exercise.

Furthermore, the three columns containing time stamps (`raw_timestamp_part_1`, `raw_timestamp_part_2`, and `cvtd_timestamp`) will not be used.

The factors `new_window` and `num_window` are not related to sensor data. They will be removed too.


```r
# find columns not containing sensor measurement data
idx <- grep("^ï..Column1$|user_name|timestamp|window", names(pmlTrainReduced))
# check
length(idx)
```

```
## [1] 7
```

```r
# remove columns
pmlTrainReduced2 <- pmlTrainReduced[-idx]
```


### Preparing the data for training

Now, the dataset contains one outcome column (`classe`) and 59 feature columns. The function `createDataPartition` of the `caret` package is used to split the data into a training and a cross-validation data set. Here, 70% of the data goes into the training set.


```r
library(caret)
```

```r
inTrain <- createDataPartition(y = pmlTrainReduced2$classe, p = 0.7, list = FALSE)
```

The index `inTrain` is used to split the data.


```r
training <- pmlTrainReduced2[inTrain, ]
# the number of columns on the training set
nrow(training)
```

```
## [1] 13737
```

```r
crossval <- pmlTrainReduced2[-inTrain, ]
# the number of rows in the cross-validation set
nrow(crossval)
```

```
## [1] 5885
```


### Train a model

A *random-forest* technique generates a predictive model. In sum, 10 models were trained. 


```r
library(randomForest)
```


```r
trControl <- trainControl(method = "cv", number = 2)
modFit <- train(classe ~ ., data = training, method = "rf", prox = TRUE, trControl = trControl)
```

### Evaluate the model (out-of-sample error)

First, the final model is used to predict the outcome in the cross-validation dataset.


```r
pred <- predict(modFit, newdata = crossval)
```

Second, the function `confusionMatrix` is used to calculate the accuracy of the prediction.


```r
coMa <- confusionMatrix(pred, reference = crossval$classe)
acc <- coMa$overall["Accuracy"]
acc
```

```
##  Accuracy 
## 9916737
```

The accuracy of the prediction is 99.17%. Hence, the *out-of-sample error* is 0.83%.


### Variable importance

The five most important variables in the model and their relative importance values are:


```r
vi <- varImp(modFit)$importance
vi[head(order(unlist(vi), decreasing = TRUE), 5L), , drop = FALSE]
```

```
##                     Overall
## roll_belt         100.00000
## magnet_dumbbell_z  93.63891
## yaw_belt           84.66317
## magnet_dumbbell_y  82.31206
## roll_forearm       69.45428
```

***************************************************************************

#### The source of the data

For this study the data is used from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. It has been published:

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. [Qualitative Activity Recognition of Weight Lifting Exercises](http://groupware.les.inf.puc-rio.br/har#ixzz34irPKNuZ). *Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13)*. Stuttgart, Germany: ACM SIGCHI, 2013.
The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har.
