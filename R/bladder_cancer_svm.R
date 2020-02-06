#######load library

library("ggplot2")
library("e1071")
library("CardinalWorkflows")
library(tensorflow)
library(magrittr)
library("gtools")

########load data
load("data/bladder_cancer.rdata")

mse<-bladder_cancer

#######split into training and validation

mse_train <- mse[,mse$sample_name=="TMA1"]
mse_test <- mse[,mse$sample_name=="TMA2"]

#######set x-axis and y-axis for imaging

xaxis_train <- c(range(coord(mse_train)$x)[1]-5, range(coord(mse_train)$x)[2]+5)
yaxis_train <- c(range(coord(mse_train)$y)[1]-5, range(coord(mse_train)$y)[2]+5)
xaxis_test <- c(range(coord(mse_test)$x)[1]-5, range(coord(mse_test)$x)[2]+5)
yaxis_test <- c(range(coord(mse_test)$y)[1]-5, range(coord(mse_test)$y)[2]+5)

###############define accuracy and balanced accuracy

balance_accu<-function(prediction, truth)
{
  xx<-table(prediction,truth)
  
  return(0.5*xx[1,1]/sum(xx[,1])+0.5*xx[2,2]/sum(xx[,2]))
}




accu<-function(prediction, truth)
{
  xx<-table(prediction,truth)
  
  return((xx[1,1]+xx[2,2])/sum(xx))
}


###############SVM

print('SVM')

train_x<-t(spectra(mse_train))
train_y<-mse_train$label

test_x<-t(spectra(mse_test))
test_y<-mse_test$label

naive_svm<-svm(x=train_x,y=train_y, kernel = 'sigmoid')

label_svm<-predict(naive_svm,train_x)
mse_train$svm <- ifelse(label_svm=="Tumor", 'Cancer', 'Normal')

label_svm<-predict(naive_svm,test_x)
mse_test$svm<-ifelse(label_svm=="Tumor", 'Cancer', 'Normal')

image(mse_train,svm~x*y, xlim=xaxis_train, ylim=yaxis_train, main="SVM: Training")

image(mse_test,svm~x*y, xlim=xaxis_test, ylim=yaxis_test, main="SVM: Validation")

print("The accuracy of SVM on training data wrt label is:\\" )
print(accu(mse_train$svm, mse_train$diagnosis))
print("The balanced accuracy of SVM on training data wrt label is:\\" )
print(balance_accu(mse_train$svm, mse_train$diagnosis))

print("The accuracy of SVM on training data wrt truth is:\\" )
print(accu(mse_train$svm, mse_train$truth))
print("The balanced accuracy of SVM on training data wrt truth is:\\" )
print(balance_accu(mse_train$svm, mse_train$truth))

print("The accuracy of SVM on testing data wrt truth is:\\" )
print(accu(mse_test$svm, mse_test$truth))
print("The balanced accuracy of SVM on testing data wrt truth is:\\" )
print(balance_accu(mse_test$svm, mse_test$truth))



##########mi-SVM

print('mi-SVM')
#####split cancer and normal samples
n_samples<-NULL
c_samples<-NULL
for (s in unique(mse_train$sample))
{
  if ('Tumor'%in% mse_train$label[mse_train$sample==s])
  {
    c_samples<-c(c_samples, s)
  }else
    n_samples<-c(n_samples, s)
}

threshhold=0.002*dim(mse_train)[2]/2
train_x<-t(spectra(mse_train))
train_y<-mse_train$label

test_x<-t(spectra(mse_test))
test_y<-mse_test$label

iteration<-200
for (i in 1:iteration)
{
  mi_svm<-svm(x=train_x,y=train_y, kernel = 'sigmoid',  gamma = 0.00125, probability = T)
  label_svm<-predict(mi_svm,train_x)
  print(paste('iteration:', i, 'accuracy wrt tissue-level label:', accu(label_svm, mse_train$label)))
  print("====================================")
  if (sum(train_y[mse_train$label=="Tumor"]!=label_svm[mse_train$label=="Tumor"])<threshhold)
  {
    break;
  }
  train_y[mse_train$label=="Tumor"]<-label_svm[mse_train$label=="Tumor"]
  for (s in c_samples)
  {
    if (!('Tumor' %in% train_y[mse_train$sample==s]))
    {
      pred<-predict(mi_svm,train_x[mse_train$sample==s,],probability=T)
      predictions <- attr(pred, "probabilities")
      j=which(predictions[,'Tumor']==max(predictions[,'Tumor']))
      train_y[mse_train$sample==s][j]<-'Tumor'       
    }
  }
  
}

mse_train$mi_svm <- ifelse(label_svm=="Tumor", 'Cancer', 'Normal')

label_svm<-predict(mi_svm,test_x)
mse_test$mi_svm<-ifelse(label_svm=="Tumor", 'Cancer', 'Normal')

image(mse_train,mi_svm~x*y, xlim=xaxis_train, ylim=yaxis_train)

image(mse_test,mi_svm~x*y, xlim=xaxis_test, ylim=yaxis_test)



print("The accuracy of mi-SVM on training data wrt label is:\\" )
print(accu(mse_train$mi_svm, mse_train$diagnosis))
print("The balanced accuracy of mi-SVM on training data wrt label is:\\" )
print(balance_accu(mse_train$mi_svm, mse_train$diagnosis))

print("The accuracy of mi-SVM on training data wrt truth is:\\" )
print(accu(mse_train$mi_svm, mse_train$truth))
print("The balanced accuracy of mi-SVM on training data wrt truth is:\\" )
print(balance_accu(mse_train$mi_svm, mse_train$truth))

print("The accuracy of mi-SVM on testing data wrt truth is:\\" )
print(accu(mse_test$mi_svm, mse_test$truth))
print("The balanced accuracy of mi-SVM on testing data wrt truth is:\\" )
print(balance_accu(mse_test$mi_svm, mse_test$truth))