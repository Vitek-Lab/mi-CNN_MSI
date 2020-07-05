#######load library

library("ggplot2")
library("e1071")
library("CardinalWorkflows")
library("gtools")

########load data and pre-process
save.plot <- F

load("data/rccTissue.rdata")

rcc<-rcc.tissue

register(SerialParam())

rcc.resample<-rcc%>%
  normalize(method = "tic")%>%
  peakBin(c(seq(from=151, to = 1000, by = 1))) %>%
  process()


mse<-rcc.resample

mse$sample <- run(mse)

mse$diagnosis <- as.factor(mse$diagnosis)

mse$sample_diag <- paste0(mse$sample,'_', mse$diagnosis)
#####split
mse_train<-mse[,mse$sample%in%unique(mse$sample)[c(1:5, 8)]]

mse_test<-mse[,mse$sample%in%unique(mse$sample)[6:7]]



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
train_y<-mse_train$diagnosis

test_x<-t(spectra(mse_test))
test_y<-mse_test$diagnosis

naive_svm<-svm(x=train_x,y=train_y)

label_svm<-predict(naive_svm,train_x)
mse_train$svm <- label_svm

label_svm<-predict(naive_svm,test_x)
mse_test$svm<-label_svm

if (save.plot == T)
{
  pdf("results/rcc_SVM.pdf")
  print(image(mse_train,svm~x*y, xlim=xaxis_train, ylim=yaxis_train, main="SVM: Training"))
  
  print(image(mse_test,svm~x*y, xlim=xaxis_test, ylim=yaxis_test, main="SVM: Validation"))
  dev.off()
}



##########mi-SVM

print('mi-SVM')
#####split cancer and normal samples
n_samples<-NULL
c_samples<-NULL
for (s in unique(mse_train$sample_diag))
{
  if ('cancer'%in% mse_train$diagnosis[mse_train$sample_diag==s])
  {
    c_samples<-c(c_samples, s)
  }else
    n_samples<-c(n_samples, s)
}

threshhold=0.002*dim(mse_train)[2]
train_x<-t(spectra(mse_train))
train_y<-mse_train$diagnosis

test_x<-t(spectra(mse_test))
test_y<-mse_test$diagnosis

iteration<-200
for (i in 1:iteration)
{
  mi_svm<-svm(x=train_x,y=train_y, probability = T)
  label_svm<-predict(mi_svm,train_x)
  print(paste('iteration:', i, 'accuracy wrt tissue-level label:', accu(label_svm, mse_train$diagnosis)))
  print("====================================")
  if (sum(train_y[mse_train$diagnosis=="cancer"]!=label_svm[mse_train$diagnosis=="cancer"])<threshhold)
  {
    break;
  }
  train_y[mse_train$diagnosis=="cancer"]<-label_svm[mse_train$diagnosis=="cancer"]
  for (s in c_samples)
  {
    if (!('cancer' %in% train_y[mse_train$sample==s]))
    {
      pred<-predict(mi_svm,train_x[mse_train$sample==s,],probability=T)
      predictions <- attr(pred, "probabilities")
      j=which(predictions[,'cancer']==max(predictions[,'cancer']))
      train_y[mse_train$sample==s][j]<-'cancer'       
    }
  }
  
}

mse_train$mi_svm <- label_svm

label_svm<-predict(mi_svm,test_x)
mse_test$mi_svm<-label_svm

if (save.plot == T)
{
  pdf(file="results/rcc_mi_SVM.pdf")
  
  print(image(mse_train,mi_svm~x*y, xlim=xaxis_train, ylim=yaxis_train,main="mi-SVM: Training"))
  
  print(image(mse_test,mi_svm~x*y, xlim=xaxis_test, ylim=yaxis_test, main="mi-SVM: Validation"))
  dev.off()
}



