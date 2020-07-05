#######load library
library("ggplot2")
library("e1071")
library("CardinalWorkflows")
library(tensorflow)
library(magrittr)
library("gtools")

#####################prepare data

load("data/bladder_cancer.rdata")

mse<-bladder_cancer

save.plot <- F
#######split into training and validation

mse_train <- mse[,mse$sample_name=="TMA1"]
mse_test <- mse[,mse$sample_name=="TMA2"]

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

#######set x-axis and y-axis for imaging

xaxis_train <- c(range(coord(mse_train)$x)[1]-5, range(coord(mse_train)$x)[2]+5)
yaxis_train <- c(range(coord(mse_train)$y)[1]-5, range(coord(mse_train)$y)[2]+5)
xaxis_test <- c(range(coord(mse_test)$x)[1]-5, range(coord(mse_test)$x)[2]+5)
yaxis_test <- c(range(coord(mse_test)$y)[1]-5, range(coord(mse_test)$y)[2]+5)


source(paste0(getwd(),'/R/model.R'))

## Gradient Descent and learning rate setting
learning_rate <- 0.000008 
train_step_by_GD <- tf$train$AdamOptimizer(learning_rate)$minimize(cross_entropy)

## Session setting
rm(sess)
sess <- tf$Session()
init <- tf$global_variables_initializer()
sess$run(init)




####data put into model
train_x<-t(spectra(mse_train))
train_y<-mse_train$label

test_x<-t(spectra(mse_test))
test_y<-mse_test$label
train_y<-ifelse(train_y=="Tumor",1 ,0)
train_y<-as.numeric(train_y)
train_y<-cbind(train_y,1-train_y)
test_y<-ifelse(test_y=="Tumor",1 ,0)
test_y<-as.numeric(test_y)
test_y<-cbind(test_y,1-test_y)

######ground truth
true_train_y<-mse_train$histology
true_test_y<-mse_test$histology
true_train_y<-ifelse(true_train_y=="Tumor",1 ,0)
true_train_y<-as.numeric(true_train_y)
true_train_y<-cbind(true_train_y,1-true_train_y)
true_test_y<-ifelse(true_test_y=="Tumor",1 ,0)
true_test_y<-as.numeric(true_test_y)
true_test_y<-cbind(true_test_y,1-true_test_y)


#############start train
threshhold=0.002*dim(mse_train)[2]/2
print("training start:==============")
##############Training
for (n in 1:200)
{
  # Running
  for (i in 1:1000){
    batch_seq <- round(30) %>% sample(seq_len(nrow(train_x)), size = .) 
    batches_xs <- train_x[batch_seq,]
    batches_ys <- train_y[batch_seq,]
    sess$run(train_step_by_GD, feed_dict = dict(xs = batches_xs, ys = batches_ys, keep_prob_s= 0.95))
    if(i %% 50 == 0){
      print(paste("iteration =", i, "||  Accuracy wrt train_y =", compute_accuracy(output_result, train_x, train_y), sep = " "))
      print("---")
      print(paste("iteration =", i, "||  Accuracy wrt ground truth=", compute_accuracy(output_result, train_x, true_train_y), sep = " "))
      print("=================================================")
    }
    
  }
  ######MIL
  pred_prob <- sess$run(output_result, feed_dict = dict(xs = train_x, keep_prob_s = 1))
  pred_prob <- data.frame(pred_prob)
  names(pred_prob) <- paste(c(1,0))
  pred_label <- c()
  for(i in 1:nrow(pred_prob)){
    n_answer <- names(which.max(pred_prob[i,]))
    pred_label <- c(pred_label, n_answer)
  }
  
  pred_label<-as.numeric(pred_label)
  
  if (sum(train_y[mse_train$label=="Tumor",1]!=pred_label[mse_train$label=="Tumor"])<threshhold)
  {
    break;
  }
  
  train_y[mse_train$label=="Tumor",1]<-pred_label[mse_train$label=="Tumor"]
  train_y[mse_train$label=="Tumor",2]<-1-pred_label[mse_train$label=="Tumor"]
  
  for (s in c_samples)
  {
    if (sum(train_y[mse_train$sample==s,1])==0)
    {
      j=which(pred_prob[mse_train$sample==s,1]==max(pred_prob[mse_train$sample==s,1]))
      train_y[mse_train$sample==s,][j,]<-c(1,0)       
    }
  }
  image(mse_train, train_y[,1]~x*y)
}

pred_prob <- sess$run(output_result, feed_dict = dict(xs = test_x, keep_prob_s = 1))
pred_prob <- data.frame(pred_prob)
names(pred_prob) <- paste(c(1,0))
pred_label <- c()
for(i in 1:nrow(pred_prob)){
  n_answer <- names(which.max(pred_prob[i,]))
  pred_label <- c(pred_label, n_answer)
}

mse_test$mi_cnn<-ifelse(pred_label=='1',"Cancer", "Normal")

pred_prob <- sess$run(output_result, feed_dict = dict(xs = train_x, keep_prob_s = 1))
pred_prob <- data.frame(pred_prob)
names(pred_prob) <- paste(c(1,0))
pred_label <- c()
for(i in 1:nrow(pred_prob)){
  n_answer <- names(which.max(pred_prob[i,]))
  pred_label <- c(pred_label, n_answer)
}
mse_train$mi_cnn<-ifelse(pred_label=='1',"Cancer", "Normal")

if (save.plot == T)
{
  pdf('results/bladder_mi-CNN.pdf')
  
  print(image(mse_train,mi_cnn~x*y, xlim=xaxis_train, ylim=yaxis_train, main='Training Set'))
  
  print(image(mse_test,mi_cnn~x*y, xlim=xaxis_test, ylim=yaxis_test, main = 'Testing Set'))
  
  dev.off()
}

print(paste0("The accuracy of mi-CNN on training data wrt label is: ", 
             accu(mse_train$mi_cnn, mse_train$diagnosis)))

print(paste0("The balanced accuracy of mi-CNN on training data wrt label is: ", 
             balance_accu(mse_train$mi_cnn, mse_train$diagnosis)))

print(paste0("The accuracy of mi-CNN on training data wrt truth is: ", 
             accu(mse_train$mi_cnn, mse_train$truth)))

print(paste0("The balanced accuracy of mi-CNN on training data wrt truth is:",
             balance_accu(mse_train$mi_cnn, mse_train$truth)))


print(paste0("The accuracy of mi-CNN on testing data wrt truth is: ",
             accu(mse_test$mi_cnn, mse_test$truth)))

print(paste0("The balanced accuracy of mi-CNN on testing data wrt truth is: ",
             balance_accu(mse_test$mi_cnn, mse_test$truth)))


#################CNN
learning_rate <- 0.000005 
train_step_by_GD <- tf$train$AdamOptimizer(learning_rate)$minimize(cross_entropy)

sess <- tf$Session()
init <- tf$global_variables_initializer()
sess$run(init)

#########data
train_x<-t(spectra(mse_train))
train_y<-mse_train$label

test_x<-t(spectra(mse_test))
test_y<-mse_test$label
train_y<-ifelse(train_y=="Tumor",1 ,0)
train_y<-as.numeric(train_y)
train_y<-cbind(train_y,1-train_y)
test_y<-ifelse(test_y=="Tumor",1 ,0)
test_y<-as.numeric(test_y)
test_y<-cbind(test_y,1-test_y)

for (i in 1:10000){
  batch_seq <- round(30) %>% sample(seq_len(nrow(train_x)), size = .) 
  batches_xs <- train_x[batch_seq,]
  batches_ys <- train_y[batch_seq,]
  sess$run(train_step_by_GD, feed_dict = dict(xs = batches_xs, ys = batches_ys, keep_prob_s= 0.95))
  if(i %% 50 == 0){
    print(paste("iteration =", i, "||  Accuracy wrt label =", compute_accuracy(output_result, train_x, train_y), sep = " "))
    print("---")
    print(paste("iteration =", i, "||  Accuracy wrt ground truth=", compute_accuracy(output_result, train_x, true_train_y), sep = " "))
    print("=================================================")
  }
  
}


pred_prob <- sess$run(output_result, feed_dict = dict(xs = test_x, keep_prob_s = 1))
pred_prob <- data.frame(pred_prob)
names(pred_prob) <- paste(c(1,0))
pred_label <- c()
for(i in 1:nrow(pred_prob)){
  n_answer <- names(which.max(pred_prob[i,]))
  pred_label <- c(pred_label, n_answer)
}

mse_test$cnn<-ifelse(pred_label=='1',"Cancer", "Normal")

pred_prob <- sess$run(output_result, feed_dict = dict(xs = train_x, keep_prob_s = 1))
pred_prob <- data.frame(pred_prob)
names(pred_prob) <- paste(c(1,0))
pred_label <- c()
for(i in 1:nrow(pred_prob)){
  n_answer <- names(which.max(pred_prob[i,]))
  pred_label <- c(pred_label, n_answer)
}
mse_train$cnn<-ifelse(pred_label=='1',"Cancer", "Normal")


if (save.plot == T)
{
  pdf('results/bladder_CNN.pdf')
  
  print(image(mse_train, cnn~x*y, xlim=xaxis_train, ylim=yaxis_train, main='Training Set'))
  
  print(image(mse_test, cnn~x*y, xlim=xaxis_test, ylim=yaxis_test, main = 'Testing Set'))
  
  dev.off()
}

print(paste0("The accuracy of mi-CNN on training data wrt label is: ", 
             accu(mse_train$cnn, mse_train$diagnosis)))

print(paste0("The balanced accuracy of mi-CNN on training data wrt label is: ", 
             balance_accu(mse_train$cnn, mse_train$diagnosis)))

print(paste0("The accuracy of mi-CNN on training data wrt truth is: ", 
             accu(mse_train$cnn, mse_train$truth)))

print(paste0("The balanced accuracy of mi-CNN on training data wrt truth is:",
             balance_accu(mse_train$cnn, mse_train$truth)))


print(paste0("The accuracy of mi-CNN on testing data wrt truth is: ",
             accu(mse_test$cnn, mse_test$truth)))

print(paste0("The balanced accuracy of mi-CNN on testing data wrt truth is: ",
             balance_accu(mse_test$cnn, mse_test$truth)))
