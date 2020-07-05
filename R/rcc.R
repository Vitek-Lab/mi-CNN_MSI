library("ggplot2")
library("e1071")
library("CardinalWorkflows")
library(tensorflow)
library(magrittr)
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

xaxis_train <- c(range(coord(mse_train)$x)[1]-5, range(coord(mse_train)$x)[2]+5)
yaxis_train <- c(range(coord(mse_train)$y)[1]-5, range(coord(mse_train)$y)[2]+5)
xaxis_test <- c(range(coord(mse_test)$x)[1]-5, range(coord(mse_test)$x)[2]+5)
yaxis_test <- c(range(coord(mse_test)$y)[1]-5, range(coord(mse_test)$y)[2]+5)



source(paste0(getwd(),'/R/model.R'))

print('train mi-CNN model--------------------')

set.seed(1238)
## Gradient Descent and learning rate setting
learning_rate <- 0.00001 # Set learning rate = 0.001
train_step_by_GD <- tf$train$AdamOptimizer(learning_rate)$minimize(cross_entropy)
## Session setting
rm(sess)
sess <- tf$Session()
init <- tf$global_variables_initializer()
sess$run(init)
saver <- tf$train$Saver()

#########data
train_x<-t(spectra(mse_train))
train_y<-mse_train$diagnosis

test_x<-t(spectra(mse_test))
test_y<-mse_test$diagnosis
train_y<-ifelse(train_y=="cancer",1 ,0)
train_y<-as.numeric(train_y)
train_y<-cbind(train_y,1-train_y)
test_y<-ifelse(test_y=="cancer",1 ,0)
test_y<-as.numeric(test_y)
test_y<-cbind(test_y,1-test_y)

threshhold=0.002*dim(mse_train)[2]/2
threshhold=0
for (n in 1:100)
{
  #sess$run(init)
  #Running
  for (i in 1:1000){
    batch_seq <- round(30) %>% sample(seq_len(nrow(train_x)), size = .) 
    
    batches_xs <- train_x[batch_seq,]
    batches_ys <- train_y[batch_seq,]
    sess$run(train_step_by_GD, feed_dict = dict(xs = batches_xs, ys = batches_ys, keep_prob_s= 0.95))

    if(i %% 50 == 0){
      print(paste("Step =", i, "|| Accuracy wrt label for training data =", compute_accuracy(output_result, train_x, train_y), sep = " "))
      print("---")
      print(paste("Step =", i, "|| Accuracy wrt label for testing data =", compute_accuracy(output_result, test_x, test_y), sep = " "))
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
  
  if (sum(train_y[mse_train$diagnosis=="cancer",1]!=pred_label[mse_train$diagnosis=="cancer"])<threshhold)
  {
    break;
  }
  #print(sum(train_y[mse_train$label=="cancer",1]!=pred_label[mse_train$label=="cancer"]))
  train_y[mse_train$diagnosis=="cancer",1]<-pred_label[mse_train$diagnosis=="cancer"]
  train_y[mse_train$diagnosis=="cancer",2]<-1-pred_label[mse_train$diagnosis=="cancer"]
  
  for (s in c_samples)
  {
    if (sum(train_y[mse_train$sample_diag==s,1])==0)
    {
      j=which(pred_prob[mse_train$sample_diag==s,1]==max(pred_prob[mse_train$sample_diag==s,1]))
      train_y[mse_train$sample_diag==s,][j,]<-c(1,0)       
    }
  }
  
  print(image(mse_train, train_y[,1]~x*y))
  print(image(mse_train, pred_label~x*y))
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


image(mse_train,mi_cnn~x*y, xlim=xaxis_train, ylim=yaxis_train)

image(mse_test,mi_cnn~x*y, xlim=xaxis_test, ylim=yaxis_test)


if (save.plot == T)
{
  pdf('results/rcc_mi-CNN.pdf')
  
  print(image(mse_train,mi_cnn~x*y, xlim=xaxis_train, ylim=yaxis_train, main='Training Set'))
  
  print(image(mse_test,mi_cnn~x*y, xlim=xaxis_test, ylim=yaxis_test, main = 'Testing Set'))
  
  dev.off()
}

################CNN
print('train CNN model------------------')


## Gradient Descent and learning rate setting
learning_rate <- 0.0002 # Set learning rate = 0.001
train_step_by_GD <- tf$train$AdamOptimizer(learning_rate)$minimize(cross_entropy)
## Session setting
rm(sess)
sess <- tf$Session()
init <- tf$global_variables_initializer()
sess$run(init)
saver <- tf$train$Saver()

#########data
train_x<-t(spectra(mse_train))
train_y<-mse_train$diagnosis

test_x<-t(spectra(mse_test))
test_y<-mse_test$diagnosis
train_y<-ifelse(train_y=="cancer",1 ,0)
train_y<-as.numeric(train_y)
train_y<-cbind(train_y,1-train_y)
test_y<-ifelse(test_y=="cancer",1 ,0)
test_y<-as.numeric(test_y)
test_y<-cbind(test_y,1-test_y)



for (i in 1:3000){
  batch_seq <- round(30) %>% sample(seq_len(nrow(train_x)), size = .) 
  
  batches_xs <- train_x[batch_seq,]
  batches_ys <- train_y[batch_seq,]
  sess$run(train_step_by_GD, feed_dict = dict(xs = batches_xs, ys = batches_ys, keep_prob_s= 0.95))
  
  if(i %% 50 == 0){
    print(paste("Step =", i, "|| Accuracy wrt label for training data =", compute_accuracy(output_result, train_x, train_y), sep = " "))
    print("---")
    print(paste("Step =", i, "|| Accuracy wrt label for testing data =", compute_accuracy(output_result, test_x, test_y), sep = " "))
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


image(mse_train,cnn~x*y, xlim=xaxis_train, ylim=yaxis_train)

image(mse_test,cnn~x*y, xlim=xaxis_test, ylim=yaxis_test)


if (save.plot == T)
{
  pdf('results/rcc_CNN.pdf')
  
  print(image(mse_train,cnn~x*y, xlim=xaxis_train, ylim=yaxis_train, main='Training Set'))
  
  print(image(mse_test,cnn~x*y, xlim=xaxis_test, ylim=yaxis_test, main = 'Testing Set'))
  
  dev.off()
}


#####################save checkpoint  

saver<- tf$train$Saver()
saver$save(sess, save_path='/checkpoints/rcc', write_meta_graph = FALSE)

sess$run(init)

saver$restore(sess, '/checkpoints/rcc')



