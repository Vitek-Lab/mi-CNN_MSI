#######load library
library("ggplot2")
library("e1071")
library("CardinalWorkflows")
library(tensorflow)
library(magrittr)
library("gtools")

############define function of CNN layers


########function for filter
add_conv_filter <- function(filterShape){ 
  filterForConvLayer <- tf$truncated_normal(filterShape, stddev = 0.1) %>% tf$Variable() 
  return(filterForConvLayer)
}
# function for bias
add_bias <- function(BiasShape){
  bias <- tf$constant(0.1, shape = BiasShape) %>% tf$Variable()
  return(bias)
}
# function for convolution layer
add_convolutionLayer <- function(inputData, filter_weight, activation_function = "None"){
  conv1dLayer <- tf$nn$conv1d(input = inputData, 
                              filters = filter_weight, 
                              stride = 1L, 
                              padding = 'SAME'
  )
  if(activation_function == "None"){
    output_result <- conv1dLayer %>% tf$nn$dropout(., keep_prob = keep_prob_s)
  }else{
    output_result <- conv1dLayer %>% tf$nn$dropout(., keep_prob = keep_prob_s) %>% activation_function()
  }
  return(output_result)
}
# function for Max pooling layer
add_maxpoolingLayer <- function(inputData){
  MaxPooling <- tf$nn$max_pool1d(inputData, 
                                 ksize = shape(2L), 
                                 strides = shape(1L), 
                                 padding = 'SAME')
  return(MaxPooling)
}
# function for flatten layer
add_flattenLayer <- function(inputData, numberOfFactors){
  flatten_layer <- tf$reshape(inputData, shape(-1, numberOfFactors))
  return(flatten_layer)
}
# function for fully connected layer
add_fullyConnectedLayer <- function(inputData, Weight_FCLayer, bias_FCLayer, activation_function = "None"){
  Wx_plus_b <- tf$matmul(inputData, Weight_FCLayer)+bias_FCLayer
  if(activation_function == "None"){
    FC_output_result <- Wx_plus_b %>% tf$nn$dropout(., keep_prob = keep_prob_s)
  }else{
    FC_output_result <- Wx_plus_b %>% tf$nn$dropout(., keep_prob = keep_prob_s) %>% activation_function()
  }
  return(FC_output_result)
}

# function for accuracy
compute_accuracy <- function(model_result, v_xs, v_ys){
  y_pre <- sess$run(model_result, feed_dict = dict(xs = v_xs, keep_prob_s= 1))
  correct_prediction <- tf$equal(tf$argmax(y_pre, 1L), tf$argmax(v_ys, 1L))
  accuracy <- tf$cast(correct_prediction, tf$float32) %>% tf$reduce_mean(.)
  result <- sess$run(accuracy, feed_dict = dict(xs = v_xs, ys = v_ys, keep_prob_s= 1))
  return(result)
}

#####################prepare data

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



##########define placeholders

#####input length
mz_len <- length(mz(mse))

xs <- tf$placeholder(tf$float32, shape(NULL,mz_len)) 
ys <- tf$placeholder(tf$float32, shape(NULL,2L)) 
keep_prob_s <- tf$placeholder(tf$float32)
x_image <- tf$reshape(xs, shape(-1L, mz_len, 1L)) 

## Convolution layer 1 
convolayer1 <- add_convolutionLayer(
  inputData = x_image,
  filter_weight = shape(38L,1L,1L) %>% add_conv_filter(),
  activation_function = tf$nn$relu
)
## Max pooling layer 1
maxPooling_1 <- add_maxpoolingLayer(
  convolayer1
)
## Convolution layer 2 
convolayer2 <- add_convolutionLayer(
  inputData = maxPooling_1, 
  filter_weight = shape(18L,1L,1L) %>% add_conv_filter(),
  activation_function = tf$nn$relu
) 
## Max pooling layer 2 
maxPooling_2 <- add_maxpoolingLayer(
  inputData = convolayer2
)

##convolution layer 3
convolayer3 <- add_convolutionLayer(
  inputData = maxPooling_2, 
  filter_weight = shape(16L,1L,1L) %>% add_conv_filter(),
  activation_function = tf$nn$relu
) 
## Max pooling layer 3 
maxPooling_3 <- add_maxpoolingLayer(
  inputData = convolayer3
)
## Flatten layer
flatLayer_output <- add_flattenLayer(
  inputData = maxPooling_3,
  numberOfFactors = c(ceiling(mz_len/1)) %>% as.numeric()
)

## Fully connected layer 1
output_result <- add_fullyConnectedLayer(
  inputData = flatLayer_output,
  Weight_FCLayer = shape(ceiling(mz_len/1), 2L) %>% 
    tf$random_normal(., stddev = 0.1) %>% 
    tf$Variable(), # Set output layer ouput = 10 labels
  bias_FCLayer = shape(2L) %>% add_bias(),
  activation_function = tf$nn$softmax
)


## Loss function (cross entropy)
cross_entropy <- tf$reduce_mean(-tf$reduce_sum(ys*tf$log(output_result), 
                                               reduction_indices = 1L))

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


image(mse_train,mi_cnn~x*y, xlim=xaxis_train, ylim=yaxis_train)

image(mse_test,mi_cnn~x*y, xlim=xaxis_test, ylim=yaxis_test)



print("The accuracy of mi-CNN on training data wrt label is:\\" )
print(accu(mse_train$mi_cnn, mse_train$diagnosis))
print("The balanced accuracy of mi-CNN on training data wrt label is:\\" )
print(balance_accu(mse_train$mi_cnn, mse_train$diagnosis))

print("The accuracy of mi-CNN on training data wrt truth is:\\" )
print(accu(mse_train$mi_cnn, mse_train$truth))
print("The balanced accuracy of mi-CNN on training data wrt truth is:\\" )
print(balance_accu(mse_train$mi_cnn, mse_train$truth))

print("The accuracy of mi-CNN on testing data wrt truth is:\\" )
print(accu(mse_test$mi_cnn, mse_test$truth))
print("The balanced accuracy of mi-CNN on testing data wrt truth is:\\" )
print(balance_accu(mse_test$mi_cnn, mse_test$truth))

#################CNN
learning_rate <- 0.000008 # Set learning rate = 0.001
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


image(mse_train,cnn~x*y, xlim=xaxis_train, ylim=yaxis_train)

image(mse_test,cnn~x*y, xlim=xaxis_test, ylim=yaxis_test)

print("The accuracy of CNN on training data wrt label is:\\" )
print(accu(mse_train$cnn, mse_train$diagnosis))
print("The balanced accuracy of CNN on training data wrt label is:\\" )
print(balance_accu(mse_train$cnn, mse_train$diagnosis))

print("The accuracy of CNN on training data wrt truth is:\\" )
print(accu(mse_train$cnn, mse_train$truth))
print("The balanced accuracy of CNN on training data wrt truth is:\\" )
print(balance_accu(mse_train$cnn, mse_train$truth))

print("The accuracy of CNN on testing data wrt truth is:\\" )
print(accu(mse_test$cnn, mse_test$truth))
print("The balanced accuracy of CNN on testing data wrt truth is:\\" )
print(balance_accu(mse_test$cnn, mse_test$truth))