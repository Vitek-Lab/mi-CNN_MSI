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
    tf$Variable(), 
  bias_FCLayer = shape(2L) %>% add_bias(),
  activation_function = tf$nn$softmax
)


## Loss function (cross entropy)

cross_entropy <- -tf$reduce_mean(abs(1-ys-tf$reduce_sum(ys)/60)*ys*tf$log(output_result + 1e-10), 
                                               reduction_indices = 1L)


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



