#######load library

library("ggplot2")
library("e1071")
library("CardinalWorkflows")
library(tensorflow)
library(magrittr)
library("gtools")

########load data
script.dir <- dirname(sys.frame(1)$ofile)