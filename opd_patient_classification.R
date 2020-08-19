
#Orthopedic Patient classification project
# Author: Rajeev Kumar Rajesh
# ----------------------------------------------------------

# Read the csv File

dat <- read.csv("column_2C_weka.csv", stringsAsFactors = F)

# Loading all needed libraries

library(knitr)
library(ggplot2)
library(plyr)
library(dplyr)
library(corrplot)
library(caret)
library(gridExtra)
library(scales)
library(Rmisc)
library(ggrepel)
library(randomForest)
library(psych)
library(xgboost)
library(rpart)


head(dat)

dim(dat)

str(dat)

#Total Normal and abnormal people
ggplot(dat,aes(x=class,fill=class))+geom_bar(stat = 'count')+labs(x = 'Number of normal and abnormal peopl') +
  geom_label(stat='count',aes(label=..count..), size=7) +theme_grey(base_size = 18)

table(dat$class)

