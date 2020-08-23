
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


#Histogram explain all the detail of all six variable

s1<-ggplot(dat,aes(x=pelvic_incidence ))+geom_histogram(binwidth = 5, fill='blue') + theme_grey() +
  scale_x_continuous(breaks= seq(0, 150, by=10))
s2<-ggplot(dat,aes(x=pelvic_tilt.numeric ))+geom_histogram(binwidth = 5, fill='blue') + theme_grey() +
  scale_x_continuous(breaks= seq(0, 150, by=10))
s3<-ggplot(dat,aes(x=lumbar_lordosis_angle ))+geom_histogram(binwidth = 5, fill='blue') + theme_grey() +
  scale_x_continuous(breaks= seq(0, 150, by=10))
s4<-ggplot(dat,aes(x=sacral_slope ))+geom_histogram(binwidth = 5, fill='blue') + theme_grey() +
  scale_x_continuous(breaks= seq(0, 150, by=10))
s5<-ggplot(dat,aes(x=pelvic_radius ))+geom_histogram(binwidth = 5, fill='blue') + theme_grey() +
  scale_x_continuous(breaks= seq(0, 150, by=10))
s6<-ggplot(dat,aes(x=degree_spondylolisthesis ))+geom_histogram(binwidth = 5, fill='blue') + theme_grey() 
grid.arrange(s1,s2,s3,s4,s5,s6)






set.seed(1, sample.kind = 'Rounding')

#modrf<-randomForest(dat$class~.,data = dat,na.action = na.roughfix)
#imp_RF <- importance(modrf)
#class(imp_RF)
#imp_DF <- data.frame(Variables = row.names(imp_RF), MSE = imp_RF[,1])
#imp_DF <- imp_DF[order(imp_DF$MSE, decreasing = TRUE),]
#ggplot(imp_DF[1:6,], aes(x=reorder(Variables, MSE), y=MSE, fill=MSE)) + geom_bar(stat = 'identity') + labs(x = 'Variables', y= '% increase MSE if variable is randomly permuted') + coord_flip() + theme(legend.position="none")


#We have noticed that degree spondylolisthesis is the most important of the variables to explain the normal and abnormal of the patients. Let's focus more on this variable starting with its density compared to the classification of the patients.


ggplot(dat,aes(x=degree_spondylolisthesis,fill=class))+geom_density(alpha=0.5, aes(fill=factor(class))) + labs(title="degree spondylolisthesis") +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 10)) + theme_grey()




#we can see it also on the decision tree
library(rpart)
arbre <- rpart(dat$class~., method="class", minsplit=150, xval=10000, data=dat)
plot(arbre, uniform=TRUE, margin=0.1, main="Decision Tree")
text(arbre, fancy=TRUE, use.n=TRUE, pretty=0, all=TRUE)

#We can also view it on the boxplot.
ggplot(dat,aes(x=factor(class),y=degree_spondylolisthesis))+geom_boxplot()



#We can also visualize the density of the set of variables of importance that explains the class of patients.

s1<-ggplot(dat,aes(x=pelvic_incidence,fill=class))+geom_density(alpha=0.5, aes(fill=factor(class))) + labs(title="pelvic_incidence")  + theme_grey()
s2<-ggplot(dat,aes(x=pelvic_tilt.numeric,fill=class))+geom_density(alpha=0.5, aes(fill=factor(class))) + labs(title="pelvic tilt")  + theme_grey()
s3<-ggplot(dat,aes(x=lumbar_lordosis_angle,fill=class))+geom_density(alpha=0.5, aes(fill=factor(class))) + labs(title="lumbar lordosis angle") + theme_grey()
s4<-ggplot(dat,aes(x=sacral_slope,fill=class))+geom_density(alpha=0.5, aes(fill=factor(class))) + labs(title="sacral slope") +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 10)) + theme_grey()
s5<-ggplot(dat,aes(x=pelvic_radius,fill=class))+geom_density(alpha=0.5, aes(fill=factor(class))) + labs(title="pelvic radius") + theme_grey()
s6<-ggplot(dat,aes(x=degree_spondylolisthesis,fill=class))+geom_density(alpha=0.5, aes(fill=factor(class))) + labs(title="grade of spondylolisthesis") + theme_grey()
grid.arrange(s1,s2,s3,s4,s5,s6)




# Boxplot for all variables
s1<-ggplot(dat,aes(x=factor(class),y=pelvic_incidence ))+geom_boxplot()
s2<-ggplot(dat,aes(x=factor(class),y=pelvic_tilt.numeric))+geom_boxplot()
s3<-ggplot(dat,aes(x=factor(class),y=lumbar_lordosis_angle))+geom_boxplot()
s4<-ggplot(dat,aes(x=factor(class),y=sacral_slope))+geom_boxplot()
s5<-ggplot(dat,aes(x=factor(class),y=pelvic_radius))+geom_boxplot()
s6<-ggplot(dat,aes(x=factor(class),y=degree_spondylolisthesis))+geom_boxplot()
grid.arrange(s1,s2,s3,s4,s5,s6)


#For this part we are interested in the correlation of numeric variables to better understand their evolution between, we will take degree spondylolisthesis as target. Altogether, there all numeric variables with a correlation greater than zero with degree spondylolisthesis .

library(corrplot)
numericVars <- which(sapply(dat, is.numeric)) #index vector numeric variables
dat_numVar <- dat[, numericVars]
cor_numVar <- cor(dat_numVar, use="pairwise.complete.obs") #correlations of all numeric variables
#sort on decreasing correlations with degree_spondylolisthesis
cor_sorted <- as.matrix(sort(cor_numVar[,'degree_spondylolisthesis'], decreasing = TRUE))
#select only high corelations
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0)))
cor_numVar <- cor_numVar[CorHigh, CorHigh]
corrplot.mixed(cor_numVar, tl.col="black", tl.pos = "lt", tl.cex = 0.7,cl.cex = .7, number.cex=.7)





#we notice that,there are 3 numeric variables with a correlation of at least 0.5 with degree_spondylolisthesis,and there is no correlation between pelvic_radius and degree_spondylolisthesis.
#It also becomes clear the multicollinearity is an issue. For example: the correlation between sacral_slope and pelvic_incidence is very high (0.81), and both have similar (high) correlations with degree_spondylolisthesis. 
#Now let us visualize the correlation of these variables with respect to our target

library(gridExtra)
s1<-ggplot(data=dat, aes(x=pelvic_incidence, y=degree_spondylolisthesis))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black", aes(group=1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=100000))
s2<-ggplot(data=dat, aes(x=lumbar_lordosis_angle, y=degree_spondylolisthesis))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black", aes(group=1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=100000))
s3<-ggplot(data=dat, aes(x=sacral_slope, y=degree_spondylolisthesis))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black", aes(group=1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=100000))
s4<-ggplot(data=dat, aes(x=pelvic_tilt.numeric, y=degree_spondylolisthesis))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black", aes(group=1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=100000))
s5<-ggplot(data=dat, aes(x=pelvic_radius, y=degree_spondylolisthesis))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black", aes(group=1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=100000))
grid.arrange(s1,s2,s3,s4,s5)


#Visualize the high corelation between sacral_slope and pelvic_incidence


ggplot(data=dat, aes(x=sacral_slope, y=pelvic_incidence))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black", aes(group=1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=100000))


#Decision tree
library(rpart.plot)
class.tree <- rpart(dat$class~.,data = dat,control = rpart.control(cp = 0.01))
rpart.plot(class.tree, 
           box.palette="GnBu",
           branch.lty=10, shadow.col="gray", nn=TRUE)

# Random FOrest
caret_matrix <- train(x=dat[,1:6], y=dat[,7], data=dat, method='rf', trControl=trainControl(method="cv", number=5))
caret_matrix

#Support Vector Machine (SVM) model
caret_svm <- train(x=dat[,1:6], y=dat[,7], data=dat, method='svmRadial', trControl=trainControl(method="cv", number=5))
caret_svm

#Gradient Boosting Machine (GBM) model
caret_boost <- train(class~pelvic_incidence+pelvic_tilt.numeric+lumbar_lordosis_angle+sacral_slope+pelvic_radius+degree_spondylolisthesis, data=dat, method='gbm', preProcess= c('center', 'scale'), trControl=trainControl(method="cv", number=7), verbose=FALSE)
print(caret_boost)


#KNN Model to visualize the data
index <- sample(2,nrow(dat),replace= TRUE,prob=c(0.7,0.3))
trainClean <- dat[index==1,]
testClean <- dat[index==2,]

caret_knn <- train(class~., data=dat, method='knn', trControl=trainControl(method="cv", number=5),tuneLength = 20)
caret_knn

caret_knn <- train(class~., data=trainClean, method='knn', trControl=trainControl(method="cv", number=5),tuneLength = 20)
caret_knn

caret_knn <- train(class~., data=testClean, method='knn', trControl=trainControl(method="cv", number=5),tuneLength = 20)
caret_knn
