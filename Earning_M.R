#import required libraries
library(psych)
library(caTools)
library(smotefamily)
library(ROSE)
library(ROCR)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ggplot2)
library(caret)
library(e1071)
library(randomForest)

#descriptive analytics

earning <- read.csv("Earning_M.csv" , stringsAsFactors = T)
str(earning)
summary(earning)
head(earning)

ggplot (data = earning , aes(x = Manipulater)) + geom_bar() + 
  ggtitle("Bar chart of Number of Manipulator")

multi.hist(earning[, 2:9], bcol = "aliceblue"  ,
           density = TRUE, dcol=c("red","blue"),dlty=c("solid","dotted") , breaks = 16)

#correlation between different fields
pairs.panels(earning[,2:9]) 
 
ggplot (data = earning , aes(x = DSRI , color = Manipulater)) + geom_boxplot()
ggplot (data = earning , aes(x = GMI , color = Manipulater)) + geom_boxplot()
ggplot (data = earning , aes(x = AQI , color = Manipulater)) + geom_boxplot()
ggplot (data = earning , aes(x = SGI , color = Manipulater)) + geom_boxplot()
ggplot (data = earning , aes(x = DEPI , color = Manipulater)) + geom_boxplot()
ggplot (data = earning , aes(x = SGAI , color = Manipulater)) + geom_boxplot()
ggplot (data = earning , aes(x = ACCR , color = Manipulater)) + geom_boxplot()
ggplot (data = earning , aes(x = LEVI , color = Manipulater)) + geom_boxplot()

#________________Q1________________

#building beneish model
earning$beneish <- -4.84 + 0.92*earning$DSRI + 0.528*earning$GMI + 0.404*earning$AQI +
                    0.892*earning$SGI + 0.115*earning$DEPI - 0.172*earning$SGAI+
                    4.679*earning$ACCR- 0.327*earning$LEVI
head(earning$beneish)

#calculating total accuracy of beneish model
earning$ben <- ifelse(earning$beneish > -1.78 , 1 , 0)

#splitting data
set.seed(72)
split <- sample.split(earning$Manipulater, SplitRatio = 0.7)
split[1:20]
train <- subset(earning, split==TRUE)
test <- subset(earning, split==FALSE)

#calculating test accuracy of beneish model
b_table <- table(test$Manipulater , test$ben)
b_table
test_ben_accuracy <- sum(diag(b_table))/nrow(test)
test_ben_accuracy

#checking the ration in test and train
length(which(train$C.MANIPULATOR == 1))/length(train$C.MANIPULATOR)
length(which(test$C.MANIPULATOR == 1))/length(test$C.MANIPULATOR)
#________________Q2________________
#method1: SMOTE
train_smote = SMOTE(train[,2:9],train[,10],K=7)$data
colnames(train_smote)[9] <- "Manipulater"
train_smote$Manipulater <- as.factor(train_smote$Manipulater)
table(train_smote$Manipulater)
View(train_smote)  
#method2: under sampling
train_under = ovun.sample(Manipulater ~ DSRI+GMI+AQI+SGI+DEPI+SGAI+ACCR+LEVI, data=train, 
                          p=0.2, seed=1, method="under")$data
table(train_under$Manipulater)
#method3: over sampling
train_over = ovun.sample(Manipulater ~ DSRI+GMI+AQI+SGI+DEPI+SGAI+ACCR+LEVI, data=train, 
                          p=0.4, seed=1, method="over")$data
table(train_over$Manipulater)
#method4: both sampling
train_both = ovun.sample(Manipulater ~ DSRI+GMI+AQI+SGI+DEPI+SGAI+ACCR+LEVI, data=train, 
                         p=0.5, seed=1, method="both")$data
table(train_both$Manipulater)
View(train_both)
#method5: ROSE
train_rose <- ROSE(Manipulater ~ DSRI+GMI+AQI+SGI+DEPI+SGAI+ACCR+LEVI, data=train, seed=1)$data
table(train_rose$Manipulater)

#________________Q3________________

#logmod with train (imbalance) dataset
set.seed(72)
logmod <- glm (Manipulater ~ . -Company.ID -C.MANIPULATOR -beneish -ben -LEVI -DEPI , data=train , family = binomial)
summary(logmod)
#draw ROC curve and find the best threshold
pred_train_log <- predict(logmod, type="response")
head(pred_train_log)
pred_ROC_train_log <- prediction(pred_train_log, train$Manipulater)
perf_ROC_train_log <- performance(pred_ROC_train_log, "tpr", "fpr")
plot(perf_ROC_train_log ,print.cutoffs.at=seq(0,1,by=0.2), text.adj=c(-0.5,0.5) , colorize=TRUE
     , main = "ROC curve for imbalance train data")
abline(0,1,lty=2)
table(train$Manipulater, pred_train_log > 0.03)

as.numeric(performance(pred_ROC_train_log, "auc")@y.values)

#test
pred_test_log <- predict(logmod, type = "response", newdata = test)
table(test$Manipulater, pred_test_log > 0.03)

pred_ROC_log <- prediction(pred_test_log, test$Manipulater)
perf_ROC_log <- performance(pred_ROC_log, "tpr", "fpr")
plot(perf_ROC_log , main = "test ROC curve for imbalance dataset")
abline(0,1,lty=2)
as.numeric(performance(pred_ROC_log, "auc")@y.values)

#----------------SMOTE
#logmod with smote dataset
set.seed(72)
logmod_smote <- glm(Manipulater ~ . -DEPI, data=train_smote , family = binomial)
summary(logmod_smote)
#draw ROC curve and find the best threshold
pred_train_log_smote <- predict(logmod_smote, type="response")
pred_ROC_train_log_smote <- prediction(pred_train_log_smote, train_smote$Manipulater)
perf_ROC_train_log_smote <- performance(pred_ROC_train_log_smote, "tpr", "fpr")
plot(perf_ROC_train_log_smote,print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.5,1) , colorize=TRUE
     , main = "ROC curve for SMOTE train dataset")
abline(0,1,lty=2)

table(train_smote$Manipulater, pred_train_log_smote > 0.4)
as.numeric(performance(pred_ROC_train_log_smote, "auc")@y.values)

#test
pred_test_log_smote <- predict(logmod_smote, type = "response", newdata = test)
table(test$Manipulater, pred_test_log_smote > 0.4)

pred_ROC_log_smote <- prediction(pred_test_log_smote, test$Manipulater)
perf_ROC_log_smote <- performance(pred_ROC_log_smote, "tpr", "fpr")
plot(perf_ROC_log_smote , main = "test ROC curve for SMOTE dataset")
abline(0,1,lty=2)
as.numeric(performance(pred_ROC_log_smote, "auc")@y.values)

#----------------Under Sampling
#logmod with undersampling dataset
set.seed(72)
logmod_under <- glm(Manipulater ~ . -LEVI -DEPI -SGAI -GMI, data=train_under , family = binomial)
summary(logmod_under)
#draw ROC curve and find the best threshold
pred_train_log_under <- predict(logmod_under, type="response")
pred_ROC_train_log_under <- prediction(pred_train_log_under, train_under$Manipulater)
perf_ROC_train_log_under <- performance(pred_ROC_train_log_under, "tpr", "fpr")
plot(perf_ROC_train_log_under,print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.3,1) , colorize=TRUE
     , main = "ROC curve for undersampling train dataset")
abline(0,1,lty=2)

table(train_under$Manipulater, pred_train_log_under > 0.29)
as.numeric(performance(pred_ROC_train_log_under, "auc")@y.values)

#test
pred_test_log_under <- predict(logmod_under, type = "response", newdata = test)
table(test$Manipulater, pred_test_log_under > 0.29)

pred_ROC_log_under <- prediction(pred_test_log_under, test$Manipulater)
perf_ROC_log_under <- performance(pred_ROC_log_under, "tpr", "fpr")
plot(perf_ROC_log_under, main = "test ROC curve for undersampling dataset")
abline(0,1,lty=2)
as.numeric(performance(pred_ROC_log_under, "auc")@y.values)

#----------------Over Sampling
#logmod with oversampling dataset
set.seed(72)
logmod_over <- glm(Manipulater ~ . -DEPI , data=train_over , family = binomial)
summary(logmod_over)
#draw ROC curve and find the best threshold
pred_train_log_over <- predict(logmod_over, type="response")
pred_ROC_train_log_over <- prediction(pred_train_log_over, train_over$Manipulater)
perf_ROC_train_log_over <- performance(pred_ROC_train_log_over, "tpr", "fpr")
plot(perf_ROC_train_log_over,print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.5,1) , colorize=TRUE
     , main = "ROC curve for oversampling train dataset")
abline(0,1,lty=2)

table(train_over$Manipulater, pred_train_log_over > 0.25)
as.numeric(performance(pred_ROC_train_log_over, "auc")@y.values)

#test
pred_test_log_over <- predict(logmod_over, type = "response", newdata = test)
table(test$Manipulater, pred_test_log_over > 0.25)

pred_ROC_log_over <- prediction(pred_test_log_over, test$Manipulater)
perf_ROC_log_over <- performance(pred_ROC_log_over, "tpr", "fpr")
plot(perf_ROC_log_over, main = "test ROC curve for oversampling dataset")
abline(0,1,lty=2)
as.numeric(performance(pred_ROC_log_over, "auc")@y.values)

#----------------Both Sampling
#logmod with bothsampling dataset
set.seed(72)
logmod_both <- glm(Manipulater ~ . -DEPI -LEVI -SGAI, data=train_both , family = binomial)
summary(logmod_both)
#draw ROC curve and find the best threshold
pred_train_log_both <- predict(logmod_both, type="response")
pred_ROC_train_log_both <- prediction(pred_train_log_both, train_both$Manipulater)
perf_ROC_train_log_both <- performance(pred_ROC_train_log_both, "tpr", "fpr")
plot(perf_ROC_train_log_both,print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.5,1) , colorize=TRUE
     , main = "ROC curve for bothside sampling train dataset")
abline(0,1,lty=2)

table(train_both$Manipulater, pred_train_log_both > 0.35)
as.numeric(performance(pred_ROC_train_log_both, "auc")@y.values)

#test
pred_test_log_both <- predict(logmod_both, type = "response", newdata = test)
table(test$Manipulater, pred_test_log_both > 0.35)

pred_ROC_log_both <- prediction(pred_test_log_both, test$Manipulater)
perf_ROC_log_both <- performance(pred_ROC_log_both, "tpr", "fpr")
plot(perf_ROC_log_both, main = "test ROC curve for both side sampling dataset")
abline(0,1,lty=2)
as.numeric(performance(pred_ROC_log_both, "auc")@y.values)

#----------------ROSE
#logmod with rose dataset
set.seed(72)
logmod_rose <- glm(Manipulater ~ . -LEVI -GMI, data=train_rose , family = binomial)
summary(logmod_rose)
#draw ROC curve and find the best threshold
pred_train_log_rose <- predict(logmod_rose, type="response")
pred_ROC_train_log_rose <- prediction(pred_train_log_rose, train_rose$Manipulater)
perf_ROC_train_log_rose <- performance(pred_ROC_train_log_rose, "tpr", "fpr")
plot(perf_ROC_train_log_rose,print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.5,1) , colorize=TRUE
     , main = "ROC curve for ROSE train dataset")
abline(0,1,lty=2)

table(train_rose$Manipulater, pred_train_log_rose > 0.41)
as.numeric(performance(pred_ROC_train_log_rose, "auc")@y.values)

#test
pred_test_log_rose <- predict(logmod_rose, type = "response", newdata = test)
table(test$Manipulater, pred_test_log_rose > 0.41)

pred_ROC_log_rose <- prediction(pred_test_log_rose, test$Manipulater)
perf_ROC_log_rose <- performance(pred_ROC_log_rose, "tpr", "fpr")
plot(perf_ROC_log_rose, main = "test ROC curve for ROSE dataset")
abline(0,1,lty=2)
as.numeric(performance(pred_ROC_log_rose, "auc")@y.values)

#________________Q4________________
#since the model is going to be run several times with different data sets,
#a function is defined which takes the training dataset, makes the model and evaluates it
tree_model <- function(train_dataset) {
  #cross validation for parameter cp
  numFolds <- trainControl(method="cv", number=5)
  set.seed(72)
  cpGrid <- expand.grid(.cp=seq(0.001, 0.1, 0.001))
  tree_cp <- train(Manipulater ~., data=train_dataset, method="rpart", trControl= numFolds, tuneGrid= cpGrid)
  #choosing the best cp
  best_cp <- tree_cp$bestTune$cp
  
  set.seed(72)  
  #make tree decision model
  CV_tree <- rpart(Manipulater ~., data=train_dataset, method="class", cp=best_cp)
  prp(CV_tree)
  
  #draw ROC curve and calculate area under the curve-train
  pred_train <- predict(CV_tree, type="prob")
  pred_ROC_train <- prediction(pred_train[, 2], train_dataset$Manipulater)
  perf_ROC_train <- performance(pred_ROC_train, "tpr", "fpr")
  plot(perf_ROC_train,print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.5,1) , colorize=TRUE
       , main = "ROC curve for train dataset")
  abline(0,1,lty=2)
  
  #find best threshold based on tpr and fpr
  cutoffs <- data.frame(cut=perf_ROC_train@alpha.values[[1]],
                        J=perf_ROC_train@y.values[[1]]-perf_ROC_train@x.values[[1]])
  cutoffs <- cutoffs[order(cutoffs$J, decreasing=TRUE),]
  threshold <- cutoffs[1,1]
  
  #draw ROC curve and calculate area under the curve-test
  pred_test_tree <- predict(CV_tree, newdata = test, type="prob")
  pred_ROC_tree <- prediction(pred_test_tree[,2], test$Manipulater)
  perf_ROC_tree <- performance(pred_ROC_tree, "tpr", "fpr")
  plot(perf_ROC_tree, main = "ROC curve for test dataset")
  auc <- as.numeric(performance(pred_ROC_tree, "auc")@y.values)
  
  #draw confusion matrix using the threshold
  confusinMatrix <- table(test$Manipulater, pred_test_tree[,2]>threshold)
  #evaluation criteria
  accuracy_ <- sum(diag(confusinMatrix))/nrow(test)
  #in some cases, all the records are predicted FALSE, and confusion matrix has only one column
  #which means specificity=0(TP=0) and sensitivity=1(FP=0)
  if(ncol(confusinMatrix)==2) {
    specificity_ <- confusinMatrix[1,1]/(confusinMatrix[1,1]+confusinMatrix[1,2])
    sensitivity_ <- confusinMatrix[2,2]/(confusinMatrix[2,1]+confusinMatrix[2,2])
  } else {
    specificity_ <- 0
    sensitivity_ <- 1
  }
  
  #draw confusion matrix for train dataset in order to check overfitting
  train_matrix = table(train_dataset$Manipulater, pred_train[,2]>threshold)
  
  return(list(confusinMatrix, best_cp, auc, threshold, accuracy_, specificity_, sensitivity_, train_matrix))
}

#----------------ROSE
tree_model_rose = tree_model(train_rose)
#train confusion matrix
tree_model_rose[8]
#ROSE confusion matrix
tree_model_rose[1]
print(paste0("(ROSE) cp:", round(as.numeric(tree_model_rose[2]),3)," , auc:",round(as.numeric(tree_model_rose[3]),3) , " , threshold:", 
             round(as.numeric(tree_model_rose[4]),3), " , accuracy:",round(as.numeric(tree_model_rose[5]),3)," , specificity:",
             round(as.numeric(tree_model_rose[6]),3), " , sensitivity:",round(as.numeric(tree_model_rose[7]),3)))

#----------------Over Sampling
tree_model_over = tree_model(train_over)
#train confusion matrix
tree_model_over[8]
#Over Sampling confusion matrix
tree_model_over[1]
print(paste0("(Over Sampling) cp:", round(as.numeric(tree_model_over[2]),3)," , auc:",round(as.numeric(tree_model_over[3]),3) , " , threshold:", 
             round(as.numeric(tree_model_over[4]),3), " , accuracy:",round(as.numeric(tree_model_over[5]),3)," , specificity:",
             round(as.numeric(tree_model_over[6]),3), " , sensitivity:",round(as.numeric(tree_model_over[7]),3)))

#----------------Under Sampling
tree_model_under = tree_model(train_under)
#train confusion matrix
tree_model_under[8]
#Under Sampling confusion matrix
tree_model_under[1]
print(paste0("(Under Sampling) cp:", round(as.numeric(tree_model_under[2]),3)," , auc:",round(as.numeric(tree_model_under[3]),3) , " , threshold:", 
             round(as.numeric(tree_model_under[4]),3), " , accuracy:",round(as.numeric(tree_model_under[5]),3)," , specificity:",
             round(as.numeric(tree_model_under[6]),3), " , sensitivity:",round(as.numeric(tree_model_under[7]),3)))

#----------------both(over+under)
tree_model_both = tree_model(train_both)
#train confusion matrix
tree_model_both[8]
#Both(Over+Under) confusion matrix
tree_model_both[1]
print(paste0("(Both(Over+Under)) cp:", round(as.numeric(tree_model_both[2]),3)," , auc:",round(as.numeric(tree_model_both[3]),3) , " , threshold:", 
             round(as.numeric(tree_model_both[4]),3), " , accuracy:",round(as.numeric(tree_model_both[5]),3)," , specificity:",
             round(as.numeric(tree_model_both[6]),3), " , sensitivity:",round(as.numeric(tree_model_both[7]),3)))

#----------------SMOTE
tree_model_smote = tree_model(train_smote)
#train confusion matrix
tree_model_smote[8]
#SMOTE confusion matrix
tree_model_smote[1]
print(paste0("(SMOTE) cp:", round(as.numeric(tree_model_smote[2]),3)," , auc:",round(as.numeric(tree_model_smote[3]),3) , " , threshold:", 
             round(as.numeric(tree_model_smote[4]),3), " , accuracy:",round(as.numeric(tree_model_smote[5]),3)," , specificity:",
             round(as.numeric(tree_model_smote[6]),3), " , sensitivity:",round(as.numeric(tree_model_smote[7]),3)))

#________________Q5________________

#set mtry
def_mtry <- sqrt(ncol(earning)-4)

#since the model is going to be run several times with different data sets,
#a function is defined which takes the training dataset, makes the model and evaluates it
rf_model <- function(train_dataset) {
  #to prevent model from doing regression, target column is transformed to factor
  train_dataset$Manipulater = as.factor(train_dataset$Manipulater)
  #make random forest model
  set.seed(72)
  rfmod <- randomForest(Manipulater ~ . , data=train_dataset, method="class", nodsize=5, ntree=500, mtry=def_mtry)
  
  pred_train <- predict(rfmod, type="class")
  #draw confusion matrix-train
  confusinMatrix_tr <- table(train_dataset$Manipulater, pred_train)
  
  pred_test_rf <- predict(rfmod, newdata = test, type="class")
  
  #draw confusion matrix-test
  confusinMatrix <- table(test$Manipulater, pred_test_rf)
  #evaluation criteria
  accuracy_ <- sum(diag(confusinMatrix))/nrow(test)
  #in some cases, all the records are predicted FALSE, and confusion matrix has only one column
  #which means specificity=0(TP=0) and sensitivity=1(FP=0)
  if(ncol(confusinMatrix)==2) {
    specificity_ <- confusinMatrix[1,1]/(confusinMatrix[1,1]+confusinMatrix[1,2])
    sensitivity_ <- confusinMatrix[2,2]/(confusinMatrix[2,1]+confusinMatrix[2,2])
  } else {
    specificity_ <- 0
    sensitivity_ <- 1
  }
  
  return(list(confusinMatrix, accuracy_, specificity_, sensitivity_, confusinMatrix_tr))
}

#----------------ROSE
rf_model_rose = rf_model(train_rose)
#draw confusion matrix for train dataset in order to check overfitting
rf_model_rose[5]
#ROSE confusion matrix
rf_model_rose[1]
print(paste0("(ROSE) accuracy:",round(as.numeric(rf_model_rose[2]),3), " , specificity:", round(as.numeric(rf_model_rose[3]),3), " , sensitivity:",round(as.numeric(rf_model_rose[4]),3)))

#----------------Over Sampling
rf_model_over = rf_model(train_over)
#train confusion matrix
rf_model_over[5]
#Over Sampling confusion matrix
rf_model_over[1]
print(paste0("(OverSampling) accuracy:",round(as.numeric(rf_model_rose[2]),3), " , specificity:", round(as.numeric(rf_model_rose[3]),3), " , sensitivity:",round(as.numeric(rf_model_rose[4]),3)))

#----------------Under Sampling
rf_model_under = rf_model(train_under)
#train confusion matrix
rf_model_under[5]
#Under Sampling confusion matrix
rf_model_under[1]
print(paste0("(UnderSampling) accuracy:",round(as.numeric(rf_model_rose[2]),3), " , specificity:", round(as.numeric(rf_model_rose[3]),3), " , sensitivity:",round(as.numeric(rf_model_rose[4]),3)))

#----------------both(over+under)
rf_model_both = rf_model(train_both)
#train confusion matrix
rf_model_both[5]
#Both(Over+Under) confusion matrix
rf_model_both[1]
print(paste0("(Both) accuracy:",round(as.numeric(rf_model_rose[2]),3), " , specificity:", round(as.numeric(rf_model_rose[3]),3), " , sensitivity:",round(as.numeric(rf_model_rose[4]),3)))

#----------------SMOTE
rf_model_smote = rf_model(train_smote)
#train confusion matrix
rf_model_smote[5]
#SMOTE confusion matrix
rf_model_smote[1]
print(paste0("(SMOTE) auc:",round(as.numeric(rf_model_smote[2]), 3) ," , accuracy:",round(as.numeric(rf_model_smote[3]),3),
             " , specificity:", round(as.numeric(rf_model_smote[4]),3), " , sensitivity:",round(as.numeric(rf_model_smote[5]),3)))
