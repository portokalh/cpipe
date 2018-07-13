library(dplyr)
library(e1071)   #svm
library(randomForest) #randomforest

library(readr) #read csv
test_data <- read_csv("/Users/wenlin_wu/Desktop/Book8.csv")
# View(test_data)
library(data.table)
test_data <- as.data.table(test_data)
test_data[genotype==3, genotype:=1]
test_data[genotype==4, genotype:=0]
variable_data <- test_data[, 5:7]
gen_data <- cbind(test_data[, list(genotype)], variable_data)
#sex_data <- cbind(test_data[, list(sex)], variable_data)
#age_data <- cbind(test_data[, list(age)], variable_data)
#age_data[age == 120, age := 0]
#age_data[age >120, age := 1]



# Choose data

df <- gen_data
setnames(df, 'genotype', 'Y')

#df <- sex_data
#setnames(df, 'sex', 'Y')

# df <- age_data
# setnames(df, 'age', 'Y')



# calculate accuracy
Accu <- function(true, pred){
  n <- length(pred)
  pred <- as.numeric(pred>=.5) #p>=0.5 --> yhat=1, p<0.5, --> yhat=0
  return(sum(true==pred)/n)
}


# Cross Validation
set.seed(123)
n = nrow(df)
index = sample(n, n)


k=n
# save results
Test = as.data.frame(matrix(nrow = k, ncol = 4))
colnames(Test) = c("Y", "glm", "rf", "svm")

for (i in 1:k) {
  test.cv <- df[index[i], ]
  Test[i,1] <- test.cv$Y
  train.cv <- df[-index[i], ]
  
  #GLM
  glm.cv = glm(Y ~ ., family = binomial(), data = train.cv, control=list(maxit=50))
  pred.glm.cv = predict(glm.cv, newdata = test.cv[,-1], type = "response")
  Test[i, 2] <- pred.glm.cv
  
  #Random forest
  rf.cv = randomForest(Y ~ .,data = train.cv)
  pred.rf.cv = predict(rf.cv, newdata = select(test.cv, -Y))
  Test[i,3]<-pred.rf.cv

  
  #SVM
  svm.cv = svm(Y ~ .,data = train.cv, mfinal = 10, rfs = TRUE, control = rpart.control(cp = -1))
  pred.svm.cv = predict(svm.cv, newdata = select(test.cv, -Y))
  Test[i,4] <- pred.svm.cv
  
}


Accu(Test$Y, Test$glm)
Accu(Test$Y, Test$rf)
Accu(Test$Y, Test$svm)




