library(e1071)
library(caret)
library(xgboost)
library(Boruta)

# Load data
load("C:/Users/vasun/Downloads/class_data.RData")
Rdata = cbind(y, x)
set.seed(24)

# Split data into training and testing sets
train_index = sample(1:nrow(Rdata),320)
train = Rdata[train_index,]
test = Rdata[-train_index,]

# Run Boruta feature selection on the training data
boruta.train = Boruta(y~., data = train, doTrace = 0)
final.boruta = TentativeRoughFix(boruta.train)

# Get selected attributes and create a train subset with those attributes only
selected_attributes = getSelectedAttributes(final.boruta, withTentative = FALSE)
# plot(final.boruta, xlab = "Attributes", main = "Boruta Graph")
selected_attributes_list = c("y", selected_attributes)
train_subset = train[, selected_attributes_list]

x_boruta_train = train_subset[, -1]
y_boruta_train = train_subset[, 1]

best_param = list()
best_seednumber = 1234
best_logloss = Inf
best_logloss_index = 0

for (iter in 1:100) {
  # Set hyperparameters
  param = list(objective = "binary:logistic",
                eval_metric = "logloss",
                max_depth = sample(6:10, 1),
                eta = runif(1, .01, .3),
                gamma = runif(1, 0.0, 0.2), 
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, .8), 
                min_child_weight = sample(1:40, 1),
                max_delta_step = sample(1:10, 1),
                num_class = 1
  )
  # Set cross-validation parameters
  cv.nround = 1000
  cv.nfold = 5
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv = xgb.cv(data = as.matrix(x_boruta_train), label = as.numeric(y_boruta_train),
                 params = param, nthread=6, 
                 nfold=cv.nfold, nrounds=cv.nround,
                 verbose = T, early_stopping_rounds=8, maximize=FALSE)
  
  min_logloss_index  =  mdcv$best_iteration
  min_logloss =  mdcv$evaluation_log[min_logloss_index]$test_logloss_mean
  
  # Find the hyperparameters with the lowest cross-validation log loss
  if (min_logloss < best_logloss) {
    best_logloss = min_logloss
    best_logloss_index = min_logloss_index
    best_seednumber = seed.number
    best_param = param
  }

}

# Training the final model with best parameters
nround = best_logloss_index
set.seed(best_seednumber)
xg_mod = xgboost(data = as.matrix(x_boruta_train), label = as.numeric(y_boruta_train), params = best_param, nround = nround, verbose = F)

test_boruta_subset = test[, selected_attributes_list]
x_boruta_test = test_boruta_subset[,-1]
y_boruta_test = test_boruta_subset[,1]

# Predict on testing subset
XgBoost.prob = predict(xg_mod, as.matrix(x_boruta_test))
XgBoost.pred=ifelse(XgBoost.prob>=0.5,1,0)
mean(XgBoost.pred == y_boruta_test)
test_error = mean(XgBoost.pred != y_boruta_test)
Boruta_XgBoost_error = test_error
cat(paste("Boruta + XgBoost error: ", Boruta_XgBoost_error, "\n",sep = ""))


##Testing
x_new_boruta = xnew[,selected_attributes]
XgBoost.test.prob = predict(xg_mod, as.matrix(x_new_boruta))
XgBoost.test.pred=ifelse(XgBoost.test.prob>=0.5,1,0)
ynew = XgBoost.test.pred
ynew = as.factor(ynew)

save(ynew,test_error,file="11.RData")


