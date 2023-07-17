require(ISLR)
require(boot)
require(glmnet)
library(e1071)
library(tree)
library(randomForest)
library(gbm)
library(caret)
library(xgboost)
library(MASS)
library(dplyr)
library("devtools")
library(factoextra)
library(ggcorrplot)
library(keras)
library(tensorflow)
library(Boruta)

load("C:/Users/vasun/Downloads/class_data.RData")
set.seed(24)
Rdata = cbind(y, x)
train_index = sample(1:nrow(Rdata),320)
train = Rdata[train_index,]
test = Rdata[-train_index,]

#####################################
#Supervised Learning
#####################################

#####################################
#Logistic Regression
#####################################
LR.fit=glm(y~.,data=train,family=binomial)
LR.probs=predict(LR.fit,newdata=test[,-1])
LR.pred=ifelse(LR.probs >=0.5,1,0)
table(LR.pred,test[,1])
LR_error = mean(LR.pred!=test[,1]) # 0.5 accuracy
#algorithm did not converge, have to look for another method or reduce variables

#####################################
#LDA
#####################################
LDA.fit=lda(y~.,data=train)
LDA.pred=predict(LDA.fit,test[,-1])
table(LDA.pred$class,test[,1])
LDA_error = mean(LDA.pred$class!=test[,1]) # 0.525 accuracy
#Variables are collinear, need to remove predictors

#####################################
#SVM
#####################################
svm.model = svm(y~., data=train, cost=1, method="C-classification", kernel="linear")
summary(svm.model)
svm.probs = predict(svm.model, test[,-1])
svm.pred = ifelse(svm.probs >=0.5,1,0)
svm.pred = as.integer(svm.pred)
table(svm.pred,test[,1])
SVM_error = mean(svm.pred!=test[,1]) # 0.475 accuracy

#####################################
#Neural Network
#####################################

x_nn_train = as.matrix(train[,-1])
y_nn_train = as.matrix(train[,1])
x_nn_test = as.matrix(test[,-1])
y_nn_test = as.matrix(test[,1])

model_nn = keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = ncol(x_nn_train)) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model_nn %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_adam(),
  metrics = list("accuracy")
)

history = model_nn %>% fit(
  x = x_nn_train,
  y = y_nn_train,
  epochs = 50,
  batch_size = 64,
  validation_split = 0.2
)

nn.prob = model_nn %>% predict(x_nn_test)
nn.pred = round(nn.prob)

NN_error = mean(nn.pred != y_nn_test)# 0.5125 accuracy

#####################################
#PCA + Random Forest
#####################################
options(max.print = 10000)
corr_matrix = cor(train[,-1])
# ggcorrplot(corr_matrix)
data_pca = princomp(corr_matrix)
fviz_eig(data_pca, addlabels = TRUE)
# summary(data_pca)
pca_data = predict(data_pca, newdata = train[,-1])[,1:320]
train_labels = ifelse(train[, 1]==1,"Yes","No")
train_labels = as.factor(train_labels)
pca_data_df = as.data.frame(pca_data)
train_new = cbind(train_labels, pca_data_df)

m = 42
oob.err=double(m)
test.err=double(m)
mtry = m
fit=randomForest(train_labels~.,data=train_new,mtry=m,ntree=400, type = "classification")
new_data = predict(data_pca, newdata = test[,-1])[,1:320]
pred=predict(fit,new_data)
pred.convert=ifelse(pred=="Yes",1,0)
RF_PCA_error = mean(pred.convert != test[,1])# 0.625 accuracy


#####################################
#Boruta + XgBoost
#####################################
boruta.train = Boruta(y~., data = train, doTrace = 0)
final.boruta = TentativeRoughFix(boruta.train)
selected_attributes = getSelectedAttributes(final.boruta, withTentative = FALSE)
selected_attributes_list = c("y", selected_attributes)
train_subset = train[, selected_attributes_list]

x_baruta_train = train_subset[, -1]
y_baruta_train = train_subset[, 1]

best_param = list()
best_seednumber = 1234
best_logloss = Inf
best_logloss_index = 0

for (iter in 1:100) {
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
  cv.nround = 1000
  cv.nfold = 5
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv = xgb.cv(data = as.matrix(x_baruta_train), label = as.numeric(y_baruta_train),
                 params = param, nthread=6, 
                 nfold=cv.nfold, nrounds=cv.nround,
                 verbose = T, early_stopping_rounds=8, maximize=FALSE)
  
  min_logloss_index  =  mdcv$best_iteration
  min_logloss =  mdcv$evaluation_log[min_logloss_index]$test_logloss_mean
  
  if (min_logloss < best_logloss) {
    best_logloss = min_logloss
    best_logloss_index = min_logloss_index
    best_seednumber = seed.number
    best_param = param
  }
  
}

nround = best_logloss_index
set.seed(best_seednumber)
xg_mod = xgboost(data = as.matrix(x_baruta_train), label = as.numeric(y_baruta_train), params = best_param, nround = nround, verbose = F)

test_baruta_subset = test[, selected_attributes_list]
x_baruta_test = test_baruta_subset[,-1]
y_baruta_test = test_baruta_subset[,1]

XgBoost.prob = predict(xg_mod, as.matrix(x_baruta_test))

XgBoost.pred=ifelse(XgBoost.prob>=0.5,1,0)
Boruta_XgBoost_error = mean(XgBoost.pred != y_baruta_test) # 0.775 Accuracy

cat(paste(
  "Logistic Regression error: ", LR_error, "\n",
  "LDA error: ", LDA_error, "\n",
  "SVM error: ", SVM_error, "\n",
  "Neural Network error: ", NN_error, "\n",
  "PCA + Random Forest error: ", RF_PCA_error, "\n",
  "Boruta + XgBoost error: ", Boruta_XgBoost_error, "\n",
  sep = ""
))


#####################################
# Unsupervised Learning
#####################################

################################################################################
# Clear objects from the workspace before running the Unsupervised Learning Part
################################################################################

# Load Data
dataset = load("C:/Users/vasun/Downloads/cluster_data.RData")
data = y 

##################################################
## Part 1: Dimensionality Reduction
##################################################

library(FactoMineR)
library(factoextra)
library(caret)

# Check for near zero variance
nzv = nearZeroVar(data, saveMetrics = TRUE)
print(paste('Range:',range(nzv$percentUnique)))
head(nzv) 
# Observation: Range->100 to 100, no non-zero var found. 
# No values dropped

## Dimensionality Reduction 1  PCA Method 1
# Apply PCA using the FactoMineR package
pca_result = PCA(data, scale.unit = TRUE)  # Scale data before applying PCA
# Print the main results
print(pca_result)
# Summary of the principal components
summary(pca_result)
# Visualize the percentage of variance explained by each principal component
barplot(pca_result$eig[, 2], 
        names.arg = 1:nrow(pca_result$eig),
        xlab = "Principal Components",
        ylab = "Percentage of variance",
        main = "Percentage of variance explained by each principal component")
# Plot the scree plot
fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 100))
#Cumulative variance for 10 components is less than 40%. 
#PCA might need more components to capture all variance


## PCA Method 2 using PRCOMP
pca.out=prcomp(data, scale=TRUE)
# Calculate the proportion of variance explained
pca.out$sdev
pr.var=pca.out$sdev^2
pr.var
pve=pr.var/sum(pr.var)
pve
plot(pve, xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1),type='b')
plot(cumsum(pve), xlab="Principal Component", ylab="Cumulative Proportion of Variance Explained", ylim=c(0,1),type='b')
# Plot 2 shows we need close to 300+ Components to capture significant variance
# Other dimensionality Reduction methods explored below

# Scale data. 
data = scale(data)

## Dimensionality Reduction 2 using RTSNE 
## (t-SNE): t-distributed stochastic neighbor embedding 
library(Rtsne)
# Perform t-SNE with 2 dimensions
set.seed(42)
tsne_result = Rtsne(data, dims = 2, perplexity = 30, verbose = TRUE, max_iter = 1000)
# Visualize t-SNE results
plot(tsne_result$Y, xlab = "t-SNE 1", ylab = "t-SNE 2", main = "t-SNE Reduction Viz")

## Dimensionality Reduction 3 usig UMAP
## UMAP: Uniform Manifold Approximation and Projection
library(uwot)
library(cluster)
library(NbClust)
library(factoextra)
# Set UMAP parameters
n_neighbors = 15     
min_dist = 0.1       
n_components = 2     
# Perform UMAP dimensionality reduction
umap_result = umap(data, n_neighbors = n_neighbors, min_dist = min_dist, n_components = n_components, )
plot(umap_result,  main = "UMAP Reduction Viz")

##################################################
## Asssessing Cluster Tendency
##################################################

# Note: we will select results from tsne and umap for further cluster analysis
library(clustertend)
library(hopkins)

# Method 1: Hopkins Statistics to find Statistical Significance
hopkins(tsne_result$Y)
hopkins(umap_result)

# Method 2: Visual Methods
fviz_dist(dist(tsne_result$Y), show_labels = FALSE)+
  labs(title = "TSNE Dims")
fviz_dist(dist(umap_result), show_labels = FALSE)+
  labs(title = "UMAP Dims")

##################################################
## Find Optimal Clusters
##################################################

# fviz_nbclust - clustering methods 1-4 used - kmeans, pam, clara, hcut
# Values Changed for different iterations manually to capture results and graphs

## Dataset 1: TSNE Dataset
# Elbow method
fviz_nbclust(tsne_result$Y, hcut, method = "wss") +
  geom_vline(xintercept = 4, linetype = 2)+
  labs(subtitle = "Elbow method")
# Silhouette method
fviz_nbclust(tsne_result$Y, hcut, method = "silhouette")+
  labs(subtitle = "Silhouette method")
# Gap statistic
fviz_nbclust(tsne_result$Y, hcut, nstart = 25, method = "gap_stat", nboot = 500)+
  labs(subtitle = "Gap statistic method")

## Dataset 2: UMAP Dataset
# Elbow method
fviz_nbclust(umap_result, hcut, method = "wss") +
  geom_vline(xintercept = 4, linetype = 2)+
  labs(subtitle = "Elbow method")

# Silhouette method
fviz_nbclust(umap_result, hcut, method = "silhouette")+
  labs(subtitle = "Silhouette method")

# Gap statistic
fviz_nbclust(umap_result, hcut, nstart = 25, method = "gap_stat", nboot = 500, iter.max=30)+
  labs(subtitle = "Gap statistic method")

## Clustering Method 5: DBSCAN 
library(dbscan)
db = fpc::dbscan(umap_result, eps = 0.35, MinPts = 5)
fviz_cluster(db, data = as.data.frame(umap_result), stand = FALSE,
             ellipse = FALSE, show.clust.cent = FALSE,
             geom = "point",palette = "jco", ggtheme = theme_classic())

db = fpc::dbscan(tsne_result$Y, eps = 2.56, MinPts = 5)
fviz_cluster(db, data = as.data.frame(tsne_result$Y), stand = FALSE,
             ellipse = FALSE, show.clust.cent = FALSE,
             geom = "point",palette = "jco", ggtheme = theme_classic())


## Clustering Method 6: Model-Based Clustering
library(mclust)
mc = Mclust(umap_result) # UMAP Dataset 
summary(mc)
mc$modelName # Optimal selected model 
mc$G
fviz_mclust(mc, "BIC", palette = "jco", xlab = "Number of Clusters", ylab = "BIC Value")
fviz_mclust(mc, "classification", geom = "point", pointsize = 1.5, palette = "jco", xlab = "UMAP1", ylab = "UMAP2")
fviz_mclust(mc, "uncertainty", palette = "jco", xlab = "UMAP1", ylab = "UMAP2")

mc = Mclust(tsne_result$Y) # UMAP Dataset
summary(mc)
mc$modelName # Optimal selected model 
mc$G
fviz_mclust(mc, "BIC", palette = "jco", xlab = "Number of Clusters", ylab = "BIC Value")
fviz_mclust(mc, "classification", geom = "point", pointsize = 1.5, palette = "jco", xlab = "TSNE1", ylab = "TSNE2")
fviz_mclust(mc, "uncertainty", palette = "jco", xlab = "UMAP1", ylab = "UMAP2")


## Clustering Method 7: Fuzzy Clustering
## Clusters values changed manually to capture results and plots
res.fanny = fanny(umap_result, k=5) # Best at 5
fviz_cluster(res.fanny, ellipse.type = "norm", repel = TRUE, label = FALSE, geom = "point", 
             palette = "jco", ggtheme = theme_minimal(),
             legend = "right", xlab = "UMAP1", ylab = "UMAP2")
fviz_silhouette(res.fanny, palette = "jco",
                ggtheme = theme_minimal())

res.fanny = fanny(tsne_result$Y, k=5) #Best at 5
fviz_cluster(res.fanny, ellipse.type = "norm", repel = TRUE, label = FALSE, geom = "point", 
             palette = "jco", ggtheme = theme_minimal(),
             legend = "right", xlab = "TSNE1", ylab = "TSNE2")
fviz_silhouette(res.fanny, palette = "jco",
                ggtheme = theme_minimal())

## Clustering Method 8: Hierarchical K-Means Clustering
## Clusters values changed manually to capture results and plots
res.hk =hkmeans(umap_result, 6)
fviz_cluster(res.hk, palette = "jco", repel = TRUE, label = FALSE, geom = "point",  
             ggtheme = theme_classic(), xlab = "UMAP1", ylab = "UMAP2")
silhouette_scores = silhouette(res.hk$cluster, dist(umap_result))
fviz_silhouette(silhouette_scores, palette = "jco",
                ggtheme = theme_minimal())

res.hk =hkmeans(tsne_result$Y, 6)
fviz_cluster(res.hk, palette = "jco", repel = TRUE, label = FALSE, geom = "point",  
             ggtheme = theme_classic(), xlab = "TSNE1", ylab = "TSNE2")
silhouette_scores = silhouette(res.hk$cluster, dist(tsne_result$Y))
fviz_silhouette(silhouette_scores, palette = "jco",
                ggtheme = theme_minimal())
