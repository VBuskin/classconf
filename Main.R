# Main script for bootstrap estimation
# Change


library("boot")
library("tidyverse")
library("caret")
library("mlbench")
library("ranger")
library("xgboost")

data(PimaIndiansDiabetes)


# XGB ---------------------------------------------------------------------

set.seed(123)  # For reproducibility

data_for_xgb <- function(data, response) {

  # Features (all columns except the last one)
  X <- data[, -ncol(data)]
  # Response variable (last column)
  y <- as.numeric(response)
  
  train_indices <- sample(nrow(data), 0.7 * nrow(data))
  X_train <<- X[train_indices, ]
  y_train <<- y[train_indices]
  X_test <<- X[-train_indices, ]
  y_test <<- y[-train_indices]
  
  # Convert to binary format (1 for positive class, 0 for negative class)
  y_train_binary <- ifelse(y_train == 2, 1, 0)

}

data_for_xgb(PimaIndiansDiabetes, PimaIndiansDiabetes$diabetes)

# Fit model (actually works!)

xgb_model <- xgboost(data = as.matrix(X_train), label = y_train_binary, nrounds = 20, objective = "binary:logistic")


# Bootstrapping ---------------------------------------------------------

# Function to obtain yhat

prepare_preds <- function(data, indices) {
  
  # Bootstrapping sample
  d <- data[indices, ]
  
  # Remove response variable from X_train
  features <- d[,-1]
  
  # Response variable
  response <- as.numeric(d$Object_FE_Realisation) - 1
  
  # Convert columns to proper numeric integers
  columns_to_convert <- c("Telicity", "Text_category", "lemma", "frame", "Object_Definiteness", "Object_FE_Animacy", "Cluster_vec", "Cluster_clara")
  
  # Loop through each column and convert to numeric
  for (col in columns_to_convert) {
    features[[col]] <- as.integer(features[[col]])
  }
  
  # Split the data into features and labels
  index <- createDataPartition(response, times = 1, p = 0.7, list = FALSE)
  X_train <- features[index, ]
  y_train <- response[index]
  X_test <- features[-index, ]
  y_test <- response[-index]
  
  # Fit the xgboost model
  xgb_model_i <- xgboost(data = as.matrix(X_train), label = y_train, missing = NA, nrounds = 50, objective = "binary:logistic", verbose = 0)
  
  # Get probability predictions
  predictions <- predict(xgb_model_i, as.matrix(X_test))
  
  # Create a data frame with the predictions for a feature of interest
  predictions_df <- data.frame(predictions = predictions, Lemma = X_test$lemma)
  
  # Compute means by feature
  # means_df <- predictions_df %>%
  #dplyr::group_by(Lemma) %>%
  #dplyr::summarise(mean_prediction = mean(predictions)) %>% 
  #dplyr::ungroup()
  
  # Compute mean predictions by concreteness
  means_df <- aggregate(predictions ~ Lemma, data = predictions_df, FUN = mean)
  
  #mean_yhat <- means_df$predictions[1:90] # for all concreteness values, but impossible to interpret
  
  #mean_yhat <- mean(predictions_df$predictions)
  
  yhat <- means_df$predictions[1:115]
  
  return(yhat)
}

prepare_preds <- function(data, response, indices) {
  
  # Prepare data
  data_for_xgb(data, response)
  
  # Fit model
  xgb_model <- xgboost(data = as.matrix(X_train), label = y_train_binary, nrounds = 20, objective = "binary:logistic")

  # Get probability predictions
  predictions <- predict(xgb_model, as.matrix(X_test))
  
  # Create a data frame with the predictions for a feature of interest
  predictions_df <- data.frame(predictions = predictions, Lemma = X_test$lemma)
  
}


