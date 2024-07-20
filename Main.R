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

#set.seed(123)  # For reproducibility


# Split data into response Y and features X; then coerce all categorical variables to integers 

prep_data <- function(data, response) {
  
  # Define response variable y and convert to a numeric vector
  y <- as.numeric(data[, which(names(data) == response)]) - 1
  
  # Define feature space X (should not include response!)
  X <- data[, -which(names(data) == response)]
  
  # Define which columns of X should be converted to numeric vectors (= all of them)
  columns_to_convert <- colnames(X)
  
  # Loop through each column and convert it
  for (col in columns_to_convert) {
    X[[col]] <- as.integer(X[[col]])
  }
  
  # Combine X and y
  full_df <- data.frame(X, y)
  
  # Return objects
  return(full_df)
}

# Example

test_output <- prep_data(PimaIndiansDiabetes, "diabetes")

test_output[[1]] # X
test_output[[2]] # y


# Split data further into X train/test and y train/test
partition_data <- function(features, response, index) {
  # Note that 
  
  # Partition data into train and test sets; requires library("caret")
  index <- caret::createDataPartition(response, times = 1, p = index, list = FALSE)
  
  # Training data (output is assigned to global environment)
  X_train <- features[index, ]
  y_train <- response[index]
  
  # Test data (output is assigned to global environment)
  X_test <- features[-index, ]
  y_test <- response[-index]
  
  # Return values
  return(list(X_train, X_test, y_train, y_test)) # List object
}

# Example

#index <- caret::createDataPartition(test_output$y, times = 1, p = index, list = FALSE)

test_partition <- partition_data(test_output[, -9], test_output$y, 0.75)


# Fixed order of objects -- maybe create an S3/S4 object that allows $ subsetting?
test_partition[[1]] # X train
test_partition[[2]] # X test
test_partition[[3]] # y train
test_partition[[4]] # y test etc

# Function to (i) convert all variables of interest to numeric factors and (ii) then perform data partitioning in one go

# ignore this
prep_for_xgb <- function(data, response, indices) {
  
## This part should precede the partitioning function  
  # Define response variable y and convert to a numeric vector
  y <- as.numeric(data[, which(names(data) == response)]) - 1
  
  # Define feature space X (should not include response!)
  X <- data[, -which(names(data) == response)]
  
  # Define which columns of X should be converted to numeric vectors (= all of them)
  columns_to_convert <- colnames(X)
  
  # Loop through each column and convert it
  for (col in columns_to_convert) {
    X[[col]] <- as.integer(X[[col]])
  }
  
  # Run partitioning function
  partition_data(data = y, indices = indices)
  
  # Done
}


# New test

input_prep <- prep_data(PimaIndiansDiabetes, "diabetes")

input_part <- partition_data(input_prep[, -9], input_prep$y, 0.75)

# Fit model (actually works!)
xgb_model <- xgboost(data = as.matrix(input_part[[1]]), label = input_part[[3]], nrounds = 20, objective = "binary:logistic")

# Bootstrapping ---------------------------------------------------------

# Version 20-07-2024

pred_fun <- function(model, test_data, col_index, indices) {

 # Get probability predictions
  predictions_model <- predict(model, as.matrix(X_test))
  
  # Create a data frame with the predictions for a feature of interest
  predictions_df <- data.frame(predictions = predictions_model, Var = X_test[, col_index])
  
  # Compute mean predictions by concreteness
  means_df <- aggregate(predictions ~ predictions_df$Var, data = predictions_df, FUN = mean)
  
  # Store statistic of interest
  yhat <- means_df$predictions[1:20]
  
  # Function return
  return(yhat)
}

# Example 

model <- xgb_model
X_test <- input_part[[2]]
col_index <- 2

pred_fun(xgb_model, input_part[[2]], 2) # okay, this works

# Run bootstrapping procedure
boot_results <- boot(data = input_part[[2]],
                     model = xgb_model,
                     statistic = pred_fun, 
                     R = 10,
                     col_index = 2 # Additional function arguments; maybe add everything here?
                     )

boot_results # No results because this has to be an iterative process. The bootstrapping function should the part where I create random subsamples of X.


# Combine everything ------------------------------------------------------

class_ci <- function(data, response, col_index, indices, model, prediction_function, iterations) { # or here
  
## Part 1: Data preparation
    
    # Define response variable y and convert to a numeric vector
    y <- as.numeric(data[, which(names(data) == response)]) - 1
    
    # Define feature space X (should not include response!)
    X <- data[, -which(names(data) == response)]
    
    # Define which columns of X should be converted to numeric vectors (= all of them)
    columns_to_convert <- colnames(X)
    
    # Loop through each column and convert it
    for (col in columns_to_convert) {
      X[[col]] <- as.integer(X[[col]])
    }
    
    # Combine X and y
    full_df <- data.frame(X, y)
    
## Part 2: Create test and training data
  
    # Version 2: Sample the full df
    train_indices <- sample(1:nrow(full_df), 0.75 * nrow(full_df))
    
    X_train_full <- full_df[train_indices, ]
    X_test_full <- full_df[-train_indices, ]
      
    # Training data (separated)
    X_train <- X_train_full %>% dplyr::select(!(y))
    y_train <- X_train_full %>% dplyr::pull(y)
      
    # Test data
    X_test <- X_test_full %>% dplyr::select(!(y))
    y_test <- X_test_full %>% dplyr::pull(y)
      
## Part 3: Prediction function
      
pred_fun <- function(model, test_data, col_index, indices, iterations) {
  
   # Get probability predictions
    predictions_model <- predict(model, as.matrix(X_test))
    
    # Create a data frame with the predictions for a feature of interest
    predictions_df <- data.frame(predictions = predictions_model, Var = X_test[, col_index])
    
    # Compute mean predictions by concreteness
    means_df <- aggregate(predictions ~ predictions_df$Var, data = predictions_df, FUN = mean)
    
    # Store statistic of interest
    yhat <- means_df$predictions[1:20]
    
    # Function return
   return(yhat)
  }
    
## Part 4: Bootstrapping

# Define a function that will be used by the boot function to ensure random subsamples
boot_fn <- function(data, indices) {
  # Sample the data
  sample_data <- data[indices, ]
  # Separate features and response
  X_sample <- sample_data %>% dplyr::select(-y)
  y_sample <- sample_data$y
  # Fit the model on the sampled data
  xgb_model_sample <- xgboost(data = as.matrix(X_sample),
                              label = y_sample,
                              nrounds = 20,
                              objective = "binary:logistic",
                              verbose = 0)
  # Apply the prediction function
  return(pred_fun(xgb_model_sample, X_test, col_index, indices, iterations))
}

# Perform bootstrap
boot_results <- boot(data = full_df,
                     statistic = boot_fn, 
                     R = iterations)

# Calculate 95% confidence intervals (first element only)
ci <- boot.ci(boot_results, type = "basic", conf = 0.95)

# Extract the bootstrap statistics
bootstrap_stats <- boot_results$t

# Compute the basic 95% confidence intervals for each statistic
ci <- apply(bootstrap_stats, 2, function(x) quantile(x, c(0.025, 0.975), na.rm = TRUE))

# Convert the confidence intervals to a tibble
ci_tibble <- tibble(
  original = boot_results$t0,
  lower = ci[1, ],
  upper = ci[2, ],
  variable = X_test[, col_index]
)

# Return df
return(ci_tibble)
  
  # Perform bootstrap
  boot_results <- boot(data = X_test,
                       model = xgb_model,
                       statistic = pred_fun, 
                       R = iterations,
                       col_index = 2# Additional function arguments; maybe add everything here?
  )
  
  # Calculate 95% confidence intervals (first element only)
  ci <- boot.ci(boot_results, type = "basic", conf = 0.95)
  
  # Extract the bootstrap statistics
  bootstrap_stats <- boot_results$t
  
  # Extract the bootstrap statistics
  bootstrap_stats <- boot_results$t
  
  # Compute the basic 95% confidence intervals for each statistic
  ci <- apply(bootstrap_stats, 2, function(x) quantile(x, c(0.025, 0.975), na.rm = TRUE))
  
  # Get the column name using col_index
  variable_name <- colnames(X_test)[col_index]
  
  # Get the unique levels of the variable
  variable_levels <- unique(X_test[, col_index])

  
  # Convert the confidence intervals to a tibble
  ci_tibble <- tibble(
    original = boot_results$t0,
    lower = ci[1, ],
    upper = ci[2, ],
    variable = X_test[, col_index]
  )
  
  # Add unique variable levels
  
  
  # Should also contain the variable
  
  # Return df
  return(ci_tibble)
}


# Testing -----------------------------------------------------------------

#class_ci <- function(data, response, index, features, col_index, indices, model, prediction_function, iterations) # ...

test_cis <- class_ci(data = PimaIndiansDiabetes,
         response = "diabetes",
         col_index = 2,
         model = xgb_model,
         iterations = 50)

test_cis$index <- 1:nrow(test_cis)

ggplot(test_cis, aes(x = index, y = original)) +
  geom_point() +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  labs(
    title = "Original Values with Error Bars",
    x = "Index",
    y = "Original Value"
  ) +
  theme_minimal()


# Older functions
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


