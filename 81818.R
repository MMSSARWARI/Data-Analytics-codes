library(caret)
library(xgboost)

set.seed(123)  # For reproducibility(Resampling)

# Load & preprocess training data
train_data <- read.csv("train_data.csv")
train_data$V5 <- as.numeric(gsub('"', '', train_data$V5))
train_data$target <- as.factor(train_data$target)
train_data$ID <- NULL 
train_data$V4 <- NULL # because the V4 has zero variance and it's existence is not important 

# Remove zero variance features
train_features <- train_data[, -which(names(train_data) == "target")]
zero_var <- nearZeroVar(train_features, saveMetrics = TRUE)
train_features_filtered <- train_features[, !zero_var$zeroVar]


# Preprocess: center and scale
preproc <- preProcess(train_features_filtered, method = c("center", "scale"))
train_scaled_features <- predict(preproc, train_features_filtered)


# Re-attach target
train_scaled <- train_scaled_features
train_scaled$target <- train_data$target

# Split data for model validation
train_index <- createDataPartition(train_scaled$target, p = 0.75, list = FALSE)
train_data_split <- train_scaled[train_index, ]
test_data_split <- train_scaled[-train_index, ]

# Cross-validation control
control <- trainControl(method = "cv", number = 10)

# XGBoost model tuning grid (single parameter set)
xgb_grid <- expand.grid(nrounds = 100,
                        max_depth = 6,
                        eta = 0.1,
                        gamma = 0,
                        colsample_bytree = 1,
                        min_child_weight = 1,
                        subsample = 1)

# Train with CV
model_xgb_cv <- train(target ~ ., data = train_data_split,
                      method = "xgbTree",
                      trControl = control,
                      tuneGrid = xgb_grid)

# Predict on validation split
xgb_preds <- predict(model_xgb_cv, newdata = test_data_split)
print("Validation Results:")
print(confusionMatrix(xgb_preds, test_data_split$target))

# Train final model on full training data
X_full <- as.matrix(train_scaled_features)
y_full <- train_scaled$target
y_numeric_full <- as.numeric(y_full) - 1
dfull <- xgb.DMatrix(data = X_full, label = y_numeric_full)

final_model <- xgb.train(
  params = list(
    objective = "multi:softmax",
    num_class = length(levels(y_full)),
    eval_metric = "merror",
    max_depth = 6,
    eta = 0.1
  ),
  data = dfull,
  nrounds = 100,
  verbose = 1
)

# Load and preprocess test data
rawData <- read.csv("test_features.csv")
rawData$V5 <- as.numeric(gsub('"', '', rawData$V5))
test_ids <- rawData$ID
rawData$ID <- NULL
rawData$V4 <- NULL

# Apply zero variance filter and scaling
test_features_filtered <- rawData[, !zero_var$zeroVar]
test_scaled_features <- predict(preproc, test_features_filtered)

# Predict test labels
dtest <- xgb.DMatrix(data = as.matrix(test_scaled_features))
test_preds <- predict(final_model, dtest)
test_labels <- levels(y_full)[test_preds + 1]

# Save submission file
submission <- data.frame(ID = test_ids, Prediction = test_labels)
write.csv(submission, "81818_predictions.csv", row.names = FALSE)







