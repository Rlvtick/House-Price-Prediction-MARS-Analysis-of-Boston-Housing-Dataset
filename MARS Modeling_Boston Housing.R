# ----------------------------------- 1. Install and Load Necessary Libraries -----------------------------------

required_packages <- c("earth", "MASS", "caret", "ggplot2", "dplyr", "gridExtra")
installed_packages <- rownames(installed.packages())

for (pkg in required_packages) {
  if (!(pkg %in% installed_packages)) {
    install.packages(pkg, dependencies = TRUE)
  }
}

update.packages(ask = FALSE)
packageVersion("ggplot2")
packageVersion("plotly")
install.packages("ggplot2")
install.packages("plotly")
install.packages("corrplot")

library(earth)
library(MASS)      # Contains the Boston dataset
library(ggplot2)  # Load ggplot2 first
library(plotly)   # Then load plotly
library(caret)    # Finally, load caret
library(dplyr)     # For data manipulation
library(gridExtra) # For arranging multiple plots
library(corrplot) # For correlation plot

# ----------------------------------- 2. Load and Explore the Boston Housing Dataset -----------------------------------


data("Boston", package = "MASS")
head(Boston)

summary(Boston)
sum(is.na(Boston))

# ----------------------------------- 3. Exploratory Data Analysis (EDA) -----------------------------------
-----------------------------------

# Correlation Matrix
cor_matrix <- cor(Boston)
print(round(cor_matrix, 2))

# Visualize the correlation matrix
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8, addCoef.col = "black")

# Pairwise Scatter Plots for selected features vs. MEDV
selected_features <- c("rm", "lstat", "ptratio", "indus", "nox")

# Create scatter plots
plot_list <- lapply(selected_features, function(feature) {
  ggplot(Boston, aes_string(x = feature, y = "medv")) +
    geom_point(color = "blue", alpha = 0.6) +
    geom_smooth(method = "loess", color = "red") +
    labs(title = paste("MEDV vs", toupper(feature)),
         x = toupper(feature),
         y = "Median Value (MEDV)") +
    theme_minimal()
})

# Arrange plots in a grid
do.call(grid.arrange, c(plot_list, ncol = 2))

# ----------------------------------- 4. Data Preprocessing -----------------------------------

names(Boston) <- make.names(names(Boston))

Boston$chas <- as.factor(Boston$chas) # Convert 'chas' to a factor as it's a categorical variable

# Useful for interpretation)
preProcValues <- preProcess(Boston, method = c("center", "scale"))
Boston_scaled <- predict(preProcValues, Boston)

# ----------------------------------- 5. Split the Data into Training and Testing Sets -----------------------------------

set.seed(123)

# Create a 70-30 train-test split
train_index <- createDataPartition(Boston_scaled$medv, p = 0.7, list = FALSE)
train_data <- Boston_scaled[train_index, ]
test_data  <- Boston_scaled[-train_index, ]

# ----------------------------------- 6. Implementing MARS using the 'earth' Package -----------------------------------

# Fit the MARS model
mars_model <- earth(medv ~ ., data = train_data, 
                    degree = 2,  # Degree of interaction
                    nprune = NULL, # Algorithm will decide the number of terms
                    trace = 2)     # To see the fitting process

summary(mars_model)

# ----------------------------------- 7. Model Evaluation -----------------------------------

# Predict on the test set
predictions <- predict(mars_model, newdata = test_data)

# Calculate Performance Metrics
mae <- mean(abs(test_data$medv - predictions))
rmse <- sqrt(mae)
r_squared <- 1 - sum((test_data$medv - predictions)^2) / sum((test_data$medv - mean(test_data$medv))^2)

cat("Model Performance on Test Set:\n")
cat(sprintf("Mean Absolute Error (MAE): %.3f\n", mae))
cat(sprintf("Root Mean Squared Error (RMSE): %.3f\n", rmse))
cat(sprintf("R-squared (R²): %.3f\n", r_squared))


# ----------------------------------- 8. Cross-Validation (Robust Evaluation) -----------------------------------

train_control <- trainControl(method = "cv", number = 5)

set.seed(123)
cv_model <- train(medv ~ ., data = train_data, method = "earth",
                  trControl = train_control,
                  tuneGrid = expand.grid(degree = 2, nprune = 19:50)) 

print(cv_model)
plot(cv_model)

# ----------------------------------- 9. Feature Importance and Interpretation -----------------------------------

# Variable Importance
var_imp <- evimp(mars_model)
print(var_imp)

# Plot Variable Importance
plot(var_imp, main = "Variable Importance from MARS Model")

top_features <- rownames(var_imp)[1:3]

# Generate Partial Dependence Plots
par(mfrow = c(1, 3)) 
for (feature in top_features) {
  plot(mars_model, which.terms = feature, main = paste("Partial Dependence of", feature))
}
par(mfrow = c(1,1)) 

# ----------------------------------- 10. Residual Analysis -----------------------------------

residuals <- test_data$medv - predictions

# Residual Plot
ggplot(data = NULL, aes(x = predictions, y = residuals)) +
  geom_point(alpha = 0.6, color = "darkgreen") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residuals vs Predicted Values",
       x = "Predicted MEDV",
       y = "Residuals") +
  theme_minimal()

# Histogram of Residuals
ggplot(data = NULL, aes(x = residuals)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black", alpha = 0.7) +
  labs(title = "Histogram of Residuals",
       x = "Residuals",
       y = "Frequency") +
  theme_minimal()