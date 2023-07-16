# Reading in packages
library("readr")
library("dplyr")
library("corrplot")
library("caret")
library("stats")
library("MASS") 

# Import dataframes
train <- "https://github.gatech.edu/raw/MGT-6203-Summer-2023-Canvas/Team-116/main/Data/SplitData/TrainingSet.csv?token=GHSAT0AAAAAAAACXHOOOXLN47K5KJVZCKLCZFUIGTQ"
train_df <- read_csv(url(train)) # Selecting training dataframe

test <- "https://github.gatech.edu/raw/MGT-6203-Summer-2023-Canvas/Team-116/main/Data/SplitData/TestSet.csv?token=GHSAT0AAAAAAAACXHOPWAPO4OOHBZBD3CA2ZFUGRSQ"
test_df <- read_csv(url(test)) # Selecting test dataframe

# Filtering out COVID year, error columns

train_df <- train_df %>%
  filter(Year != 2020) %>%                                   # Removing COVID Year
  dplyr::select(-contains("Error")) %>%                      # Removing Error Columns
  dplyr::select(-contains("Mean")) %>%                       # Selecting Median Columns
  dplyr::select(-contains("Past")) %>%                       # Removing Past 12 Months Data
  dplyr::select(-c("Market", "Year", "Dome", "NumTeams", "Valuation")) # Removing Extraneous Data

test_df <- test_df %>%
  filter(Year != 2020) %>%                                   # Removing COVID Year
  dplyr::select(-contains("Error")) %>%                      # Removing Error Columns
  dplyr::select(-contains("Mean")) %>%                       # Selecting Median Columns
  dplyr::select(-contains("Past")) %>%                       # Removing Past 12 Months Data
  dplyr::select(-c(Market, Year, Dome, NumTeams, Valuation)) # Removing Extraneous Data

# Cleaning train columns 
colnames(train_df) <- gsub(" ", "_", colnames(train_df))  # Remove all spaces
colnames(train_df) <- gsub("-", "_", colnames(train_df))  # Remove all -
colnames(train_df) <- gsub("\\$", "", colnames(train_df)) # Remove all $
colnames(train_df) <- gsub("\\.", "", colnames(train_df)) # Remove all .
colnames(train_df) <- gsub(",", "", colnames(train_df))   # Remove all ,
colnames(train_df) <- gsub("\\(", "", colnames(train_df)) # Remove all ()
colnames(train_df) <- gsub("\\)", "", colnames(train_df)) # Remove all ()

# Cleaning test columns
colnames(test_df) <- gsub(" ", "_", colnames(test_df))  # Remove all spaces
colnames(test_df) <- gsub("-", "_", colnames(test_df))  # Remove all -
colnames(test_df) <- gsub("\\$", "", colnames(test_df)) # Remove all $
colnames(test_df) <- gsub("\\.", "", colnames(test_df)) # Remove all .
colnames(test_df) <- gsub(",", "", colnames(test_df))   # Remove all ,
colnames(test_df) <- gsub("\\(", "", colnames(test_df)) # Remove all ()
colnames(test_df) <- gsub("\\)", "", colnames(test_df)) # Remove all ()

# Remove outliers, calculate the z scores
z_scores <- apply(train_df, 2, function(x) abs(scale(x)))

# Z-score threshold
threshold <- 3

# Removing rows that violate threshold of 3
train_df <- train_df[rowSums(z_scores <= threshold) == ncol(train_df), ]

# Select the variables that show higher correlation with the attendance variable
y <- train_df %>% dplyr::select("Attendance")    # Selecting Y
cor_matrix_y <- data.frame(cor(train_df)) # Find values of X that have high correlation with Y

columns <- filter(cor_matrix_y, abs(cor_matrix_y$Attendance) >= .4) # Filtering for columns that have high correlation with Y

x <- data.frame(train_df  %>% 
                  dplyr::select(c(row.names(columns))) %>% 
                  dplyr::select(-"Attendance")) # Selecting X

test_df_x <- data.frame(test_df  %>% 
                          dplyr::select(c(row.names(columns))) %>% 
                          dplyr::select(-"Attendance")) # Selecting x values

test_df_y <- test_df %>% dplyr::select("Attendance") # Selecting y value

# Apply a boxcox transformation to the nonlinear data
box_cox <- boxcox(y$Attendance~., data=x) # Applying transformation with attendnace as y

lambda <- box_cox$x[which.max(box_cox$y)] # Storing lambda value

# Reduce dimensionality and correlation with PCA
PCA <- prcomp(x, scale=TRUE) # Scaled data and generated PCs for selected variables

# Print summary of PCA
summary(PCA)
PCA_x <- data.frame(PCA$x) # Creating new dataframe with PC values

# Fitting a linear model with the applied transformations selecting the PCs that show statistical significance
lm_model <- lm(((y$Attendance^lambda-1)/lambda)~PC1+PC3+PC4+PC5+PC6,
                data=PCA_x)

# Print summary of model
summary(lm_model) # Generating summary for linear regression performance
plot(lm_model)    # Generating plots for linear regression performance

# Looking at transformed alphas
rotations <- PCA$rotation[, c("PC1", "PC3", "PC4", "PC5", "PC6")] # Grabbing the rotation values from PCA
alpha <- rbind(lm_model$coefficients[2:6] %*% t(rotations) )      # Looking at our transformed coefficients
alpha

# Apply the same transformations to the new data
transformed_pca <- predict(PCA, newdata = test_df_x) # Applying PC transformation 

# Combine the transformed variables
transformed_data <- cbind(test_df_y, transformed_pca) # Combining attendance values with PC transformed values

# Evaluate the model on our testing data
predictions <- predict(lm_model, newdata = transformed_data) # Generating predictions based on our model

# Undo Box-Cox transformation
predicted_attendance <- (predictions * lambda + 1)^(1 / lambda) # Applying lambda to our projections

# Create a new column for our predicted attendance
test_df_y$Predicted_Attendance <- predicted_attendance # Appending projected attendance to recorded attendance

# Calculate evaluations for the model
mae <- mean(abs(test_df_y$Predicted_Attendance - test_df_y$Attendance)) # MAE
rmse <- sqrt(mean((test_df_y$Predicted_Attendance - test_df_y$Attendance)^2)) #RMSE
ss_total <- sum((test_df_y$Attendance - mean(test_df_y$Attendance))^2)
ss_residual <- sum((test_df_y$Attendance - test_df_y$Predicted_Attendance)^2)
r_squared <- 1 - (ss_residual / ss_total) #R^2
mae
rmse
r_squared

# Gather residuals
residuals <- resid(lm_model)

# Plot residuals
plot(residuals, main = "Residuals", xlab = "Observation", ylab = "Residuals") # No patterns
hist(residuals, col = "blue", main = "Residuals", xlab = "Residuals") # Normal distribution

# Testing the data on new markets
predict_url <- "https://github.gatech.edu/raw/MGT-6203-Summer-2023-Canvas/Team-116/main/Data/CSVData/cleaned_new_markets_data.csv?token=GHSAT0AAAAAAAACXHOO4PCUII2B7YPEZHOIZFUIJSQ"
predict_df <- read_csv(url(predict_url))

# Filtering out COVID year, error columns
predict_df <- predict_df %>%
  filter(Year != 2020) %>%              # Removing COVID Year
  dplyr::select(-contains("Error")) %>% # Removing Error Columns
  dplyr::select(-contains("Mean")) %>%  # Selecting Median Columns
  dplyr::select(-contains("Past"))      # Removing Past 12 Months Data

# Cleaning column names
colnames(predict_df) <- gsub(" ", "_", colnames(predict_df))  # Remove all spaces
colnames(predict_df) <- gsub("-", "_", colnames(predict_df))  # Remove all -
colnames(predict_df) <- gsub("\\$", "", colnames(predict_df)) # Remove all $
colnames(predict_df) <- gsub("\\.", "", colnames(predict_df)) # Remove all .
colnames(predict_df) <- gsub(",", "", colnames(predict_df))   # Remove all ,
colnames(predict_df) <- gsub("\\(", "", colnames(predict_df)) # Remove all ()
colnames(predict_df) <- gsub("\\)", "", colnames(predict_df)) # Remove all ()

# Selecting columns we used for our model
columns <- c(row.names(columns))
predict_df <- subset(predict_df, select = columns[-1]) # Removing attendance variables

# Apply the same transformations to the new data
transformed_pca <- predict(PCA, newdata = predict_df) # Applying PC transformation
transformed_data <- data.frame(transformed_pca)       # Creating new dataframe with PC values

# Evaluate the model on our testing data
predictions <- predict(lm_model, newdata = transformed_data) # Predicting on our transformed data

# Undo Box-Cox transformation
predicted_attendance <- (predictions * lambda + 1)^(1 / lambda) # Applying lambda to our predicted attendance

# Create a new column for our predicted attendance
predict_df_new <- read_csv(url(predict_url))
predict_df_new <- predict_df_new %>% filter(Year != 2020) %>%
  dplyr::select(-c(Dome, NumTeams))

predict_df_new$Predicted_Attendance <- predicted_attendance # Creating new dataframe with our predicted attendance

# Calculating Average Attendance Yearly per New Market
highest_projected_attendance <- predict_df_new %>% 
  group_by(Market) %>% 
  summarize(Projected_Average = mean(Predicted_Attendance))

# Selecting Top 5 Project Markets
head(highest_projected_attendance[order(highest_projected_attendance$Projected_Average, 
                                        decreasing = TRUE), ], n = 5)
