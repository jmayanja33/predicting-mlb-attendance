# Reading in packages
library("readr")
library("dplyr")
library("corrplot")
library("stats")
library("tidyverse")

# Import dataframe

# urlfile <- "INPUT CURRENT RAW LINK TO GITHUB URL CURRENT MARKET DATA HERE"
urlfile <- "https://github.gatech.edu/raw/MGT-6203-Summer-2023-Canvas/Team-116/main/Data/CSVData/cleaned_current_markets_data.csv?token=GHSAT0AAAAAAAACXHOPU24YEMPJTDDIHXEKZF5ROZQ"
df <- read_csv(url(urlfile))

# Filtering out COVID year
df <- df %>% filter(Year != 2020) # Filtering out COVID year

df <- df %>%
  dplyr::select(-contains("Error")) %>% # Removing Error Columns
  dplyr::select(-contains("Mean")) %>%  # Selecting Median Columns
  dplyr::select(-contains("Past")) %>%  # Removing Past 12 Months Data
  dplyr::select(-c(Market, Year))       # Removing Extraneous Data


# Cleaning df columns 
colnames(df) <- gsub(" ", "_", colnames(df))  # Remove all spaces
colnames(df) <- gsub("-", "_", colnames(df))  # Remove all -
colnames(df) <- gsub("\\$", "", colnames(df)) # Remove all $
colnames(df) <- gsub("\\.", "", colnames(df)) # Remove all .
colnames(df) <- gsub(",", "", colnames(df))   # Remove all ,
colnames(df) <- gsub("\\(", "", colnames(df)) # Remove all ()
colnames(df) <- gsub("\\)", "", colnames(df)) # Remove all ()

# Plotting distirbutions
for (col in names(df)) {
  hist(df[[col]], main=col, xlab=col)
}


# Plotting scatterplots
for (col in names(df)) {
  plot(df[[col]], df$Attendance, main=col, xlab=col)
}

# Analyzing correlations between attributes and dependent variable
cor_matrix <- cor(df)
corrplot(cor_matrix, method = "color", tl.cex = 0.45)

# Calculating Z-Scores
z_scores <- as.data.frame(scale(df)) # Calculate all Z Scores

# Function for gathering all outliers in all columns
outliers <- function(dataframe){
  dataframe %>%
    select_if(is.numeric) %>% 
    map(~ boxplot.stats(.x)$out) 
}

outliers <- outliers(df) # Store outliers
max(z_scores) # Maximum outlier returned
