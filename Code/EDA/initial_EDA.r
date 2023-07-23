# Loading necessary packages
library("readr")
library("dplyr")
library("corrplot")
library("MASS")

# Import csv df
urlfile <- "https://github.gatech.edu/raw/MGT-6203-Summer-2023-Canvas/Team-116/main/Data/CSVData/cleaned_current_markets_data.csv?token=GHSAT0AAAAAAAACXHOOSYMU7GDMGE6S75BWZFJ2ITQ"
df <- read_csv(url(urlfile))

# Filtering out COVID year
df <- df %>% filter(Year != 2020)

# Renaming columns for analysis
names(df)[names(df) == "Households Median Income (Dollars)"] <- "Households_Median_Income_Dollars"
names(df)[names(df) == "Families Median Income (Dollars)"] <- "Families_Median_Income_Dollars"
names(df)[names(df) == "Nonfamily Households Median Income (Dollars)"] <- "Nonfamily_Households_Median_Income_Dollars"
names(df)[names(df) == "Married-Couple Families Median Income (Dollars)"] <- "Married_Couple_Families_Median_Income_Dollars"
names(df)[names(df) == "Married-Couple Families"] <- "Married_Couple_Families"
names(df)[names(df) == "Nonfamily Households"] <- "Nonfamily_Households"

# Selecting columns to compare
df <- subset(df, select = c( "Attendance", "Valuation", "Households", 
                             "Households_Median_Income_Dollars", 
                             "Families_Median_Income_Dollars", 
                             "Married_Couple_Families_Median_Income_Dollars", 
                             "Nonfamily_Households_Median_Income_Dollars",
                             "Families",
                             "Married_Couple_Families",
                             "Nonfamily_Households"))

# Plotting distirbutions
for (col in names(df)) {
  hist(df[[col]], main=col, xlab=col)
}

# Attempting to normalize data
df$Families_log <- log(df$Families)
df$Married_Couple_Families_log <- log(df$Married_Couple_Families)
df$Nonfamily_Households_log <- log(df$Nonfamily_Households)
df$Married_Couple_Families_log <- log(df$Married_Couple_Families)
df$Households_Median_Income_Dollars_log <- log(df$Households_Median_Income_Dollars)

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

# First look at linear model
summary(lm(Attendance~Families_log+Married_Couple_Families_log+Nonfamily_Households_log+Households_Median_Income_Dollars_log,
           data=df))

