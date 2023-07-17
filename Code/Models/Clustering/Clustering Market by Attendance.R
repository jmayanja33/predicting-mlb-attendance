library("readr")
library("dplyr")
library("ggplot2")


# Import dataframs
current <- "https://github.gatech.edu/raw/MGT-6203-Summer-2023-Canvas/Team-116/main/Data/CSVData/cleaned_current_markets_data.csv?token=GHSAT0AAAAAAAACXGNNZYYXCYS5IRGDKJGUZFU5OVA"
current_market <- read_csv(url(current)) # Selecting current market dataframe
current_market <- as.data.frame(current_market)

# Filtering out year 2021, select Market, Attendance columns

current_market <- current_market %>%
  filter(Year == 2021) %>%                                   # Selecting Year 2021
  dplyr::select(c(Market, Attendance))                  # Selecting Market, Year, Attendace columns


current_market

# Find the optimal number of clusters using Elbow method

wssplot <- function(data, nc=15, seed=1234){
  wss <- (nrow(data)-1)*sum(apply(data,2,var))
  for (i in 2:nc){
    set.seed(seed)
    wss[i] <- sum(kmeans(data, centers=i)$withinss)}
  plot(1:nc, wss, type="b", xlab="Number of Clusters",
       ylab="Within groups sum of squares")
  wss
}

# Plot the Elbow 
wssplot(current_market['Attendance'])


# Perform k-means clustering with k = 3 clusters
km <- kmeans(current_market['Attendance'], 3)
km

# Add a Size column to the dataframe according corresponding to its culster
Size <- as.character(km$cluster)
current_market$Size <- Size

current_market

# Plot the cluster graph, color by market size

ggplot(data = current_market, aes(x= 1:nrow(current_market), y = Attendance, color = Size)) + 
  geom_point() +
  geom_label(
    label=current_market$Market, 
    nudge_x = 0.25, nudge_y = 0.25
  ) +
  xlab("Market #")


