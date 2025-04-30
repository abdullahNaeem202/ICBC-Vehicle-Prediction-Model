library(tidyverse)
library(ggplot2)
library(stringr)

# Setting seed so we get the same random sample
set.seed(145)

# Loading raw data in R
raw_data <- read.csv("Vehicle_data.csv")

# Printing First few rows from the top
head(raw_data)

# Removing first columns
filter_data <- select(raw_data, 2:15) 

# Filtered the data so we can do analysis on data with specific attributes
filter_data <- filter(filter_data, (Vehicle.Use == 'Personal') & 
                        (Model.Year > 2018) & 
                        (Owner.Type == 'Person') &
                        (Vehicle.Count == 1))

# Drawing random sample of 10,000 rows 
data <- filter_data[sample(nrow(filter_data), size = 10000), ]

# Checking if there any missing values
anyNA.data.frame(data)


data$Make <- str_remove(data$Make, " .*")

summary(data)

write.csv(data, file = "cleaned_vehicle_data", row.names = FALSE)
