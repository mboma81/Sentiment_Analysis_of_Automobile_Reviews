#CONSUMER CAR RATING AND REVIEWS

setwd("~/DISSERTATION/Consumers Car rating and reviews")

#Retrieving the csv files from the directory

review_files <- list.files(pattern = "*.csv")

#Combining the csv files
combined_review_files <- list()

#Storing the data frame in the combined_review_files list:

for (file in review_files) {
  df <- read.csv(file, header = TRUE)
  combined_review_files[[file]] <- df
}


#Combining the data frames into one using

combined_df <- do.call(rbind, combined_review_files)

#The do.call() function applies the rbind() function to the list of data frames, 
#effectively combining them row-wise.

#Exporting the combined data frame to a new CSV file

#Let's save the combined_df as a txt and csv file in our machine and read it using read_delim
#First as txt file
write.table(combined_df, file = 'combined_df.txt', row.names = F, sep = '\t')

#Second as csv file
write.csv(combined_df, file = 'combined_df.csv', row.names = F)


#DATA CLEANING

# Loading required libraries
library(dplyr)
library(stringr)
library(tm)



# Data cleaning for text columns
text_columns <- c("Author_Name", "Review_Title", "Review")  # Specify the column names containing text reviews

# Remove leading and trailing whitespaces from text columns
combined_df[, text_columns] <- lapply(combined_df[, text_columns], str_trim)

# Convert text to lowercase
combined_df[, text_columns] <- lapply(combined_df[, text_columns], tolower)

# Remove punctuation from text columns
combined_df[, text_columns] <- lapply(combined_df[, text_columns], function(x) str_replace_all(x, "[[:punct:]]", ""))

# Remove numbers from text columns
combined_df[, text_columns] <- lapply(combined_df[, text_columns], function(x) str_replace_all(x, "\\d+", ""))

# Let's remove stopwords using the tm package
stopwords_list <- stopwords("english")  # List of stopwords
combined_df[, text_columns] <- lapply(combined_df[, text_columns], function(x) removeWords(x, stopwords_list))

# Perform additional data cleaning and preprocessing steps as needed for the text columns

# Identifying the complete rows
complete_rows <- complete.cases(combined_df)

# Subseting the data to keep only complete rows
complete_data <- combined_df[complete_rows, ]

# Remove rows with NA values
clean_data <- complete_data[complete.cases(complete_data), ]

# Removing the text on date column

library(dplyr)
library(stringr)

# Read the CSV file into a data frame
cleandata1 <- read.csv("cleandata.csv")

# Remove the text from the date column
cleandata1$Review_Date <- str_replace(cleandata1$Review_Date, "on", "")

# Convert the 'date_column' to a Date format
cleandata1$Review_Date <- as.Date(cleandata1$Review_Date, format = "%Y-%m-%d")

# Save the modified data frame back to a CSV file
write.csv(cleandata1, "clean_data", row.names = FALSE)

# Save the cleaned and complete data to a new CSV file
write.csv(clean_data, file = 'cleandata.csv', row.names = F)

