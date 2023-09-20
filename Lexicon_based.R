library(NLP)
library(tm)
library(RColorBrewer)
library(SnowballC)
library(stringr)
library(syuzhet)
library(tidytext)
library(dplyr)



# read in review.csv as input file
reviews <- read.csv("C:/Users/nkemj/Documents/DISSERTATION/NEW DATA/review.csv", sep = ",", header = TRUE)

am <- as.matrix(reviews)

head(am)
tail(am)

#---------------------------------------------------
#text data cleaning

# stringr functions for removing symbols
am <- str_remove_all(am,"–")
am <- str_remove_all(am,"’")
am <- str_remove_all(am,"—")
am <- str_remove_all(am,"“")
am <- str_remove_all(am,"”")

# tm functions for text cleaning
am<-removeNumbers(am)
am<-removePunctuation(am)
am<-tolower(am)
am<-removeWords(am,c("now", "one", "will", "may", "says", "said", 
                       "also", "figure", "etc", "re", "can"))
stopwords<-c("the", "and", stopwords("en"))
am<-removeWords(am, stopwords("en"))
am<-stripWhitespace(am)
am<-wordStem(am)        #function from SnowballC

review_text<-am
head(review_text)
tail(review_text)

#---------------------------------------------------

# Sentiment analysis:
# sentiment score using get_sentiment() function & scoring method
# scoring mehods: syuzhet, bing, afinn, nrc 
# Each method may have different scale

syuzhet_score <- get_sentiment(review_text, method="syuzhet")
head(syuzhet_score)
summary(syuzhet_score)

bing_score <- get_sentiment(review_text, method="bing")
head(bing_score)
summary(bing_score)

afinn_score <- get_sentiment(review_text, method="afinn")
head(afinn_score)
summary(afinn_score)

nrc_score <- get_sentiment(review_text, method="nrc")
head(nrc_score)
summary(nrc_score)

#Let's combine the scores from the different methods
comb_score <- cbind(syuzhet_score, bing_score, afinn_score, nrc_score)
dimnames(comb_score) <- list(1:nrow(comb_score), c("s1", "s2", "s3", "s4"))
df <- as.data.frame(comb_score)
head(df,20)

#---------------------------------------------------

# simple analysis based on syuzhet_score
min(df$s1)
max(df$s1)

#view text docs with extreme negative sentiment score
syuz_neg <- which(syuzhet_score<=(-5))
txt<-review_text[syuz_neg]
result<-cbind(syuz_neg,txt)
result

#view text docs with high posive sentiment score
syuz_posit <- which(syuzhet_score>=4.5)
txt<-review_text[syuz_posit]
result1<-cbind(syuz_posit,txt)
result1

# View text docs with a neutral sentiment score
syuz_neutral <- which(syuzhet_score > -2.5 & syuzhet_score < 2.5)
txt_neutral <- review_text[syuz_neutral]
result_neutral <- data.frame(syuz_neutral, txt_neutral)
result_neutral

#---------------------------------------------------

# Analysis given above has limitations wrt scale
# scale used by the text mining methods differ

# sentiment score normalized with sign function
# sign function assigns +1 for values > 0
# sign function assigns -1 for values < 0
# sign function assigns 0 for values == 0

norm_score <- cbind(
  sign(syuzhet_score), 
  sign(bing_score), 
  sign(afinn_score),
  sign(nrc_score))

dimnames(norm_score)<-list(1:nrow(norm_score), c("x1", "x2", "x3", "x4"))
head(norm_score)

z<-as.data.frame(norm_score)
head(z,20)

round(prop.table(table(z$x1)),2)    #syuzhet score


#---------------------------------------------------
#  Emotion classification & positive, negative and neutralsentiments

#  "Emotion classification is built on the 
#  NRC Word-Emotion Association Lexicon (aka EmoLex)"

#  "The NRC Emotion Lexicon is a list of English words and 
#  their associations with eight basic emotions (anger, fear, 
#  anticipation, trust, surprise, sadness, joy, and disgust) and 
#  two sentiments (negative and positive)"


# Emotion classification & positive, negative, and neutral sentiments

#______________________________________________________________________________________________

#Let's calculate sentiment using the other three methods and do some comparism

# Load the required libraries
library(tidytext)
library(dplyr)
library(ggplot2)
library(reshape2)


# Perform sentiment analysis using the syuzhet method
syuzhet_scores <- get_sentiment(review_text, method = "syuzhet")
syuzhet_sentiments <- ifelse(syuzhet_scores > 0, "positive", ifelse(syuzhet_scores < 0, "negative", "neutral"))

# Perform sentiment analysis using the bing method
bing_scores <- get_sentiment(review_text, method = "bing")
bing_sentiments <- ifelse(bing_scores > 0, "positive", ifelse(bing_scores < 0, "negative", "neutral"))

# Perform sentiment analysis using the afinn method
afinn_scores <- get_sentiment(review_text, method = "afinn")
afinn_sentiments <- ifelse(afinn_scores > 0, "positive", ifelse(afinn_scores < 0, "negative", "neutral"))

# Perform sentiment analysis using the nrc method
nrc_scores <- get_sentiment(review_text, method = "nrc")
nrc_sentiments <- ifelse(nrc_scores > 0, "positive", ifelse(nrc_scores < 0, "negative", "neutral"))

# Combine the sentiment scores and sentiments into a data frame
sentiment_data <- data.frame(Syuzhet = syuzhet_scores, Bing = bing_scores, Afinn = afinn_scores, NRC = nrc_scores,
                             Syuzhet_Sentiment = syuzhet_sentiments, Bing_Sentiment = bing_sentiments,
                             Afinn_Sentiment = afinn_sentiments, NRC_Sentiment = nrc_sentiments)

# Plot sentiment distribution for each method
plot_syuzhet <- ggplot(sentiment_data, aes(Syuzhet_Sentiment, fill = Syuzhet_Sentiment)) +
  geom_bar() +
  labs(title = "Sentiment Distribution - Syuzhet Method")

plot_bing <- ggplot(sentiment_data, aes(Bing_Sentiment, fill = Bing_Sentiment)) +
  geom_bar() +
  labs(title = "Sentiment Distribution - Bing Method")

plot_afinn <- ggplot(sentiment_data, aes(Afinn_Sentiment, fill = Afinn_Sentiment)) +
  geom_bar() +
  labs(title = "Sentiment Distribution - Afinn Method")

plot_nrc <- ggplot(sentiment_data, aes(NRC_Sentiment, fill = NRC_Sentiment)) +
  geom_bar() +
  labs(title = "Sentiment Distribution - NRC Method")

# Display the individual sentiment plots and the combined plot
print(plot_syuzhet)
print(plot_bing)
print(plot_afinn)
print(plot_nrc)

#BAR PLOT

# Load the required libraries
library(ggplot2)


# Create a data frame with the sentiment results from the four methods
sentiment_data <- data.frame(
  Method = c(rep("Syuzhet", length(syuzhet_scores)),
             rep("Bing", length(bing_scores)),
             rep("Afinn", length(afinn_scores)),
             rep("NRC", length(nrc_scores))),
  Score = c(syuzhet_scores, bing_scores, afinn_scores, nrc_scores)
)

# Create the box plot
plot_sentiment <- ggplot(sentiment_data, aes(x = Method, y = Score, fill = Method)) +
  geom_boxplot() +
  labs(title = "Sentiment Results Comparison", x = "Method", y = "Sentiment Score") +
  theme_minimal()

# Display the box plot
print(plot_sentiment)



#WORD CLOUD

# Load the required libraries
library(wordcloud)
library(tm)

# Create a corpus from the review text
corpus <- Corpus(VectorSource(review_text))

# Preprocess the corpus
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
corpus <- tm_map(corpus, stripWhitespace)

# Filter out empty documents
corpus <- corpus[!sapply(corpus, function(doc) length(doc) == 0)]

# Load the required libraries
library(tidytext)
library(dplyr)

# Create a data frame with the review text
data <- data.frame(review_text)

# Tokenize the text and calculate word frequencies
word_freq <- data %>%
  unnest_tokens(word, review_text) %>%
  count(word, sort = TRUE)

# Print the top 10 words by frequency
top_words <- head(word_freq, 10)
print(top_words)

# Create a data frame with 'Words' and 'Frequency' as headers for the top 20 words
word_freq_table <- data.frame(Words = word_freq$word[1:20], Frequency = word_freq$n[1:20])

# Print the table
print(word_freq_table)


# Create word cloud for Syuzhet method
wordcloud_syuzhet <- wordcloud(review_text[syuzhet_scores > 0], max.words = 100,
                               random.order = FALSE, colors = brewer.pal(8, "Dark2"))

# Create word cloud for Bing method
wordcloud_bing <- wordcloud(review_text[bing_scores > 0], max.words = 100,
                            random.order = FALSE, colors = brewer.pal(8, "Dark2"))

# Create word cloud for Afinn method
wordcloud_afinn <- wordcloud(review_text[afinn_scores > 0], max.words = 100,
                             random.order = FALSE, colors = brewer.pal(8, "Dark2"))

# Create word cloud for NRC method
wordcloud_nrc <- wordcloud(review_text[nrc_scores > 0], max.words = 100,
                           random.order = FALSE, colors = brewer.pal(8, "Dark2"))

# Combine the four word clouds
combined_wordcloud <- wordcloud(c(review_text[syuzhet_scores > 0],
                                  review_text[bing_scores > 0],
                                  review_text[afinn_scores > 0],
                                  review_text[nrc_scores > 0]), max.words = 100,scale=c(4, 0.5),
                                random.order = FALSE, colors = brewer.pal(8, "Dark2"))

# Display the individual word clouds and the combined word cloud
print(wordcloud_syuzhet)
print(wordcloud_bing)
print(wordcloud_afinn)
print(wordcloud_nrc)
print(combined_wordcloud)


#SAVE THE SENTIMENT RESULT

# Create a data frame with the sentiment scores
sentiment_result <- data.frame(syuzhet_scores, bing_scores, afinn_scores, nrc_scores)

# Save the data frame as a CSV file
write.csv(sentiment_result, file = "lexicon_result.csv", row.names = FALSE)


#LINE PLOT

# Create a data frame with the sentiment scores
lineplot_data <- data.frame(syuzhet_scores, bing_scores, afinn_scores, nrc_scores)

# Compute the total number of positive, negative, and neutral sentiments
total_sentiments <- data.frame(
  Positive = colSums(lineplot_data > 0),
  Negative = colSums(lineplot_data < 0),
  Neutral = colSums(lineplot_data == 0)
)

# Reshape the data to long format
total_sentiments_long <- tidyr::pivot_longer(total_sentiments, cols = c(Positive, Negative, Neutral),
                                             names_to = "Sentiment", values_to = "Count")

# Add a Method column to the data frame
total_sentiments_long$Method <- rep(names(lineplot_data), each = 3)

# Define custom colors for the lines
line_colors <- c("Positive" = "#00BFC4", "Negative" = "#F8766D", "Neutral" = "#619CFF")

# Create the line plot
ggplot(total_sentiments_long, aes(x = Method, y = Count, color = Sentiment, group = Sentiment)) +
  geom_line(size = 1.5) +
  geom_point(size = 3, shape = 21, fill = "white") +
  scale_color_manual(values = line_colors) +
  theme_minimal() +
  labs(x = "Method", y = "Count", color = "Sentiment") +
  ggtitle("Comparison of Total Sentiments by Method") +
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        legend.position = c(0.9, 0.9),
        legend.justification = c(0.5, 0.9),
        legend.title = element_blank(),
        legend.text = element_text(size = 12),
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 14),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())


#assignning sentiment labels to the results

# Assuming you have the review_text variable containing the text data

library(syuzhet)
library(sentimentr)
library(lexicon)
library(tidytext)

# Function to assign sentiment labels based on the sentiment scores
assign_sentiment_labels <- function(scores) {
  labels <- ifelse(scores > 0, "positive", ifelse(scores < 0, "negative", "neutral"))
  return(labels)
}

# Assign sentiment labels for each method
syuzhet_labels <- assign_sentiment_labels(syuzhet_scores)
bing_labels <- assign_sentiment_labels(bing_scores)
afinn_labels <- assign_sentiment_labels(afinn_scores)
nrc_labels <- assign_sentiment_labels(nrc_scores)

# Combine the sentiment scores and labels into a data frame
sentiment_data <- data.frame(
  Review_Text = review_text,
  Syuzhet_Score = syuzhet_scores,
  Syuzhet_Label = syuzhet_labels,
  Bing_Score = bing_scores,
  Bing_Label = bing_labels,
  Afinn_Score = afinn_scores,
  Afinn_Label = afinn_labels,
  NRC_Score = nrc_scores,
  NRC_Label = nrc_labels
)

# Print the sentiment data
head(sentiment_data)

#Top 10 words

# Convert the sentiment scores into sentiment labels
sentiment_labels <- data.frame(
  Syuzhet_Sentiment = ifelse(syuzhet_scores > 0, "positive", ifelse(syuzhet_scores < 0, "negative", "neutral")),
  Bing_Sentiment = ifelse(bing_scores > 0, "positive", ifelse(bing_scores < 0, "negative", "neutral")),
  Afinn_Sentiment = ifelse(afinn_scores > 0, "positive", ifelse(afinn_scores < 0, "negative", "neutral")),
  NRC_Sentiment = ifelse(nrc_scores > 0, "positive", ifelse(nrc_scores < 0, "negative", "neutral"))
)

# Create a new dataframe with the appropriate number of rows
num_rows <- min(nrow(text), nrow(sentiment_labels))
sentiment_data <- data.frame(
  text = text$text[1:num_rows],
  sentiment_labels[1:num_rows, ]
)

# Get the top 10 words for each sentiment in each method
top_words <- sentiment_data %>%
  unnest_tokens(word, text) %>%
  count(Syuzhet_Sentiment, word, sort = TRUE) %>%
  group_by(Syuzhet_Sentiment) %>%
  slice_max(n = 10, order_by = n) %>%
  ungroup()

# Plot the top 10 words for each sentiment in each method
plot_top_words <- ggplot(top_words, aes(word, n, fill = Syuzhet_Sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ Syuzhet_Sentiment, scales = "free_y") +
  labs(y = "Count", x = NULL) +
  coord_flip()

print(plot_top_words)
# Save the sentiment result as a CSV file
write.csv(sentiment_data, file = "sentiment _results_with_labels.csv", row.names = FALSE)
