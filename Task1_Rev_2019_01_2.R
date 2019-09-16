## Data Science Capstone Project
## Coursera-Swiftkey Prediction Model
## getcleandata.R

## 01 LOAD DATA ########################################################################
## Load libraries

#library(rJava)
library(stringi)
library(stringr)
library(caret)
library(tm)
library(qdap)
library(wordcloud)
library(ngram)
library(data.table)
library(quanteda)
library(magrittr)
library(ggplot2)
library(dplyr)
#library(RWeka)

## Set seed and working directory

set.seed(20190827)
#setwd("~/Documents/GIT/Coursera/MODULE10")
setwd("~/900_Personale/300_My_Training/100_Coursera/300_Data_Science/MODULE10/TASK1")

## Get data

# files path
file.blog      <- "en_US/en_US.blogs.txt"
file.news      <- "en_US/en_US.news.txt"
file.twitter   <- "en_US/en_US.twitter.txt"
file.profanity <- "profanity.csv"
file.abbreviations <- "abbreviations.csv"
file.short <- "Short_form.csv"

# dimension of files
size.blog    <- round(file.info(file.blog)$size / 1024^2 ,2)
size.news    <- round(file.info(file.news)$size / 1024^2 ,2)
size.twitter <- round(file.info(file.twitter)$size / 1024^2 ,2)
size.total   <- size.blog + size.news + size.twitter

txt.vector.blog      <- readLines(file.blog,encoding="UTF-8", skipNul = TRUE)
txt.vector.news      <- readLines(file.news,encoding="UTF-8", skipNul = TRUE)
txt.vector.twitter   <- readLines(file.twitter,encoding="UTF-8", skipNul = TRUE)
data.profanity       <- read.csv(file.profanity, header = F)
data.abbreviation    <- read.csv(file.abbreviations, header = T, sep=";")
data.short           <- read.csv(file.short, header = T, sep=";")

## Count lines in sources
rows.blog    <- length(txt.vector.blog)
rows.news    <- length(txt.vector.news)
rows.twitter <- length(txt.vector.twitter)
rows.total   <- rows.blog + rows.news + rows.twitter

## Count words in sources
## Require *** library stringi
word.count.blog    <- stri_count_words(txt.vector.blog)
word.count.news    <- stri_count_words(txt.vector.news)
word.count.twitter <- stri_count_words(txt.vector.twitter)

word.sum.blog    <- sum(word.count.blog)
word.sum.news    <- sum(word.count.news)
word.sum.twitter <- sum(word.count.twitter)
word.sum.total   <- sum(word.sum.blog + word.sum.news + word.sum.twitter)

## Summarize sources
source.dimen <- c(size.blog, size.news, size.twitter, size.total)
source.lines <- c(rows.blog, rows.news, rows.twitter, rows.total)
source.words <- c(word.sum.blog, word.sum.news, word.sum.twitter, word.sum.total)
source.words.lines <- c(word.sum.blog/rows.blog,
                        word.sum.news/rows.news,
                        word.sum.twitter/rows.twitter,
                        word.sum.total/rows.total
)

source.summary <- data.frame(source.dimen, source.lines, source.words, source.words.lines)
rownames(source.summary) <- c("Blogs", "News", "Twitter", "Total")
colnames(source.summary) <- c("Dimension_file", "Number_of_Lines", "Number_of_Words", "Words_per_Line")
source.summary

## 02 Sample, Splitting and Cleaning ######################

## Sample and Split Data


## Sampling factor for sample size = sampling.factor * total number of lines
sampling.factor = 0.01

## Sample and combine the sources with a sampling factor
sample.blog <- sample(txt.vector.blog, size=length(txt.vector.blog)*sampling.factor, replace=FALSE)
sample.news <- sample(txt.vector.news, size=length(txt.vector.news)*sampling.factor, replace=FALSE)
sample.twitter <- sample(txt.vector.twitter, size=length(txt.vector.twitter)*sampling.factor, replace=FALSE)

## Combine samples and permutate
rawData <- c(sample.blog, sample.news, sample.twitter)
rawData <- sample(rawData, size=length(rawData), replace=FALSE)

## Training ratio
train.Percent <- 0.6
validate.Percent <- 0.2
test.Percent <- 1 - train.Percent - validate.Percent
test.Split <- test.Percent
train.Split <- train.Percent/(train.Percent + validate.Percent)

## Split the raw data into test and non-test data sets
## *** Require library caret
inTest <- createDataPartition(seq_len(NROW(rawData)), p=test.Split , list=FALSE)
rawTestData <- rawData[inTest]
rawNonTestData <- rawData[-inTest]

## Split the non-test data into training and validation data sets
inTrain <- createDataPartition(seq_len(NROW(rawNonTestData)), p=train.Split,
                               list=FALSE)
rawTrainData <- rawNonTestData[inTrain]
rawValidateData <- rawNonTestData[-inTrain]

## Count lines in data sets
trainLines <- length(rawTrainData)
validateLines <- length(rawValidateData)
testLines <- length(rawTestData)
rawLines <- trainLines + validateLines + testLines

## Count words in data sets
## Require library stringi
trainWords <- sum(stri_count_words(rawTrainData))
validateWords <- sum(stri_count_words(rawValidateData))
testWords <- sum(stri_count_words(rawTestData))
rawWords <- trainWords + validateWords + testWords

## Summarize data sets
datasetLines <- c(trainLines, validateLines, testLines, rawLines)
datasetWords <- c(trainWords, validateWords, testWords, rawWords)
datasetWordsPerLine <- c(trainWords/trainLines,
                         validateWords/validateLines,
                         testWords/testLines,
                         rawWords/rawLines
)
datasetSummary <- data.frame(datasetLines, datasetWords, datasetWordsPerLine)
rownames(datasetSummary) <- c("Training", "Validation", "Testing", "Total")
colnames(datasetSummary) <- c("Number_of_Lines", "Number_of_Words", "Words_per_Line")
datasetSummary

## Save data to files and remove auxiliary data
#save(blogs, news, twitter, sourceSummary, file="originalData.rda")
#rm(inTest, inTrain, blogs, news, twitter)

## CLEAN DATA

## preprocessData: clean a corpus by converting its text to plain text document
## and lower case, replacing contractions with their full forms, and
## removing profanities, numbers and punctuation. Removing English stopwords
## is optional.
##
## rawCorpus = input text to be preprocessed.
## isStopWordsRemoved = FALSE, set to TRUE to remove stopwords
## cleanCorpus = output preprocessed corpus
##

preprocessData <- function(rawCorpus, bannedWords, isStopWordsRemoved=FALSE) {
  
  if (!require(tm)) {
    stop("Library tm is missing.")
  }
  ## Convert to plain text
  cleanCorpus <- tm_map(rawCorpus, PlainTextDocument)
  
  ## Replace contractions with their full form using qdap dictionary
  ## from https://trinkerrstuff.wordpress.com/my-r-packages/qdap/
  if (!require(qdap)) {
    stop("Library qdap is missing.")
  }
  cleanCorpus <- tm_map(cleanCorpus, content_transformer(replace_contraction))
  
  remove.decimals     <- function(x) {gsub("([0-9]*)\\.([0-9]+)", "\\1 \\2", x)}
  remove.hashtags     <- function(x) {gsub("#[a-zA-z0-9]+", " ", x)}
  remove.noneng       <- function(x) {gsub("\\W+", " ",x)}
  
  cleanCorpus <- tm_map(cleanCorpus, remove.decimals)
  cleanCorpus <- tm_map(cleanCorpus, remove.noneng)
  cleanCorpus <- tm_map(cleanCorpus, remove.hashtags)
  ## Remove numbers, puntuation and strip white space
  cleanCorpus <- tm_map(cleanCorpus, removeNumbers)
  cleanCorpus <- tm_map(cleanCorpus, stripWhitespace)
  cleanCorpus <- tm_map(cleanCorpus, removePunctuation)
  cleanCorpus <- tm_map(cleanCorpus, tolower)
  
  ## Remove profanities defined in the banned word list
  cleanCorpus <- tm_map(cleanCorpus, removeWords, bannedWords)
  
  ## Remove stopwords
  if (isStopWordsRemoved) {
    cleanCorpus <- tm_map(cleanCorpus, removeWords, stopwords("en"))
  }
  
  return(cleanCorpus)
}


## Get data WITH English stopwords

## Convert training data to corpus and inspect a few documents
trainCorpus <- SimpleCorpus(VectorSource(rawTrainData))
inspect(trainCorpus[c(1,length(trainCorpus)%/%2, length(trainCorpus)-20)])

## Clean the training, validate, and test corpuses
trainCorpus <- preprocessData(trainCorpus, data.profanity$V1,
                                   isStopWordsRemoved=FALSE)
inspect(trainCorpus[c(1,length(trainCorpus)%/%2, length(trainCorpus)-20)])

validateCorpus <- SimpleCorpus(VectorSource(rawValidateData))
validateCorpus <- preprocessData(validateCorpus, data.profanity$V1,
                                 isStopWordsRemoved=FALSE)

testCorpus <- SimpleCorpus(VectorSource(rawTestData))
testCorpus <- preprocessData(testCorpus, data.profanity$V1,
                             isStopWordsRemoved=FALSE)



## Get data WITHOUT English stopwords


## Convert training data to corpus and inspect a few documents
trainNwsCorpus <- SimpleCorpus(VectorSource(rawTrainData))
inspect(trainNwsCorpus[c(1,length(trainNwsCorpus)%/%2, length(trainNwsCorpus)-20)])

## Clean the training, validate, and test corpuses
trainNwsCorpus <- preprocessData(trainNwsCorpus, data.profanity$V1,
                                 isStopWordsRemoved=TRUE)
inspect(trainNwsCorpus[c(1,length(trainNwsCorpus)%/%2, length(trainNwsCorpus)-20)])

validateNwsCorpus <- SimpleCorpus(VectorSource(rawValidateData))
validateNwsCorpus <- preprocessData(validateNwsCorpus, data.profanity$V1,
                                    isStopWordsRemoved=TRUE)

testNwsCorpus <- SimpleCorpus(VectorSource(rawTestData))
testNwsCorpus <- preprocessData(testNwsCorpus, data.profanity$V1,
                                isStopWordsRemoved=TRUE)


## 03 Generate Ngrams ####################################################
## Generate Ngrams and Explore the data WITH stopwords

## getTermFrequency: get frequencies of terms in a corpus, in decreasing order.
##
## corpus = input a corpus
## sparse = input a sparse fraction 0 to 1
## freqDf = output a data frame of terms and frequencies
##
getTermFrequency <- function(corpus, sparse) {
  if (!require(tm)) {
    stop("Library tm is missing.")
  }
  ## Convert to DocumnetTermMatrix
  dtm <- DocumentTermMatrix(corpus)
  
  ## Get frequencies of terms
  if (sparse >= 1) {
    freq <- colSums(as.matrix(dtm))
  } else {
    freq <- colSums(as.matrix(removeSparseTerms(dtm, sparse=sparse)))
  }
  
  ## Sort in decreasing order and output
  freq <- sort(freq, decreasing=TRUE)
  freqDf <- data.frame(ngram=names(freq), freq=freq)
}

## getNgram: extracts ngrams with the ngram library
## corpus = input, a SimpleCorpus
## n = input, number of word in ngrams
## ng = output, a dataframe of ngram, freq, and probability
##
getNgram <- function(corpus, n=2) {
  if (!require(ngram)) {
    stop("Library ngram is missing.")
  }
  
  ## Convert corpus to a string
  str <- concatenate(corpus$content)
  ng <- ngram(str, n=n)
  return(get.phrasetable(ng))
}

## summaryNgram: summarizes the ngram generated by getNgram
## ngram = input, a dataframe from getNgram
## summaryTable = output, a dataframe of number of ngrams 
##   at frequencies >= 1, 2, 3, and 4
##
summaryNgram <- function(ngram) {
  freqNWords <- function(n) sum(ngram$freq >= n)
  freqNOccurence <- function(n) sum(ngram[ngram$freq >= n, "freq"])
  n2 = 10
  n3 = 25
  n4 = 50
  n5 = 100
  totalWords <- dim(ngram)[1]
  freq2Words <- freqNWords(n2)
  freq3Words <- freqNWords(n3)
  freq4Words <- freqNWords(n4)
  freq5Words <- freqNWords(n5)
  numWords <- c(totalWords, freq2Words, freq3Words, freq4Words, freq5Words)
  fracWords <- c(1.0, round(freq2Words/totalWords, 4),
                 round(freq3Words/totalWords, 4), 
                 round(freq4Words/totalWords, 4),
                 round(freq5Words/totalWords, 4))
  totalOccurrence <- sum(ngram$freq)
  freq2Occurrence <- freqNOccurence(n2)
  freq3Occurrence <- freqNOccurence(n3)
  freq4Occurrence <- freqNOccurence(n4)
  freq5Occurrence <- freqNOccurence(n5)
  probWords <- c(1.0, round(freq2Occurrence/totalOccurrence, 4),
                 round(freq3Occurrence/totalOccurrence, 4),
                 round(freq4Occurrence/totalOccurrence, 4),
                 round(freq5Occurrence/totalOccurrence, 4))
  summaryData <- data.frame(numWords, fracWords, probWords)
  colnames(summaryData) <- c("Number_of_Words", "Fraction_of_Total", "Probability")
  rownames(summaryData) <- c("All", 
                             paste("Freq >= ", as.character(n2), collapse=''),
                             paste("Freq >= ", as.character(n3), collapse=''),
                             paste("Freq >= ", as.character(n4), collapse=''),
                             paste("Freq >= ", as.character(n5), collapse=''))
  
  return(summaryData)
}

##*****************************************************
## Generate Ngrams and Explore the data WITH stopwords
##*****************************************************

system.time(train1Gram <- getNgram(trainCorpus, n=1))
summaryNgram(train1Gram)
wordcloud(train1Gram$ngram, train1Gram$freq, scale=c(7, 1), 
          max.words=20, random.order=FALSE)
train1Gram.data <- data.frame(ngram=train1Gram$ngrams[1:30],freq=train1Gram$freq[1:30])
p <- ggplot(train1Gram.data, aes(y=freq, x=reorder(ngram,freq))) +
  geom_bar(stat='identity') +
  coord_flip() +
  theme_bw() +
  labs(y = "Word Counts", x = "Top Words")
ggtitle(paste("Top"))
p

system.time(train2Gram <- getNgram(trainCorpus, n=2))
summaryNgram(train2Gram)
wordcloud(train2Gram$ngram, train2Gram$freq, scale=c(3, 1), 
          max.words=20, random.order=FALSE)
train2Gram.data <- data.frame(ngram=train2Gram$ngrams[1:30],freq=train2Gram$freq[1:30])
p <- ggplot(train2Gram.data, aes(y=freq, x=reorder(ngram,freq))) +
  geom_bar(stat='identity') +
  coord_flip() +
  theme_bw() +
  labs(y = "Word Counts", x = "Top Words")
ggtitle(paste("Top"))
p

system.time(train3Gram <- getNgram(trainCorpus, n=3))
summaryNgram(train3Gram)
wordcloud(train3Gram$ngram, train3Gram$freq, scale=c(1, 1), 
          max.words=20, random.order=FALSE)
train3Gram.data <- data.frame(ngram=train3Gram$ngrams[1:30],freq=train3Gram$freq[1:30])
p <- ggplot(train3Gram.data, aes(y=freq, x=reorder(ngram,freq))) +
  geom_bar(stat='identity') +
  coord_flip() +
  theme_bw() +
  labs(y = "Word Counts", x = "Top Words")
ggtitle(paste("Top"))
p

## Get validate and test ngrams
validate1Gram <- getNgram(validateCorpus, n=1)
validate2Gram <- getNgram(validateCorpus, n=2)
validate3Gram <- getNgram(validateCorpus, n=3)

test1Gram <- getNgram(testCorpus, n=1)
test2Gram <- getNgram(testCorpus, n=2)
test3Gram <- getNgram(testCorpus, n=3)

##*****************************************************
## Generate Ngrams and Explore the data WITHOUT stopwords
##*****************************************************

system.time(trainNws1Gram <- getNgram(trainNwsCorpus, n=1))
summaryNgram(trainNws1Gram)
wordcloud(trainNws1Gram$ngram, trainNws1Gram$freq, scale=c(7, 1), 
          max.words=20, random.order=FALSE)
trainNws1Gram <- data.frame(ngram=trainNws1Gram$ngrams[1:30],freq=trainNws1Gram$freq[1:30])
p <- ggplot(trainNws1Gram, aes(y=freq, x=reorder(ngram,freq))) +
  geom_bar(stat='identity') +
  coord_flip() +
  theme_bw() +
  labs(y = "Word Counts", x = "Top Words")
ggtitle(paste("Top"))
p

system.time(trainNws2Gram <- getNgram(trainNwsCorpus, n=2))
summaryNgram(trainNws2Gram)
wordcloud(trainNws2Gram$ngram, trainNws2Gram$freq, scale=c(3, 1), 
          max.words=20, random.order=FALSE)
trainNws2Gram.data <- data.frame(ngram=trainNws2Gram$ngrams[1:30],freq=trainNws2Gram$freq[1:30])
p <- ggplot(trainNws2Gram.data, aes(y=freq, x=reorder(ngram,freq))) +
  geom_bar(stat='identity') +
  coord_flip() +
  theme_bw() +
  labs(y = "Word Counts", x = "Top Words")
ggtitle(paste("Top"))
p

system.time(trainNws3Gram <- getNgram(trainNwsCorpus, n=3))
summaryNgram(trainNws3Gram)
wordcloud(trainNws3Gram$ngram, trainNws3Gram$freq, scale=c(1, 1), 
          max.words=20, random.order=FALSE)
trainNws3Gram.data <- data.frame(ngram=trainNws3Gram$ngrams[1:30],freq=trainNws3Gram$freq[1:30])
p <- ggplot(trainNws3Gram.data, aes(y=freq, x=reorder(ngram,freq))) +
  geom_bar(stat='identity',fill = "#FF6666") +
  theme(axis.text.x = element_text(angle = 90))+
  labs(y = "Ngram Counts", x = "Top Ngram")
ggtitle(paste("Top"))
p + scale_x_reverse()


validateNws1Gram <- getNgram(validateNwsCorpus, n=1)
validateNws2Gram <- getNgram(validateNwsCorpus, n=2)
validateNws3Gram <- getNgram(validateNwsCorpus, n=3)

testNws1Gram <- getNgram(testNwsCorpus, n=1)
testNws2Gram <- getNgram(testNwsCorpus, n=2)
testNws3Gram <- getNgram(testNwsCorpus, n=3)




## 04 Plots --------------------------------------

##************
## Word clouds
##************

## Set up plot for word clouds of unigrams
layout(matrix(c(1, 2), nrow=1), heights=c(4, 1))
par(mar=c(1, 1, 1, 1))

## Word cloud of unigrams WITH stopwords
wordcloud(train1Gram$ngram, train1Gram$freq, scale=c(7, 1), 
          max.words=20, random.order=FALSE, main="Title")
mtext("Unigrams of training data with English stopwords", font=2)

## Word cloud of unigrams WITHOUT stopwords
wordcloud(trainNws1Gram$ngram, trainNws1Gram$freq, scale=c(5, 1), 
          max.words=20, random.order=FALSE, main="Title")
mtext("Unigrams of training data without English stopwords", font=2)


## Set up plot for word clouds of bigrams
layout(matrix(c(1, 2), nrow=1), heights=c(4, 1))
par(mar=c(1, 1, 1, 1))

## Word cloud of bigrams WITH stopwords
wordcloud(train2Gram$ngram, train2Gram$freq, scale=c(3.5, 1), 
          max.words=20, random.order=FALSE)
mtext("Bigrams of training data with English stopwords", font=2)

## Word cloud of bigrams WITHOUT stopwords
wordcloud(trainNws2Gram$ngram, trainNws2Gram$freq, scale=c(3, 1), 
          max.words=20, random.order=FALSE)
mtext("Bigrams of training data without English stopwords", font=2)


## Set up plot for word clouds for trigrams
layout(matrix(c(1, 2), nrow=1), heights=c(4, 1))
par(mar=c(1, 1, 1, 1))

## Word cloud of trigrams WITH stopwords
wordcloud(train3Gram$ngram, train3Gram$freq, scale=c(2.6, 1), 
          max.words=20, random.order=FALSE)
mtext("Trigrams of training data with English stopwords", font=2)

## Word cloud of trigrams WITHOUT stopwords
wordcloud(trainNws3Gram$ngram, trainNws3Gram$freq, scale=c(2.0, 1), 
          max.words=20, random.order=FALSE)
mtext("Trigrams of training data without English stopwords", font=2)

##************
## Density plots
##************

##???TO BE REVIEWED


## 05 ------------------ ########################

## Get frequencies smoothed by Simple Good Turing method
## by Willian A. Gale
## https://faculty.cs.byu.edu/~ringger/CS479/papers/Gale-SimpleGoodTuring.pdf

## freqSimpleGT = data table of pairs(freq, freqGT) with key = freq
## the first row is the pair(0, freqGT_at_0) for the unobserved instance.
## 

simpleGT <- function(freqhist) {
  
  ## nrzest: Averaging transformation
  ## Replace nr by zr = nr/(0.5*(t - q))
  ## where q, r, t are successive indices of non-zero values
  ## in gtfuncs.S by William A. Gale
  ##
  nrzest <- function(r, nr) {
    d <- c(1, diff(r))
    dr <- c(0.5 * (d[-1] + d[-length(d)]), d[length(d)])
    return(nr/dr)
  }
  
  ## rstest: Linear Good-Turing estimate
  ## log(nr) = a + b * log(r)
  ## b = coef[2]
  ## rstest r(star)est = r *(1 + 1/r)^(b + 1)
  ## b < -1
  ## in gtfuncs.S by William A. Gale
  ##
  rstest <- function(r, coef) {
    return(r * (1 + 1/r)^(1 + coef[2]))
  }
  
  ## The following code comes from gtanal.S by William A. Gale
  
  ## Get the input xr and xnr    
  xm <- freqhist
  xr <- xm[, 1]
  xnr <- xm[, 2]
  xN <- sum(xr * xnr)
  
  ## make averaging transform
  xnrz <- nrzest(xr, xnr)
  
  ## get Linear Good-Turing estimate
  xf <- lsfit(log(xr), log(xnrz))
  xcoef <- xf$coef
  xrst <- rstest(xr, xcoef)
  xrstrel <- xrst/xr
  
  ## get Turing estimate
  xrtry <- xr == c(xr[-1]-1, 0)
  xrstarel <- rep(0, length(xr))
  xrstarel[xrtry] <- (xr[xrtry]+1) / xr[xrtry] * c(xnr[-1], 0)[xrtry] / xnr[xrtry]
  
  ## make switch from Turing to LGT estimates
  tursd <- rep(1, length(xr))
  for (i in 1:length(xr)) {
    tursd[i] <- (i+1) / xnr[i] * sqrt(xnr[i=1] * (1 + xnr[i+1] / xnr[i]))
  }
  xrstcmbrel <- rep(0, length(xr))
  useturing <- TRUE
  for (r in 1:length(xr)) {
    if (!useturing) {
      xrstcmbrel[r] <- xrstrel[r]
    } else if (abs(xrstrel - xrstarel)[r] * r / tursd[r] > 1.65) {
      xrstcmbrel[r] <- xrstarel[r]
    } else {
      useturing <- FALSE
      xrstcmbrel[r] <- xrstrel[r]
    }
  }
  
  ## renormalize the probabilities for observed objects
  sumpraw <- sum(xrstcmbrel * xr * xnr / xN)
  xrstcmbrel <- xrstcmbrel * (1 - xnr[1]/xN) / sumpraw
  
  ## output to file
  # cat(xN, sum(xnr), file="gtanal", sep=",")
  # cat(0, xnr[1]/xN, file="gtanal", sep=",", append=TRUE)
  # for (i in 1:length(xr)) {
  #     cat(xr[i], xr[i]*xrstcmbrel[i], file="gtanal", append=TRUE)
  # }
  
  ## output matrix (0, r0est) + (xr, xnrstarnormalized)
  rrstar <- cbind(c(0, xr), c(xnr[1]/xN, xr*xrstcmbrel))
  
  ## output data table by pairs = (r = freq, rstar = freqGT)
  ## keyed (ordered) by freq.
  rrstar <- data.table(rrstar)
  colnames(rrstar) <- c("freq", "freqGT")
  setkey(rrstar, freq)
  return(rrstar)
}

cleanNgram <- function(ngramDf, minFreq) {
  if (!require(stringr)) {
    stop("Library stringr is missing.")
  }
  
  ## Convert the data frame to data table and rename columns
  colnames(ngramDf) <- c("ngram", "freq", "prob")
  ngramDt <- data.table(ngramDf)
  
  ## Remove ngrams with frequencies below the cutoff minFreq
  ngramDt <- ngramDt[freq >= minFreq]
  
  ## Get frequency (Nr) of frequency r,
  freqNf <- data.table(table(ngramDt[, "freq"]))
  colnames(freqNf) <- c("freq", "NFreq")
  freqNf <- sapply(freqNf, as.numeric)
  
  ## Get frequencies smoothed by Simple Good Turing method
  ## by Willian A. Gale
  ## https://faculty.cs.byu.edu/~ringger/CS479/papers/Gale-SimpleGoodTuring.pdf
  
  ## freqSimpleGT = data table of pairs(freq, freqGT) with key = freq
  ## the first row is the pair(0, freqGT_at_0) for the unobserved instance.
  ## 
  freqSimpleGT <- simpleGT(freqNf)
  
  ## Merge bgramDt with freqSimpleGT with bgramDt
  setkey(ngramDt, freq)
  ngramDt <- merge(ngramDt, freqSimpleGT)
  
  ## Calculate probability of zero frequency
  pZero <- freqSimpleGT[1, 2]/sum(c(1, freqNf[, "NFreq"]) * freqSimpleGT[, 2])
  
  ## Output the clean smoothed ngrams and probability of zero frequency
  ngramAndPzero <- list(ngramDt=ngramDt, pZero=pZero)
  return(ngramAndPzero)
}

split1Gram <- function(ugramDt) {
  if (!require(stringr)) {
    stop("Library stringr is missing.")
  }
  
  ## Trim trailing spaces
  ugramDt[, word1 := str_trim(ngram)]
  setkey(ugramDt, word1)
  
  ## Reset frequencies and calculate words' probability (unsmoothed)
  ugramDt <- ugramDt[, freq := sum(freq), by=c("word1")]
  
  ugramTotalFreq <- sum(ugramDt$freq)
  ugramDt[, prob := freq/ugramTotalFreq]
  
  ## Reset frequencies and calculate words' probability (smoothed)
  ugramDt <- ugramDt[, freqGT := sum(freqGT), by=c("word1")]
  ugramTotalFreqGT <- sum(ugramDt$freqGT)
  ugramDt[, probGT := freqGT/ugramTotalFreqGT]
  
  ## Set key column
  setkey(ugramDt, word1)
  
  ## Reorder the columns in bigrams
  setcolorder(ugramDt, c("ngram", "word1",
                         "freq", "prob", "freqGT", "probGT"))
  return(ugramDt)
}
split2Gram <- function(bgramDt, vocab) {
  if (!require(stringr)) {
    stop("Library stringr is missing.")
  }
  
  ## Split the bigram into words
  bgramSplits <- str_split(bgramDt$ngram, boundary("word"))
  bgramDt[, word1 := sapply(bgramSplits, function(m) m[1])]
  bgramDt[, word2 := sapply(bgramSplits, function(m) m[2])]
  
  ## Set words not in the vocabulary list to <UNK>
  bgramDt[!(word1 %in% vocab), word1 := "<UNK>"]
  bgramDt[!(word2 %in% vocab), word2 := "<UNK>"]
  
  ## Count instances of word1-word2 and word1 by freq (unsmoothed)
  bgramDt[, count_w1_w2 := sum(freq), by=c("word1", "word2")]
  bgramDt[, count_w1 := sum(freq), by=c("word1")]
  
  ## Calculate p(w2|w1) = count(w1,w2)/count(w1)
  bgramDt[, prob := count_w1_w2/count_w1]
  
  ## Count instances of word1-word2 and word1 by freqGT (smoothed)
  bgramDt[, count_w1_w2_GT := sum(freqGT), by=c("word1", "word2")]
  bgramDt[, count_w1_GT := sum(freqGT), by=c("word1")]
  
  ## Calculate p(w2|w1) = count(w1,w2)/count(w1) by freqGT
  bgramDt[, probGT := count_w1_w2_GT/count_w1_GT]
  
  ## Remove temporary columns
  bgramDt[, c("count_w1_w2", "count_w1", "count_w1_w2_GT", "count_w1_GT") := NULL]
  
  ## Set key columns
  setkey(bgramDt, word1, word2)
  
  ## Reorder the columns in bigrams
  setcolorder(bgramDt, c("ngram", "word1", "word2", 
                         "freq", "prob", "freqGT", "probGT"))
  return(bgramDt)
}
split3Gram <- function(tgramDt, vocab) {
  if (!require(stringr)) {
    stop("Library stringr is missing.")
  }
  
  ## Split the bigram into words
  tgramSplits <- str_split(tgramDt$ngram, boundary("word"))
  tgramDt[, word1 := sapply(tgramSplits, function(m) m[1])]
  tgramDt[, word2 := sapply(tgramSplits, function(m) m[2])]
  tgramDt[, word3 := sapply(tgramSplits, function(m) m[3])]
  
  ## Set words not in the vocabulary list to <UNK>
  tgramDt[!(word1 %in% vocab), word1 := "<UNK>"]
  tgramDt[!(word2 %in% vocab), word2 := "<UNK>"]
  tgramDt[!(word3 %in% vocab), word3 := "<UNK>"]
  
  ## Count instances of word1-word2-word3 and word1-word2 by freq (unsmoothed)
  tgramDt[, count_w1_w2_w3 := sum(freq), by=c("word1", "word2", "word3")]
  tgramDt[, count_w1_w2 := sum(freq), by=c("word1", "word2")]
  
  ## Calculate p(w3|w1w2) = count(w1,w2,w3)/count(w1,w2)
  tgramDt[, prob := count_w1_w2_w3/count_w1_w2]
  
  ## Count instances of word1-word2-word3 and word1-word2 by freqGT (smoothed)
  tgramDt[, count_w1_w2_w3_GT := sum(freqGT), by=c("word1", "word2", "word3")]
  tgramDt[, count_w1_w2_GT := sum(freqGT), by=c("word1", "word2")]
  
  ## Calculate p(w2|w1) = count(w1,w2)/count(w1) by freqGT
  tgramDt[, probGT := count_w1_w2_w3_GT/count_w1_w2_GT]
  
  ## Remove temporary columns
  tgramDt[, c("count_w1_w2_w3", "count_w1_w2", 
              "count_w1_w2_w3_GT", "count_w1_w2_GT") := NULL]
  setkey(tgramDt, word1, word2, word3)
  
  ## Reorder the columns in bigrams
  setcolorder(tgramDt, c("ngram", "word1", "word2", "word3", 
                         "freq", "prob", "freqGT", "probGT"))
  return(tgramDt)
}

unigram <- function(ugramDf, minFreq) {
  ugramPzero <- cleanNgram(ugramDf, minFreq)
  ugramPzero$ngramDt <- split1Gram(ugramPzero$ngramDt)
  return(ugramPzero)
}
bigram <- function(bgramDf, minFreq, vocab) {
  bgramPzero <- cleanNgram(bgramDf, minFreq)
  bgramPzero$ngramDt <- split2Gram(bgramPzero$ngramDt, vocab)
  return(bgramPzero)
}
trigram <- function(tgramDf, minFreq, vocab) {
  tgramPzero <- cleanNgram(tgramDf, minFreq)
  tgramPzero$ngramDt <- split3Gram(tgramPzero$ngramDt, vocab)
  return(tgramPzero)
}

biNextWords <- function(bgramDt, newWord1) {
  bNextWords <- subset(bgramDt, word1==newWord1)
  bNextWords <- bNextWords[word2 != "<UNK>", ]
  return(bNextWords)
}
triNextWords <- function(tgramDt, newWord1, newWord2) {
  tNextWords <- subset(tgramDt, word1==newWord1 & word2==newWord2)
  tNextWords <- tNextWords[word3 != "<UNK>", ]
  return(tNextWords)
}

bimodel <- function(lastWord, coef, ugramPzero, bgramPzero, tgramPzero) {
  newWord1 <- lastWord
  
  if ("<UNK>" %in% newWord1) return(data.table())
  
  ## Get bigrams
  bNextWords <- biNextWords(bgramPzero$ngramDt, newWord1)
  if (dim(bNextWords)[1] > 0) {
    ## Get probabilities of unigrams = nextWord
    setkey(bNextWords, word2)
    setkey(ugramPzero$ngramDt, word1)
    bNextWords <- bNextWords[ugramPzero$ngramDt, nomatch=0L]
    names(bNextWords) <- gsub("i.", "u", names(bNextWords))
    
    ## Add probability of trigram at zero frequency
    bNextWords[, tprobGT := tgramPzero$pZero]
    
    ## Calculate trigram probabilities
    bNextWords[, predictProb := coef[1]*uprobGT + coef[2]*probGT 
               + coef[3]*tprobGT]
    
    ## Sort predicted probabilities in decreasing order
    setorder(bNextWords, -predictProb)
    predictions <- bNextWords
  } else {
    ## Get the most frequent word if trigrams and bigrams not found
    uNextWords <- ugramPzero$ngramDt[order(-probGT)][1]
    
    ## Add probabilities of bigrams and trigrams at zero frequency
    uNextWords[, bprobGT := bgramPzero$pZero]
    uNextWords[, tprobGT := tgramPzero$pZero]
    
    ## Calculate trigram probabilities
    uNextWords[, predictProb := coef[1]*probGT + coef[2]*bprobGT 
               + coef[3]*tprobGT]
    predictions <- uNextWords
  }
  
  return(predictions)
}
trimodel <- function(lastWords, coef, ugramPzero, bgramPzero, tgramPzero, nKeep) {
  newWord1 = lastWords[[1]]
  newWord2 = lastWords[[2]]
  
  if ("<UNK>" %in% newWord2) return(data.table())
  
  ## Get trigrams of two new words
  tNextWords <- triNextWords(tgramPzero$ngramDt, newWord1, newWord2)
  if (nrow(tNextWords) > 0) {
    ## Get probabilities of bigrams = newWord2-nextWord
    setkey(tNextWords, word2, word3)
    setkey(bgramPzero$ngramDt, word1, word2)
    tNextWords <- tNextWords[bgramPzero$ngramDt, nomatch=0L]
    names(tNextWords) <- gsub("i.", "b", names(tNextWords))
    
    ## Get probabilities of unigrams = nextWord
    setkey(tNextWords, word3)
    setkey(ugramPzero$ngramDt, word1)
    tNextWords <- tNextWords[ugramPzero$ngramDt, nomatch=0L]
    names(tNextWords) <- gsub("i.", "u", names(tNextWords))
    
    ## Calculate trigram probabilities as the sum of three weighted 
    ## probabilities: unigram, bigram, and trigram.
    ## The Jelinek & Mercer method
    ## see https://english.boisestate.edu/johnfry/files/2013/04/bigram-2x2.pdf
    ## p'(wi|w_i-2, w_i-1) = c1*p(wi) + c2*p(wi|w_i-1) + c3*p(wi|w_i_2,w_i-1)
    ## 
    ## Suggeted coef=c(0.1, 0.3, 0.6)
    ##
    tNextWords[, predictProb := coef[1]*uprobGT + coef[2]*bprobGT 
               + coef[3]*probGT]
    
    ## Sort predicted probabilities in decreasing order
    setorder(tNextWords, -predictProb)
    predictions <- tNextWords
  } else {
    ## Get bigrams if trigrams not found
    predictions <- bimodel(newWord2, coef, ugramPzero, bgramPzero, tgramPzero)
  }
  
  if (nrow(predictions) > nKeep) {
    predictions <- predictions[1:nKeep, ]
  }
  return(predictions)
}
tgramProb <- function(triWords, predictions, triPzero) {
  cnames <- colnames(predictions)
  checkWords <- c("word1", "word2", "word3")
  if (sum(checkWords %in% cnames) == 3) {
    prediction <- predictions[word1 == triWords[[1]] 
                              & word2 == triWords[[2]] 
                              & word3 == triWords[[3]]]
    triProb <- ifelse(dim(prediction)[1] > 0, 
                      prediction[1, "predictProb"], triPzero)
  } else {
    triProb <- triPzero
  }
  return(triProb)
}
validateModel <- function(testData, coef, 
                          ugramPzero, bgramPzero, tgramPzero, nKeep, blockSize) {
  
  ## Get the probability, first, second and third matches (T,F) for each row
  fmodel <- function(x) {
    predictions <- trimodel(x[, 1:2], coef, ugramPzero, bgramPzero, tgramPzero,
                            nKeep)
    triProb <- tgramProb(x, predictions, tgramPzero$pZero)
    firstMatch <- isNthMatch(1, x[[1,3]], predictions)
    secondMatch <- isNthMatch(2, x[[1,3]], predictions)
    thirdMatch <- isNthMatch(3, x[[1,3]], predictions)
    return(c(triProb[[1]], firstMatch[[1]], secondMatch[[1]], thirdMatch[[1]]))
  }
  
  dataLength <- nrow(testData)
  
  ## Run fmodel by rows
  predictAll <- testData[, fmodel(.SD), by=1:nrow(testData)]
  #predictAll <- testData[, fmodel(.SD)]
  
  ## Reshape to form 4 by dataLength matrix
  predictAll <- matrix(predictAll$V1, nrow = 4, byrow=FALSE)
  rownames(predictAll) <- c("perplexity", "firstMatch", 
                            "secondMatch", "thirdMatch")
  
  modelPerplexity <- perplexity(predictAll["perplexity", ])
  firstAccuracy <- sum(predictAll["firstMatch", ]) / dataLength
  secondAccuracy <- sum(predictAll["secondMatch", ]) / dataLength
  thirdAccuracy <- sum(predictAll["thirdMatch", ]) / dataLength
  
  results <- c(modelPerplexity, firstAccuracy, secondAccuracy, thirdAccuracy)
  return(results)
}

isNthMatch <- function(n, testWord3, predictions) {
  cnames <- colnames(predictions)
  isMatched <- FALSE
  if (nrow(predictions) >= n) {
    if ("word3" %in% cnames) {
      isMatched <- testWord3 == predictions[[n, "word3"]]
    } else if ("word2" %in% cnames) {
      isMatched <- testWord3 == predictions[[n, "word2"]]
    } else {
      isMatched <- testWord3 == predictions[[n, "word1"]]
    }
  }
  return(isMatched)
}
perplexity <- function(predictProbs) {
  totalLogProb <- sum(log(predictProbs))
  logPerplexity <- (-1/length(predictProbs)) * totalLogProb
  return(exp(logPerplexity))
}

simpleGT <- function(freqhist) {
  
  ## nrzest: Averaging transformation
  ## Replace nr by zr = nr/(0.5*(t - q))
  ## where q, r, t are successive indices of non-zero values
  ## in gtfuncs.S by William A. Gale
  ##
  nrzest <- function(r, nr) {
    d <- c(1, diff(r))
    dr <- c(0.5 * (d[-1] + d[-length(d)]), d[length(d)])
    return(nr/dr)
  }
  
  ## rstest: Linear Good-Turing estimate
  ## log(nr) = a + b * log(r)
  ## b = coef[2]
  ## rstest r(star)est = r *(1 + 1/r)^(b + 1)
  ## b < -1
  ## in gtfuncs.S by William A. Gale
  ##
  rstest <- function(r, coef) {
    return(r * (1 + 1/r)^(1 + coef[2]))
  }
  
  ## The following code comes from gtanal.S by William A. Gale
  
  ## Get the input xr and xnr    
  xm <- freqhist
  xr <- xm[, 1]
  xnr <- xm[, 2]
  xN <- sum(xr * xnr)
  
  ## make averaging transform
  xnrz <- nrzest(xr, xnr)
  
  ## get Linear Good-Turing estimate
  xf <- lsfit(log(xr), log(xnrz))
  xcoef <- xf$coef
  xrst <- rstest(xr, xcoef)
  xrstrel <- xrst/xr
  
  ## get Turing estimate
  xrtry <- xr == c(xr[-1]-1, 0)
  xrstarel <- rep(0, length(xr))
  xrstarel[xrtry] <- (xr[xrtry]+1) / xr[xrtry] * c(xnr[-1], 0)[xrtry] / xnr[xrtry]
  
  ## make switch from Turing to LGT estimates
  tursd <- rep(1, length(xr))
  for (i in 1:length(xr)) {
    tursd[i] <- (i+1) / xnr[i] * sqrt(xnr[i=1] * (1 + xnr[i+1] / xnr[i]))
  }
  xrstcmbrel <- rep(0, length(xr))
  useturing <- TRUE
  for (r in 1:length(xr)) {
    if (!useturing) {
      xrstcmbrel[r] <- xrstrel[r]
    } else if (abs(xrstrel - xrstarel)[r] * r / tursd[r] > 1.65) {
      xrstcmbrel[r] <- xrstarel[r]
    } else {
      useturing <- FALSE
      xrstcmbrel[r] <- xrstrel[r]
    }
  }
  
  ## renormalize the probabilities for observed objects
  sumpraw <- sum(xrstcmbrel * xr * xnr / xN)
  xrstcmbrel <- xrstcmbrel * (1 - xnr[1]/xN) / sumpraw
  
  ## output to file
  # cat(xN, sum(xnr), file="gtanal", sep=",")
  # cat(0, xnr[1]/xN, file="gtanal", sep=",", append=TRUE)
  # for (i in 1:length(xr)) {
  #     cat(xr[i], xr[i]*xrstcmbrel[i], file="gtanal", append=TRUE)
  # }
  
  ## output matrix (0, r0est) + (xr, xnrstarnormalized)
  rrstar <- cbind(c(0, xr), c(xnr[1]/xN, xr*xrstcmbrel))
  
  ## output data table by pairs = (r = freq, rstar = freqGT)
  ## keyed (ordered) by freq.
  rrstar <- data.table(rrstar)
  colnames(rrstar) <- c("freq", "freqGT")
  setkey(rrstar, freq)
  return(rrstar)
}

##########
## Case 1: Vocabulary size = 10K, minFreq = 2 for bigrams and trigrams
##########

## Set min freq for train1Gram data frame
## minFreq1 = 23 vocab_size = 10263
minFreq1 <- 23
vocabSize1 <- nrow(train1Gram[train1Gram$freq >= minFreq1,])

## Set words with frequencies less than minFreq1 to <UNK>
ugramPzero.f23 <- unigram(train1Gram, minFreq1)

## Set the vocabulary
Vocab.f23 <- ugramPzero.f23$ngramDt$word1

## Clean the bigrams
minFreq2 <- 2
bgramPzero.f2.f23 <- bigram(train2Gram, minFreq2, Vocab.f23)

## Clean the trigrams
minFreq3 <- 2
tgramPzero.f2.f23 <- trigram(train3Gram, minFreq3, Vocab.f23)

## Clean validation trigrams
## Keep all trigrams in the validation data
vminFreq3 <- 1
vtgramPzero.f2.f23 <- trigram(validate3Gram, vminFreq3, Vocab.f23)

vtgram.f2.f23.Words <- vtgramPzero.f2.f23$ngramDt[, c("word1", "word2", "word3")]

## Sampling factor for sample size = samplingFactor * total number of lines
testSamplingFactor = 0.01

## Samples of validation words
vtgramWords1 <- sample_frac(vtgram.f2.f23.Words, 
                            size=testSamplingFactor, replace=FALSE)

trial <- 1
coef <- c(0.0, 0.0, 1.0)
startTime <- proc.time()
performance1 <- validateModel(vtgramWords1, coef,
                              ugramPzero.f23, bgramPzero.f2.f23, tgramPzero.f2.f23, 
                              nKeep=100, blockSize=1000)
stopTime <- proc.time()
print("Elapsed time of validateModel")
stopTime - startTime

## Save performances
performances1 <- data.table(matrix(numeric(), 10, 4))
names(performances1) <- c("perplexity", "firstAccuracy", 
                          "secondAccuracy", "thirdAccuracy")
performances1[trial, names(performances1) <- as.list(performance1)]








freqDf.train <- getTermFrequency(trainCorpus, .99)

p<-ggplot(data=freqDf.train[1:20,], aes(x=reorder(ngram, -freq), y=freq)) +
  geom_bar(stat="identity") 
p



## 
# qua.trainCorpus.SWord <- corpus(trainCorpus.SWord)
# toks_immig <- tokens(qua.trainCorpus.SWord)
# toks_ngram <- tokens_ngrams(toks_immig, n = 2)
# head(toks_ngram)
# 
# cps.dfm.ng1 <- quanteda::dfm(toks_immig, ngrams = 3, skip = 0, verbose=TRUE,
#                              toLower=TRUE, removeNumbers=TRUE, removePunct=TRUE, removeSeparators=TRUE, removeURL=TRUE,
#                              removeTwitter=TRUE, language = "english") 
# topfeatures(cps.dfm.ng1)








