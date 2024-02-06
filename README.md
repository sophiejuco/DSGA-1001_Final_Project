# DSGA-1001_Final_Project
DSGA-1001 Final Project: Data science project using song ratings data &amp; Spotify song information data
<br>
This repository contains the code notebook and data used for this project as well as the results of our analysis.
* <em>Capstone.ipynb</em> is the Python notebook.
* <em>spotify52kData.csv</em> is the Spotify song data used.
* <em>starRatings.csv.zip</em> is the ratings data, uploaded here as a .zip file.
<br>

There are also a folder:<br>
* <em>Plots</em> contains the plots and tables used to show the results of the project.
<br>

# Project Report

In this capstone project, we analyzed a dataset of 52,000 songs and in order to understand what makes music popular as well as the audio features that make up specific genres. Kitty worked on Q1-Q5, Joy worked on Q6-Q8, Sophie worked on Q9-Q10 and the extra credit problem, ChatGPT: suggested possible visualization for Q5 (Actual vs. Predicted Values Plot). Question-specific data handling is detailed in the following responses. Random seed 14844467 (Joy Fan’s N number) was used for randomization. 


**Question 1:** 
For determining whether there is a relationship between song length and popularity of a song, we choose to look at a simple linear regression model. To clean the duration data set, we choose to remove outliers that are above 3 standard deviations and their corresponding popularity values. Doing so, the COD we obtained is COD = 0.00299 (rounded) and the plot is attached below. As only about 0.3% of the variance in the popularity can be explained by the duration, we conclude that there isn't a relationship between song length and popularity of a song, and thus it does not make sense for us to interpret the negative slope of the line.

![alt_text]()

**Question 2:**
The two groups we are testing here are: songs that are explicitly rated and songs that are not explicitly rated, and whether songs are more popular is determined by the number of plays. To compare the two groups, we first remove the missing data, then separate the dataset by the first 5k rows and the rest, as songs with explicit feedback correspond to the first 5k rows in our spotify52kData.csv file. As we are using the mean as our test statistic and comparing two groups, we consider the independent samples t-test to be a reasonable choice. Here, since we are testing for whether explicitly rated songs are more popular, we want to do a one-tailed test. The test result gives us a p-value of 0.0019, and therefore we can reject the null hypothesis and conclude that explicitly rated songs are more popular than songs that are not explicit.

![alt_text]()

**Question 3:**
Similar to what we did in Q2, we have two groups: songs in major key and songs in minor key. We clean the dataset by removing the missing data and their corresponding popularity data, then separating the dataset into two groups depending on the mode of a song. Again, the independent samples t-test is the most suitable in this case, and since we are asked to determine whether songs in major key are more popular, we are doing a one-tailed test. The test result gives us a p-value of 0.9999993 (rounded), which is saying there really isn't much of a difference between the two groups. Thus, we fail to reject the null hypothesis and conclude that there is no significant difference in popularity between songs in major key or minor key. 

![alt_text]()

**Question 4:**
To determine which of the songs features(duration, danceability, energy, loudness, speechiness,
acousticness, instrumentalness, liveness, valence and tempo) predicts popularity best, we want to build 10 simple linear regression models and compare the performance for each one of them. We first clean the data by dropping all the missing values, then do a 80/20 train/test split using the RNG linked to our group. When we build 10 simple linear regression models, we standardize all the feature data to deal with possible skewness of the data distribution. We choose not to store all the models but instead only compare the COD value computed using the training set, which gives us the best predictor feature instrumentalness. 

Then we build a model only using the instrumentalness feature as a predictor, and the new COD value is computed using the test set. The value we obtained is COD = 0.01907 (rounded), meaning that the best predictor can only explain about 2% of the variance in the popularity. Therefore, this model is not a good model and doesn't help much with making predictions. 

![alt_text]()

**Question 5:**
As we are building a model that uses all of the song features in Q4, we don't need to re-clean the data. Again, we do a 80/20 train/test split using the RNG linked to our group. We then build a multiple regression model with all the features using the training set and compute the COD value using the testing set, which gives us 0.043339 (rounded). This means that our multiple regression model can only explain about 4% of the variance, which is still not a reliable model. If we solely compare the COD of the model we build in Q5 and the one from Q4, the COD value has doubled. However, it is expected for the COD in Q5 to be higher as we add more predictors into the model, which doesn't necessarily mean that the model is better. Due to the low COD value, it is not that meaningful to say how much this model has improved if it has improved at all. If we regularize the model using Ridge Regression, picking the best alpha value using RidgeCV, the COD value we obtain is 0.043336 (rounded). Again, the COD value is too low for us to make meaningful comments, which is expected as doing regularization is dealing with potential overfitting while our model most likely doesn’t have an overfitting problem. 

![alt_text]()

**Question 6:**
For the purpose of this analysis, we focused on the following 10 song features: duration, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence and tempo. In Q6, we use PCA to reduce dimensionality and use K-means clustering to find groupings. 

Prior to dimensionality reduction, the dataset has a shape of (52000,10), but after principal components extraction, the Kaiser Criterion (drop components with eigenvalues less than 1) determined that only 3 principal components should be included. These 3 principal components explained 57.4% of the variance. 

![alt_text]()

K Means is an algorithm for unsupervised clustering, to find clusters in data based on the data attributes alone instead of the labels:

![alt_text]()

We used Silhouette analysis to apply the K-means clustering to a range of plausible values K (2-10) and find out that the optimal cluster number to be 2. 

![alt_text]()

Since the optimal cluster number we have is 2, these clusters don’t really correspond to the genre labels. 

**Question 7:**
In this question we tried to predict whether a song is in major or minor key from valence using logistic regression. As seen in the ROC curve, AUC (area under ROC curve) is very close to 0.5, indicating a poor classification performance. It indicates valence doesn’t do a very good job in separating major/minor music. 

![alt_text]()

We tried different features, the best result is when we use “energy” or “loudness” as predictors. But the AUC score is still below 0.6.

Logistic regression doesn’t work well here, so we decided to build an SVM using “loudness” and “energy”. 

The accuracy we got from SVM is 62%. From the graph below, we conclude that SVM is not doing a great job here either, the model may not be able to perfectly separate all the data points, meaning the major/minor key data is not linearly separable. 

![alt_text]()

**Question 8:**
In this question, we are trying to predict the genre by using the 10 song features mentioned from above.

The first thing we did is to clean up the genre column. Initially we have 52 unique values. 

![alt_text]()

However, some of those can be combined together, for example, hard-rock, hardcore,hardstyle can be all under “hard-rock”, dancehall and dance can be all under “dance”, etc. After cleaning up the genre, we ended up with 13 genres:

![alt_text]()

We built a neural network and trained it using the 10 features to classify data into one of 13 genres: [dance-electronic, rock, alternative, acoustic, folk, chill, anime, children, classical, comedy, disney, gospel and hip-hop]. The neural network is a feedforward network and uses a backpropagation algorithm to adjust the weights and biases to minimize the cost function. The activation function we used here is Sigmoid function. The input layer has 10 neurons for the 10 features, and the output layer has 13 neurons. The feature data was standardized to improve accuracy.

If we use the “uncleaned” genre, which is 52 different genres, below is the result we get:

![alt_text]()

The precision rate is only 28%. 

If we use the “cleaned” genre, which is 13 different genres, below is the result we get:

![alt_text]()

The precision rate is largely improved by cleaning up the genre.

**Question 9:**
In order to assess if there is a relationship between popularity (how many times a song is listened to) and average star rating for the 5k songs we have explicit feedback for, I first found the average star ratings for each song using the Pandas mean() function. The first assessment of these data that I conducted was correlation tests. I used the Spearman correlation method which looks for linear relationships between the data. The result of the Spearman correlation was a score of 0.54. I also used the Pearson correlation method which looks for monotonic relationships which is when one variable, the other either increases or decreases. This is a more appropriate correlation method if the data isn’t linear. The Pearson correlation score is 0.57. From these results, we can conclude that there is a slight positive linear relationship and slight positive poisson relationship between average star rating and popularity. Despite the monotonic relationship being stronger than the linear relationship, neither is particularly large.

To further explore why this is the case, a scatter plot of the data, histogram of the two variables, and regression models were calculated and plotted.

![alt_text]()

The scatter plot shows that the data is quite noisy but there is a vaguely upward trend. The histograms show that the average ratings approach a normal distribution while the popularity data has multiple spikes with low popularity values being the highest spike. While popularity data is not skewed, it is definitely noisier than the average ratings data. 

![alt_text]()

A linear regression of the data was performed to assess the validity of a linear relationship between the data. The COD score is 0.33 which is not very high and indicates there is more variance in the data than what is explained by a linear model. A poisson regression of the data was performed as it is a general linear model that is specifically catered towards count data which is relevant as popularity is the count value of plays of a given song. Similar to the simple linear regression, there is a positive upward trend but the D^2 score, or percent of deviation explained by the poisson model, is still quite low at 0.24.

There is a slight positive monotonic relationship between average and popularity such that as one increases, so does the other but this relationship is not strong enough to explain large percentages of deviation or variance in the data.

The top 10 “greatest hits” of the 5k songs with explicit feedback can be found according to the popularity based model by ordering the average ratings of each song from high to low and taking the top 10 highest average rated songs. 

Doing this produced the following list of 10 “greatest hits” but some of these 10 songs are duplicates. In other words, they are the same song audibly but are given a different value in some column(s). For example, I Love You. by The Neighborhood is listed twice with different genres - alt-rock and alternative.

![alt_text]()

In order to handle duplicates, songs that shared the same track_name, artist, and album_name had any duplicates dropped as these songs would audibly be the exact same song. The first one of these duplicates were kept as they captured the highest average rating of that song. Had the song been the same artist and track but a different album, these duplicates would have also been dropped as they, again, would be the same song audibly just featured on multiple albums (i.e. if an artist’s original song is featured on a movie soundtrack). However, if the artist or track name was different, I would have considered these as different songs as they would be different versions audibly (i.e. a cover of a song by another artist or an acoustic version of an existing song).

The resulting top 10 greatest hits are as follows:

![alt_text]()

**Question 10:**
In order to create a recommender system that generates a personalized mixtape of the 10 top songs a listener would enjoy most, I used collaborative filtering based on user similarity to build this system.

Before generating the correlation or similarity between listeners, I first mean normalized the data to account for any users who have a tendency to consistently over or under rate songs. If this is not done, it would be more likely that a user who over-rates songs is interpreted by the system as liking all songs and it could be more difficult to find another listener who is similar. Correlation between users looks at each user in comparison to every other user and produces a score of how similar the user is to every other user. In this sense, a “similar” user would be another user who not only rated similar songs but also gave those songs a similar rating to the user we’re referencing. The following heatmap shows the correlation between the first 30 users:

![alt_text]()

After finding user similarity, a latent model is used to generate predicted ratings for each song for a given user. For this latent model, I set the cut off of similarity to zero. So, any user with a correlation score of 0 or lower cannot be used to generate predictions. I opted for this threshold because a 0 correlation score indicates that the two users are not at all similar while a negative correlation score indicates two users who have opposite tastes.

All predicted scores were stored back in a table with user ratings per song along with the existing ratings. The mixtapes were saved to a new dataframe that is 10x10000 in dimension - representing the top 10 recommended songs for each user. Both files are exported as .csv files.

For brevity’s sake, the mixtapes for the first 10 users will be used to show the generated recommendations and their evaluations. The mixtapes are as follows with the column indices representing the user’s number, the row indices representing the top 10 recommended songs from highest to lowest predicted rating, and the values being the songNum:

![alt_text]()

To compare how the mixtapes compare to the “greatest hits” determined in Question 9, I evaluated precision (how many songs from the greatest hits list were recommended in the same rank order for the mixtape) and recall (how many songs overall from the greatest hits list were recommended in the mixtape, regardless of order). Again, for brevity’s sake, we will just look at the results from the same set of users above:

![alt_text]()

From these results, we can see that for the first 10 users, recall is 0.5 or higher but precision is 0.2 or lower. So, while the greatest hits songs are recommended often in the top 10 songs, they are less often recommended in the same popularity ranking.

The system can perform the task of making a top 10 recommended songs list for each user but with the threshold of user similarity set to greater than zero, not all songs are given a predicted rating for every user. In future iterations of this recommender system, methods to improve the speed should be taken and methods to fill in the remaining empty predictions should be implemented. Perhaps a model that also incorporates song metrics would be able to do this well.

**Extra Credit:**
The question we want to explore is: Can you predict explicitness from speechiness using logistic regression?

The motivation behind this question is that purely instrumental songs, say classical music, don’t have any words so they cannot be explicit whereas songs that could be more speech-forward like rap songs, could have a tendency to be more explicit.

The first step was to explore the “explicit” and “speechiness” columns to assess if there are any missing values so they can be handled. There are no missing values in either column so no additional steps are necessary to alter the data. 

The data were randomly split into training and testing data and a logistic regression with cross validation to avoid overfitting was fit to the train data. Predictions were generated using the testing data and the beta value and AUC score were calculated. The beta value is 4.24 and the AUC score is 0.77.

![alt_text]():

Despite the decent AUC score, we can see that speechiness is not a good predictor of explicitness since there are a large number of songs that were misclassified in the test set. Looking at the data further, it makes sense that speechiness would be a poor predictor since there are a large number of songs that have a low level of speechiness but are still explicit as is shown in the following conditioned histograms:

![alt_text]()

