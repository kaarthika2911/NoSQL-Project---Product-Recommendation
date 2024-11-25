This project presents a Sentiment-Enhanced Collaborative Filtering system designed 
to improve product recommendations in e-commerce by integrating sentiment 
analysis into traditional recommendation algorithms. While conventional 
collaborative filtering methods rely solely on user similarity and purchase history, 
they often overlook the emotional context embedded in customer reviews. To 
address this limitation, we propose a system that combines sentiment analysis with 
collaborative filtering techniques, utilizing MongoDB for scalable data management. 
Sentiment analysis is conducted on user-generated reviews using NLP algorithms 
like VADER, categorizing reviews into positive, neutral, or negative sentiments. 
These sentiment scores are integrated with collaborative filtering methods, including 
User-Based and Item-Based Collaborative Filtering, to enrich the recommendation 
process. By considering both sentiment and traditional ratings, the system prioritizes 
products that are not only highly rated but also positively reviewed by users with 
similar preferences. MongoDB’s document-oriented database efficiently stores user 
data, including IDs, product IDs, ratings, review texts, sentiment scores, and 
timestamps, allowing for robust handling of unstructured data and enabling complex 
queries required for this approach. By combining sentiment analysis with 
collaborative filtering, the system produces more personalized and emotionally 
resonant recommendations, enhancing the user experience and fostering greater 
engagement on e-commerce platforms. This project demonstrates the value of 
integrating machine learning with sentiment analysis to advance recommendation 
systems. 
