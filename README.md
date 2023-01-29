# Seniority_model

Used unsupervised learning techniques to predict the level of seniority.
* Scrapped publicly available employment history data
* Processed the data using several word embeddings, e.g. Spacy, FastText, GloVe and BERT 
* Implemented different ranking models using PyTorch: RankNet, ListNet and my own approaches 
* Used bubble-sort distance to evaluate different models' performance
* The best model (RankNet + FastText) resulted in 87\% sorting accuracy on validation sample
* Created an app on Flask and deployed it to Heroku
* \href{http://f233-194-75-216-247.ngrok.io}{Link to test}
