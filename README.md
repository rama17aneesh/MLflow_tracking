# ml_flow
- NLP_Project
- Mlflow is an open source platform for managing the end-toend machine learning life cycles.
- It can track experiments to record and compare parameters and result.

- Track.py, is about **Sentiment_analysis** of movie review with the dataset "IMDB_Dataset" from uci_machine_learning" repositry, dependent variable has two classes "positive and negative" and i am going to use **logistic regression machine learning model** to train this dataset. But in the dataset, the **sentiment is string**, we need **integers as our labels to train our model**. In feature parameter, I am going to choose **vectorizer** from **Scikit-learn** library, in order to convert text into some kind of features that will be used for training. I will use **TfidfVectorizer**, the vectorizer mesures how important a word in a text is giving more weight to rare terms and less weight to common terms, in track.py i have made test dataframe to identify unique statements using ".unique" and converted them to a list.
  - After specifying the vectorizer, i neeed to tansform the string(review) into a specified range on which machine learning model can work upon and it is given by **fit_transform** method and dont forget to **transform** test_set as the model will cause error if the model trained on different values and tested on differnent values. for evaluation, i gave **metric as accuracy**. 

- In this experiment i am using **MLflow tracking** to track parameters used to run this experiment and **compare the experiment results**. first i have intialized the default values for logistic regression [C=1.0, penalty = l2, solver=lbfgs] and i tracked how changing the regularization parameter 'C' impacts model's performance. i have executed through **Command line** by changing 'C' value [0.001, 1.0, 10, 25, 35, 50]. MLflow provides built-in visualization tools to create comparison charts. You can see how the accuracy changes as you increase or decrease C values.

- please check **Scatter_plot.png** folder to view the vizualized comparison got from mlflow ui.

Evaluation:-  
- Lower C values **C=0.001 which gave me an accuracy 0.767 and C=0.1 gave 0.891**, The performance improves as C increases from 0.001 to 1.0, This suggests that the model benefits from less regularization up to this point.
- The optimal C appears to be around when **C = 10 gave me an accuracy 0.892**, balancing the regularization and model complexity. This suggests that the model needs some degree of freedom to fit the sentiment data, but too much freedom leads to diminishing returns or slight overfitting, that's what i oberserved as the increase in C beyond 10, eg : **C=25, C=35 and C=50**, the performance stays around **0.88**, slightly below the peak at C=10. This indicates that once regularization is reduced too much, the model starts to overfit slightly without improving generalization.
- C=10 provides the best trade-off between bias and variance for this sentiment analysis task. The dataset likely contains some complexity that benefits from reducing regularization, but excessive flexibility doesn’t continue to improve the model’s performance.

- Since the model is using TFIDFVectorizer, it creates a lot of features from the text data. Regularization (controlled by the C value) helps the model avoid overfitting by not giving too much importance to these many features. Without proper regularization, the model might become too complex and perform worse on new data.
