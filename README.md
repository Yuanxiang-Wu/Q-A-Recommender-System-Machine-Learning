# Q-A-Recommender-System-Machine-Learning
This is a matching strategy between questions and experts in a question recommender system. 
It can:
1. Forecast which expert is more likely to answer the question given Toutiao Q&A’s training and test data. 
2. Implement Principle Component Analysis to convert users and questions info into linear combination of orthogonal basis with respect to users’ labels and calculated the match between questions and labels. 
3. Use neural networks to train the model and predict the probability of a certain expert answering the question. 
4. Analyzed feature importance to identify the major factors that influenced the results to overcome overfitting.

Data Description:

user_info.txt: expert tag file

     Each line represents one expert user, which has four properties separated by /tab. The four properties are:
     1、Anonymized expert user ID: the unique identifier of each expert user.
     2、Expert user tags: there will be multiple tags, i.e. 18/19/20 may represent baby/ pregnancy/ parenting.
     3、Word ID sequence: User descriptions (excluding modal particles and punctuation) are first segmented,then each word will be replaced by the Character ID, i.e.284/42 may represent "Dont Panic".
     4、Character ID sequence: User descriptions (excluding modal particles and punctuation) are first segmented, then each character will be replaced by the Character ID,i.e. 284/42 may represent “BE”.
     note:when a property is null we will use a placeholder "/" to represent it.

question_info.txt: Question data file

     Each line represents one question, which has seven properties separated by /tab. The seven properties are:
     1、Anonymized question ID: the unique identifier of each question.
     2、Question tag: there will be single tags, i.e. 2 may represent fitness.
     3、Word ID sequence: User descriptions (excluding modal particles and punctuation) are first segmented, then each word will be replaced by the Character ID, i.e. 284/42 may represent “Dont Panic”.
     4、Character ID sequence: User descriptions (excluding modal particles and punctuation) are first segmented, then each character will be replaced by the Character ID, i.e.284/42 may represent “BE”.
     5、Number of upvotes: The total number of upvotes of all answers to this question. It may indicate the popularity of the question.
     6、Number of answers: The total number of answers to this question. It may indicate the popularity of the question.
     7、Number of top quality answers: The total number of top quality answers to this question. It may indicate the popularity of the question. 
    note:when a property is null we will use a placeholder "/" to represent it.
    
The training set will contain one file with the following format:

Invited_info_train.txt: Question distribution data

     Each line represents one question push notification record, which includes the encrypted ID of the question, the encrypted ID of the expert user and if the expert user answered the question (0=ignored, 1=answered), separated by /tab.

     Validation set and test set will each contain one file (invited_info_validation.txt and invited_info_test.txt) with the same format as the invited_info_train.txt. Each of the files will contain a part of the push records.
