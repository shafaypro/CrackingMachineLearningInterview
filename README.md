## CrackingMachineLearningInterview

These questions are guranteed to be asked in Machine Learning Engineer or ML based interviews for upto 80% of your job interviews. Make sure that you cover and know about machine learning. These questions can Ace your interviews regarding ML even if you are a fresh graduate or upto 5 years of experience.
##### Ultimate Machine Learning Engineer Interview Questions.
Helpful for:
* Machine Learning Engineer

* Data Scientist

* Deep Learning Engineer

* AI Engineer

* Software Development Engineer (focused on AI/ML)

## About:
* Github Profile :  [Shafaypro](https://github.com/shafaypro)
* ML Interview Questions repository : [ML interview Repository Link](https://github.com/shafaypro/CrackingMachineLearningInterview "ML Interview Repository")
## Questions

#### Difference between SuperVised and Unsupervised Learning?
        Supervised learning is when you know the outcome and you are provided with the fully labeled outcome data while in unsupervised you are not provided with labeled outcome data. Fully labeled means that each example in the training dataset is tagged with the answer the algorithm should come up with on its own. So, a labeled dataset of flower images would tell the model which photos were of roses, daisies and daffodils. When shown a new image, the model compares it to the training examples to predict the correct label.
#### What is Reinforcment Learning and how would you define it?
        A learning differs from supervised learning in not needing labelled input/output pairs be presented, and in not needing sub-optimal actions to be explicitly corrected. Instead the focus is on finding a balance between exploration (of uncharted territory) and exploitation (of current knowledge) .Semisupervised learning is also known as Reinforcment learning, in reinforcment learning each learning steps involved a penalty criteria whether to give model positive points or negative points and based on that penalizing the model.
#### What is Deep Learning ?
        Deep learning is defined as algorithms inspired by the structure and function of the brain called artificial neural networks(ANN).Deep learning most probably focuses on Non Linear Analysis and is recommend for Non Linear problems regarding Artificial Intelligence.
#### Difference between Machine Learning and Deep Learning?	
        Since DL is a subset of ML and both being subset of AI.While basic machine learning models do become progressively better at whatever their function is, they still need some guidance. If an AI algorithm returns an inaccurate prediction, then an engineer has to step in and make adjustments. With a deep learning model, an algorithm can determine on its own if a prediction is accurate or not through its own neural network.
#### Difference between SemiSupervised and Reinforcment Learning?

#### Difference between Bias and Variance?
        Bias is definned as over simpliciation assumption assumed by the model, 
        Variance is definned as ability of a model to learn from Noise as well, making it highly variant.
        There is always a tradeoff between these both, hence its recommended to find a balance between these two and always use cross validation to 
        determine the best fit.
#### What is Linear Regressions ? How does it work?
        Fitting a Line in the respectable dataset when drawn to a plane, in a way that it actually defines the correlation between your dependent
        variables and your independent variable. Using a simple Line/Slope Formulae. Famously, representing f(X) = M(x) + b.
        Where b represents bias
        X represent the input variable (independent ones)
        f(X) represents Y which is dependent(outcome).

        The working of linear regression is Given a data set of n statistical units, a linear regression model assumes that the relationship between the dependent variable y and the p-vector of regressors x is linear. This relationship is modeled through a disturbance term or error variable ε — an unobserved random variable that adds "noise" to the linear relationship between the dependent variable and regressors. Thus the model takes the form Y = B0 + B1X1 + B2X2 + ..... + BNXN
        This also emplies : Y(i) = X(i) ^ T + B(i)
        Where T : denotes Transpose
        X(i) : denotes input at the I'th record in form of vector
        B(i) : denotes vector B which is bias vector.

#### UseCases of Regressions:
        Poisson regression for count data.
        Logistic regression and probit regression for binary data.
        Multinomial logistic regression and multinomial probit regression for categorical data.
        Ordered logit and ordered probit regression for ordinal data.
#### What is Logistic Regression? How does it work?
        Logistic regression is a statistical technique used to predict probability of binary response based on one or more independent variables. 
        It means that, given a certain factors, logistic regression is used to predict an outcome which has two values such as 0 or 1, pass or fail,
        yes or no etc
        Logistic Regression is used when the dependent variable (target) is categorical.
        For example,
            To predict whether an email is spam (1) or (0)
            Whether the tumor is malignant (1) or not (0)
            Whether the transaction is fraud or not (1 or 0)
        The prediction is based on probabilties of specified classes 
        Works the same way as linear regression but uses logit function to scale down the values between 0 and 1 and get the probabilities.

#### What is Logit Function? or Sigmoid function/ where in ML and DL you can use it?
        The sigmoid might be useful if you want to transform a real valued variable into something that represents a probability. While the Logit function
        is to map probaliticvalues from -Inf to +inf to either real numbers representing True or False towards 1 or 0 (real number). This is commonly used
        in Classification having base in  Logistic Regression along with Sigmoid based functions in Deep learning used to find a nominal outcome in a
        layer or output of a layer.

#### What is Support Vector Machine ? how is it different from  OVR classifiers?
        Support Vector Machine is defineed as a Technique which is classification and regression model. Which uses hyper plan estimation and best hyper plane
        fitting the estimate on linear lines drawn same a linear one. Although it can also work for non Linear using kernal tricks on SVM.
        SVM is totally based on Marginal lines (with difference between two classes in the best way possible).
        One Vs rest is the base classifier concept which is used in all the Ml algorithms which involves classification based on Class A vs Classes REst approach. Since here are only two heuristic approaches which are enhancment of Multiclass classificaton to make the binary classifier perform
        well on multi class problems and solve the problem in hand.

        The algorithms which uses OVO are:
            1) Extreme learning Machines(ELM's)
            2) Support Vector Machine(Classifiers)
            3) K Nearest Neighbours.(for neighbouring classes based on distances)
            4) Naive Bayes (based on MAP : Maximum Posterior )
            5) Decision Trees(decision in subnodes after parent node has one feature)
            6) Neural Networks (different nets)

#### Types of SVM kernels
        Think of kernels as definned filters each for their own specific usecases.

        1) Polynomial Kernels (used for image processing)
        2) Gaussian Kernel (When there is no prior knowledge for data)
        3) Gaussian Radial Basis Function(same as 2)
        4) Laplace RBF Kernel ( recommend for higher training set more than million)
        5) Hyperbolic Tangent Kernel (neural network based kernel)
        6) Sigmoid Kernel(proxy for Neural network)
        7) Anova Radial Basis Kernel (for Regression Problems)

#### What is different type of Evauation metrics in Regression?
        There are multiple evaluation metrics for Regression Analysis
        1) Mean Squared Error ( the average squared difference between the estimated values and the actual value)
        2) Mean Absolute Error (Absolute of the Average difference)

#### How would you define Mean absolute error vs Mean squared error?
        MAE : Use MAE when you are doing regression and don’t want outliers to play a big role. It can also be useful if you know that your distribution is multimodal, and it’s desirable to have predictions at one of the modes, rather than at the mean of them.
        MSE : use MSE the other way around, when you want to punish the outliers.

#### How would you evaluate your classifier?
        A classifier can be evaluated through multiple case, having the base case around its confusion metrics and its attributes which are TP, TN , FP and FN. Along with the Accuracy metrics which can be derived alongside Precision, Recall scores.

#### What is Classification?
        Classification is defined as categorizing classes or entities based on the specified categories either that category exists or not in the respectable data. The concept is quite common for Image based classification or Data Based Classification. The answer in form of Yes or No;
        alongside answers in form of types of objects/classes.

#### How would you differentiate between Multilabel and MultiClass classification?
        A multiclass defines as a classification outcome which can be of multiple classes either A or B or C but not   two or more than one.
        While in MultiLabel classification, An outcome can be of either one or more than two classes i.e A or A and B or A and B and C. 

#### Which Algorithms are High Biased Algorithms?
        Bias is the simplifying assumptions made by the model to make the target function easier to approximate.
        1) High bias algorithms are most probably Linear Algorithm, which are concerned with linear relationships or linear distancing. Examples are 
        2) Linear, Logistic or Linear Discrimenant Analysis.

#### Which Algorithms are High and low Variance Algorithms?	
        Variance is the amount that the estimate of the target function will change given different training data

        1) High Variance Algorithms are Decision Trees, K Nearest Neigbours and SVMs
        2) Low Variance Algorithms are Linear Regression, Logistic Regression and LDA's

#### Why are the above algorithms are High biased or high variance?
        Linear machine learning algorithms often have a high bias but a low variance.
        Nonlinear machine learning algorithms often have a low bias but a high variance.

#### What are root case of Prediction Bias?
        Possible root causes of prediction bias are:

        1) Incomplete feature set
        2) Noisy data set
        3) Buggy pipeline
        4) Biased training sample
        5) Overly strong regularization

#### What is Randomforest and Decision Trees?
        A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event
        outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements.

        Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operate by
        constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean
        prediction (regression) of the individual trees. Used to remove the Overfitting occured due to single Decision Trees.
#### What is Process of Splitting?
        Splitting up your data in to subsets based on provided data facts. (can come in handy for decision Trees)

#### What is the process prunning?
        The Shortening of branches of Decision Trees is termed as Prunning. The process is done in case to reach the decision quite earlier than
        expected. Reducing the size of the tree by turning some branch nodes into leaf nodes, and removing the leaf nodes under the original branch.

#### How do you do Tree Selection?
        Tree selection is mainly done from the following
        1) Entropy 
                A decision tree is built top-down from a root node and involves partitioning the data into subsets that contain instances with similar 
                values (homogeneous). ID 3 algorithm uses entropy to calculate the homogeneity of a sample. If the sample is completely homogeneous the
                entropy is zero and if the sample is an equally divided it has entropy of one.
                Entropy(x) -> -p log(p) - qlog(q)  with log of base 2
        2) Information Gain  
                The information gain is based on the decrease in entropy after a dataset is split on an attribute. Constructing a decision tree is all about finding attribute that returns the highest information gain (i.e., the most homogeneous branches).

                2.1) Calculate entropy of the target.
                2.2) The dataset is then split on the different attributes. The entropy for each branch is calculated. Then it is added proportionally, to get total entropy for the split. The resulting entropy is subtracted from the entropy before the split. The result is the Information Gain, or decrease in entropy.
                2.3) Choose attribute with the largest information gain as the decision node, divide the dataset by its branches and repeat the same process on every branch.

#### Pseudocode for Entropy in Decision Trees:
        '''      
        from math import log

        def calculateEntropy(dataSet):
        number = len(dataSet)
        labelCounts = {}
        for featureVector in dataSet:
            currentLabel = featureVector[-1]
            if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
            labelCounts[currentLabel] +=1
        entropy = 0
        for i in labelCounts:
            probability = float(labelCounts[keys])/number
            entropy -=probability*log(probability,2)
        return entropy
        '''
#### How does RandomForest Works and Decision Trees?
        -* Decision Tree *- A Simple Tree compromising of the process defined in selection of Trees.
        -* RandomForest *- Combination of Multiple N number of Decision Trees and using the aggregation to determine the final outcome.
        The classifier outcome is based on Voting of each tree within random forest while in case of regression it is based on the 
        averaging of the tree outcome.

#### What is Gini Index? Explain the concept?
        The Gini Index is calculated by subtracting the sum of the squared probabilities of each class from one. It favors larger partitions.
        Imagine, you want to draw a decision tree and wants to decide which feature/column you should use for your first split?, this is probably defined
        by your gini index.

#### What is the process of gini index calculation?
        Gini Index:
        for each branch in split:
            Calculate percent branch represents .Used for weighting
            for each class in branch:
                Calculate probability of class in the given branch.
                Square the class probability.
            Sum the squared class probabilities.
            Subtract the sum from 1. #This is the Ginin Index for branch
        Weight each branch based on the baseline probability.
        Sum the weighted gini index for each split.

#### What is the formulation of Gini Split / Gini Index?
        Favors larger partitions.
        Uses squared proportion of classes.
        Perfectly classified, Gini Index would be zero.
        Evenly distributed would be 1 – (1/# Classes).
        You want a variable split that has a low Gini Index.
        The algorithm works as 1 – ( P(class1)^2 + P(class2)^2 + … + P(classN)^2)

#### What is probability? How would you define Likelihood?
        Probability defines the percentage of Succes occured. or Success of an event. Can be described Chance of having an event is 70% or etc.
        We suppose that the event that we get the face of coin in success, so the probability of success now is 0.5 because the probability of face and back of a coin is equal. 0.5 is the probability of a success.

        Likelihood is the conditional probability. The same example, we toss the coin 10 times ,and we suppose that we get 7 success ( show the face) and
        3 failed ( show the back). The likelihood is calculated (for binomial distribution, it can be vary depend on the distributions).


        Likelihood(Event(success)) - > L(0.5|7)= 10C7 * 0.5^7 * (1-0.5)^3 = 0.1171 

        L(0.5 | 7) : means event likelihood of back( given number of successes)
        10C7 -> Combination based on total 10 Events, and having the success outcome be 7 events
        In general:
            Event(X | Y)  -> C(Total Event| Success Event) * [(Prob of X) ^ (Success Event X)] * [(1 - Prob of X) ^ (1 - Success Event X)]

#### What is Entropy? and Information Gain ? there difference ?
        Entropy: Randomness of information being processed.

        Information Gain multiplies the probability of the class times the log (base=2) of that class probability.  Information Gain favors smaller partitions with many distinct values.  Ultimately, you have to experiment with your data and the splitting criterion.
        IG depends on Entropy Change (decrease represent increase in IG)


#### What is KL divergence, how would you define its usecase in ML?
        Kullback-Leibler divergence calculates a score that measures the divergence of one probability distribution from another
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/4958785faae58310ca5ab69de1310e3aafd12b32)

#### How would you define Cross Entropy, What is the main purpose of it ? 
        Entropy: Randomness of information being processed.

        Cross Entropy: A measure from the field of information theory, building upon entropy and generally calculating the difference between two
        probability distributions. It is closely related to but is different from KL divergence that calculates the relative entropy between two
        probability distributions, whereas cross-entropy can be thought to calculate the total entropy between the distributions.

        Cross-entropy can be calculated using the probabilities of the events from P and Q, as follows:
                H(P, Q) = – sum x in X P(x) * log(Q(x))
#### How would you define AUC - ROC Curve?
        ROC is a probability curve and AUC represents degree or measure of separability. AUC - ROC curve is a performance measurement for classification problem at various thresholds settings.

        It tells how much model is capable of distinguishing between classes. Mainly used in classification problems for measure at different thresholds.
        Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s. By analogy, Higher the AUC, better the model is at distinguishing between patients with disease and no disease.
    
![](https://miro.medium.com/max/722/1*pk05QGzoWhCgRiiFbz-oKQ.png)
#### How would you define False positive or Type I error and False Negative or Type II Error ?
        False positive : A false positive is an outcome where the model incorrectly predicts the positive class.(was A but got predicted B) aka Type I error.

        False Negative : A false negative is an outcome where the model incorrectly predicts the negative class. (was not A but predicted A) aka Type II error.

#### How would you define precision() and Recall(True positive Rate)?
        Take a simple Classification example of "Classifying email messages as spam or not spam"

        Precision measures the percentage of emails flagged as spam that were correctly classified—that is, the percentage of dots to the right of the threshold line, it is also defined as % of event being Called at positive rates e.g 
                Precision = True Positive / (True Positive + False positive) 
        
        Recall measures the percentage of actual spam emails that were correctly classified
                Recall = True Postives / (True Positive + False Negative)
        
        There is always a tradeoff between precision and Recall same is the case of Bias and Variance.


#### Which one would you prefer for you classification model Precision or Recall?
        This totally depends on Business Usecase or SME usecase. In case of Fraud Detection Business domains such as banks, online ecommerce websites
        recommends of better recall score than precision. While in other cases such as word suggestions or Multi label Categorization it can be precision.
        In general, totally dependent on your use case.

#### What is F1 Score? which intution does it gives ?
        The F1 score is the harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall).
        Also known as Dice Similarity Coefficent. 

        David (Scientist Statistician): The widespread use of the F1 score since it gives equal importance to precision and recall. In practice, different types of mis-classifications incur different costs. In other words, the relative importance of precision and recall is an aspect of the problem

#### What is difference between Preceptron and SVM?
        The major practical difference between a (kernel) perceptron and SVM is that perceptrons can be trained online (i.e. their weights can be updated
        as new examples arrive one at a time) whereas SVMs cannot be. Perceptron is no more than hinge loss (loss function) + stochastic gradient descent (optimization).
        
        SVM has almost the same goal as L2-regularized perceptron.
        SVM can be seen as hinge loss + l2 regularization (loss + regularization) + quadratic programming or other fancier optimization algorithms like SMO (optimization).

#### What is the difference between Logsitic and Liner Regressions?
        LogR is Classifier, LR is Regression.
        LogR values are between 0 and 1 and probabilty in between as well.
        LR values are in realnumbers from 1 to postive N (where N is known)
#### What are outliers and How would you remove them?
        An outlier is an observation that lies an abnormal distance from other values in a random sample from a population.
![](https://www2.southeastern.edu/Academics/Faculty/dgurney/Outlier.jpg)

        Outliers can be removed by following:
            1) Use Inter Quantile Range (IQR * 1.5)
            2) Use Z-score Scale removal (so that any point much away from mean gets removed)
            3) Combination of Z Score and IQR (custom scores)

#### What is Regulization?
        Regularizations are techniques used to reduce the error by fitting a function appropriately on the given training set and avoid overfitting.

#### Difference between L1 and L2 Regulization?
        1) L1 Regulization (Lasso Regression)
            (Least Absolute Shrinkage and Selection Operator) adds “absolute value of magnitude” of coefficient as penalty term to the loss function.

        2) L2 Regulization (Ridge Regression)
            (adds “squared magnitude” of coefficient as penalty term to the loss function. Here the highlighted part represents L2 regularization element.)

        The key difference between these techniques is that Lasso shrinks the less important feature’s coefficient to zero thus, removing some feature
        altogether. So, this works well for feature selection in case we have a huge number of features.

#### What are different Technique of Sampling your data?
        Data Sampling statistical analysis technique used to select, manipulate and analyze a representative subset of data points to identify patterns and trends in the larger data set being examined.
        There are different techniques of sampling your data
        1) Simple Random Sampling (records are picked at random)
        2) Stratified Sampling (subsets based on common factor with equal ratio distribution)
        3) Cluster Sampling (largest set is breaken down in form of clusters based on defined factors and SRS is applied)
        4) MultiStage Sampling (cluster on Cluster sampling)
        5) Symentaic Sampling (Sample created by setting interval)
        
#### Can you define the concept of Undersampling and Oversampling?
        Undersampling is the concept of downsizing the class based sample from a Bigger range to smaller range i.e 1Million records to 0.1 Million records,
        keeping the ratio of information intact

        Oversampling represents the concept of using a smaller class sample i.e 100K to scale upto million keeping the trend and the property to make up
        datasets.

#### What is Imbalanced Class?
        Imbalancment is when you don't have balance in between classes.
        Imabalnced class is when the normal distribution/support count of multiple classes or classes being considered are not the same or almost same.
        E.G:
            Class A has 1 Million Record
            Class B has 1000 Record
        This is imbalanced data set and Class B is UnderBalanced Class.

#### How would you resolve the issue of Imbalancment data set?
        The techniques such as 
            1) OverSampling
            2) UnderSampling
            3) Smote combination of both
            4) bringing in more dataset
            5) doing more trend analysis
        can resolve the issue of Imbalancment dataset

#### How would you define Weighted Moving Averages ?
        A incremental increase of Moving Average having a Weighted Multiple to keep the values which gets repeated during a certain time with High 
        priority/ Impact.

#### What is meant by ARIMA Models?
        A Regressive and Moving Average Model combination is termed as ARIMA. To be exact, Auto Regressive Intergerated Moving Avearges.
        A techniques which does regression analysis along with moving averages which fits time series analysis and gets trend analysis with 
        acceptable scores.

#### How would you define Bagging and Boosting? How would XGBoost differ from RandomForest?
        Bagging : A way to decrease the variance in the prediction by generating additional data for training from dataset using combinations with repetitions to produce multi-sets of the original data. 
                Example : Random Forest  (uses random Sampling subsets)
        Boosting: An iterative technique which adjusts the weight of an observation based on the last classification
                Example: AdaBoost, XGboost  (using gradient descent as main method)
#### What is IQR, how can these help in Outliers removal?
        IQR is interquantile range which specifies the range between your third quantile and the first one.
        Quantile are 4 points of your data represented by percentage(should be four equal parts )
            Q1: 0-25%
            Q2: 25-50%
            Q3: 50-75%
            Q4: 75- 100%
        IQR :- Q3 - Q1
#### What is SMOTE?	
        Synthetic Minority Over-sampling TEchnique also known as SMOTE. 
        A very popular oversampling method that was proposed to improve random oversampling but 
        its behavior on high-dimensional data has not been thoroughly investigated. 
        KNN algorithm gets benefits from SMOTE.
#### How would you resolve Overfitting or Underfitting? 
        Underfitting
            1) Increase complexity of model
            2) Increasing training time
            3) decrease learning rate
        Overfitting:
            1) Cross Validation
            2) Early Stops
            3) increased learning rates(hops)
            4) Ensembling 
            5) Bring in More data
            6) Remove Features
#### Mention some techniques which are to avoid Overfitting?
        1) Cross Validation
        2) Early Stops
        3) increased learning rates(hops)
        4) Ensembling 
        5) Bring in More data
        6) Remove Features
#### What is a Neuron?
        A "neuron" in an artificial neural network is a mathematical approximation of a biological neuron.
        It takes a vector of inputs, performs a transformation on them, and outputs a single scalar value.
         It can be thought of as a filter. Typically we use nonlinear filters in neural networks.

#### What are Hidden Layers and Input layer?
![](https://www.i2tutorials.com/wp-content/uploads/2019/05/Hidden-layrs-1-i2tutorials.jpg)

        1) Input Layer: Initial input for your neural network
        2) Hiddent layers: a hidden layer is located between the input and output of the algorithm, 
        in which the function applies weights to the inputs and directs them through an activation function as the output.
        In short, the hidden layers perform nonlinear transformations of the inputs entered into the network. 
        Hidden layers vary depending on the function of the neural network, and similarly, the layers may vary depending 
        on their associated weights.

#### What are Output Layers?	


#### What are activation functions ? 	

#### What is Convolotional Neural Network?	

#### What is recurrent Neural Network?	

#### What is Convolotional Layer?	

#### What is MaxPooling Layer? How does it work?	

#### What is Kernel or Filter?	

#### What is Activation functions?	

#### What is Segmentation?	

#### What is poster analysis?	

#### What is Forward propagation?	

#### What is backward propagation?	

#### what are dropout neurons?	

#### what are flattening layers?	

#### How is backward propagation dealing an improvment in the model?	

#### What is correlation? and covariance?	

#### What is Anova? when to use Anova?	

#### How would you define dimentionality reduction? why we use dimentionality reduction?	

#### What is Principle componenet analysis? how does PCA work in Dimentonality reduction?	

#### What is Maximum Likelihood estimation?	

#### What is Naive Bayes? How does it works?	

#### What is KNN how does it works? what is neigbouring criteria? How you can change it ?	

#### What is Conditional Probability? what is distributive Probability?

#### Which one would you prefer low FN or FP's based on Fraudial Transaction?

#### Differentiate between KNN and KMean?	

#### What is Attention ? Give Example ?	

#### What are AutoEncoders? and what are transformers?

        Autoencoders take input data, compress it into a code, then try to recreate the input data from that summarized code. It’s like starting with Moby Dick, creating a SparkNotes version and then trying to rewrite the original story using only SparkNotes for reference. While a neat deep learning trick, there are fewer real-world cases where a simple autocoder is useful. But add a layer of complexity and the possibilities multiply: by using both noisy and clean versions of an image during training, autoencoders can remove noise from visual data like images, video or medical scans to improve picture quality.

#### How would you define GAN(Generative Adversarial Networks) ?	

#### What are Gausian Processes?	

#### What is Graph Neural Network?	

#### Define Text Classification, Give some usecase examples?	

#### What is Language Modelling(LM), give examples?	

#### Define Named Entity recognition? Give some usecases where it can come in handy?	

#### What is Image Captioning?	

#### Give some example of Text summarization.	

#### Define Style Transfer?

#### Define Image Segmentation and Pose Analysis?	

#### Define Semantic Segmentation?	

#### What is Instance Segmentation?	

#### What is Imperative and Symbolic Programming?	
