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
* Github Profile :  [Shafaypro](https://github.com/shafaypro) &copy;
* Linkedin Profile : [MShafayAmjad](https://www.linkedin.com/in/imshafay/) 
* ML Interview Questions repository : [ML interview Repository Link](https://github.com/shafaypro/CrackingMachineLearningInterview "ML Interview Repository")

#### Images References
* The Image references are only for Educational Purposes. These are totally made by the creators, specified in the reference.md file.

#### Sharing is Caring:
        Feel Free to Share the Questions/References Link into your blog.


## Questions

#### Difference between SuperVised and Unsupervised Learning?
        Supervised learning is when you know the outcome and you are provided with the fully labeled outcome data while in unsupervised you are not provided with labeled outcome data. Fully labeled means that each example in the training dataset is tagged with the answer the algorithm should come up with on its own. So, a labeled dataset of flower images would tell the model which photos were of roses, daisies and daffodils. When shown a new image, the model compares it to the training examples to predict the correct label.
![](https://miro.medium.com/max/2800/0*Uzqy-gqZg77Wun0e.jpg)
#### What is Reinforcment Learning and how would you define it?
        A learning differs from supervised learning in not needing labelled input/output pairs be presented, and in not needing sub-optimal actions to be explicitly corrected. Instead the focus is on finding a balance between exploration (of uncharted territory) and exploitation (of current knowledge) .Semisupervised learning is also known as Reinforcment learning, in reinforcment learning each learning steps involved a penalty criteria whether to give model positive points or negative points and based on that penalizing the model.
![](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Reinforcement_learning_diagram.svg/250px-Reinforcement_learning_diagram.svg.png)
#### What is Deep Learning ?
        Deep learning is defined as algorithms inspired by the structure and function of the brain called artificial neural networks(ANN).Deep learning most probably focuses on Non Linear Analysis and is recommend for Non Linear problems regarding Artificial Intelligence.

#### Difference between Machine Learning and Deep Learning?	
        Since DL is a subset of ML and both being subset of AI.While basic machine learning models do become progressively better at whatever their function is, they still need some guidance. If an AI algorithm returns an inaccurate prediction, then an engineer has to step in and make adjustments. With a deep learning model, an algorithm can determine on its own if a prediction is accurate or not through its own neural network.
![](https://lawtomated.com/wp-content/uploads/2019/04/MLvsDL.png)
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
![](https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/300px-SVM_margin.png)
        The algorithms which uses OVO are:
            1) Extreme learning Machines(ELM's)
            2) Support Vector Machine(Classifiers)
            3) K Nearest Neighbours.(for neighbouring classes based on distances)
            4) Naive Bayes (based on MAP : Maximum Posterior )
            5) Decision Trees(decision in subnodes after parent node has one feature)
            6) Neural Networks (different nets)
![](https://i.ytimg.com/vi/OmTu0fqUsQk/maxresdefault.jpg)
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
![](https://econbrowser.com/wp-content/uploads/2019/07/msemae.png)
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
![](https://4.bp.blogspot.com/-sCcOrQsTH9Q/XG1yv7mhERI/AAAAAAAAAJI/aEj6Jf1lookERHqPQS_Y6Q9bxBcTV7TIwCLcBGAs/s1600/multiclass-multilabel.png)

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

![](https://miro.medium.com/max/1200/1*5dq_1hnqkboZTcKFfwbO9A.png)
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
![](https://miro.medium.com/proxy/0*gKOV65tvGfY8SMem.png)
#### What are outliers and How would you remove them?
        An outlier is an observation that lies an abnormal distance from other values in a random sample from a population.
![](https://www2.southeastern.edu/Academics/Faculty/dgurney/Outlier.jpg)

        Outliers can be removed by following:
            1) Use Inter Quantile Range (IQR * 1.5)
            2) Use Z-score Scale removal (so that any point much away from mean gets removed)
            3) Combination of Z Score and IQR (custom scores)

#### What is Regulization?
        Regularizations are techniques used to reduce the error by fitting a function appropriately on the given training set and avoid overfitting.
        Add Lambda * Biasness value at the end.
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
        Output layer in ANN determines the final layer which is responsible for the final outcome, the outcome totally depends
        on the usecase provided and the function which is being used to scale the values. By default, Linear, Sigmoid and Relu are 
        most common choices.
        Linear for Regression.
        Sigmoid/Softmax for Classification.

#### What are activation functions ?
        Activation functions perform a transformation on the input received, in order to keep values within a manageable range depending
        on the limitation of the activation function. Its more of a mathematical scale filter applied to a complete layer (Vector)
        to scale out values.
        Some common examples for AF are:
            1) Sigmoid or SoftMax Function (has Vanish Gradient problem)
                Softmax outputs produce a vector that is non-negative and sums to 1. It's useful when you have mutually exclusive categories 
                ("these images only contain cats or dogs, not both"). You can use softmax if you have 2,3,4,5,... mutually exclusive labels.

            2) Tanh function (has Vanish Garident problem)
                if the outputs are somehow constrained to lie in [−1,1], tanh could make sense.
            3) Relu Function
                ReLU units or similar variants can be helpful when the output is bounded above or below. 
                If the output is only restricted to be non-negative, it would make sense to use a ReLU 
                activation as the output function. (0 to Max(x))

            4) Leaky Relu Function (to fix the dying relu problem in the Relu function within hidden layers)

#### What is Convolotional Neural Network?
        convolutional-neural-network is a subclass of neural-networks which have at least one convolution layer. 
        They are great for capturing local information (e.g. neighbor pixels in an image or surrounding words in a text) 
        as well as reducing the complexity of the model (faster training, needs fewer samples, reduces the chance of overfitting).
        . A convolution unit receives its input from multiple units from the previous layer which together create a proximity.
        Therefore, the input units (that form a small neighborhood) share their weights.
![](https://miro.medium.com/max/3288/1*uAeANQIOQPqWZnnuH-VEyw.jpeg)

#### What is recurrent Neural Network?	
        A class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence.
        This allows it to exhibit temporal dynamic behavior. Derived from feedforward neural networks, RNNs can use their 
        internal state (memory) to process variable length sequences of inputs. This makes them applicable to tasks such 
        as unsegmented, connected handwriting recognition or speech recognition.
![ImageAddress](https://www.i2tutorials.com/wp-content/uploads/2019/09/Neural-network-62-i2tutorials.png)

#### What is LSTM network?
        Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture.
        Unlike standard feedforward neural networks, LSTM has feedback connections. It can not only process single
        data points (such as images), but also entire sequences of data (such as speech or video). 
        For example, LSTM is applicable to tasks such as unsegmented, connected handwriting recognition, Anomly detection in network
        traffic or IDS.
![](https://miro.medium.com/max/1400/1*qn_quuUSYzozyH3CheoQsA.png)

#### What is Convolotional Layer?	
        A convolution is the simple application of a filter to an input that results in an activation. Repeated application of the 
        same filter to an input results in a map of activations called a feature map, indicating the locations and strength of a 
        detected feature in an input, such as an image.
        You can use Filters which are based on Horizental Lines or Verticial Lines or Gray Scale conversion or other conversion filters.
![](https://qph.fs.quoracdn.net/main-qimg-29982be98b548e9a3256b68c6ecbcb60.webp)

#### What is Pooling Layer?
        Pooling layers provide an approach to down sampling feature maps by summarizing the presence of features in patches of the feature map.
        Two common pooling methods are average pooling and max pooling that summarize the average presence of a feature and the most activated 
        presence of a feature respectively.      

        This is required to downsize your feature scale (e.g You have detected vertical lines, now remove some of the feature to go in grain)
![](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/image_folder_4/Pooling_Simple_max.png)

#### What is MaxPooling Layer? How does it work?
        Max polling uses the Maximum value found in a considered size metrics. Maximum pooling, or max pooling, is a pooling operation that calculates the maximum, or largest, value in each patch of each feature map.in a way.
![](https://computersciencewiki.org/images/8/8a/MaxpoolSample2.png)
#### What is Kernel or Filter?
        kernel methods are a class of algorithms for pattern analysis, whose best known member is the support vector machine (SVM)
        Kernel functions have been introduced for sequence data, graphs, text, images, as well as vectors.
        A Kernel is used to solve Non- Linear problem by Linear Classifiers in a way that its useable.
![](https://2.bp.blogspot.com/-iNPVcxMHMNg/WdDnyLPY9QI/AAAAAAAAAZU/hgwQOQ1liyE4nhKVYzOyuUprjHNEx7aygCLcBGAs/s1600/kernel.png)


#### What is Segmentation?
        The process of partitioning a digital source into multiple segments.
        If you refer Image, Imagine Image source being converted into multiple segments such as Airplane object.
        The goal of segmentation is to simplify and/or change the representation of an image into something that
         is more meaningful and easier to analyze.

#### What is Pose Estimation?	
        Detection of poster from an Image is represented as Post Estimation.

#### What is Forward propagation?	
        The input data is fed in the forward direction through the network. Each hidden layer accepts the input data,
        processes it as per the activation function and passes to the successive layer.
![](https://miro.medium.com/max/3652/1*FczAiD6e8zWjWupOQkP_-Q.png)

#### What is backward propagation?
        Back-propagation is the essence of neural net training. It is the practice of fine-tuning the weights 
        of a neural net based on the error rate (i.e. loss) obtained in the previous epoch (i.e. iteration).
        Proper tuning of the weights ensures lower error rates, making the model reliable by increasing its generalization
![](https://i.ytimg.com/vi/An5z8lR8asY/maxresdefault.jpg)

#### what are dropout neurons?	
        The term “dropout” refers to dropping out units (both hidden and visible) in a neural network.
        Simply put, dropout refers to ignoring units (i.e. neurons) during the training phase of certain 
        set of neurons which is chosen at random. By “ignoring”, I mean these units are not considered during
        a particular forward or backward pass.
        More technically, At each training stage, individual nodes are either dropped out of the net with 
        probability 1-p or kept with probability p, so that a reduced network is left; incoming and 
        outgoing edges to a dropped-out node are also removed.

#### what are flattening layers?	
        A flatten layer collapses the spatial dimensions of the input into the channel dimension. 
        For example, if the input to the layer is an H-by-W-by-C-by-N-by-S array (sequences of images),
        then the flattened output is an (H*W*C)-by-N-by-S array.

#### How is backward propagation dealing an improvment in the model?
        practice of fine-tuning the weights of a neural net based on the error rate (i.e. loss) 
        obtained in the previous epoch (i.e. iteration). Proper tuning of the weights ensures lower error rates, 
        making the model reliable by increasing its generalization

#### What is correlation? and covariance?
        “Covariance” indicates the direction of the linear relationship between variables. 
        “Correlation” on the other hand measures both the strength and direction of the linear relationship between two variables.

        When comparing data samples from different populations, covariance is used to determine how much two random variables
        vary together, whereas correlation is used to determine when a change in one variable can result in a change in another. 
        Both covariance and correlation measure linear relationships between variables.
![](https://miro.medium.com/max/716/1*T52-LSuLQyq-6I2c1lkj-A.png)

#### What is Anova? when to use Anova?	
        Analysis of variance (ANOVA) is a collection of statistical models and their associated estimation procedures 
        (such as the "variation" among and between groups) used to analyze the differences among group means in a sample.

        Use a one-way ANOVA when you have collected data about one categorical independent variable and 
        one quantitative dependent variable. The independent variable should have at least three levels
         (i.e. at least three different groups or categories)

#### How would you define dimentionality reduction? why we use dimentionality reduction?	
        Dimensionality reduction or dimension reduction is the process of reducing the number of random variables 
        under consideration by obtaining a set of principal variables. Approaches can be divided into feature 
        selection and feature extraction.
        The reason we use it is because
                1) Immensive dataset 
                2) Longer Trainnig time/gathering time
                3) Too much complex assumptions/ Model overfitting
        Types of Dimentionality reductions are 
                1) Feature Selection
                2) Feature Projection( transform data from higher dimention to lower space of fewer dimention)
                3) Principle component Analysis
                        Linear Technique for DR, performs linear mapping of data to lower dimention
                        space in such a way variance is maximized.
                4) Non Negative Metrics Factorization
                5) Kernel PCA ( Non linear way of utilization of Kernel Trick)
                6) Graph Based Kernel PCA ( locally linear embedding, Eigen Embeddings)

                7) Linear Discrimenant Analysis
                        A method used in statistics, pattern recognition and machine learning to find a 
                        linear combination of features that characterizes or separates two or more 
                        classes of objects or events.
                8) Generalized Discrimenant Analysis 
                8) TSNE (is a non-linear dimensionality reduction technique useful for visualization of high-dimensional datasets.)
                9) U-Map
                        Uniform manifold approximation and projection (UMAP) is a nonlinear dimensionality reduction technique. 
                        Visually, it is similar to t-SNE, but it assumes that the data is uniformly distributed on a locally 
                        connected Riemannian manifold and that the Riemannian metric 
                        is locally constant or approximately locally constant.
                10) Autoencoders (can learn from Non Linear dimention reduction function)

#### What is Principle componenet analysis? how does PCA work in Dimentonality reduction?

        The main linear technique for dimensionality reduction, principal component analysis, performs
         a linear mapping of the data to a lower-dimensional space in such a way that the variance of 
         the data in the low-dimensional representation is maximized. In practice, the covariance (and 
         sometimes the correlation) matrix of the data is constructed and the eigenvectors on this 
         matrix are computed. The eigenvectors that correspond to the largest eigenvalues (the 
         principal components) can now be used to reconstruct a large fraction of the variance of the 
         original data. The original space (with dimension of the number of points) has been reduced 
         (with data loss, but hopefully retaining the most important variance) to the space spanned by 
         a few eigenvectors

#### What is Maximum Likelihood estimation?
        Maximum likelihood estimation is a method that determines values for the parameters of a model. 
        The parameter values are found such that they maximise the likelihood that the process described by the model
        produced the data that were actually observed.

#### What is Naive Bayes? How does it works?
        A method of estimating the parameters of a probability distribution by maximizing a likelihood function, 
        so that under the assumed statistical model the observed data is most probable
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/52bd0ca5938da89d7f9bf388dc7edcbd546c118e)

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/d0d9f596ba491384422716b01dbe74472060d0d7)


#### What is Bayes Theorm?
        The probability of an event, based on prior knowledge of conditions that might be related to the event.

#### What is Probability?
        Probability is a number between 0 and 1, where, roughly speaking, 0 indicates impossibility and 1 indicates certainty.
        The higher the probability of an event, the more likely it is that the event will occur. 
        Example:
        A simple example is the tossing of a fair (unbiased) 
        coin. Since the coin is fair, the two outcomes ("heads" and "tails") are both equally probable; the probability of "heads" equals the probability 
        of "tails"; and since no other outcomes are possible, the probability of either "heads" or "tails" is 1/2 (which could also be written as 0.5 or 
        50%).
[ReferenceLink](https://machinelearningmastery.com/joint-marginal-and-conditional-probability-for-machine-learning/)
#### What is Joint Probability?
        Joint probability is a statistical measure that calculates the likelihood of two events occurring together and at the same point in time.
                P(A and B) or P (A ^ B) or P(A & B)
                The joint probability is detremeinded as :
                P(A and B) = P(A given B) * P(B)       
![](https://image.slidesharecdn.com/probabilitydistribution-150117052614-conversion-gate02/95/probabilitydistribution-14-638.jpg?cb=1421494048)

#### What is Marginal Probability?
        Probability of event X=A given variable Y. Single Random event probability 
        P(A) , A single probability of an independent event.

#### What is Conditional Probability? what is distributive Probability?
        Probability of event A given event B is termed as Conditional Probability.

#### What is Z score?
        Z score (also called the standard score) represents the number of standard deviations with which the 
        value of an observation point or data differ than the mean value of what is observed

#### What is KNN how does it works? what is neigbouring criteria? How you can change it ?
        KNN is dependent on distancing estimation from the points of a Class to respectable points in class, thus acting as a Vote Based Neigbouring
        Classifier, where you conclude the outcome of your input to be predicted by measuring which points come close to it.
        You can have as much as neigbours you want, the more you specify neigbours the more classes it will use to evaluate the final outcome.

        Working is quite similar than a distancing algorithm, although you draw the point and calculate all the neigbouring by looking which
        are close, when you are done with it you go with as votes, E.g Class A were 5 classes and Class B were 2 classes in that neigbour hood.
        Hence the vote would be class A.


#### Which one would you prefer low FN or FP's based on Fraudial Transaction?
        Recommended is low FN's, the reason is because if you consider Fraudly Transaction being occured and counting it as not being occured 
        This has huge impact on the Business model.

#### Differentiate between KNN and KMean?
        KMean: Unsupervised, Random points drawn, each uses distance based averages for prediction.
        KNN: Supervised, neigbouring, C values , Voting 
![](https://qph.fs.quoracdn.net/main-qimg-e50401a4bdaf033ef6b451ea72334f8b)

#### What is Attention ? Give Example ?
        A neural attention mechanism equips a neural network with the ability to focus on a subset of its inputs (or features).
        1) Hard Attention (Image Cropping)
        2) Soft Attention (Highlight attentional area keeping the image size same)
![](https://distill.pub/2016/augmented-rnns/assets/show-attend-tell.png)

#### What are AutoEncoders? and what are transformers?

        Autoencoders take input data, compress it into a code, then try to recreate the input data from that summarized code. It’s like starting with Moby Dick, creating a SparkNotes version and then trying to rewrite the original story using only SparkNotes for reference. While a neat deep learning trick, there are fewer real-world cases where a simple autocoder is useful. But add a layer of complexity and the possibilities multiply: by using both noisy and clean versions of an image during training, autoencoders can remove noise from visual data like images, video or medical scans to improve picture quality.

#### What is Image Captioning?	
        Image Captioning is the process of generating textual description of an image. It uses both Natural Language Processing and Computer Vision to
        generate the captions
![](https://cdn.analyticsvidhya.com/wp-content/uploads/2018/03/example.png)

#### Give some example of Text summarization.	
        Summarization is the task of condensing a piece of text to a shorter version, reducing the size of the initial text while preserving the meaning.
        Some examples are :
                1) Essay Summarization
                2) Document Summarization
                etc

#### Define Style Transfer?

#### Define Image Segmentation and Pose Analysis?	
        Image Segmentation : In digital image processing and computer vision, image segmentation is the process of partitioning a digital image into 
        multiple segments (sets of pixels, also known as image objects). The goal of segmentation is to simplify and/or change the representation of 
        an image into something that is more meaningful and easier to analyze.

        Pose Analysis:
                The process of determining the location and the orientation of a Human Entity (pose).

![PoseSegmentation](https://i.ytimg.com/vi/KYNDzlcQMWA/maxresdefault.jpg)


#### Define Semantic Segmentation?
        Semantic Segmentation is the Segmentation of an image based on Type of Objects
![](https://miro.medium.com/max/2436/0*QeOs5RvXlkbDkLOy.png)
#### What is Instance Segmentation?
        Same as Semantic, although with Objects (with respectable ID's)

#### What is Imperative and Symbolic Programming?
![](https://slideplayer.com/slide/14913960/91/images/17/Imperative+vs+Symbolic+Programming.jpg)


#### Define Text Classification, Give some usecase examples?
        Text classification also known as text tagging or text categorization is the process of categorizing text into organized groups. 
        By using Natural Language Processing (NLP), text classifiers can automatically analyze text and then assign a set of pre-defined
        tags or categories based on its content.
        UseCases:
                1) Document Classification
                2) Document Categorization
                3) Point of Interest in Document
                4) OCR
                etc
![](https://www.researchgate.net/profile/Raghava_Rao_Mukkamala/publication/321892732/figure/fig3/AS:574016848764930@1513867689381/Text-Classification-Architecture.png)

#### which algorithms to use for Missing Data?
![](https://www.researchgate.net/publication/330704615/figure/fig2/AS:720385997815812@1548764814471/Machine-learning-with-missing-data-Conventional-single-imputation-methods-for-handling.ppm)

#### How would you define GAN(Generative Adversarial Networks) ?

#### What are Gausian Processes?

#### What is Graph Neural Network?



#### What is Language Modelling(LM), give examples?


#### Define Named Entity recognition? Give some usecases where it can come in handy?