



# Metrics

- Classification

  - Accuracy: true_positive + true_negative

  - Recall/Precision/F. F1 is balance between recall and precision

  - NegativeLogLikelyhood 

  - Area Under ROC Curve(AUC) for binary classification

    he AUC for a classifier with no power, essentially random guessing, is 0.5, because the curve follows the diagonal. The AUC for that mythical being, the perfect classifier, is 1.0. Most classifiers have AUCs that fall somewhere between these two values.

  - Confusion Matrix

- Regression

  - Mean Absolute Error
  - Mean Square Error. It gives more punishment for large error
  - R^2 (R Sqaure)





# Model

- Loss Function 

  need to be non-negative

  0 = perfect prediction

- No Free Lunch Theorem

  > No single model that works best!
  >
  > Model makes assumptions!

- Model Selection

  - Cross-validation

  - Bayesian model selection

    $P(m|D) = \frac{P(D|m_0)}{P(D|m_1)} * \frac{P(m_0)}{P(m_1)}$ (Usually same prior)

- Bias-Variance Tradeoff

  - Bias: error from mean 
  - Variance: spread around center

- Bayesian Information Criterion (BIC)

  $BIC = DOF(\theta)log N - 2logP(D|\theta)$ : penalise high DOF model, mitigated by its likelyhood 



# Linear Regression

- Assumptions:

  1. Linearity: data is linear
  2. Independence of error: data is IDD sampled 
  3. Residue(error) is Gaussian
  4. Equal variance of errors

- Loss Function: 

  - Based on Gaussian Error 
  - MSE 

- Optimization (Solution):

  - Derivation of close form solution $(X^TX)^{-1}X^Ty$

- Gaussian MLE always biased: $E[\sigma^2] = \frac{n-1}{n} \sigma^2$
  Smaller than the true error. Because we measure from sample 

- Expectation of Error = bias * bias + variance

- Regularization

  - More regularization -> high bias term -> more bias

  - Less regularization -> more overfitting -> more variance

    ​

# KNN

- Idea: choose class based on K nearst neighbor 

- Caveats:

  - Need to normalise vector

- Model Complexity:

  - K too small: overfitting 

  - K too high: underfit

    ​

# Logistic Regression

- Idea: $p(y=1|x) = \frac{1}{1 + e^{- w^T x}} y \in \{0, 1\}$
- Why not use Linear Regression for Classification?
  - Ans: Classification does not care about the magnitude of loss, outlier will hur linear regression for classification badly.
- Derivation
  - odds of two class $\frac{P(y=1|x)}{P(y=0|x)} = e^{w^Tx}$
  - then we have $\frac{p}{1-p} = e^{w^Tx}$
  - solve for $p$
- Soft-max interpretation
  - $\frac{e^{G(x,y)}}{e^{G(x)} + e^{0}}$
- Loss function:
  - LIkelyhood function: MLE of every data $\sum log[y_i P(y=1|x_i) + (1-y_i)P(y_i=0|x_i)]$
- Optimization:
  - Gradient: $-\sum(y_i-p(y_i|x_i, w_i)) x_i$
  - Gradient Ascent
    - SGD or Batch or Whole Data

# Perceptron

- Idea: $y = sign(w^Tx) y \in \{-1,1\}$
-  Loss function: 
  - 1 loss if incorrect, 0 otherwise
  - or $L = max(0, -y_iw^Tx_i)$
- Optimization:
  - Gradient: 
    - $\nabla L=0$ if $y_iw^Tx_i > 0$, else $-y_ix_i$
  - Weight Update:
    - $ w^{i+1} = w^{i} + \eta (y_i - \hat{y}_i)x_i$, ($y-\hat{y_i} $ is either 2 or -2)
- Online-Learning vs. Batch Learning
  - Assumptions behind `batch learning`:
    - **one optimal hypothesis fits all of the data**
    - data are IID
    - Advantage:
      - Guarentee convergency
  - Online Learning:
    - No train-test data split. Infinite data steam
    - **No consistency in hypothesis**
    - Advantage
      - Handle large data
      - Faster for lots of data

# SVM

- Idea: find the best plane that separate data by maximise the margin to each of example

- Derivation (highly recommend reading: [Stanford CS229 Notes](http://cs229.stanford.edu/notes/cs229-notes3.pdf))
  - Functional Margin: $\hat{\gamma}_i =y_i (w^Tx_i + b)$

  - Geometric Margin (norm of vector AB), and it satisfy following function
     $\gamma \dot{\frac{W}{||w||}} = A - B$. Also since $B$ is on the decision boundary, $w^TB = 0$, we can derive geo margin $\gamma_i = y_i [(\frac{w}{||w|})^T x_i + \frac{b}{||w||}]$
      We can see that if $||w|| = 1$, then $\gamma = \hat{\gamma}$, which implies $\frac{\hat{\gamma}}{\gamma} = ||w||$
      ​
      ​

  - ![alt text](SVM_margin.png "Logo Title Text 1")

  - Therefore, our objective is to:

    > max $\gamma$, s.t. $y_i (w^Tx_i + b) \geq \gamma, \forall i$ and $||w||=1$

    Then, since $\frac{\hat{\gamma}}{\gamma} = ||w||$, we can rewrite objective function to:

    > max $\frac{\hat{\gamma}}{||w||}$, s.t. $y_i (w^Tx_i + b) \geq \hat{\gamma}, \forall i$

    Finally, since maximise max $\frac{\hat{\gamma}}{||w||}$ is equivalent to min $\frac{1}{2}||w||^2$, and $\hat{\gamma}$ can be arbitrarily scaled, we set $\hat{\gamma} = 1$, then we have

    > max $\frac{1}{2}||w||^2$, s.t. $y_i (w^Tx_i + b) \geq 1, \forall i$

- Loss function

  - Hinge Loss: $max(0, 1-y \dot{} wx)$



## SVM Duality

- Dual vs Primal max (Please refer to Andrew Ng CS229 notes)

  ![alt text](svm_duality.png "Logo Title Text 1")

  - Note $w = \sum [a_iy_ix_i]$ and only some of $a_i$ are non-zero, which are the support vector

- Slack Variable

  - Add regulariser term $min_w 1/2 ||w||^2 + C\sum \epsilon_i$ s.t $(w^Tx_i)y_i + \epsilon_i \geq 1$
  - Larger C means more penalty on slacking the margin
    - Better fit to data. More overfitting.

- New Prediction for test data $x_j$ is given by $x_j^T\dot{}w = x_j^T\sum[a_iy_ix_i] = \sum a_iy_i(x^T_j x_i)$

## Kernel

- How to use kernel?

  1. Feature mapping: $\phi(x)$ to higher dimension to make data linearly separable. e.g $\phi(x)=[x_{(1)}^2, \sqrt{2}x_{(1)}x_{(2)}, x^2_{(2)}]$

     In primal form, we need to learn $w$ for each feature, which could be very large

     While in dual form, we don't have $w$

  2. We can use $\phi(x)$ instead of $x$ in our computation in order to learn non-linear boundary.

  ![alt text](svm_feature_mapping.png "Logo Title Text 1")

  3. Then we can define Kernel $K(x, x') = \phi(x)^T\phi(x')$. 

     Why we want to define it as kernel? Because we can often have simpler solution to Kernel without actually computing the $\phi$ in a possibly very high dimension

     ![alt text](svm_kernel.png "Logo Title Text 1")


For example above, we **don't** need to actually compute the inner product of $\phi(x)$, which is $(x^2_{1}x'^2_{(1)} + …… + 2x_{(1)}x_{(2)}x'_{(1)}x'_{(2)})$,  All we need to compute is $(x \dot{} x')^2$

- Kernel as similarity Function

  ![alt text](svm_kernel_sim.png "Logo Title Text 1")

- What qualify a Kernel?

  - You can decompose it into $\phi(x)^T\phi(x')$
  - Or it satisfy Mercer's Theorem : kernel matrix is symmetric positive semi-definite
    - Kernel Matrix $\mathbf{K}_{ij} = \mathbf{K}_{ji}$
    - $\mathbf{x}^TK\mathbf{x} \geq 0, \forall\mathbf{x} \in R^m$

- Kernel formation

  ![alt text](svm_kernel_formation.png "Logo Title Text 1")

- Gaussian Kernel

  ![alt text](svm_gaussian_kernel.png "Logo Title Text 1")

  Larger $\sigma$ means the support vector has greater influence area

- Summary

  - Good
    - Arbitrarily high dimension
    - Extension to other data types
    - Non-linearity
  - -Bad
    - Choose kernel is not easy
    - Can not handle large data

# Decision Tree

- ID3 Algorithm

  ![alt text](decision_tree_id3.png "Logo Title Text 1")

  - Terminating when:
    - All data has same label
    - Unseen example or feature value
    - No further splits possible (run out of splitting feature or feature value for data are the same, but somehow label are not)

- IG and Entropy

  ![alt text](decision_tree_ig.png "Logo Title Text 1")

- Overfitting and Underfitting

  - Overfitting: complete tree remembers every data
  - Underfitting: one level tree returns majority label

- Tree Pruning 

  ![alt text](decision_tree_prune.png "Logo Title Text 1")

- Summary

  ![alt text](decision_tree_procon.png "Logo Title Text 1")

# Ensemble Method

## Random Forrest

-  Data Sampling: With dataset of size N, create M different samples of size N by resampling wit replacement
  - Bootstrapping and Bagging
-  Feature Sampling: random subset of feature at each node

## AdaBoost

- Idea: weighted combined decision for weak learners

- Algorithm:

  ![alt text](adaboost_1.png "Logo Title Text 1")

  ![alt text](adaboost_2.png "Logo Title Text 1")

  - Note:
    1. Loss function is different for different classifier
    2. $a_m$ increases as error rate decreases, giving this classifier more weight in the final product

- Why AdaBoos t is not prune to over-fittting

  1. Implicit L1 regularization through stopping at a finite number of weak learners
  2. AdaBoost continues to maximize the margin, leading to better generalization (this yields the similarities with SVMs)

- AdaBoost is able to find better margin even though it training error goes down fast.

  - Other practical Advantages
    - Not prune to over-fitting
    - Fast
    - Easy to implement
    - No parameter tunning 
    - Not specific to any weak learner
    - Well-motivated by learning theory
    - Can identify outliers

## Diversity and Weighted Experts

- Idea: Diversity in classifiers, datasets
  - Data diversity: feature/instance bagging



# Neural Net

- Two layer NN classification

  - $y_k(x, w) = \sigma(\sum_j^M w_{kj}[h(\sum_{i}^D w_{ji}x_i)+w_{j0}] + w_{k0})$

    Where D is number of input;

    M is number of hidden layer neurones

    K is number of classes

    $\sigma$ Is sigmoid function

    $h$ Is a non-linear activation function

- Loss function:

  - Cross Entropy for classification: [Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy) is the number of bits we'll need if we encode symbols from yy using the *wrong* tool ŷ 

    ![alt text](cross_entropy.png "Logo Title Text 1")

    - Sidenotes: KL divergence is just the difference between cross and entropy

  - OLS  

- Optimisation:

  - Gradient descent 
  - Back propagation: Chain rule of derivation through computation graph.

###Problems with NN:

1. Required Large Labelled Data otherwise easy to overfit.
2. Non-convex objective -> easy stuck in local optima
3. "Vanishing Gradient": as gradient propagate back, gradient become even smaller

### How To Solve?

- Idea1: Back-propragate + SGD -> not always work

- Idea 2: Supervised Pre-training

  - Iteratively add hidden layer. Train current hidden layer with previous hidden layer fixed
  - Then train once together

- Idea 3: Unsupervised Pre-training via Auto-Encoder

  - Auto-encoder: learn input structure by training to encoding input.

  - Better than idea 2 because it is trying to perform a different task: learning the structure in the data which can resulting in a better starting point

    ![alt text](nn_auto_encoder.png "Logo Title Text 1")

# Transfer Learning

Pretrain ConvNet with a large dataset and then use it as one of following 3 ways:

 	1. **ConvNet as fixed feature extractor**. Remove last layer fo ConvNet and use the output of the last hidden layer(firstly pass through ReLu function if in original ConvNet there is ReLU ). Then use the output as features to feed SVM or SoftMax or other ML models
	2. **Fine-Tune the on new dataset**. It is possible to fine-tune all the layers of the ConvNet, or it’s possible to keep some of the earlier layers fixed (due to overfitting concerns) and only fine-tune some higher-level portion of the network. This is motivated by the observation that the earlier features of a ConvNet contain more generic features .
	3. **Pretrained models**. Since modern ConvNets take 2-3 weeks to train across multiple GPUs on ImageNet, it is common to see people release their final ConvNet checkpoints for the benefit of others who can use the networks for fine-tuning. 

More to look at http://cs231n.github.io/transfer-learning/

##Dimension Reduction

###Background

####Covariance

 It is a measure of the extent to which corresponding elements from two sets of ordered data move in the same direction. Formula is shown above denoted by *cov(x,y)* as the covariance of *x* and *y*.

$cov(x, y) = \frac{\sum(x_i - \bar{x}) (y_i -\bar{y})}{N}$

![covariance](/Users/jiaojingping/Developer/ml_notes/covariance.png)



#### Eigen Value and Eigen Vector

http://setosa.io/ev/eigenvectors-and-eigenvalues/

Assume $A$ is matrix and $v$ is a vector. $Av$ Is essentially transform $v$ by applying transformation $A$. 

Then if $Av = \lambda v$ for some scalar value $\lambda$, it means transformation of $v$ by $A$ is only changing magnitude of $v$ (stretching or shrinking $ v$ while maintaining its direction).

Given a fix $A$, we can find the its eigen vectors (by selecting different $\lambda$) .

![covariance](/Users/jiaojingping/Developer/ml_notes/eigen_val_example.png)

### PCA

####Intuitions

- Firstly, What is the goal of PCA ?

  find some low independent dimension of projected data, such that the error from reconstruction of the original data error by projected data is minimised. 


- Then How to minimise reconstruction error?

  Reconstruction error is minimised when variance of projected data on each dimension is maximised. Therefore, we want to transform the original data points such that the covariance matrix of transformed data points is a diagonal matrix (zero covariance and high variance ). 

- What is Covariance Matrix? And why it should be diagonal

  We want the data to be spread out i.e. it should have high variance along dimensions. Also we want to remove correlated dimensions i.e. covariance among the dimensions should be zero (they should be linearly independent). Therefore, our covariance matrix should have -
  - large numbers as the main diagonal elements.
  - zero values as the off diagonal elements.![covariance](./cov_equation.png)![covariance](./cov_mat.png)

- Finally, how should we ensure that projected data's covariance matrix is diagonal?

  If we find the matrix of eigen vectors of $C_x$ and use that as $P$ ($P$ is used for transforming $X$ to $Y$, see the image below) , then $C_y$ (covariance of transformed points) will be a diagonal matrix. Hence $Y$ will be the set of new/transformed data points.

  ![covariance](./pca_equation.png)

#### Processpc

1. Calculate the covariance matrix *X* of data points.
2. Calculate eigen vectors and corresponding eigen values.
3. Sort the eigen vectors according to their eigen values in decreasing order.
4. Choose first k eigen vectors and that will be the new k dimensions.
5. Transform the original n dimensional data points into k dimensions.



# Time Serie

## Generalised Additive Model(GAM)

The principle behind GAMs is similar to that of [regression](https://algobeans.com/2016/01/31/regression-correlation-tutorial/), except that instead of summing effects of *individual predictors*, GAMs are a sum of *smooth functions*. Functions allow us to model more complex patterns, and they can be averaged to obtain smoothed curves that are more generalizable.



### Why GAM?

- Interpretability 
- Non-parametric, avoids the pitfalls of dealing higher order polynomial terms in linear models

### Project

- FB Prophet Package
- Tesla vs. General Motor(GM)



# Data Science Stack

!![kafka](/Users/jiaojingping/Developer/601.475-notes/kafka.png)

## Kafka

**Key takeaways:**

1. Kafka maintains feeds of messages in categories called topics.
2. We'll call processes that publish messages to a Kafka topic producers.
3. We'll call processes that subscribe to topics and process the feed of published messages consumers..
4. Kafka is run as a cluster comprised of one or more servers each of which is called a broker.

[![enter image description here](https://i.stack.imgur.com/lnhn5.png)](https://i.stack.imgur.com/lnhn5.png)

Communication between the clients and the servers is done with a simple, high-performance, language agnostic TCP protocol.

**Use Cases:**

1. **Messaging**: *Kafka works well as a replacement for a more traditional message broker*. In this domain Kafka is comparable to traditional messaging systems such as ActiveMQ or RabbitMQ
2. **Website Activity Tracking:** The original use case for Kafka was to be able to rebuild a user activity tracking pipeline as a set of real-time publish-subscribe feeds
3. **Metrics**: Kafka is often used for operational monitoring data, which involves aggregating statistics from distributed applications to produce centralized feeds of operational data
4. **Log Aggregation**
5. **Stream Processing**
6. **Event sourcing** is a style of application design where state changes are logged as a time-ordered sequence of records.
7. **Commit Log**: *Kafka can serve as a kind of external commit-log* for a distributed system. The log helps replicate data between nodes and acts as a re-syncing mechanism for failed nodes to restore their data



## Churn User Analyse

### Process

1. Understand the problem
   - **Who should be considered as churn user**
   - Is ML the most efficient solution?  Would simple heuristic would do like a rule-based system?
   - How interpretable should the system be?
   - How the system is going to be used in future?
   - Scalability?
2. Understand data we get
   - Visualize data to get a quick sense
   - Understand statistical property of data
   - Handle missing data
   - Handle outliers
3. Choose model
   - Interpretability?
   - Time/Space Constrain?
   - Data assumption met for this model?
   - Existing researches/solutions
4. Developing
   - **Baseline system**
   - Feature Engineering
   - Training and Tuning model
5. Deploy
   - Evaluate and improve
   - **Interpret model and build insight to make suggestions to relevant stake-holders**