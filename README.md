# Machine learning in Octave/Matlab: typical machine learning algorithms ##

###### 1:  Linear regression with one variable:
   1. Computing the cost, minimizing the cost function using Gradient Descent;
   2. Visualizing J(theta_0, theta_1)
   3. Feature Normalization - scaling features and seting them to zero mean
   4. Linear regression with multiple variables - Gradient Descent - prediction
   5. Selecting learning rates
   6. Normal Equations - prediction

###### 2: Logistic Regression
   1. Compute Cost and Gradient
   2. Optimizing using fminunc (Octave-Matlab) - find optival theta
   3. Plot decising Boundary
   4. Using the logistic regression model to predict the probability, compute the training and test set accuracies
   5. Regularization with logistic regression
   6. Dataset with data points that are notlinearly separable - using more features - polynomial features
   7. Initial cost and gradient for regularized logistic regression
   8. Different lambda - how regularization affects the decision boundary and accuracy
   
###### 3: Multi-class Classication and Neural Networks
   1. Vectorizing logistic regression, one-vs-all classification
   2. Predict for One-Vs-All
   3. NN - predict,  training set accuracy

###### 4: Neural Network Learning
    1. Compute Cost (Feedforward)
    2. Regularization
    3. Sigmoid Gradient
    4. Random initializing Pameters
    5. Backpropagation
    6. Regularization gradient
    7. Training NN (using "fmincg")
    8. Using different values of lambda and max iter
    9.  Visualize Weights
    10. Training set accuracy

###### 5: Regularized Linear Regression and Bias vs. Variance
   1. Regularized Linear Regression (cost function, gradient): y=theta0+ theta1*x1 (dimention=1)
   2. Bias-variance tradeoff: plot training and test errors on a learning curve to diagnose bias-variance problems (use different amount of testing data m:=1:m)
   Indicator of variance is gab between training and test errors cureve.
   3. Move to Polynomial regression: use biger, dimention=8: y=theta0+ theta1*x1 +...+theta8*x1^8 (+ Feature Normalization)
   4. Adjusting the regularization parameter lambda (0..100)
   5. Find best lambda automatically.

###### 6: SVM
   1. Training Linear SVM with different C
   2. Implementing Gaussian Kernel
   3. K-Means Clustering on Pixels
   3. Find best C and sigma automatically
   4. Spam filter

###### 7:  Principle Component Analysis and K-Means Clustering
   1. K-Means
   2. K-Means Clustering on Pixels - image compression
   3. Principle Component Analysis (PCA)
     
###### 8:  Anomaly Detection and Collaborative Filtering
   1. Anomaly Detection in server statistics across several machines
   2. Selecting the threshold epsolon via F1 alg (using precision and recall)
   3. Collaborative Filtering on movie ratings dataset: "visualize" the ratings, Cost Function, gradient for Theta and X,    regularization, tarain model
   4. Set own movie rating for short movie list - get  movie recommendation for new user
  
