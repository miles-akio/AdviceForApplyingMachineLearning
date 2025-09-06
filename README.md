# Advice for Applying Machine Learning

```markdown

This project explores techniques for evaluating and improving machine learning models.  
I walk through different approaches using **polynomial regression** and **neural networks**, focusing on concepts such as bias and variance, model complexity, regularization, and evaluation strategies.  

The goal of the project is to demonstrate how I assess model performance and tune models to generalize better on unseen data.  

---

## Outline
- [1 - Packages](#1)
- [2 - Evaluating a Learning Algorithm (Polynomial Regression)](#2)
  - [2.1 Splitting the data set](#2.1)
  - [2.2 Error calculation for linear regression](#2.2)
  - [2.3 Training vs test performance](#2.3)
- [3 - Bias and Variance](#3)
  - [3.1 Visualizing train, CV, and test sets](#3.1)
  - [3.2 Finding the optimal polynomial degree](#3.2)
  - [3.3 Tuning regularization](#3.3)
  - [3.4 Effect of training set size](#3.4)
- [4 - Evaluating a Learning Algorithm (Neural Network)](#4)
  - [4.1 Data set](#4.1)
  - [4.2 Classification error](#4.2)
- [5 - Model Complexity](#5)
  - [5.1 Complex vs simple models](#5.1)
- [6 - Regularization](#6)
- [7 - Iterating to find optimal regularization](#7)

---

<a name="1"></a>
## 1 - Packages

I used the following libraries throughout the project:
- [numpy](https://numpy.org/) for numerical operations
- [matplotlib](http://matplotlib.org) for visualization
- [scikit-learn](https://scikit-learn.org/stable/) for regression models and preprocessing
- [tensorflow](https://www.tensorflow.org/) for neural network models  

---

<a name="2"></a>
## 2 - Evaluating a Learning Algorithm (Polynomial Regression)

When building a machine learning model, fitting the training data well is not enough—the model must also generalize to new data.  

The process I used:
1. Split the dataset into **training** and **test** sets.
2. Train the model on the training data.
3. Evaluate error on unseen test data.

---

<a name="2.1"></a>
### 2.1 Splitting the data set
I split the dataset using `train_test_split` from scikit-learn, reserving around 30–40% for testing.  
This ensures I can evaluate how well the model generalizes.  

---

<a name="2.2"></a>
### 2.2 Error calculation for linear regression
I used **Mean Squared Error (MSE)** to evaluate predictions:  

\[
J_\text{test}(w, b) = \frac{1}{2m}\sum_{i=0}^{m-1}(f_{w,b}(x^{(i)}) - y^{(i)})^2
\]

This provides a simple but effective metric to compare training and test performance.  

---

<a name="2.3"></a>
### 2.3 Training vs test performance
I trained a high-degree polynomial regression model and compared training error to test error.  
The results showed:
- Low training error
- High test error  

This is a classic case of **overfitting**: the model memorized training data but failed to generalize.  

To address this, I also introduced a **cross-validation (CV) set** and used the standard split:  
- Training: 60%  
- Cross-validation: 20%  
- Test: 20%  

---

<a name="3"></a>
## 3 - Bias and Variance

Bias and variance were explored by varying polynomial degrees and regularization.  

<a name="3.1"></a>
### 3.1 Visualizing train, CV, and test sets
By plotting the data, I could visually assess where the model was underfitting (high bias) or overfitting (high variance).  

<a name="3.2"></a>
### 3.2 Finding the optimal polynomial degree
I trained models with different polynomial degrees and measured both training and CV error.  
- Low degree → underfitting (high bias)  
- High degree → overfitting (high variance)  
- Optimal degree → balanced performance  

<a name="3.3"></a>
### 3.3 Tuning regularization
I applied **ridge regression** with different values of lambda (λ).  
- Small λ → risk of overfitting  
- Large λ → risk of underfitting  
- Optimal λ → best generalization  

<a name="3.4"></a>
### 3.4 Effect of training set size
When overfitting, adding more data helped reduce variance.  
However, for underfitting models, adding data did not help.  

---

<a name="4"></a>
## 4 - Evaluating a Learning Algorithm (Neural Network)

Next, I built a neural network to classify clustered data.  

<a name="4.1"></a>
### 4.1 Data set
The dataset was generated using synthetic clusters, split into training, cross-validation, and test sets.  

<a name="4.2"></a>
### 4.2 Classification error
For classification, I measured error as the **fraction of incorrect predictions**.  
This provided a direct way to evaluate how well the neural network separated the clusters.  

---

<a name="5"></a>
## 5 - Model Complexity

I compared:
- A **complex neural network** with multiple dense layers  
- A **simple neural network** with fewer parameters  

Results:
- The complex model overfit the training data.  
- The simple model generalized better on the CV set.  

---

<a name="6"></a>
## 6 - Regularization

I applied L2 regularization to the complex neural network.  
This significantly reduced overfitting and improved CV performance, making it behave similarly to the “ideal” model.  

---

<a name="7"></a>
## 7 - Iterating to Find Optimal Regularization

I tested multiple λ values and observed how training and CV errors converged.  
For this dataset, λ > 0.01 produced the best tradeoff between bias and variance.  

---

## Key Takeaways

Through this project, I explored:
- **Bias vs Variance** tradeoff  
- Importance of **training, cross-validation, and test splits**  
- How **regularization** improves generalization  
- Differences in **complex vs simple models**  
- The value of **adding more data** when combating variance  

This workflow provides a strong foundation for evaluating and improving machine learning models in real-world applications.  
```

---
