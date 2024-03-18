# Introduction
Author: Kaiwen Bian & Bella Wang

This project demonstrate how we may draw insights from a highly unbalanced data set using ensemble learning. Predictive model detecting user preference using **textual features** in combnation with other **numerical features** is the key first step prior to building a reconmander system or doing any other further analysis. The challenge that is addressed in this project is related to the highly imbalance nature of the `recipe` data set that we are using.

### Random Forest Algorithm
In this project, we will adapt ideas of **homogenous ensemble learning** where we will use multipl **Decision Trees**, and making them into a **Random Forest** for more  robust predictions of the data.

A **Decision Tree** essentially learns to come up with questions or decisions at an high dimensional space (depending on the number of features) and then separate the data using "boxes" or "lines" in that way. The core mechanism that allows it to happen is using *entropy minimization* where the model tries to reduce the entropy, or uncertainty of each split, making one catagory fit to one side and the other catagory to the other side.

A **Random Forest** essentially is when at the splitting point of data to train/test/val, **a random subset of features** is taken out instead of choosing from all of them and then spliting the tree base on this subset of the feature, usually speaking $m = sqrt(d)$ seems to work well in practice and it is also the default that `sk_learn` uses. **This allows each decision trees to come up with different prediction rules for later on voting an best one**
- Notice that we are not doing simple boostrap of the data as each decision tree may not resemble too great of a difference in that way, instead, we are taking different features directly using the same type of model (decision tree), making it a homogenous ensemble learning method.
- We want the individual predictors to have low bias, high variance, and be uncorrelated with each other. In this way, when averaging (taking votes) them together, low bias and low variance would occur.

<p align="center">
  <img src="assets/rfc.png" alt="random forest classifier" width="800" height="600"/>
</p>

[Full report Document](assets/report.pdf)

<a href="https://github.com/KevinBian107/ensemble_imbalance_data" style="background-color: #007bff; color: white; padding: 10px 15px; text-decoration: none; border-radius: 5px; font-size: 16px;">Visit Developer Repository</a>


### Feature Importantness
<iframe src="assets/feature.html" width="800" height="600" frameborder="0"></iframe>