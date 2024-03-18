# Introduction
Author: Kaiwen Bian & Bella Wang

[Full Report Document](assets/report.pdf)

This project demonstrate how we may draw insights from a highly unbalanced data set using ensemble learning. Predictive model detecting user preference using **textual features** in combnation with other **numerical features** is the key first step prior to building a reconmander system or doing any other further analysis. The challenge that is addressed in this project is related to the highly imbalance nature of the `recipe` data set that we are using.

## Random Forest Algorithm
In this project, we will adapt ideas of **homogenous ensemble learning** where we will use multipl **Decision Trees**, and making them into a **Random Forest** for more  robust predictions of the data.

A **Decision Tree** essentially learns to come up with questions or decisions at an high dimensional space (depending on the number of features) and then separate the data using "boxes" or "lines" in that way. The core mechanism that allows it to happen is using *entropy minimization* where the model tries to reduce the entropy, or uncertainty of each split, making one catagory fit to one side and the other catagory to the other side.

A **Random Forest** essentially is when at the splitting point of data to train/test/val, **a random subset of features** is taken out instead of choosing from all of them and then spliting the tree base on this subset of the feature, usually speaking $m = sqrt(d)$ seems to work well in practice and it is also the default that `sk_learn` uses. **This allows each decision trees to come up with different prediction rules for later on voting an best one**
- Notice that we are not doing simple boostrap of the data as each decision tree may not resemble too great of a difference in that way, instead, we are taking different features directly using the same type of model (decision tree), making it a homogenous ensemble learning method.
- We want the individual predictors to have low bias, high variance, and be uncorrelated with each other. In this way, when averaging (taking votes) them together, low bias and low variance would occur.

## Content for this Project
1. Introduction
2. Data Cleaning, Transformation, and EDA
    - Transformation
    - Univariate & Bivariate Analysis
    - Aggreagted Analysis
    - Textual Feature Analysis
3. Assessment of Missingness Mechanism
    - MAR Anlaysis
    - NMAR Analysis
4. Permutation Testing of TF-IDF
5. Framing a Predictive Question
6. Baseline Model: An Naive Approach
    - Handling Missingness in Data
    - Train/Val/Test Split
    - Feature Engineering
7. Final Model: Homogenous Ensemble Learning
    - Feature Engineering (Back to EDA)
    - Model Pipeline
    - Hyperparameter Tuning
    - Evaluation
        - Feature Importantness
        - Confusion Matrix, Evaluation Metrics, and ROC_AUC
8. Fairness Analysis

# Data Cleaning, Transformation, and EDA
## Merging & Transformation
Initial merging is needed for the two dataset (`interaction` and `recipe`) to form one big data set. We performed a series of merging as follows:
1. Left merge the recipes and interactions datasets together.
2. In the merged dataset, we also filled all ratings of 0 with `np.NaN` as `rating` of zero doesn't make sense, we will be evaluating this in the `missingness mechanism` section.
3. We then find the average rating per recipe (as a series) and add this series containing the average rating per recipe back to the recipes dataset.

We also performed a series of follow up transformations to fit our needs for the data set as follows:
1. Some columns, like `nutrition`, contain values that look like lists, but are actually strings that look like lists. We turned the strings into actual columns for every unique value in those lists
2. Convert to list for `steps`, `ingredients`, and `tags`
3. Convert `date` and `submitted` to Timestamp object and rename as `review_date` and `recipe_date`
4. Convert Types
5. Drop same `id` (same with `recipe_id`)
6. Replace 'nan' with np.NaN

After the transformation, we have types of each of the columns as the following:
1. `String`: [name, contributor_id, user_id, recipe_id, ]
    - quantitative or qualitative, but cannot perform mathamatical operations (**quntitative discrete**)
    - `name` is the name of recipe
    - `contributor_id` is the author id of the recipe _(shape=7157)_
    - `recipe_id` is the id of teh recipe _(shape=25287)_
        - `id` from the original dataframe also is the id of the recipe, dropped after merging
    - `user_id` is the id of the reviewer _(shape=8402)_
2. `List`: [tags, steps, description, ingredients, review]
    - qualitative, no mathamatical operation (**qualitative discrete**)
3. `int`: [n_steps, minutes, n_ingredients, rating]
    - quantitative mathamatical operations allowed (**quantitative continuous**)
4. `float`: [avg_rating, calories, total_fat sugar, sodium, protein, sat_fat, carbs]
    - quantitative mathamatical operations allowed (**quantitative continuous**)
5. `Timestamp`: [recipe_date, review_date]
    - quantitative mathamatical operations allowed (**quantitative continuous**)

## Univariate & Bivariate Analysis
We will be performing **Explorative Data Analysis** for our data set:

img

Looks like that our data have a lot of outliers! we might want to write a function to deal with that. Here we are writing the function `outlier`, which will be used quite often later on.

img

Looks like the data are kind of **imbalanced** in `rating` (at this point, we thought that this wouldn't effect our modle too much, but it turns out later to be one of the main challenge that we need to deal with during the moeling phase).

img

Seems like there is a **threshold point** for `n_ingredients` and `n_steps`, this will be utilized later in our **feature engineering** section.

img

It also seems like more `sugar` and more `total_fat` (transformed from `nutrition`) seems to be related to higher `rating`! This is quite suprising!

img

Seems like there is some sort of relationships between `n_steps`, `n_ingredients`, and the `rating` column. However, this relationship doesn't seem to be that exact. In a later section we might use this idea.

## Aggreagted Analysis
Now we can first use the groupby function that we have implemented to look at some aggregated data first before using it for the next few sections

img

Looking at the right column of graph, it seems like the previous relationships taht we observed in no aggregation data is still preserved in the aggregated version where higher `calories` seems to be correlated to higher `rating` and `n_ingredients` and `n_steps` seems to have some relationships with `rating` as well.

img

When aggregating by user, something interesting appears, it seems like that `rating` column is not so much correlated with teh `n_steps` and `n_ingrredients` column though it is still quite correlated with the `calories` column. **Though we will not be working with this version of the aggregated data frame firectly when we are making our predictive model, this ideas may be taken into considerations when choosing features.**

img


## Textual Feature Analysis
We actually made more edas and feature engineering with **textual features**, but we will introduce those later in the section as it is much more relevant to our modeling process. For now, we will show some technique with TF-IDF that we will use later on in this project by checking the top 5 **most important** words in each of the rows (recipe_id) in the **original cleaned** data frame filtered by getting only the **5 rating recipes**(note, recipe_id is not unique here).
- We will probably not directly use this approach here as it runs really slow! But we may use a similar approach that have a better runtime complexity!

img

# Assessment of Missingness Mechanism

## MAR Anlaysis

## NMAR Analysis

# Permutation Testing of TF-IDF

# Framing a Predictive Question

# Baseline Model: An Naive Approach

## Handling Missingness in Data

## Train/Val/Test Split

## Feature Engineering

# Final Model: Homogenous Ensemble Learning

## Feature Engineering (Back to EDA)

## Model Pipeline

## Hyperparameter Tuning

## Evaluation

### Feature Importantness

### Confusion Matrix, Evaluation Metrics, and ROC_AUC

## Fairness Analysis

<p align="center">
  <img src="assets/rfc.png" alt="random forest classifier" width="800" height="600"/>
</p>

<a href="https://github.com/KevinBian107/ensemble_imbalance_data" style="background-color: #007bff; color: white; padding: 10px 15px; text-decoration: none; border-radius: 5px; font-size: 16px;">Visit Developer Repository</a>