# Introduction
This project demonstrate how we may draw insights from a highly unbalanced data set using ensemble learning. Predictive model detecting user preference using **textual features** in combnation with other **numerical features** is the key first step prior to building a reconmander system or doing any other further analysis. The challenge that is addressed in this project is related to the high unbalance nature of the `recipe` data set that we are using.

[Full report Document](assets/report.pdf)

<a href="https://github.com/KevinBian107/ensemble_unbalance_data" style="background-color: #007bff; color: white; padding: 10px 15px; text-decoration: none; border-radius: 5px; font-size: 16px;">Visit Developer Repository</a>

## Data Cleaning & Transformations
### Type Logic
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
