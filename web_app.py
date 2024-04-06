import streamlit as st
from fuzzywuzzy import process
import pandas as pd
import numpy as np
import scipy as sp
import warnings
import joblib

import clean
import process as p
#from lfm_resizable import LightFMResizable

###################################################################
############## Cleaning data and instantiating variables ##########

if 'selected_combinations' not in st.session_state:

    OPTIMAL_MATRIX = {"method": "zero_one", "threshold": 2.5}

    reviews = pd.read_parquet("raw-data.pq")
    cleaned_df = clean.merge_similar_name_breweries(reviews)
    cleaned_df = clean.merge_brewery_ids(cleaned_df)
    cleaned_df = clean.remove_dup_beer_rows(cleaned_df)
    cleaned_df = clean.remove_null_rows(cleaned_df)
    cleaned_df = clean.remove_duplicate_reviews(cleaned_df)

    cleaned_df["beer_id"] = cleaned_df["beer_beerid"].astype("category").cat.codes
    
    beers = cleaned_df.copy().drop(
        [
            "review_time",
            "review_overall",
            "review_aroma",
            "review_appearance",
            "review_profilename",
            "review_palate",
            "review_taste",
        ],
        axis=1,
    )  # drop all columns having to do with a specific review

    beers = beers[
        ["beer_id", "beer_name", "brewery_name"]
    ]  # and then reorder the columns to prioritize the most important info
    beers = beers.drop_duplicates()  # and then eliminate duplicates

    # load the model and build the interaction matrix
    model = joblib.load(
        "final-model.joblib"
    )  # LightFMResizable object with the fitted model
    int_matrix_trans = p.InteractionMatrixTransformer(cleaned_df)
    matrix = int_matrix_trans.fit(**OPTIMAL_MATRIX)

    
    st.session_state.cleaned_df = cleaned_df
    st.session_state.beers = beers
    st.session_state.matrix = matrix
    st.session_state.model = model
    st.session_state.selected_combinations = []
    st.session_state.selected_beer_ids = []

####################################################################
########################  Helper methods ###########################
    
# determines which breweries to show depending on what is typed into the box
def on_brewery_search(query):

    # Suppress UserWarning related to the empty string query
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Applied processor reduces input query to empty string.*",
        )

        # Perform fuzzy matching to find breweries that match the user's input
        matched_breweries = process.extract(
            query, st.session_state.beers["brewery_name"].unique(), limit=20
        )

    return [brewery for brewery, _ in matched_breweries]
    

# adds a beer to the list of selected beers when you select it from the dropdown menu
def on_beer_select(selected_brewery, selected_beer):

    # Add the selected combination to the list
    st.session_state.selected_combinations.append((selected_brewery, selected_beer))
    st.session_state.selected_beer_ids.append(
        st.session_state.beers.loc[
            (st.session_state.beers.brewery_name == selected_brewery)
            & (st.session_state.beers.beer_name == selected_beer),
            "beer_id",
        ].values[0]
    )
    s = "Selected Combinations:" + str(
        st.session_state.selected_combinations
    )  # + '\n' + "Selected Beer IDs:" + str(selected_beer_ids)
    st.session_state.chosen_beers_df = pd.DataFrame(st.session_state.selected_combinations, columns=['Brewery','Beer'])
    #container.table(st.session_state.chosen_beers_df)


# filters which beers are shown based on what is written in the textbox
def on_beer_search(brewery, query):

    # Suppress UserWarning related to the empty string query
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Applied processor reduces input query to empty string.*",
        )

        # Perform fuzzy matching to find breweries that match the user's input
        matched_beers = process.extract(
            query,
            st.session_state.beers.loc[st.session_state.beers.brewery_name == brewery, "beer_name"].unique(),
            limit=20,
        )

    return [beer for beer, _ in matched_beers]


# runs the model and prints your personalized recommendations
def get_beer_recs():
    new_row = np.zeros(st.session_state.matrix.shape[1])
    beer_ids = st.session_state.selected_beer_ids
    for beer_id in beer_ids:
        new_row[beer_id] = 1
    new_row = sp.sparse.csr_matrix(new_row)
    new_matrix = sp.sparse.vstack([st.session_state.matrix, new_row], format="csr")

    st.session_state.model.fit_partial(new_matrix)
    preds = st.session_state.model.predict(33363 * np.ones(65357), [i for i in range(65357)])

    for beer_id in beer_ids:
        preds[beer_id] = -100
    num_recs = st.session_state.slider
    ind = np.argpartition(preds, -num_recs)[-num_recs:]
    recs = st.session_state.beers.loc[st.session_state.beers.beer_id.isin(ind), ["brewery_name", "beer_name"]]
    recs.rename(columns={'brewery_name':'Brewery','beer_name':'Beer'}, inplace=True)
    with recs_container:
        st.write("Here are your personalized recommendations! :tada:")
        st.dataframe(recs, hide_index=True) 

#####################################################################
####################  Designing the app #############################

st.title("Beer Recommender")
#instructions = "Instructions:\n1. Find the brewery of the beer you have in mind by typing in all or part of the brewery's name in the first box and then select the desired brewery from the dropdown menu. \n2. Type in all of part of the beer name you have in mind and select the desired beer using the associated dropdown menu. \n3. Click the Add Beer button to add this to your list of liked beers.\n4. Repeat steps 1-3 as many times as desired -- we recommend adding at least 5 beers that you like. The more beers you list, the better your recommendations will be. \n5. Click 'Get Beer Recommendations' to get your personalized beer recommendations!"
instructions = "Add beers that you like using the boxes below -- the more you beers you add, the better your recommendations will be. When you're done adding all the beers you want, you can choose how many recommendations you want and click \"Get recommendations!\""
overview = "This project was done as part of the [Fall 2023 Erdös Institute Data Science Bootcamp](https://www.erdosinstitute.org/). Our model uses matrix factorization and can suggest beers you might like by comparing your beer preferences to those of ~33,000 users in this BeerAdvocate [data set](https://data.world/socialmediadata/beeradvocate). To learn more about the project, visit the [github repository](https://github.com/b-butler/beer-recommender-erdos-fall-2023) or our [Erdös Institute project page](https://www.erdosinstitute.org/project-database/fall-2023/data-science-boot-camp/brewsavvy)."
st.write(overview)
st.write(instructions)

# Input control for filtering by partial name
brewery_text = st.text_input("Enter all or part of the brewery name to filter results", "")

brewery_filtered = on_brewery_search(brewery_text)
brewery_name = st.selectbox("Select the brewery from the drop down menu", brewery_filtered)

beer_text = st.text_input("Enter all or part of the beer name to filter results")
beer_filtered = on_beer_search(brewery_name, beer_text)
beer_name = st.selectbox("Select the beer you have in mind from the drop down menu", beer_filtered)

add_beer = st.button("Add Beer", on_click = lambda: on_beer_select(brewery_name, beer_name))

container = st.container()
container.write(" ")
if 'chosen_beers_df' in st.session_state:
    container.write("List of beers you like:")
    container.table(st.session_state.chosen_beers_df)
    



num_recs = st.slider("How many recommendations do you want?",
                     min_value = 1, 
                     max_value = 30, 
                     value = 10,
                     key = 'slider')

get_recs = st.button("Get Recommendations!", on_click=get_beer_recs)

recs_container = st.container()
recs_container.write(" ")

