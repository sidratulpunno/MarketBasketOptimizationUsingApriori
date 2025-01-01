import streamlit as st
import pandas as pd
from apyori import apriori


st.title("Product Recommendation System")
st.write("This app uses the Apriori algorithm to suggest products frequently bought together.")

def load_data():
    try:
        # Load the dataset
        dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
        transactions = []
        for i in range(0, len(dataset)):
            transactions.append(
                [str(dataset.values[i, j]) for j in range(0, dataset.shape[1]) if str(dataset.values[i, j]) != 'nan'])
        return transactions
    except FileNotFoundError:
        st.warning("Data file 'Market_Basket_Optimisation.csv' not found. Please check the file path.")
        return None

transactions = load_data()

if transactions is None:
    st.info("No recommendations available.")  # Display a generic message if data loading fails
    st.stop()

def train_apriori(transactions):
    rules = apriori(transactions=transactions,
                    min_support=0.003,
                    min_confidence=0.2,
                    min_lift=3,
                    min_length=2,
                    max_length=2)
    return list(rules)

rules = train_apriori(transactions)

def get_rules_dataframe(results):
    def inspect(results):
        lhs = [tuple(result[2][0][0])[0] for result in results]
        rhs = [tuple(result[2][0][1])[0] for result in results]
        supports = [result[1] for result in results]
        confidences = [result[2][0][2] for result in results]
        lifts = [result[2][0][3] for result in results]
        return list(zip(lhs, rhs, supports, confidences, lifts))

    return pd.DataFrame(inspect(results),
                        columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

rules_df = get_rules_dataframe(rules)

available_products = sorted(set(rules_df['Left Hand Side']))
selected_product = st.selectbox("Choose a product to get recommendations:", available_products)

if selected_product:
    suggestions = rules_df[rules_df['Left Hand Side'] == selected_product][['Right Hand Side', 'Confidence', 'Lift']]

    if not suggestions.empty:
        st.write(f"Recommendations for **{selected_product}**:")
        st.dataframe(suggestions.head(3))
    else:
        st.info(f"No recommendations available for **{selected_product}**.")