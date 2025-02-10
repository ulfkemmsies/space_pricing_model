import streamlit as st

# Custom imports 
from multipage import MultiPage
from pages import single_convergence, variables_page
# Create an instance of the app 
app = MultiPage()

#Turn off pyplot warnings
# st.image(use_container_width=True)

# Title of the main page
st.title("Fuel Pricing at Cislunar Locations")

# Add all your applications (pages) here
app.add_page("Drawing the Network", single_convergence.app)
app.add_page("Variables and Constants", variables_page.app)


# The main app
app.run()
