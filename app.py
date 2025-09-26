import streamlit as st

st.set_page_config(
    page_icon="data/logo2.png",
    page_title="Smartboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for navigation
if 'main_nav' not in st.session_state:
    st.session_state.main_nav = "EDA"

# Sidebar setup
st.sidebar.image("data/photo1.jpeg", caption="Smart-Data-board")
st.sidebar.title("Smart Dashboard for Datascience and ML")

# Define all pages in a dictionary
PAGES = {
    "EDA": {
        "Data Analysis": [
            {"title": "Overview", "path": "pages/overview_data.py"},
            {"title": "Customize Data", "path": "pages/customz.py"},
            {"title": "Visualise Data", "path": "pages/visualise_data.py"}
        ]
    },
    "RESEARCH": {
        "Model Training": [
            {"title": "Data Encoding", "path": "pages/ML_data_encoding.py"},
            {"title": "Data Training", "path": "pages/ML_models.py"}
        ]
    },
    "SMART-TOOLS": {
        "Smart Tools": [
            {"title": "Get-Best Model", "path": "pages/Training_sm.py"},
            {"title": "Generate Reports", "path": "pages/Tuning_sm.py"}
        ]
    }
}

# Create the selectbox that will update the session state
selected_section = st.sidebar.selectbox(
    "What's on your mind",
    list(PAGES.keys()),
    key="main_nav_select",
    index=list(PAGES.keys()).index(st.session_state.main_nav)
)

# Update the main navigation state when selectbox changes
if selected_section != st.session_state.main_nav:
    st.session_state.main_nav = selected_section
    st.rerun()

# Display the appropriate navigation based on selection
if st.session_state.main_nav in PAGES:
    # Convert our page dictionary to st.Page objects
    nav_pages = {
        section: [st.Page(page["path"], title=page["title"]) for page in pages]
        for section, pages in PAGES[st.session_state.main_nav].items()
    }
    
    # Create and run navigation
    pg = st.navigation(nav_pages, position="top")
    pg.run()

# Quick access toolbar - improved version
def handle_quick_access(page_section, page_path):
    """Handle quick access navigation properly"""
    if st.session_state.main_nav != page_section:
        st.session_state.main_nav = page_section
        st.rerun()
    st.switch_page(page_path)


with st.sidebar.expander("Quick access toolbar", expanded=True):
    if st.button("Cleaning tools", type="secondary" , use_container_width=True):
        handle_quick_access("EDA", "pages/overview_data.py")
    if st.button("Encoding", type="secondary", use_container_width=True):
        handle_quick_access("RESEARCH", "pages/ML_data_encoding.py")
    if st.button("Manual Train", type="secondary", use_container_width=True):
        handle_quick_access("RESEARCH", "pages/ML_models.py")
    if st.button("Smart-Train", type="secondary", use_container_width=True):
        handle_quick_access("SMART-TOOLS", "pages/Training_sm.py")
    if st.button("Visualize", type="secondary", use_container_width=True):
        handle_quick_access("EDA", "pages/visualise_data.py")

st.sidebar.page_link("https://portfolio-joysarthaks-projects.vercel.app/", label="About Me", icon=":material/language:")