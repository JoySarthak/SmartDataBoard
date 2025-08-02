import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import streamlit as st

COLOR_SCHEMES = {
    "Default": {
        "bar_colors": ["#FF4B4B", "#581A1A", "#FF9999"],
        "pie_colors": px.colors.sequential.RdBu[::-1],
        "line_color": ["#1f77b4", "#ff7f0e", "#2ca02c"],  # For mean, max, min
        "area_color": ["#1f77b4", "#a1c9f4", "#8de5a1"]   # Different shades for distinction
    },
    "Ocean": {
        "bar_colors": ["#006994", "#003366", "#5D8BF4"],
        "pie_colors": px.colors.sequential.Blues[::-1],
        "line_color": ["#00B4D8", "#0077B6", "#90E0EF"],
        "area_color": ["#0096C7", "#48CAE4", "#B8E2F2"]
    },
    "Forest": {
        "bar_colors": ["#2E8B57", "#3A5A40", "#90EE90"],
        "pie_colors": px.colors.sequential.Greens[::-1],
        "line_color": ["#588157", "#3A5A40", "#A3B18A"],
        "area_color": ["#3A5A40", "#588157", "#DAD7CD"]
    },
    "Sunset": {
        "bar_colors": ["#FF7F50", "#E25822", "#FFA07A"],
        "pie_colors": px.colors.sequential.Oranges[::-1],
        "line_color": ["#FF6B35", "#EF476F", "#FFD166"],
        "area_color": ["#EF476F", "#FF9E00", "#FFD166"]
    },
    "Violet": {
        "bar_colors": ["#6A4C93", "#4A306D", "#B399D4"],
        "pie_colors": px.colors.sequential.Purples[::-1],
        "line_color": ["#7B2CBF", "#5A189A", "#C77DFF"],
        "area_color": ["#5A189A", "#9D4EDD", "#E0AAFF"]
    },
    "Cool": {
        "bar_colors": ["#00B4D8", "#0077B6", "#90E0EF"],
        "pie_colors": px.colors.sequential.ice[::-1],
        "line_color": ["#48CAE4", "#0096C7", "#B8E2F2"],
        "area_color": ["#0096C7", "#00B4D8", "#CAF0F8"]
    },
    "Warm": {
        "bar_colors": ["#FF9E00", "#FF7B00", "#FFC971"],
        "pie_colors": px.colors.sequential.thermal[::-1],
        "line_color": ["#FFB703", "#FB8500", "#FFD166"],
        "area_color": ["#FB8500", "#FF9E00", "#FFD166"]
    }
}

def get_color_scheme():
    """Get the selected color scheme from session state or return default"""
    return st.session_state.get("color_scheme", "Default")

def color_scheme_selector(key):
    """Display color scheme selector and store in session state"""
    col1, col2 = st.columns([0.3,0.7])
    with col1:
        selected_scheme = st.selectbox(
            "Choose Color Scheme",
            options=list(COLOR_SCHEMES.keys()),
            index=list(COLOR_SCHEMES.keys()).index(get_color_scheme()),
            key=key
        )
    st.session_state.color_scheme = selected_scheme
    return COLOR_SCHEMES[selected_scheme]

def valc_tool(df, opt):
    df_view = df[opt].value_counts().reset_index()
    df_view.columns = [opt, 'count']
    return df_view
def agg_tool(df, opt):
    ag = st.selectbox("Find Max, Mean, Min wrt:",df.select_dtypes(include=['int64', 'float64']).columns)
    agg_df = df.groupby(opt)[ag].agg(['mean', 'max', 'min']).reset_index()
    return agg_df,ag
def search_tool(df : pd.DataFrame, opt, k):
    option = df[opt].unique()
    c1,c2 = st.columns(2)
    with c1:
        selected_options = st.multiselect(
                    f"Select from column",
                    option,
                    key=k,
                    default=option[0]
                )
    with c2:
        val_opt = st.selectbox("Select what to view: ",df.drop(columns=opt).columns)
        
    if selected_options and val_opt:
        f_df = df[df[opt].isin(selected_options)][val_opt].value_counts().reset_index()
        f_df.columns = [val_opt, 'count']  # First column is the values, second is counts
    return f_df
    


def viewNaN_BarC(df):
    null_counts = df.isnull().sum()
    non_null_counts = df.notnull().sum()

    # Combine into a DataFrame
    counts_df = pd.DataFrame({
        "Null": null_counts,
        "Non-null": non_null_counts
    })

    # Show stacked bar chart
    st.bar_chart(counts_df, stack=True,x_label="features",y_label="NaN vs Non_NaN",height=470, use_container_width=True, 
                 color=["#FF4B4B", "#581A1A"])
def viewNaN_Pie(df):
    column = st.selectbox("Choose a column to view NaN:",df.columns)
    null_count = df[column].isnull().sum()
    non_null_count = df[column].notnull().sum()

    # Prepare summary dataframe
    summary_df = pd.DataFrame({
        'Status': ['Non-Null', 'Null'],
        'Count': [non_null_count, null_count]
    })

    # Plot pie chart
    fig = px.pie(
    summary_df,
    values='Count',
    names='Status',
    title=f"Null vs Non-Null Count in {column}",
    color='Status',
    color_discrete_map={
        "Non-Null": "#FF4B4B",
        "Null": "#581A1A"
    }
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)
def viewNaN_line(df):
    null_counts = df.isnull().sum()
    non_null_counts = df.notnull().sum()

    # Combine into a DataFrame
    counts_df = pd.DataFrame({
        "Null": null_counts,
        "Non-null": non_null_counts
    })
    st.subheader("Graphical Overview")
    with st.container(border=True):
        st.line_chart(counts_df, use_container_width=True, height=500)

def Graph_Bar(df: pd.DataFrame, mode, opt, colors):
    c = False
    on = st.toggle("Horizontal", key=mode)
    if on:
        c = True
    if mode == "Value Counts":
        df_view = valc_tool(df, opt)
        st.bar_chart(df_view, x=opt, y='count', height=550, use_container_width=True, 
                    horizontal=c, color=colors["bar_colors"][0])
    elif mode == "Agg":
        df_view, y_l = agg_tool(df, opt)
        if c:
            x_l = y_l
            y_l = opt
        else:
            x_l = opt
        st.bar_chart(df_view, x=opt, y=['mean', 'max', 'min'], stack=True, 
                    y_label=y_l, x_label=x_l if c else None, height=650, 
                    use_container_width=True, horizontal=c, color=colors["bar_colors"])
    elif mode == "Searc":
        df_view = search_tool(df, opt, "ssch")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.bar_chart(
                df_view,
                x=df_view.columns[0],
                y='count',
                height=600,
                use_container_width=True,
                horizontal=c,
                color=colors["bar_colors"][0]
            )
        with col2:
            st.dataframe(df_view)

def Graph_line(df: pd.DataFrame, mode, opt, colors):
    if mode == "Value Counts":
        df_view = valc_tool(df, opt)
        st.line_chart(df_view, x=opt, y='count', height=550, 
                     use_container_width=True, color=colors["line_color"][0])
    elif mode == "Agg":
        df_view, y_l = agg_tool(df, opt)
        st.line_chart(df_view, x=opt, y=['mean', 'max', 'min'], y_label=y_l, 
                     height=650, use_container_width=True, color=colors["line_color"])
    elif mode == "Searc":
        df_view = search_tool(df, opt, "ssch")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.line_chart(
                df_view,
                x=df_view.columns[0],
                y='count',
                height=600,
                use_container_width=True,
                color=colors["line_color"][0]
            )
        with col2:
            st.dataframe(df_view)

def Graph_area(df: pd.DataFrame, mode, opt, colors):
    if mode == "Value Counts":
        df_view = valc_tool(df, opt)
        st.area_chart(df_view, x=opt, y='count', height=550, 
                      use_container_width=True, color=colors["area_color"][0])
    elif mode == "Agg":
        df_view, y_l = agg_tool(df, opt)
        st.area_chart(df_view, x=opt, y=['mean', 'max', 'min'], height=650, 
                      use_container_width=True, color=colors["area_color"])
    elif mode == "Searc":
        df_view = search_tool(df, opt, "ssch")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.area_chart(
                df_view,
                x=df_view.columns[0],
                y='count',
                height=600,
                use_container_width=True,
                color=colors["area_color"][0]
            )
        with col2:
            st.dataframe(df_view)

def Graph_pie(df, opt, mode, colors):
    if mode == "Value Counts":
        value_counts = valc_tool(df, opt)
    if mode == "Searc":
        value_counts = search_tool(df, opt, "pie_s")
    value_counts.columns = ['category', 'count']
    
    threshold = st.slider("Select Threshold", 20, 200, 25, key=mode)
    small_categories = value_counts[value_counts['count'] < threshold]
    big_categories = value_counts[value_counts['count'] >= threshold]
    
    if len(small_categories) > 0:
        others_sum = small_categories['count'].sum()
        others_row = pd.DataFrame({'category': ['Others'], 'count': [others_sum]})
        final_counts = pd.concat([big_categories, others_row])
        has_others = True
    else:
        final_counts = big_categories
        has_others = False
    
    fig = px.pie(
        final_counts,
        names='category',
        values='count',
        hole=0.3,
        color_discrete_sequence=colors["pie_colors"]
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        pull=[0.1 if cat == 'Others' else 0 for cat in final_counts['category']]
    )
    fig.update_layout(
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        margin=dict(t=0, b=0, l=0, r=0),
        showlegend=False
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("### Categories")
        for i, row in final_counts.iterrows():
            if row['category'] == 'Others' and has_others:
                with st.expander(f"Others ({row['count']})", expanded=False):
                    st.markdown("**Included categories:**")
                    for _, small_row in small_categories.iterrows():
                        st.markdown(f"- {small_row['category']}: {small_row['count']}")
            else:
                st.markdown(
                f"""
                <div style='display:flex; align-items:center; margin-bottom:8px;'>
                    <div style='width:12px; height:12px; background:{colors['pie_colors'][i % len(colors['pie_colors'])]}; margin-right:8px;'></div>
                    <div>{row['category']} ({row['count']})</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
