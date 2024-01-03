import pandas as pd
import streamlit as st
from datetime import datetime
# Import libraries and clean user tasks
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import time  # Import the time module for simulation
import altair as alt
import google.generativeai as genai


GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# with open('env') as f:
#     env_content = f.read()

# GEMINI_API_KEY = env_content.split('GEMINI_API_KEY=')[1].strip()

#set genai api key
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel('gemini-pro')

#create header with title and time

st.header("Comon Study")
st.write(f"Date: {datetime.today().strftime('%Y-%m-%d')}")
# First column: User input
with st.sidebar:
    st.title("Today's Study Goals")
    tasks = st.multiselect("What tasks do you want to focus on today?",
                           ["Math", "Physics", "History", "English", "Python", "javaScript", "Other"])
    goals = st.number_input("Set your overall study goal (minutes):", 
                            min_value=1, max_value=1440, value=60)
    
    


    
     # Set individual task goals
    task_goals = {}
    for task in tasks:
        task_goals[task] = st.number_input(f"Set goal for {task} (minutes):", min_value=1, max_value=240, value=30)

    # Spacer to separate study goals from the AI assistant section
    st.sidebar.markdown("---")

    # Third column: Motivational quote and progress visualization (right sidebar)
    st.sidebar.title("AI Assistant Chatbox")

    # Chatbox input
 
    user_query = st.sidebar.text_area("Ask me anything:")
    if user_query:
        input = model.generate_content(f"{user_query}")
        output = input.candidates[0].content.parts[0].text  


    # Add a button to send the query to the AI assistant
    if st.sidebar.button("Send"):
        # Make a request to the AI assistant and display the response
        # Display a spinner while processing the AI response
        with st.spinner("Processing..."):
            # Simulate processing time (replace this with your actual AI processing logic)
            time.sleep(2)
        # ai_response = "get_ai_assistant_response(user_query)"
        st.sidebar.write(f"outBard-->: { output}")
        # print(output)


# Second column: 
st.title("Task List")

# Preprocess and store training data
# (implement data collection and preprocessing)
training_data = [("Math", 30), ("Physics", 45), ("History", 20), ("Other", 15), ("English", 27), ("Chemistr", 49), ("Python", 23), ("javaScript", 25)]

# Separate tasks and times
training_tasks, training_times = zip(*training_data)

# Initialize NLP model and train it on preprocessed training data
vectorizer = TfidfVectorizer()
model = LinearRegression()
model.fit(vectorizer.fit_transform(training_tasks), training_times)

#Task list and time estimation

task_df = pd.DataFrame(list(zip(tasks, ["Not started"] * len(tasks))), columns=["Task", "Status"])

# Predict time for user's tasks
for i, task in enumerate(tasks):
    task_df.loc[i, "Estimated Time (min)"] = model.predict(vectorizer.transform([task]))[0]

# Allow users to mark tasks as started
for i, task in enumerate(tasks):
    task_status = st.checkbox(f"Start {task} ({task_df.loc[i, 'Status']}):", key=i)
    
    if task_status:
        task_df.loc[i, "Status"] = "In Progress"

        # Update individual task goals
        task_df.loc[i, "Goal"] = task_goals[task]

# Display task list with estimated times
st.write(task_df)


# Progress Tracker
st.title("Progress Tracker")

# Placeholder for progress tracking content
# This is where you would implement the actual progress tracking logic
# a simple bar chart to represent progress
progress_data = {
    "Task": tasks,
    "Status": [
        "Done" if st.checkbox(f"Start {task} ({'Not Started'}):", key=f"checkbox_{i}_{task}") else "Not Started"
        for i, task in enumerate(tasks)
    ],
    "Goal": [task_goals[task] if task in task_goals else 0 for task in tasks],
}

progress_df = pd.DataFrame(progress_data)

# Convert "Estimated Time" to numeric data type
progress_df['Estimated Time'] = [model.predict(vectorizer.transform([task]))[0] if task in task_goals else 0 for task in tasks]

# Calculate the Achieved value as the difference between Estimated Time and Goal
progress_df["Achieved"] = progress_df["Estimated Time"] - progress_df["Goal"]

# Assign positive or negative values to Achieved based on the comparison
progress_df["Achieved"] = progress_df["Achieved"].apply(lambda x: f"+{x}" if x >= 0 else x)

# Convert "Estimated Time" and "Achieved" to numeric data type
progress_df['Estimated Time'] = pd.to_numeric(progress_df['Estimated Time'], errors='coerce')
progress_df['Achieved'] = pd.to_numeric(progress_df['Achieved'], errors='coerce')

# Create a new DataFrame with correct data types
chart_df = pd.melt(progress_df, id_vars=['Task', 'Status'], value_vars=['Goal', 'Estimated Time', 'Achieved'])

# Display progress bar chart using Altair
progress_chart = alt.Chart(chart_df).mark_bar().encode(
    x='Task:N',
    y='value:Q',
    color='variable:N',
    tooltip=['Task', 'value']
).properties(width=600, height=400)

st.altair_chart(progress_chart, use_container_width=True)

# Display the progress data as a table (optional)
st.table(progress_df)