import pandas as pd
import streamlit as st
from datetime import datetime
# Import libraries and clean user tasks
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

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


# Third column: Motivational quote and progress visualization (placeholder for now)
st.title("Motivation Boost")
st.write("Stay focused! Quote of the day: (To be implemented later)")
st.title("Progress Tracker")
st.write("Track your achievements here! (To be implemented later)")