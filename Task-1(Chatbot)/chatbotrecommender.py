import tkinter as tk
import re
import random
import pickle
import pandas as pd
from recommendations import get_recommendations_by_genre
# Load the movies and ratings data
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

# Load the TF-IDF model and matrix
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

# Function to get a random selection of movies
def get_random_movies(n=5):
    return data.sample(n)['title'].tolist()

# Custom simple tokenization
def preprocess_input(user_input):
    # Use regular expressions to split by words
    tokens = re.findall(r'\b\w+\b', user_input.lower())
    return tokens

# Function to handle sending the message and getting the response
def send_message():
    user_input = entry_box.get("1.0", 'end-1c').strip()
    entry_box.delete("1.0", 'end')

    if user_input:
        chat_box.config(state=tk.NORMAL)
        chat_box.insert(tk.END, "You: " + user_input + "\n\n")
        chat_box.config(foreground="#442265", font=("Verdana", 12))

        # Preprocess user input
        stemmed_input = preprocess_input(user_input)

        # Get bot response using rule-based approach
        response = get_response(stemmed_input)

        chat_box.insert(tk.END, "Bot: " + response + "\n\n")
        chat_box.config(state=tk.DISABLED)
        chat_box.yview(tk.END)

# Rule-based responses
def get_response(stemmed_input):
    
    if any(token in stemmed_input for token in ['hi', 'hello', 'hey']):
        return "Hello! I am here to suggest movies. What you wanna watch for today?"        
    
    elif 'help' in stemmed_input:
        return "Sure, I can help you. What do you need assistance with?"
    
    elif any(token in stemmed_input for token in ['bye', 'goodbye']):
        return "Goodbye! Have a great day!"


    # Asking for movie recommendations by genre
    
    match = re.search(r'suggest (.+?) movies', ' '.join(stemmed_input))
    if match:
        genre = match.group(1)
        recommended_movies = []
        recommended_movies = get_recommendations_by_genre(genre)
        if not recommended_movies:
            random_movies = get_random_movies(5)
         
            return "Here are some suggestions." + ', '.join(random_movies)
        return "Here are some recommended " + genre + " movies: " + ', '.join(recommended_movies)

    return "Sorry, I didn't understand that. Can you please rephrase?"

# Create a simple GUI with Tkinter
root = tk.Tk()
root.title("Rule-Based AI Chatbot")
root.geometry("400x500")
root.resizable(width=False, height=False)

# Create the chat window
chat_box = tk.Text(root, bd=1, bg="white", width="50", height="8", font="Arial")
chat_box.config(state=tk.DISABLED)

# Bind scrollbar to the chat window
scrollbar = tk.Scrollbar(root, command=chat_box.yview)
chat_box['yscrollcommand'] = scrollbar.set

# Create the entry box for user input
entry_box = tk.Text(root, bd=0, bg="white", width="29", height="5", font="Arial")
entry_box.bind("<Return>", lambda event: send_message())

# Create the send button
send_button = tk.Button(root, font=("Verdana", 12, 'bold'), text="Send", width="12", height="5",
                        bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                        command=send_message)

# Place components in the window
scrollbar.place(x=376, y=6, height=386)
chat_box.place(x=6, y=6, height=386, width=370)
entry_box.place(x=6, y=401, height=90, width=265)
send_button.place(x=282, y=401, height=90)

# Start the Tkinter event loop
root.mainloop()
