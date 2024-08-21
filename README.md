# Field_feelings 😊

Welcome to Field Feelings, a Streamlit-based application designed to support mental well-being by analyzing and responding to user emotions. This project leverages AI models to understand user sentiments, provide supportive feedback, and track emotional trends over time.

## Project Overview
Field Feelings aims to create a safe space where users can express their emotions and receive empathetic responses. The application identifies user sentiments based on chat history and provides tailored responses to help improve their mood.

## Features
* Sentiment Analysis: The application segments user emotions into predefined categories and visualizes the distribution of these sentiments.
* Conversational AI: Powered by Llama3.1 and integrated with Groq, the application responds to user inputs with supportive and constructive messages.
* Historical Data Tracking: Conversations and sentiments are saved and can be analyzed to understand emotional trends over time.
* Interactive Visualizations: Users can see how their feelings are distributed over time with dynamic charts.

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:

```bash
Copy code
git clone https://github.com/your-username/field-feelings.git
cd field-feelings
```
2. Install the required dependencies:

```bash
Copy code
pip install -r requirements.txt
```

3. Set up environment variables:
Create a .env file in the project root directory and add your Groq API keys and other environment variables:

```bash
Copy code
GROQ_API_KEY=<your_groq_api_key>
GROQ_API_KEY_2=<your_secondary_groq_api_key>
```
4. Run the application:

```bash
Copy code
streamlit run app.py
```

## Usage
Visit the application at [Field Feelings](https://fieldfeelings.streamlit.app)  and start a conversation by entering your current feelings. The AI will respond with empathetic messages and track your emotions over time.

### Main Sections
* Chat Interface: Engage in a conversation with the AI by typing in your feelings or thoughts. The AI will analyze your input and provide a suitable response.
* Sentiment Distribution: View a dynamic bar chart representing the distribution of your emotions based on your chat history.
### Technical Details
* Language Models: The application uses Llama3.1 hosted on Groq for natural language processing.
* Data Storage: Sentiments and chat histories are stored in the cloud, replacing the need for local CSV files.
* Visualization: Altair is used for creating interactive data visualizations.
Contributing
### Contributions are welcome! 

Please fork the repository and submit a pull request if you want to add a feature or fix an issue.

License
This project is licensed under the MIT License.

Contact
For more information or inquiries, please contact us at manuelfelipearias1234@gmail.com

## Demo Video 🎥

Check out this demo video to see **Field Feelings** in action:

[![Field Feelings Demo](https://i9.ytimg.com/vi_webp/CJAHikEBvjs/mq2.webp?sqp=CIismLYG-oaymwEmCMACELQB8quKqQMa8AEB-AH-CYAC0AWKAgwIABABGB8gTih_MA8=&rs=AOn4CLBCVTCJtKeCXLwWMJVdh26gdTpnnA)](https://youtu.be/CJAHikEBvjs>)

Click the image above or [watch the video on YouTube](https://youtu.be/CJAHikEBvjs>).
