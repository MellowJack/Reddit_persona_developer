🎭 Reddit Persona Generator with Groq & Llama 3
This project provides a robust Python script to generate comprehensive user personas from Reddit profiles. It leverages PRAW for data scraping, spaCy and Hugging Face Transformers for initial NLP analysis, and integrates with the Groq API (using Llama 3) for advanced semantic refinement and persona synthesis.

✨ Features
Reddit Profile Scraping: Collects recent posts and comments from any public Reddit user profile.

Initial NLP Analysis: Extracts demographics (age, location, occupation, relationship status), analyzes personality traits (MBTI-like), identifies core motivations, and quantifies behavioral patterns (posting times, frequency, length).

Advanced Sentiment Analysis: Utilizes a pre-trained transformer model to identify dominant emotions expressed in user content.

Interest Categorization: Groups frequented subreddits into broader interest categories.

Groq/Llama 3 Semantic Refinement: Synthesizes the quantitative data into a rich, narrative-driven persona, inferring archetypes, goals, and frustrations using a powerful LLM.

Markdown Report Generation: Outputs a well-formatted, easy-to-read markdown report summarizing the persona.

📁 Repository Structure
.
├── enhanced_persona_generator_refined.py  # The main Python script
├── .env.example                         # Example file for environment variables
├── .gitignore                           # Specifies files/directories to ignore (e.g., venv, .env)
└── README.md                            # This file
├── sample_reports/                      # (Optional) Directory for sample persona reports
│   └── sample_user_persona.md
└── (your_venv_folder)/                  # Your Python virtual environment (ignored by Git)

🚀 Setup Instructions
Follow these steps to get the project up and running on your local machine.

Prerequisites
Git: Make sure Git is installed on your system.

Download Git

Python 3.x: Ensure Python 3.8 or newer is installed.

Download Python

1. Clone the Repository
First, clone this repository to your local machine using Git:

git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME # Replace YOUR_REPO_NAME with the actual name of your cloned repository

(Replace YOUR_USERNAME and YOUR_REPO_NAME with your actual GitHub username and repository name.)

2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies and avoid conflicts with other Python projects.

python -m venv venv

3. Activate the Virtual Environment
On Windows:

.\venv\Scripts\activate

On macOS/Linux:

source venv/bin/activate

You should see (venv) at the beginning of your terminal prompt, indicating the virtual environment is active.

4. Install Dependencies
Once your virtual environment is active, install the required Python packages:

pip install praw python-dotenv spacy numpy sentence-transformers groq transformers torch

5. Download SpaCy Language Model
The spacy library requires a language model to be downloaded separately:

python -m spacy download en_core_web_sm

6. Configure API Keys (.env file)
This script requires API keys for Reddit and Groq.

Create a .env file:
Copy the provided example environment file:

cp .env.example .env

Edit the .env file:
Open the newly created .env file in a text editor and fill in your credentials:

# Reddit API Credentials
REDDIT_CLIENT_ID="YOUR_REDDIT_CLIENT_ID"
REDDIT_CLIENT_SECRET="YOUR_REDDIT_CLIENT_SECRET"
REDDIT_USER_AGENT="YOUR_REDDIT_USER_AGENT" # e.g., "PersonaGenerator by u/YourRedditUsername"

# Groq API Key
GROQ_API_KEY="YOUR_GROQ_API_KEY"

Getting Reddit API Credentials:

Go to Reddit's App Preferences.

Click "are you a developer? create an app..." at the bottom.

Choose "script" for the app type.

Fill in a name (e.g., "PersonaGenerator"), description, and http://localhost:8080 for the redirect URI (this is a placeholder and won't be used for a script app).

Click "create app".

Your client ID is the string below "personal use script".

Your client secret is the string next to "secret".

Your user agent should be a unique string, typically in the format YourAppName by u/YourRedditUsername (e.g., PersonaGenerator by u/JohnDoe).

Getting Groq API Key:

Go to Groq Console.

Sign up or log in.

Navigate to the "API Keys" section and create a new API key.

💡 Usage
Once everything is set up, you can run the script from your terminal within the activated virtual environment.

python enhanced_persona_generator_refined.py <reddit_profile_url> [options]

Arguments:
<reddit_profile_url>: (Required) The full URL of the Reddit user's profile (e.g., https://www.reddit.com/user/spez/).

Options:
--limit <int>: Total number of posts and comments to analyze (default: 200). The script attempts to fetch an equal number of posts and comments up to this limit.

--output <filename.md>: Optional. Specifies an output filename for the markdown report (e.g., persona.md). If not provided, the report will be printed to the console.

--skip-refinement: Use this flag to skip the Groq/Llama 3 semantic refinement step. The persona will be generated using only local NLP analysis.

Examples:
Generate a persona for a user and print to console:

python enhanced_persona_generator_refined.py https://www.reddit.com/user/spez/

Generate a persona with a higher content limit and save to a file:

python enhanced_persona_generator_refined.py https://www.reddit.com/user/AutoModerator/ --limit 300 --output automoderator_persona.md

Generate a persona, skipping Groq refinement:

python enhanced_persona_generator_refined.py https://www.reddit.com/user/gallowboob/ --skip-refinement

📄 Sample Output
You can find an example of a generated persona report in the sample_reports/ directory (if provided). This will give you an idea of the depth and format of the analysis.

🤝 Contributing
Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to open an issue or submit a pull request.

📜 License
This project is licensed under the MIT License - see the LICENSE file for details (if you choose to add one).
