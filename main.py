import os
import praw
import argparse
import re
from datetime import datetime
from dotenv import load_dotenv
from collections import Counter, defaultdict
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq # Use Groq instead of OpenAI
from transformers import pipeline
import torch
import json


load_dotenv()

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    exit()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

device = 0 if torch.cuda.is_available() else -1
emotion_analyzer = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    device=device
)

def setup_reddit():
    """Initializes and returns a PRAW Reddit instance."""
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")

    if not all([client_id, client_secret, user_agent]):
        print("🔴 Reddit API credentials not found in .env file.")
        print("Please create a .env file with REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT.")
        exit()

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )
    reddit.read_only = True
    return reddit

def get_username(url: str) -> str:
    """Extracts the username from a Reddit user profile URL."""
    try:
        match = re.search(r'(?:/u/|/user/)([^/]+)', url)
        if match:
            return match.group(1)
        else:
            raise ValueError("Could not extract username from URL.")
    except (IndexError, ValueError):
        print(f"🔴 Invalid Reddit profile URL format: {url}")
        exit()

def scrape_user_content(reddit: praw.Reddit, username: str, limit: int = 100):
    """Scrapes recent posts and comments for a given Reddit user."""
    print(f"🕵️ Accessing profile for u/{username}...")
    try:
        user = reddit.redditor(username)
        _ = user.id
    except Exception as e:
        print(f"🔴 Could not retrieve user u/{username}. They may be suspended, shadowbanned, or the username is incorrect.")
        print(f"PRAW Error: {e}")
        return [], {}

    content = []
    account_info = {
        "created_utc": user.created_utc,
        "comment_karma": user.comment_karma,
        "link_karma": user.link_karma,
        "is_mod": user.is_mod,
        "is_gold": user.is_gold
    }

    try:
        for post in user.submissions.new(limit=limit // 2):
            content.append({
                "type": "post",
                "text": f"{post.title}. {post.selftext}",
                "subreddit": post.subreddit.display_name,
                "timestamp": post.created_utc,
                "score": post.score,
            })
    except Exception as e:
        print(f"⚠️ Could not fetch all submissions: {e}")

    try:
        for comment in user.comments.new(limit=limit // 2):
            content.append({
                "type": "comment",
                "text": comment.body,
                "subreddit": comment.subreddit.display_name,
                "timestamp": comment.created_utc,
                "score": comment.score,
            })
    except Exception as e:
        print(f"⚠️ Could not fetch all comments: {e}")

    return content, account_info

def extract_demographics(content: list) -> dict:
    """Extracts potential demographic information using NLP and regex."""
    locations, organizations, ages, statuses = [], [], [], []
    
    age_patterns = [
        r'\b(i\'m|i am|i was|being) (\d{1,2})\b', r'\bmy (\d{1,2})(-| )year(-| )old\b',
        r'\b(\d{1,2})f\b', r'\b(\d{1,2})m\b'
    ]
    relationship_patterns = [
        r'\bmy (wife|husband|partner|boyfriend|girlfriend|fiancé|fiancée)\b',
        r'\b(single|married|divorced|engaged)\b'
    ]
    
    full_text = " ".join([item['text'] for item in content])
    doc = nlp(full_text)

    for ent in doc.ents:
        if ent.label_ == "GPE":
            locations.append(ent.text.strip())
        elif ent.label_ == "ORG":
            organizations.append(ent.text.strip())

    for item in content:
        text = item['text'].lower()
        for pattern in age_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                age_num = next((s for s in match if s.isdigit()), None)
                if age_num:
                    ages.append(int(age_num))
    
    for pattern in relationship_patterns:
        if re.search(pattern, full_text.lower()):
            found = re.search(pattern, full_text.lower()).group(1)
            statuses.append(found)

    location = Counter(locations).most_common(1)[0][0] if locations else "Unknown"
    common_non_orgs = {'reddit', 'youtube', 'amazon', 'google', 'twitter', 'facebook', 'instagram'}
    valid_orgs = [org for org in organizations if org.lower() not in common_non_orgs]
    occupation = Counter(valid_orgs).most_common(1)[0][0] if valid_orgs else "Unknown"
    
    valid_ages = [a for a in ages if 13 < a < 90]
    if valid_ages:
        age = int(np.median(valid_ages))
    else:
        age = "Unknown"

    status = Counter(statuses).most_common(1)[0][0].title() if statuses else "Unknown"
    return {"location": location, "occupation": occupation, "age": age, "status": status}

def analyze_personality(content: list) -> dict:
    """Analyzes personality traits based on MBTI-like dichotomies using keyword analysis."""
    all_text = " ".join([item['text'].lower() for item in content])
    
    trait_keywords = {
        'introvert': ['alone', 'quiet', 'solitude', 'reading', 'introverted', 'shy', 'private'],
        'extrovert': ['party', 'social', 'friends', 'outgoing', 'energetic', 'crowd', 'events'],
        'sensing': ['practical', 'details', 'facts', 'concrete', 'experience', 'hands-on', 'realistic'],
        'intuition': ['ideas', 'future', 'possibilities', 'abstract', 'theory', 'imagine', 'potential'],
        'thinking': ['logic', 'reason', 'objective', 'analyze', 'rational', 'think', 'data'],
        'feeling': ['feel', 'emotions', 'values', 'harmony', 'empathy', 'personal', 'support'],
        'judging': ['plan', 'organized', 'schedule', 'decided', 'structure', 'control', 'closure'],
        'perceiving': ['flexible', 'spontaneous', 'adaptable', 'options', 'open', 'casual', 'explore']
    }

    def count_indicators(text, words): return sum(word in text for word in words)
    def normalize_pair(score1, score2):
        total = score1 + score2
        return 50 if total == 0 else (score1 / total) * 100

    scores = {trait: count_indicators(all_text, keywords) for trait, keywords in trait_keywords.items()}
    
    return {
        "introvert": normalize_pair(scores['introvert'], scores['extrovert']),
        "sensing": normalize_pair(scores['sensing'], scores['intuition']),
        "thinking": normalize_pair(scores['thinking'], scores['feeling']),
        "judging": normalize_pair(scores['judging'], scores['perceiving'])
    }

def analyze_motivations(content: list) -> dict:
    """Analyzes user motivations based on common psychological drivers."""
    motivation_keywords = {
        "achievement": ["success", "goal", "improve", "progress", "win", "build"],
        "social": ["friends", "community", "family", "share", "connect", "relationship"],
        "learning": ["learn", "discover", "understand", "knowledge", "study", "curious"],
        "security": ["safe", "stable", "secure", "invest", "save", "plan"],
        "entertainment": ["fun", "hobby", "enjoy", "relax", "game", "movie"],
        "creation": ["create", "make", "design", "write", "art", "develop"]
    }
    
    all_text = " ".join([item['text'].lower() for item in content])
    motivation_scores = {m: sum(k in all_text for k in keywords) for m, keywords in motivation_keywords.items()}
    
    total_score = sum(motivation_scores.values())
    if total_score > 0:
        return {k: (v / total_score) * 100 for k, v in motivation_scores.items()}
    return {m: 0 for m in motivation_keywords}

def analyze_behavior_patterns(content: list) -> dict:
    """Analyzes user posting behavior like timing, frequency, and style."""
    if not content: return {}
    
    timestamps = [item['timestamp'] for item in content]
    hours = [datetime.utcfromtimestamp(ts).hour for ts in timestamps]
    days = [datetime.utcfromtimestamp(ts).weekday() for ts in timestamps]
    post_lengths = [len(item['text']) for item in content]
    
    peak_hour = Counter(hours).most_common(1)[0][0] if hours else 12
    if 5 <= peak_hour < 12: active_period = "Morning Person ☀️"
    elif 12 <= peak_hour < 17: active_period = "Afternoon Poster ☕"
    elif 17 <= peak_hour < 22: active_period = "Evening Contributor 🌙"
    else: active_period = "Night Owl 🦉"

    avg_length = np.mean(post_lengths) if post_lengths else 0
    if avg_length < 100: engagement_style = "Concise & To-the-Point"
    elif avg_length < 400: engagement_style = "Moderately Detailed"
    else: engagement_style = "Long-Form & Detailed"
        
    if len(timestamps) > 1:
        avg_gap_hours = np.mean(np.diff(sorted(timestamps))) / 3600
        if avg_gap_hours < 12: consistency = "Highly Active"
        elif avg_gap_hours < 72: consistency = "Regularly Active"
        else: consistency = "Posts Sporadically"
    else: consistency = "Unknown"
        
    day_map = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    most_active_day = day_map[Counter(days).most_common(1)[0][0]] if days else "Unknown"

    return {
        "active_period": active_period, "engagement_style": engagement_style, "consistency": consistency,
        "avg_post_length": int(avg_length), "peak_hour_utc": peak_hour, "most_active_day": most_active_day
    }

def analyze_interests(content: list) -> tuple:
    """Analyzes user interests by categorizing frequented subreddits."""
    if not content: return [], [], {}

    subreddit_counts = Counter([item['subreddit'] for item in content])
    
    interest_categories = {
        "Technology": ["programming", "technology", "software", "sysadmin", "netsec", "hardware", "buildapc"],
        "Gaming": ["gaming", "games", "leagueoflegends", "Minecraft", "steam", "patientgamers", "rpg"],
        "Finance & Career": ["personalfinance", "investing", "stocks", "careerguidance", "cscareerquestions", "jobs"],
        "Entertainment": ["movies", "television", "books", "music", "anime", "comics", "startrek"],
        "Hobbies & DIY": ["hobbies", "DIY", "woodworking", "3dprinting", "photography", "cooking", "gardening"],
        "Lifestyle & Health": ["fitness", "nutrition", "selfimprovement", "malefashionadvice", "skincareaddiction"],
        "News & Politics": ["news", "worldnews", "politics", "geopolitics", "europe"],
        "Humor & Memes": ["memes", "funny", "dankmemes", "wholesomememes", "facepalm", "me_irl"],
        "Q&A / Help": ["AskReddit", "explainlikeimfive", "NoStupidQuestions", "whatisthisthing", "tipofmytongue"]
    }

    categorized_activity = defaultdict(int)
    for sub, count in subreddit_counts.items():
        found = False
        for category, keywords in interest_categories.items():
            if any(keyword in sub.lower() for keyword in keywords):
                categorized_activity[category] += count
                found = True
                break
        if not found:
            categorized_activity["Niche/Other"] += count
            
    sorted_interests = sorted(categorized_activity.items(), key=lambda item: item[1], reverse=True)
    primary = [cat for cat, _ in sorted_interests[:3]]
    secondary = [cat for cat, _ in sorted_interests[3:6]]
    
    return primary, secondary, dict(subreddit_counts.most_common(10))

def analyze_sentiment(content: list) -> dict:
    """Performs an advanced emotional analysis using a transformer model."""
    if not content: return {}
    
    sample_texts = [item['text'] for item in content if 50 < len(item['text']) < 1000][:25]
    if not sample_texts: return {}

    try:
        results = emotion_analyzer(sample_texts, top_k=None)
        emotion_scores = defaultdict(float)
        for result_list in results:
            for emotion in result_list:
                emotion_scores[emotion['label']] += emotion['score']
        
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            dominant_emotions = {label: (score / total_score) * 100 for label, score in emotion_scores.items()}
            return dict(sorted(dominant_emotions.items(), key=lambda item: item[1], reverse=True))
        return {}
    except Exception as e:
        print(f"⚠️ Emotion analysis failed: {e}")
        return {}

def semantic_refinement_with_groq(raw_analysis: dict, content_sample: list) -> dict:
    """Uses Groq and Llama 3 to synthesize and refine the initial analysis."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("⚠️ GROQ_API_KEY not found. Skipping semantic refinement.")
        return {"summary": "Analysis performed using local models only. Groq refinement was skipped.", "archetype": "Unknown"}

    client = Groq(api_key=groq_api_key)
    sample_text = "\n".join([f"- (From r/{item['subreddit']}): {item['text'][:250]}..." for item in content_sample[:10]])

    prompt = f"""
    Based on the following quantitative analysis and content samples of a Reddit user, synthesize a comprehensive and insightful user persona.

    ### Quantitative Analysis:
    {json.dumps(raw_analysis, indent=2)}

    ### Content Samples:
    {sample_text}

    ### Your Task:
    Act as an expert digital anthropologist. Create a rich, narrative-driven persona.
    1.  **Summary**: Write a 2-3 paragraph summary capturing the essence of this person. What do they seem like? What are their core drivers and what do they care about?
    2.  **Archetype**: Assign a descriptive archetype (e.g., "The Practical Hobbyist", "The Curious Analyst", "The Online Debater").
    3.  **Goals**: Infer 2-3 likely goals based on their content (e.g., "Seeking career advancement in tech," "Learning a new creative skill").
    4.  **Frustrations**: Infer 2-3 likely frustrations or pain points (e.g., "Dealing with inefficient software," "Feeling misunderstood in online discussions").

    Return the response as a single, valid JSON object with four keys: "summary", "archetype", "goals", "frustrations".
    """

    print("🤖 Contacting Groq with Llama 3 for persona synthesis... This should be fast!")
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192", # Using the powerful Llama 3 70B model
            messages=[
                {"role": "system", "content": "You are an expert digital anthropologist specializing in creating rich user personas from online data. Your output must be a valid JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
            max_tokens=1000
        )
        refined_persona = json.loads(response.choices[0].message.content)
        return refined_persona
    except Exception as e:
        print(f"🔴 Groq / Llama 3 refinement failed: {e}")
        return {"summary": f"Refinement failed due to an API error: {e}", "archetype": "Error", "goals": [], "frustrations": []}

def generate_persona_report(username: str, initial_analysis: dict, refined_persona: dict) -> str:
    """Formats the complete analysis into a final markdown report."""
    demographics = initial_analysis['demographics']
    personality = initial_analysis['personality']
    motivations = initial_analysis['motivations']
    behavior = initial_analysis['behavior']
    interests = initial_analysis['interests']
    emotions = initial_analysis['emotions']

    summary = refined_persona.get('summary', "Not available.")
    archetype = refined_persona.get('archetype', "Unknown")
    goals = refined_persona.get('goals', [])
    frustrations = refined_persona.get('frustrations', [])

    def make_bar(score, length=20): return "█" * int((score / 100) * length) + "░" * (length - int((score / 100) * length))

    report_header = f"# 🎭 Reddit Persona: u/{username}\n## Archetype: {archetype}\n---\n"
    report_summary = f"### Executive Summary\n{summary}\n\n"
    report_goals = "### Inferred Goals\n" + "\n".join([f"- {g}" for g in goals]) + "\n"
    report_frustrations = "### Inferred Frustrations\n" + "\n".join([f"- {f}" for f in frustrations]) + "\n---\n"
    report_details = "### Quantitative Profile\n"
    report_details += (f"**Demographics:** ~{demographics.get('age', 'N/A')}, {demographics.get('status', 'N/A')}, "
                       f"possibly in/from **{demographics.get('location', 'N/A')}**.\n")
    report_details += f"**Primary Interests:** {', '.join(interests[0])}\n"
    report_details += f"**Behavior:** A **{behavior.get('consistency', 'N/A')}** user who is most active on **{behavior.get('most_active_day', 'N/A')}s**.\n\n"

    p = personality
    report_details += (f"**Personality Profile:**\n```\n"
                       f"Introvert {make_bar(p['introvert'])} Extrovert\n  Sensing {make_bar(p['sensing'])} Intuition\n"
                       f" Thinking {make_bar(p['thinking'])} Feeling\n  Judging {make_bar(p['judging'])} Perceiving\n```\n")

    report_details += "**Core Motivations:**\n```\n"
    for m, score in sorted(motivations.items(), key=lambda x: x[1], reverse=True):
        report_details += f"{m.title():<15} {make_bar(score, 15)}\n"
    report_details += "```\n"

    report_details += "**Dominant Emotions:**\n```\n"
    for e, score in list(emotions.items())[:5]:
        report_details += f"{e.title():<12} {make_bar(score, 15)} {score:.1f}%\n"
    report_details += "```\n"

    report_footer = f"---\n*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    
    return (report_header + report_summary + report_goals + report_frustrations + report_details + report_footer)

def main():
    """Main function to run the persona generation process."""
    parser = argparse.ArgumentParser(
        description="Build a comprehensive, AI-refined user persona from a Reddit profile using Groq and Llama 3.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("profile_url", help="Full URL of the Reddit user's profile.")
    parser.add_argument("--limit", type=int, default=200, help="Total number of posts and comments to analyze (default: 200).")
    parser.add_argument("--output", type=str, help="Optional: Output filename for the markdown report (e.g., 'persona.md').")
    parser.add_argument("--skip-refinement", action="store_true", help="Skip the Groq/Llama 3 refinement step.")
    args = parser.parse_args()

    reddit = setup_reddit()
    username = get_username(args.profile_url)
    content, account_info = scrape_user_content(reddit, username, args.limit)
    
    if not content:
        print("❌ No content found to analyze. Exiting.")
        return
    print(f"✅ Collected {len(content)} items for u/{username}.")

    print("\n🔬 Performing initial analysis...")
    initial_analysis = {
        "demographics": extract_demographics(content), "personality": analyze_personality(content),
        "motivations": analyze_motivations(content), "behavior": analyze_behavior_patterns(content),
        "interests": analyze_interests(content), "emotions": analyze_sentiment(content),
    }
    print("✅ Initial analysis complete.")

    if args.skip_refinement:
        print("⏭️ Skipping Groq/Llama 3 refinement as requested.")
        refined_persona = {
            "summary": "Report generated from local analysis only. AI refinement was skipped by the user.",
            "archetype": "Analysis Incomplete", "goals": ["N/A"], "frustrations": ["N/A"]
        }
    else:
        refined_persona = semantic_refinement_with_groq(initial_analysis, content)

    print("\n📄 Generating final persona report...")
    final_report = generate_persona_report(username, initial_analysis, refined_persona)
    
    print("\n" + "="*80)
    print(final_report)
    print("="*80)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(final_report)
            print(f"\n✅ Report successfully saved to '{args.output}'")
        except IOError as e:
            print(f"\n🔴 Error saving file: {e}")

if __name__ == "__main__":
    main()