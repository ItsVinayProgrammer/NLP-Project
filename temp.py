import json
from vosk import Model, KaldiRecognizer
import pyaudio
model = Model("D:\\Python Code\\vosk-model-en-in-0.5")
recognizer = KaldiRecognizer(model, 16000)
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()
print("Say a sentence...")
while True:
    data = stream.read(4000, exception_on_overflow=False)
    if recognizer.AcceptWaveform(data):
        result = json.loads(recognizer.Result())
        print(result.get("text", ""))
        break
stream.stop_stream()
stream.close()
p.terminate()
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import google.generativeai as genai
import random
import time
import re
import os
print(os.path.exists("D:\\Python Code\\vosk-model-en-in-0.5"))
import pyaudio
from vosk import Model, KaldiRecognizer
from gtts import gTTS
import pygame
import difflib

# Initialize models, APIs, and speech components
try:
    tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
    model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")
except Exception as e:
    print(f"Error loading grammar model: {e}")
    exit(1)

# Configure Gemini API
api_key = "Replace with your valid Gemini API key"  # Replace with your valid Gemini API key
os.environ['GOOGLE_API_KEY'] = api_key
genai.configure(api_key=api_key)

# Initialize Vosk model
MODEL_PATH = "D:\\Python Code\\vosk-model-en-in-0.5"
if not os.path.exists(MODEL_PATH):
    print(f"Please ensure the Vosk model is at {MODEL_PATH}")
    exit(1)
vosk_model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(vosk_model, 16000)

# Initialize pygame mixer
pygame.mixer.init()

# Define difficulty levels
DIFFICULTY_LEVELS = {
    "beginner": {
        "description": "Simple words and grammar for beginners.",
        "complexity": 0.2,
        "grammar_strictness": 0.5,
        "speech_speed": 0.8,
        "vocabulary_level": "A1-A2",
        "exercise_types": ["basic conversation", "simple vocabulary", "common phrases"]
    },
    "intermediate": {
        "description": "Varied vocabulary and grammar for learners.",
        "complexity": 0.5,
        "grammar_strictness": 0.8,
        "speech_speed": 1.0,
        "vocabulary_level": "B1-B2",
        "exercise_types": ["daily conversations", "idioms", "conditional sentences", "past tenses"]
    },
    "advanced": {
        "description": "Complex vocabulary and natural expressions for proficient speakers.",
        "complexity": 0.9,
        "grammar_strictness": 1.0,
        "speech_speed": 1.2,
        "vocabulary_level": "C1-C2",
        "exercise_types": ["debates", "academic discussion", "business English", "complex hypotheticals"]
    }
}

# Response templates
GREETINGS = [
    "Hi! Let’s practice English!",
    "Hello! Ready to speak?",
    "Hey! Time for English!",
    "Greetings! Let’s learn!"
]

FAREWELLS = [
    "Great job! Come back soon!",
    "Goodbye! Keep practicing!",
    "Bye! You’re improving!",
    "See you next time!"
]

DIFFICULTY_CHANGE_PROMPTS = {
    "beginner": "Beginner mode: simple words, slow speech.",
    "intermediate": "Intermediate mode: varied sentences.",
    "advanced": "Advanced mode: complex vocabulary."
}

# User profile
user_profile = {
    "current_difficulty": "beginner",
    "sessions_completed": 0,
    "words_practiced": [],
    "grammar_points_reviewed": [],
    "lesson_history": [],
    "correct_sentences": []
}

# Save and load user profile
def save_user_profile():
    try:
        with open("user_profile.json", "w") as file:
            json.dump(user_profile, file)
    except Exception as e:
        print(f"Error saving profile: {e}")

def load_user_profile():
    global user_profile
    default_profile = {
        "current_difficulty": "beginner",
        "sessions_completed": 0,
        "words_practiced": [],
        "grammar_points_reviewed": [],
        "lesson_history": [],
        "correct_sentences": []
    }
    try:
        with open("user_profile.json", "r") as file:
            loaded_profile = json.load(file)
            user_profile = default_profile.copy()
            for key, value in loaded_profile.items():
                user_profile[key] = value
            if "correct_sentences" not in user_profile:
                user_profile["correct_sentences"] = []
    except FileNotFoundError:
        user_profile = default_profile.copy()
    except Exception as e:
        print(f"Error loading profile: {e}")
        user_profile = default_profile.copy()

def correct_grammar(text, strictness=0.8):
    """Correct grammar with strictness."""
    if len(text.strip().split()) < 2:
        return text
    
    # Pre-process common errors
    corrected_text = text
    if corrected_text.lower().startswith("i "):
        corrected_text = "I " + corrected_text[2:]
    if " and driving " in corrected_text.lower():
        corrected_text = corrected_text.lower().replace(" and driving ", " am driving ")
    
    clean_text = re.sub(r'[,;:\-]', ' ', corrected_text).strip()
    input_text = "gec: " + clean_text
    try:
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)
        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Grammar correction error: {e}")
        return text
    
    # Preserve correct apostrophes and capitalization
    corrected_text = corrected_text.replace("dont ", "don't ").replace("cant ", "can't ").replace("thats ", "that's ").replace("its ", "it's ")
    if text[0].isupper() and corrected_text[0].islower():
        corrected_text = corrected_text[0].upper() + corrected_text[1:]
    
    # Similarity check
    similarity = difflib.SequenceMatcher(None, text.lower().strip(), corrected_text.lower().strip()).ratio()
    if similarity > 0.95:
        return text
    
    return corrected_text.strip()

def highlight_differences(original, corrected):
    """Identify grammar differences."""
    original_clean = re.sub(r'\s+', ' ', re.sub(r'[.,!?;]', '', original)).strip().lower()
    corrected_clean = re.sub(r'\s+', ' ', re.sub(r'[.,!?;]', '', corrected)).strip().lower()
    
    if original_clean == corrected_clean:
        return None
    
    words_original = original.split()
    words_corrected = corrected.split()
    
    matcher = difflib.SequenceMatcher(None, words_original, words_corrected)
    differences = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            for pos in range(min(i2 - i1, j2 - j1)):
                original_word = words_original[i1 + pos]
                corrected_word = words_corrected[j1 + pos]
                if original_word.lower() != corrected_word.lower():
                    differences.append({
                        "original": original_word,
                        "corrected": corrected_word,
                        "position": i1 + pos
                    })
        elif tag == 'delete' and i2 - i1 <= 2:
            for pos in range(i1, i2):
                differences.append({
                    "original": words_original[pos],
                    "corrected": "",
                    "position": pos
                })
        elif tag == 'insert' and j2 - j1 <= 2:
            for pos in range(j1, j2):
                differences.append({
                    "original": "",
                    "corrected": words_corrected[pos],
                    "position": i1
                })
    
    differences = [d for d in differences if not (d["original"] in ",.!?;:" and d["corrected"] in ",.!?;:")]
    differences = [d for d in differences if d["original"] or d["corrected"]]
    differences.sort(key=lambda x: x["position"])
    
    return differences if differences else None

def explain_mistake(diff, difficulty):
    """Explain grammar mistakes."""
    if not diff:
        return ["Your sentence is correct! Great job!"]
    
    explanations = []
    for error in diff:
        orig = error["original"]
        corr = error["corrected"]
        if not orig and not corr:
            continue
            
        if difficulty == "beginner":
            if orig.lower() == "i" and corr.lower() == "i":
                explanations.append(f"'{orig}' should be '{corr}'. Always capitalize 'I'.")
            elif orig.lower() in ["its", "it's", "thats", "that's"] and corr.lower() in ["its", "it's", "thats", "that's"]:
                explanations.append(f"'{orig}' should be '{corr}'. Use 'it's' or 'that's' for 'it is'/'that is', 'its'/'thats' for possession.")
            elif orig.lower() in ["is", "are", "am"] and corr.lower() in ["is", "are", "am"]:
                explanations.append(f"'{orig}' should be '{corr}'. The verb must match the subject.")
            elif orig.lower() in ["a", "an"] and corr.lower() in ["a", "an"]:
                explanations.append(f"'{orig}' should be '{corr}'. Use 'an' before vowels.")
            elif not orig and corr:
                explanations.append(f"Add '{corr}' to your sentence.")
            elif orig and not corr:
                explanations.append(f"You don’t need '{orig}'.")
            else:
                explanations.append(f"Change '{orig}' to '{corr}'.")
        elif difficulty == "intermediate":
            if orig.lower() == "i" and corr.lower() == "i":
                explanations.append(f"'{orig}' should be '{corr}'. Capitalize 'I'.")
            elif orig.lower() in ["its", "it's", "thats", "that's"] and corr.lower() in ["its", "it's", "thats", "that's"]:
                explanations.append(f"'{orig}' should be '{corr}'. 'It's'/'That's' means 'it is'/'that is', 'its'/'thats' shows ownership.")
            elif orig.lower() in ["is", "are", "am"] and corr.lower() in ["is", "are", "am"]:
                explanations.append(f"'{orig}' should be '{corr}'. Ensure verb agreement.")
            elif orig.lower() in ["a", "an"] and corr.lower() in ["a", "an"]:
                explanations.append(f"'{orig}' should be '{corr}'. Articles match sounds.")
            elif not orig and corr:
                explanations.append(f"Add '{corr}' for correctness.")
            elif orig and not corr:
                explanations.append(f"Remove '{orig}'.")
            else:
                explanations.append(f"Use '{corr}' instead of '{orig}'.")
        else:
            if orig.lower() == "i" and corr.lower() == "i":
                explanations.append(f"'{orig}' should be '{corr}'. Pronoun 'I' needs capitalization.")
            elif orig.lower() in ["its", "it's", "thats", "that's"] and corr.lower() in ["its", "it's", "thats", "that's"]:
                explanations.append(f"'{orig}' should be '{corr}'. Use 'it's'/'that's' for contraction, 'its'/'thats' for possessive.")
            elif orig.lower() in ["is", "are", "am"] and corr.lower() in ["is", "are", "am"]:
                explanations.append(f"'{orig}' should be '{corr}'. Subject-verb agreement issue.")
            elif orig.lower() in ["a", "an"] and corr.lower() in ["a", "an"]:
                explanations.append(f"'{orig}' should be '{corr}'. Incorrect article.")
            elif not orig and corr:
                explanations.append(f"Insert '{corr}' for structure.")
            elif orig and not corr:
                explanations.append(f"'{orig}' is redundant.")
            else:
                explanations.append(f"Replace '{orig}' with '{corr}'.")
    
    return explanations

def speak_message(message, speed=1.0):
    """Speak message with gTTS."""
    try:
        speech_text = re.sub(r'[.!?]', ', ', message)
        speech_text = re.sub(r'[:,;]', ' ', speech_text)
        speech_text = speech_text.replace("'", "")
        speech_text = re.sub(r'"([^"]*)"', r'\1', speech_text)
        speech_text = re.sub(r'\s+', ' ', speech_text).strip()
        
        tts = gTTS(text=speech_text, lang='en', tld='co.uk', slow=(speed < 0.9))
        temp_file = "temp_audio.mp3"
        tts.save(temp_file)
        
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        pygame.mixer.music.unload()
        os.remove(temp_file)
    except Exception as e:
        print(f"Error in speak_message: {e}")

def get_voice_input(silence_count=0, max_silence=3, retry_count=0, max_retries=3):
    """Capture voice input with Vosk."""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
    stream.start_stream()
    
    print("Listening...")
    text = ""
    timeout = time.time() + 20
    while True:
        try:
            data = stream.read(4000, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
                    break
            if time.time() > timeout:
                silence_count += 1
                if silence_count >= max_silence:
                    speak_message("No input detected. Exiting.")
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    return None, silence_count, retry_count
                speak_message(f"Didn’t hear you. Speak clearly. {max_silence - silence_count} tries left.")
                stream.stop_stream()
                stream.close()
                p.terminate()
                return "", silence_count, retry_count
        except Exception as e:
            print(f"Recognition error: {e}")
            if retry_count < max_retries:
                tips = [
                    "Stress syllables clearly, like 'TEA' as 'tee'.",
                    "Speak one word at a time.",
                    "Keep the mic close and speak loudly.",
                    "Pause briefly after each word."
                ]
                speak_message(f"Couldn’t understand. {tips[retry_count % len(tips)]}")
                stream.stop_stream()
                stream.close()
                p.terminate()
                return "", silence_count, retry_count + 1
            speak_message("Couldn’t catch that. Try again.")
            stream.stop_stream()
            stream.close()
            p.terminate()
            return "", silence_count, 0
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    return text.strip(), silence_count, retry_count

def format_grammar_correction(original, corrected, differences, difficulty):
    """Format grammar correction."""
    if not differences:
        return None
    
    clean_corrected = re.sub(r'["\']', '', corrected)
    if difficulty == "beginner":
        return f"You said: {original}. Try: {clean_corrected}"
    elif difficulty == "intermediate":
        return f"Better: {clean_corrected}"
    else:
        return f"Corrected: {clean_corrected}"

def get_gemini_response(user_input, difficulty, conversation_history=None):
    """Generate Gemini response."""
    try:
        if conversation_history is None:
            conversation_history = []
        
        gemini_model_name = "gemini-1.5-flash"
        gemini_model = genai.GenerativeModel(gemini_model_name)
        
        complexity = DIFFICULTY_LEVELS[difficulty]["complexity"]
        vocab_level = DIFFICULTY_LEVELS[difficulty]["vocabulary_level"]
        
        prompt = f"""
        You are an English tutor for {difficulty}-level students.
        Respond in natural English for {vocab_level} CEFR level.
        {"Simple words, short sentences." if difficulty == "beginner" else ""}
        {"Varied vocabulary, some idioms." if difficulty == "intermediate" else ""}
        {"Advanced vocabulary, natural flow." if difficulty == "advanced" else ""}
        Keep it short (2-3 sentences).
        
        User's message: {user_input}
        
        Previous conversation:
        {conversation_history[:5] if conversation_history else "None"}
        """
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        return response_text if response_text else "Got it. Tell me more."
            
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        if difficulty == "beginner":
            return random.choice(["Okay. Say more?", "Nice! What next?"])
        elif difficulty == "intermediate":
            return random.choice(["Cool! More details?", "Got it. What else?"])
        else:
            return random.choice(["Interesting. Expand?", "Great. What’s next?"])

def suggest_exercise(difficulty):
    """Suggest an exercise."""
    exercise_types = DIFFICULTY_LEVELS[difficulty]["exercise_types"]
    exercise_type = random.choice(exercise_types)
    
    if exercise_type == "basic conversation":
        topics = ["family", "food", "hobbies", "weather"]
        topic = random.choice(topics)
        return f"Let’s talk about {topic}. Tell me about your {topic}."
    elif exercise_type == "simple vocabulary":
        categories = ["animals", "foods", "jobs"]
        category = random.choice(categories)
        return f"Name three {category} you know."
    elif exercise_type == "common phrases":
        situations = ["greeting a friend", "ordering food"]
        situation = random.choice(situations)
        return f"What would you say when {situation}?"
    elif exercise_type == "daily conversations":
        scenarios = ["at a shop", "with a friend"]
        scenario = random.choice(scenarios)
        return f"Imagine you’re {scenario}. What do you say?"
    elif exercise_type == "idioms":
        return "What does ‘raining cats and dogs’ mean?"
    elif exercise_type == "conditional sentences":
        return "Finish: If I could visit anywhere, I would..."
    elif exercise_type == "debates":
        topics = ["social media", "climate change"]
        topic = random.choice(topics)
        return f"What are your thoughts on {topic}?"
    elif exercise_type == "academic discussion":
        topics = ["technology", "education"]
        topic = random.choice(topics)
        return f"What influences {topic} most?"
    elif exercise_type == "business English":
        scenarios = ["job interview", "team meeting"]
        scenario = random.choice(scenarios)
        return f"What would you say in a {scenario}?"
    else:
        return "Imagine a group of friends planning an event. Who are they? What are they planning?"

def process_commands(user_input):
    """Process special commands."""
    global user_profile
    
    if re.search(r'(?:change|set|switch)\s.*\s(beginner|intermediate|advanced)', user_input, re.IGNORECASE):
        new_level = re.search(r'(beginner|intermediate|advanced)', user_input, re.IGNORECASE).group(1).lower()
        if new_level in DIFFICULTY_LEVELS:
            user_profile["current_difficulty"] = new_level
            save_user_profile()
            response = DIFFICULTY_CHANGE_PROMPTS[new_level]
            print(f"Assistant: {response}")
            speak_message(response, DIFFICULTY_LEVELS[new_level]["speech_speed"])
            return True
    
    if re.search(r'(?:give|suggest|start|what\s*are\s*the)\s.*\s(exercise|practice|activity)', user_input, re.IGNORECASE) or user_input.lower() in ["exercise", "start exercise", "what are the exercise"]:
        difficulty = user_profile["current_difficulty"]
        exercise = suggest_exercise(difficulty)
        print(f"Assistant: {exercise}")
        speak_message(exercise, DIFFICULTY_LEVELS[difficulty]["speech_speed"])
        return True
    
    if re.search(r'(?:what|describe|explain)\s.*\s(level|difficulty)', user_input, re.IGNORECASE):
        difficulty = user_profile["current_difficulty"]
        description = f"Your level is {difficulty}. {DIFFICULTY_LEVELS[difficulty]['description']}"
        print(f"Assistant: {description}")
        speak_message(description, DIFFICULTY_LEVELS[difficulty]["speech_speed"])
        return True
    
    if re.search(r'(?:what|which|list)\s.*\s(levels|difficulties)', user_input, re.IGNORECASE):
        levels_info = "Levels: " + ", ".join(f"{level} - {DIFFICULTY_LEVELS[level]['description']}" for level in DIFFICULTY_LEVELS)
        print(f"Assistant: {levels_info}")
        speak_message(levels_info, DIFFICULTY_LEVELS[user_profile["current_difficulty"]]["speech_speed"])
        return True
    
    return False

def main():
    load_user_profile()
    
    # Mic test
    speak_message("Test your mic. Say 'hello India' clearly.")
    text, _, _ = get_voice_input()
    if text.lower() == "hello india":
        speak_message("Mic is perfect! Let’s start.")
    else:
        speak_message("Didn’t catch 'hello India', but we’ll continue. Speak clearly.")
    
    greeting = random.choice(GREETINGS)
    current_difficulty = user_profile["current_difficulty"]
    print(f"Assistant: {greeting}")
    speak_message(greeting, DIFFICULTY_LEVELS[current_difficulty]["speech_speed"])
    
    difficulty_info = f"At {current_difficulty} level. Say 'change level to beginner, intermediate, or advanced', 'explain' for grammar, 'exercise' for a task, or 'exit' to stop."
    print(f"Assistant: {difficulty_info}")
    speak_message(difficulty_info, DIFFICULTY_LEVELS[current_difficulty]["speech_speed"])
    
    last_differences = None
    last_text_checked = None
    conversation_history = []
    silence_count = 0
    max_silence = 3
    
    while True:
        user_input, silence_count, retry_count = get_voice_input(silence_count, max_silence)
        
        if user_input is None:
            user_profile["sessions_completed"] += 1
            save_user_profile()
            pygame.mixer.quit()
            break
        
        if not user_input:
            continue
        
        print(f"You: {user_input}")
        time.sleep(0.5)
        
        if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
            farewell = random.choice(FAREWELLS)
            print(f"Assistant: {farewell}")
            speak_message(farewell, DIFFICULTY_LEVELS[current_difficulty]["speech_speed"])
            user_profile["sessions_completed"] += 1
            save_user_profile()
            break
        
        if process_commands(user_input):
            current_difficulty = user_profile["current_difficulty"]
            continue
            
        if user_input.lower() == "explain":
            if last_differences or last_text_checked:
                explanations = explain_mistake(last_differences, current_difficulty)
                explanation_intro = f"Reviewing: {last_text_checked}"
                print(f"Assistant: {explanation_intro}")
                speak_message(explanation_intro, DIFFICULTY_LEVELS[current_difficulty]["speech_speed"])
                for i, explanation in enumerate(explanations, 1):
                    spoken_explanation = explanation.replace("'", "")
                    print(f"{i}. {explanation}")
                    speak_message(f"{i}. {spoken_explanation}", DIFFICULTY_LEVELS[current_difficulty]["speech_speed"])
            else:
                no_issues = "No sentences to explain. Say something first!"
                print(f"Assistant: {no_issues}")
                speak_message(no_issues, DIFFICULTY_LEVELS[current_difficulty]["speech_speed"])
            continue
        
        last_text_checked = user_input
        corrected = correct_grammar(user_input, DIFFICULTY_LEVELS[current_difficulty]["grammar_strictness"])
        differences = highlight_differences(user_input, corrected)
        last_differences = differences
        
        contextual_response = get_gemini_response(corrected if differences else user_input, 
                                                current_difficulty, 
                                                conversation_history)
        
        message_for_history = corrected if differences else user_input
        conversation_history.insert(0, f"User: {message_for_history}")
        conversation_history.insert(0, f"Assistant: {contextual_response}")
        if len(conversation_history) > 10:
            conversation_history = conversation_history[:10]
        
        speech_speed = DIFFICULTY_LEVELS[current_difficulty]["speech_speed"]
        if differences:
            correction_note = format_grammar_correction(user_input, corrected, differences, current_difficulty)
            print(f"Assistant: {correction_note}\n{contextual_response}")
            if current_difficulty == "beginner":
                speak_message(correction_note, speech_speed)
                time.sleep(0.5)
                speak_message(contextual_response, speech_speed)
                speak_message("Say explain for details.", speech_speed)
            else:
                response = f"{correction_note} {contextual_response}"
                speak_message(response, speech_speed)
                if current_difficulty == "intermediate":
                    speak_message("Say explain for details.", speech_speed)
        else:
            print(f"Assistant: Your sentence is correct! Great job! {contextual_response}")
            speak_message(f"Your sentence is correct! Great job! {contextual_response}", speech_speed)
            if "correct_sentences" not in user_profile:
                user_profile["correct_sentences"] = []
            user_profile["correct_sentences"].append(user_input)
            save_user_profile()
    
    pygame.mixer.quit()

if __name__ == "__main__":
    main()

