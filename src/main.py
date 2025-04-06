import requests
import cv2
import threading
import time
from google.cloud import vision
from speak_feelings import analyze_voice_emotion

# Google Vision Emotion Likelihood Mapping
LIKELIHOOD_MAP = {
    0: "UNKNOWN",
    1: "VERY_UNLIKELY",
    2: "UNLIKELY",
    3: "POSSIBLE",
    4: "LIKELY",
    5: "VERY_LIKELY"
}

# Emotion state shared between threads
latest_emotions = []
emotion_lock = threading.Lock()

# Track last detected emotion for Gemini reactions
last_emotion_name = None
last_gemini_response = ""
last_reaction_time = 0

# --- Gemini Integration ---
def recommend_action_for_emotion(emotion_name):
    api_key = "AIzaSyAvv5z8LYoc-k2b4sesTKhxxWBH4L94Pk4"  # <-- Replace with your actual Gemini API Key
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

    emotion_prompt_map = {
        "joy": "Give a short and cheerful message to someone feeling happy.",
        "sorrow": "Give an empathetic and comforting response to someone feeling sad or down.",
        "anger": "Give a calming and supportive message to someone feeling angry or frustrated.",
        "surprise": "Give an excited or curious reaction to someone feeling surprised."
    }

    prompt = emotion_prompt_map.get(emotion_name.lower(), "Respond kindly to this emotion.")

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"I detected the user is feeling {emotion_name}. {prompt}"}
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.ok:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            print("Gemini API error:", response.text)
            return "Sorry, couldn't get a suggestion."
    except Exception as e:
        print("Gemini error:", e)
        return "Error contacting Gemini."

# --- Google Vision Analysis ---
def analyze_emotion_google_vision(image_content):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_content)
    response = client.face_detection(image=image)
    faces = response.face_annotations

    emotions = []
    for face in faces:
        emotions.append({
            "joy": face.joy_likelihood,
            "sorrow": face.sorrow_likelihood,
            "anger": face.anger_likelihood,
            "surprise": face.surprise_likelihood
        })
    return emotions

def emotion_worker(frame_queue):
    while True:
        if not frame_queue:
            time.sleep(0.1)
            continue

        frame = frame_queue.pop(0)
        _, buffer = cv2.imencode('.jpg', frame)
        image_content = buffer.tobytes()

        try:
            emotions = analyze_emotion_google_vision(image_content)
            with emotion_lock:
                latest_emotions.clear()
                latest_emotions.extend(emotions)
        except Exception as e:
            print("Emotion analysis error:", e)

        time.sleep(2)

def get_strongest_emotion(emotion_dict):
    return max(emotion_dict.items(), key=lambda item: item[1])  # (emotion, score)

# --- Draw UI and Trigger Gemini ---
def draw_emotions(frame):
    global last_emotion_name, last_gemini_response, last_reaction_time

    with emotion_lock:
        if not latest_emotions:
            return

        for idx, emotion in enumerate(latest_emotions):
            strongest, score = get_strongest_emotion(emotion)
            emotion_name = strongest.lower()
            y_base = 30 + idx * 100

            # If emotion changed and is significant, get Gemini response
            if emotion_name != last_emotion_name and score >= 4 and time.time() - last_reaction_time > 5:
                last_emotion_name = emotion_name
                last_gemini_response = recommend_action_for_emotion(emotion_name)
                last_reaction_time = time.time()
                print("Gemini Suggestion:", last_gemini_response)

            # Draw strongest in green
            label = f"💥 {strongest.upper()}: {LIKELIHOOD_MAP[score]}"
            cv2.putText(frame, label, (10, y_base),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Draw others in blue
            y = y_base + 30
            for i, (key, value) in enumerate(emotion.items()):
                if key == strongest:
                    continue
                line = f"{key}: {LIKELIHOOD_MAP[value]}"
                cv2.putText(frame, line, (30, y + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 255), 1)

        # Display Gemini's response
        if last_gemini_response:
            wrapped_text = wrap_text(last_gemini_response, frame.shape[1] - 20)
            for i, line in enumerate(wrapped_text):
                cv2.putText(frame, line, (10, frame.shape[0] - 60 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# --- Utility to wrap Gemini message if it's long ---
def wrap_text(text, max_width):
    words = text.split()
    lines, line = [], ""
    for word in words:
        test_line = f"{line} {word}".strip()
        if len(test_line) * 10 < max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines

# --- Video Loop ---
def process_frame(cap):
    prev_time = time.time()
    frame_queue = []

    threading.Thread(target=emotion_worker, args=(frame_queue,), daemon=True).start()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        if time.time() - prev_time > 2:
            frame_queue.append(frame.copy())
            prev_time = time.time()

        draw_emotions(frame)

        cv2.imshow('Real-Time Emotion Analyzer', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --- Start Video and Voice ---
def capture_and_analyze():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        process_frame(cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    print("Starting voice emotion analysis in parallel...")
    voice_thread = threading.Thread(target=analyze_voice_emotion, daemon=True)
    voice_thread.start()

    print("Starting video emotion analyzer...")
    capture_and_analyze()

if __name__ == "__main__":
    main()
