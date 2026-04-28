Lost Item Finder Robot

Voice query setup

Install the optional microphone and text-to-speech libraries:

pip install SpeechRecognition pyttsx3 pyaudio

If pyaudio fails on Ubuntu/ROS, install the system audio packages first:

sudo apt update
sudo apt install portaudio19-dev python3-pyaudio
pip install SpeechRecognition pyttsx3 pyaudio

Quick test:

python3 -c "import speech_recognition, pyttsx3, pyaudio; print('voice libs ready')"

If the test prints "voice libs ready", test.py should use the mic automatically.
If the mic libraries are missing or speech is not understood, the robot falls back
to typed questions.

Example questions:

Have you seen my water bottle?
Can you show me the backpack?
Nevermind
