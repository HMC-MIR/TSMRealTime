pyinstaller main_mac.spec
#pyinstaller main.py --onefile --collect-all librosa \
#--collect-all scipy \
#--collect-all pymongo \
#--collect-all pyaudio \
#--hidden-import=player --hidden-import=audio --hidden-import=userinfo --add-data "samples:samples" --add-data "./ffmpeg.exe:."

sleep 10