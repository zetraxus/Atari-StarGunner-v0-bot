https://gym.openai.com/envs/StarGunner-v0/

git clone https://github.com/zetraxus/Atari-StarGunner-v0-bot.git  
cd Atari-StarGunner-v0-bot/  
git clone https://github.com/openai/gym  
python3 -m venv env  
source env/bin/activate  
pip install --upgrade pip  
pip install -r requirements.txt  
pip install gym[atari]  
python3 src/main.py