# Clone repository

    $ git clone https://github.com/zetraxus/Atari-StarGunner-v0-bot.git
    $ cd Atari-StarGunner-v0-bot/
    
# Creation and activation virtual environment

    $ python3 -m venv env
    $ source env/bin/activate
    
# Installation

    $ pip install --upgrade pip
    $ pip install -r requirements.txt

# Usage

    python3 src/main.py
    
    Parametrization of the program execution is in main.py file (lines 13, 14, 15):
    
    - SAVE_WEIGHT: if neural network weights will be saved
    - LOAD_WEIGHTS: if neural network weights will be loaded
    - RENDER: if game will be rendered
    - SAVE_RESULTS: if game results will be saved
    - ALGORITHMS: array of algorithms that the agent will be trained with 
    - params:
        - TEST_NUM: number of tests
        - EPISODES_NUM: number of episodes 
        - START_LEARNING_ITERATION: iteration which the agent will be learning from
        - PENALTY: number of points for life loss
        - PREMIUM: number of points for well-aimed shot
        - LEARNING_RATE: learning rate which will be applied to neural network
        - EPSILON_MIN: minimal probability of choosing random action in learning process
        - EPSILON_DECAY: step of probability used in learning process
        - DISCOUNT_FACTOR: weight of calculated q-value in updating policy algorithm 
        - BATCH_SIZE: number of iterations which will be applied to neural network
    
# Useful links

https://gym.openai.com/envs/StarGunner-v0/
