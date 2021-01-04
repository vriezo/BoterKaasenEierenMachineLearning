import random
 
from bke import MLAgent, is_winner, opponent, RandomAgent, train_and_plot
 
 
class MyAgent(MLAgent):
    def evaluate(self, board):
        if is_winner(board, self.symbol):
            reward = 1
        elif is_winner(board, opponent[self.symbol]):
            reward = -1
        else:
            reward = 0
        return reward
    
    
random.seed(1)
 
my_agent = MyAgent(alpha=0.8, epsilon=0.2) 
#Dit zijn de hyperparameters alpha en epsilon. Alpha bepaalt hoe snel de agent iets leert. In dit geval is alpha redelijk hoog en zal dus snel nieuwe kennis als het ware opnemen. 
#De epsilon bepaalt daarentegen hoe vaak de agent iets nieuws probeert in plaats van het gebruiken van al bekende informatie. Deze factor is daarom maar 0,2. Als deze factor hoger zou zijn zou de agent radom dingen proberen in plaats van het gebruiken van de best bekende zet.
random_agent = RandomAgent()
 
train_and_plot(
    agent=my_agent,
    validation_agent=random_agent,
    iterations=50,
    trainings=100,
    validations=1000)
