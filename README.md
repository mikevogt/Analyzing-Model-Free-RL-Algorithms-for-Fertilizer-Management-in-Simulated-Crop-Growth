# Analyzing Model Free RL Algorithms for Fertilizer Management in Simulated Crop Growth
 This GitHub represents the code-base relating to the paper "Analyzing Model-Free Reinforcement Learning Algorithms for Fertilizer Management in Simulated Crop Growth".
 
 The environment used in this research has been created by Overweg et. al. and is termed [CropGym](https://github.com/BigDataWUR/crop-gym) - you can follow the instructions on their repository to get started. It is important to note that the hyperparameter optimization .py files must be placed in the \envs folder of the CropGym environment. Optimal hyperparameters are found using the highly
scalable Ray[Tune] optimization framework along with the optimization algorithms provided by Optuna.
