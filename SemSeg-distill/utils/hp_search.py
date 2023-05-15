import os
import warnings
warnings.filterwarnings("ignore")
import logging
import joblib

import optuna

from utils.train import Trainer



class HPSearcher:
      
    def __init__(self, args, trial=None):
        self.args = args
        self.trial = trial
        
        
    def get_random_hps(self):
        #self.args.lambda_kd = self.trial.suggest_categorical("lambda_kd", [0])
        self.args.lambda_dkd = self.trial.suggest_categorical("lambda_dkd", [0.01, 0.1, 1])
        #self.args.cwd = self.trial.suggest_categorical("cwd", [True])
        #self.args.lambda_srrl_reg = self.trial.suggest_categorical("lambda_srrl_reg", [0])
        #self.args.lambda_srrl_feat = self.trial.suggest_categorical("lambda_srrl_feat", [0])
        #self.args.lambda_mgd = self.trial.suggest_categorical("lambda_mgd", [1e-4, 5e-4, 1e-3])
        #self.args.alpha_mgd = self.trial.suggest_categorical("alpha_mgd", [0.1, 0.25])
        #self.args.alpha_dkd = self.trial.suggest_categorical("alpha_dkd", [1])
        self.args.beta_dkd = self.trial.suggest_categorical("beta_dkd", [1, 4, 8])
        self.args.temp_dkd = self.trial.suggest_categorical("temp_dkd", [1, 4])
        #self.args.srrl_layer = self.trial.suggest_categorical("srrl_layer", ['last', 'back'])
        #self.args.mgd_layer = self.trial.suggest_categorical("mgd_layer", ['last', 'back'])
        #self.args.mgd_mask = self.trial.suggest_categorical("mgd_mask", ['space', 'channel'])
        self.args.norm_dkd = self.trial.suggest_categorical("norm_dkd", ['space', 'channel'])
        
        
    def objective(self, trial):
        name = 'gridsearch_' + str(trial.datetime_start) + str(trial.number)
        self.trial = trial 
        
        self.get_random_hps()
        trainer = Trainer(args=self.args, trial=self.trial)   
        metr = trainer.train()

        return metr
    
    
    def hp_search(self):

        search_space = {#"lambda_kd": [0],
                        "cwd": [True],
                        #"lambda_srrl_reg": [0],
                        #"lambda_srrl_feat": [0],
                        "lambda_mgd": [1e-4, 5e-4, 1e-3],
                        "alpha_mgd": [0.1, 0.25],
                        #"srrl_layer": ['last'],
                        "mgd_layer": ['last', 'back'],
                        "mgd_mask": ['space', 'channel']}

        self.study = optuna.create_study(study_name=f'hp_search_{self.args.hp_search_name}', 
                                         direction='maximize', 
                                         #sampler=optuna.samplers.GridSampler(search_space),
                                         pruner=optuna.pruners.HyperbandPruner())
        
        if os.path.exists(f'{self.args.hp_search_path}/hp_search_{self.args.hp_search_name}.pkl'): 
            study_old = joblib.load(f'{self.args.hp_search_path}/hp_search_{self.args.hp_search_name}.pkl')
            self.study.add_trials(study_old.get_trials())
            logging.info('Study resumed!')
        
        save_callback = SaveCallback(self.args.hp_search_path)
        self.study.optimize(lambda trial: self.objective(trial), n_trials=self.args.n_trials, callbacks=[save_callback])

        pruned_trials = self.study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = self.study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])

        logging.info("Study statistics: ")
        logging.info(f"  Number of finished trials: {len(self.study.trials)}")
        logging.info(f"  Number of pruned trials: {len(pruned_trials)}")
        logging.info(f"  Number of complete trials: {len(complete_trials)}")
        logging.info("Best trial:")
        logging.info(f"  Value: {self.study.best_trial.value}")
        logging.info("  Params: ")
        for key, value in self.study.best_trial.params.items():
            logging.info(f"    {key}: {value}")

        return self.study
    
    
class SaveCallback:
    
    def __init__(self, directory):
        self.directory = directory

    def __call__(self, study, trial):
        joblib.dump(study, os.path.join(self.directory, f'{study.study_name}.pkl'))