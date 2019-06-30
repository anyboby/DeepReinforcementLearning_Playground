import threading
import constants as Constants

"""
The optimizer calls MasterNetwork.optimize() endlessly in a loop, possibly from multiple threads
"""
class Optimizer(threading.Thread):
    
    stop_signal = False
    write_summaries = False
    
    def __init__(self, master_network):
        threading.Thread.__init__(self)
        self.master_network = master_network

    def run (self):
        trainings = 0
        trained = False
        while not self.stop_signal:
            if Optimizer.write_summaries:
                trained = self.master_network.optimize(writesummaries=True)
                Optimizer.write_summaries = not trained
            else: 
                trained = self.master_network.optimize()
                if trained:
                    trainings += 1
                    if trainings % Constants.SUMMARY_STEPS == 0: 
                        Optimizer.write_summaries = True
    def stop(self):
        self.stop_signal = True
