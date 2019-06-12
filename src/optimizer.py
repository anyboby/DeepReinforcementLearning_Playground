import threading

"""
The optimizer calls MasterNetwork.optimize() endlessly in a loop, possibly from multiple threads
"""
class Optimizer(threading.Thread):
    
    stop_signal = False
    
    def __init__(self, master_network):
        threading.Thread.__init__(self)
        self.master_network = master_network

    def run (self):
        while not self.stop_signal:
            self.master_network.optimize()
    def stop(self):
        self.stop_signal = True
