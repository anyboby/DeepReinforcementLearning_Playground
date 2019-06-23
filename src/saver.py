import tensorflow as tf
import os, time, json
from savedata import SaveData
import constants as Constants


# In practical computing, wall time is the actual time, usually measured in seconds, 
# that a program takes to run or to execute its assigned tasks. When the computer is multitasking, 
# the wall time for each program is determined separately, and depends on how the microprocessor allocates 
# resources among the programs. For example, if a computer multitasks using three different programs 
# for a continuous period of 60 seconds, one program might consume 10 seconds of wall time, the second 
# program 20 seconds, and the third program 30 seconds. But these are not contiguous blocks; instead they are 
# allocated in a rotating fashion, similar to time-division multiplexing in communications practice.

class Saver:
    def __init__(self, checkpointDir=Constants.CHECKPOINT_DIR, saveFile=Constants.SAVE_FILE, lastSaveTime=time.time()):
        self.saver          =  tf.train.Saver()
        self.data           =  SaveData()
        self.checkpointDir  =  checkpointDir
        self.checkpoint     =  tf.train.get_checkpoint_state(self.checkpointDir)
        self.saveFilename   =  saveFile
        self.lastSaveTime   =  lastSaveTime
        self.lastSaveScore  =  0


    """
    saves a score array as json file to specified filepath
    """
    def saveScores(self, scores):
        with open(self.saveFilename, "w") as f:
            f.write(json.dumps(scores))

            
            #f.write(
            #    "[" +
            #    ",\n".join(json.dumps(i) for i in scores) +
            #    "]\n")

    def loadScores(self):
        try:
            with open(self.saveFilename) as f:
                self.data.setScores(json.loads(f.read().strip()))
                print("Loaded scores, last 5 shown: ", [score["reward"] for score in self.data.scores[-5:]])
        except Exception:
            print("Unable to load scores")


    def load(self, session):
        self.loadScores()
        if self.checkpoint is not None and self.checkpoint.model_checkpoint_path is not None:

            # Load checkpoint
            self.saver.restore(session, self.checkpoint.model_checkpoint_path)
            self.data.global_t = int(self.checkpoint.model_checkpoint_path.split("-")[1])

            # Set Wall time
            wall_t_fname = self.checkpointDir + "/" + "wall_t." + str(self.data.global_t)
            with open(wall_t_fname, "r") as f:
                self.data.wall_t = float(f.read())

            print("Checkpoint:", self.checkpoint.model_checkpoint_path, "\nGlobal Step:", self.data.global_t)



    """
    saves scores as well as tf checkpoint to specified filepaths
    """
    def save(self, session):
        if not os.path.exists(self.checkpointDir):
            os.mkdir(self.checkpointDir)

        # write wall time
        wall_t = time.time() - self.data.start_time
        wall_t_filename = self.checkpointDir + "/" + "wall_t." + str(self.data.global_t)
        with open(wall_t_filename, "w") as f:
            f.write(str(wall_t))

        print("saving data...")
        self.saveScores(self.data.scores)

        def truncate(n, decimals=0):
            multiplier = 10 ** decimals
            return int(n * multiplier) / multiplier

        #truncated reward
        trnc_score = truncate(self.data.scores[-1]["reward"])

        #replace - with m to prevent interference with other -
        score_string = str(trnc_score).replace("-","neg",1)

        # Save TF checkpoint
        print("saving tf checkpoint...")
        self.saver.save(session, self.checkpointDir + "/" + "checkpoint" + score_string, global_step = self.data.global_t)

    def canSave(self):
        return time.time() - self.lastSaveTime >= Constants.SAVE_FRAMES or \
                abs(self.data.scores[-1]["reward"] - self.lastSaveScore) >= 1

    def saveIfRequested(self, session):
        if self.data.saveRequested:
            self.data.saveRequested = False
            if self.canSave():
                #print("Saving checkpoint as score crossed threshold of:", Constants.MIN_SAVE_REWARD)
                self.save(session)
                self.lastSaveTime = time.time()
