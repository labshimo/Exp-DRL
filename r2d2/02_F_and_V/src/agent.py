import gym
import pickle
import os
import numpy as np
import traceback
import math
import tensorflow as tf
import multiprocessing as mp
from learner import Learner
from actor import Actor
from memory import ReplayMemory, PERGreedyMemory, SumTree, PERProportionalMemory, PERRankBaseMemory
from simlator import *
#---------------------------------------------------
# manager
#---------------------------------------------------
class R2D2Manager():
    def __init__(self, num_actors, args):
        self.args          = args["agent"]
        self.simlator_args = args["simlator"]
        self.num_actors    = num_actors
        self._create_process()
  
    def _create_process(self):
        # create each Queue
        experience_q = mp.Queue()
        model_sync_q = [[mp.Queue(), mp.Queue(), mp.Queue()] for _ in range(self.num_actors)]
        self.learner_end_q = [mp.Queue(), mp.Queue()]
        self.actors_end_q = [mp.Queue() for _ in range(self.num_actors)]
        
        # create actor ps
        self.actors_ps = []
        epsilon = self.args["epsilon"]
        epsilon_alpha = self.args["epsilon_alpha"]
        for i in range(self.num_actors):
            # get epsilon for each actor
            if self.num_actors <= 1:
                epsilon_i = epsilon ** (1 + epsilon_alpha)
            else:
                epsilon_i = epsilon ** (1 + i/(self.num_actors-1)*epsilon_alpha)
            # get simlator fot each actor
            self.simlator   = SeparationControl(self.args, self.simlator_args)
            self.nb_actions = self.simlator.nb_actions
            
            # get build_compile_model for actor 
            model_args = self.get_model_args(1)
            args = (
                self.simlator,
                i,
                epsilon_i,
                model_args,
                self.args,
                experience_q,
                model_sync_q[i],
                self.actors_end_q[i]
                )
            self.actors_ps.append(mp.Process(target=self.actor_run_cpu, args=args))
            print("Actor{} Epsilon:{}".format(i,epsilon_i))

        # get build_compile_model for learner
        model_args = self.get_model_args(self.args["batch_size"]) 
        args = (
            model_args,
            self.args,
            experience_q,
            model_sync_q,
            self.learner_end_q
            )
        # create learner ps
        self.learner_ps = mp.Process(target=self.learner_run_gpu, args=args)  
        self.model_tester_args = self.get_model_args(1) 
    
    def __del__(self):
        self.learner_ps.terminate()
        for p in self.actors_ps:
            p.terminate()

    def train(self):
        # move process
        try:
            self.learner_ps.start()
            for p in self.actors_ps:
                p.start()

            # wait for finish
            while True:
                time.sleep(1)  # polling time

                # 終了判定
                f = True
                for q in self.actors_end_q:
                    if q.empty():
                        f = False
                        break
                if f:
                    break
        except KeyboardInterrupt:
            print("interrupt")
        except Exception:
            print(traceback.format_exc())
        
        # throw end to learner 
        self.learner_end_q[0].put(1)
        time.sleep(1)
        # kill
        self.learner_ps.terminate()
        for p in self.actors_ps:
            p.terminate()

        return

    def test(self):
        # test Actor
        test_actor = Actor(
            self.simlator,
            -1,
            0,
            self.model_tester_args,
            self.args,
            None,
            None,
            training = False,
        )
        test_actor.model.load_weights(self.args["log_dir"]+self.args["load_weights_path"])
        test_actor.test()




    #---------------------------------------------------
    # learner
    #---------------------------------------------------
    def learner_run_gpu(
        self,
        model_args, 
        args, 
        experience_q,
        model_sync_q,
        learner_end_q,
        ):
        with tf.device("/device:GPU:0"):
            self.learner_run(
            model_args, 
            args, 
            experience_q,
            model_sync_q,
            learner_end_q
            )

    def learner_run(
        self,
        model_args, 
        args, 
        experience_q,
        model_sync_q,
        learner_end_q
        ):
        learner = Learner(
            model_args=model_args, 
            args=args, 
            experience_q=experience_q,
            model_sync_q=model_sync_q,
            learner_end_q=learner_end_q,
        )
        try:
            # model load
            if os.path.isfile(self.args["log_dir"]+self.args["load_weights_path"]):
                learner.model.load_weights(self.args["log_dir"]+self.args["load_weights_path"])
                learner.target_model.load_weights(self.args["log_dir"]+self.args["load_weights_path"])

            # learning
            print("Learner Starts")
            learner.train()
                
        except KeyboardInterrupt:
            pass
        except Exception:
            print(traceback.format_exc())
        finally:
            print("Learning End. Train Count:{}".format(learner.train_num))

            
            if self.args["test"]:
                # model save
                if args["save_weights_path"] != "":
                    print("save:" + args["save_weights_path"])
                    learner.model.save_weights(args["log_dir"]+args["save_weights_path"], args["save_overwrite"])
                '''# throw end to learner
                print("Last Learner weights sending...")
                learner_end_q[1].put(learner.model.get_weights())'''

    def actor_run_cpu(
        self,
        simlator,
        actor_index,
        epsilon,
        model_args,
        args,
        experience_q, 
        model_sync_q, 
        actors_end_q,
        ):
        with tf.device("/device:CPU:0"):
            self.actor_run(
                simlator,
                actor_index, 
                epsilon,
                model_args,
                args,
                experience_q, 
                model_sync_q, 
                actors_end_q)

    def actor_run(
        self,
        simlator,
        actor_index,
        epsilon,
        model_args,
        args,
        experience_q, 
        model_sync_q, 
        actors_end_q,
        ):
        print("Actor{} Starts!".format(actor_index))
        try:
            actor = Actor(
                simlator,
                actor_index,
                epsilon,
                model_args,
                args,
                experience_q,
                model_sync_q,
                training = not (args["test"])
            )

            # model load
            print(self.args["log_dir"]+self.args["load_weights_path"])
            if os.path.isfile(self.args["log_dir"]+self.args["load_weights_path"]):
                actor.model.load_weights(self.args["log_dir"]+self.args["load_weights_path"])

            # start
            actor.run_actor()
        except KeyboardInterrupt:
            pass
        except Exception:
            print(traceback.format_exc())
        finally:
            print("Actor{} End!".format(actor_index))
            actors_end_q.put(1)



    def get_model_args(self, batch_size):
        model_args  = {
            "batch_size": batch_size,
            "input_sequence": self.args["input_sequence"],
            "input_shape": self.args["input_shape"],
            "enable_image_layer": self.args["enable_image_layer"],
            "nb_actions": self.nb_actions,
            "enable_dueling_network": self.args["enable_dueling_network"],
            "dueling_network_type": self.args["dueling_network_type"],
            "enable_noisynet": self.args["enable_noisynet"],
            "dense_units_num": self.args["dense_units_num"],
            "lstm_units_num": self.args["lstm_units_num"],
            "metrics": self.args["metrics"],
        }
        return model_args