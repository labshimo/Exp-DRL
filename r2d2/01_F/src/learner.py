import gc
import time
from network import *
from memory import*

class Learner():
    def __init__(self,
        model_args, 
        args, 
        experience_q,
        model_sync_q,
        learner_end_q
        ):
        self.experience_q = experience_q
        self.model_sync_q = model_sync_q
        self.learner_end_q = learner_end_q
        self.memory_warmup_size = args["remote_memory_warmup_size"]

        self.per_alpha   = args["per_alpha"]
        per_beta_initial = args["per_beta_initial"]
        per_beta_steps   = args["per_beta_steps"]
        per_enable_is    = args["per_enable_is"]
        memory_capacity  = args["remote_memory_capacity"]
        memory_type      = args["remote_memory_type"]
        if memory_type == "replay":
            self.memory = ReplayMemory(memory_capacity)
        elif memory_type == "per_greedy":
            self.memory = PERGreedyMemory(memory_capacity)
        elif memory_type == "per_proportional":
            self.memory = PERProportionalMemory(memory_capacity, per_beta_initial, per_beta_steps, per_enable_is)
        elif memory_type == "per_rankbase":
            self.memory = PERRankBaseMemory(memory_capacity, self.per_alpha, per_beta_initial, per_beta_steps, per_enable_is)
        else:
            raise ValueError('memory_type is ["replay","per_proportional","per_rankbase"]')

        self.gamma                     = args["gamma"]
        self.batch_size                = args["batch_size"]
        self.enable_double_dqn         = args["enable_double_dqn"]
        self.target_model_update       = args["target_model_update"]
        self.multireward_steps         = args["multireward_steps"]
        self.input_sequence            = args["input_sequence"]
        self.enable_rescaling_priority = args["enable_rescaling_priority"]
        self.enable_rescaling_train    = args["enable_rescaling_train"]
        self.rescaling_epsilon         = args["rescaling_epsilon"]
        self.burnin_length             = args["burnin_length"]
        self.priority_exponent         = args["priority_exponent"]
        self.sequence_length           = args["sequence_length"]

        self.gamma_n = self.gamma**self.multireward_steps

        # log
        self.log = args["log_dir"] + args["learner_log"]
        self.print_interval = args["print_interval"]
        assert memory_capacity > self.batch_size, "Memory capacity is small.(Larger than batch size)"
        assert self.memory_warmup_size > self.batch_size, "Warmup steps is few.(Larger than batch size)"

        # local
        self.train_num   = 0
        self.total_loss  = 0
        self.total_time  = 0
        self.total_q_max = 0

        # model create
        self.model        = build_compile_model(**model_args)
        #self.second_model = build_compile_model(**model_args)
        self.target_model = build_compile_model(**model_args)
        self.model.summary()
        # lstm ful では lstmレイヤーを使う
        self.lstm        = self.model.get_layer("lstm")
        #self.second_lstm = self.second_model.get_layer("lstm")
        self.target_lstm = self.target_model.get_layer("lstm")
        # hidden state memory 
        self.model_hidden_buffer  = self.get_hidden_state(self.lstm)
        self.target_hidden_buffer = self.get_hidden_state(self.target_lstm)

    def train(self):
        # do not training until RemoteMemory size > initial memory
        if len(self.memory) <= self.memory_warmup_size:
            print("Learner is Waiting... Replay memory have {} transitions".format(len(self.memory)))
            # if Actor wants to weights send to Queue
            self.add_network()
            # if experience in Queue send to RemoteMemory
            self.add_memory()

            time.sleep(1)
            return self.train()

        while True: 
            # if Actor wants to weights send to Queue
            self.add_network()
            # if experience in Queue send to RemoteMemory
            self.add_memory()         
            # sample
            (indexes, batchs, weights) = self.memory.sample(self.batch_size, self.train_num)
            # update
            self.train_model(indexes, batchs, weights) 
            # 終了判定
            if not self.learner_end_q[0].empty():
                print("learner end")
                break

    def add_memory(self):
        while not self.experience_q.empty():
            exps = self.experience_q.get(timeout=1)
            self.memory.add(exps[0:7], exps[7])

    def add_network(self):
        for q in self.model_sync_q:
            if not q[0].empty():
                # 空にする(念のため)
                while not q[0].empty():
                    q[0].get(timeout=1)
                
                # 送る
                q[1].put(self.model.get_weights())
        

    # ノーマルの学習
    def train_model(self, indexes, batchs, weights):
        # time 
        time_start = time.time()

        state_batch      = []
        action_batch     = []
        reward_batch     = []
        next_state_batch = []
        hidden_0_batch   = []
        hidden_1_batch   = []
        burn_in_state    = []

        for batch in batchs:
            state_batch.append(batch[0])
            action_batch.append(batch[1]) 
            reward_batch.append(batch[2])
            next_state_batch.append(batch[3]) 
            hidden_0_batch.append(batch[4]) 
            hidden_1_batch.append(batch[5]) 
            burn_in_state.append(batch[6])

        state_batch      = np.asarray(state_batch)
        action_batch     = np.asarray(action_batch)
        reward_batch     = np.asarray(reward_batch)
        next_state_batch = np.asarray(next_state_batch)
        hidden_0_batch   = np.asarray(hidden_0_batch)
        hidden_1_batch   = np.asarray(hidden_1_batch)
        burn_in_state    = np.asarray(burn_in_state)
        # burn in 
        hidden_state = self.burn_in(self.lstm, self.model, burn_in_state, hidden_0_batch, hidden_1_batch)
        #self.burn_in(self.target_lstm, self.target_model, next_state_batch, hidden_0_batch, hidden_1_batch)
        #self.second_model.set_weights(self.model.get_weights())
        #self.second_lstm.reset_states(hidden_state)        
        self.target_lstm.reset_states(hidden_state)

        # sequential training
        priorities_batch = []
        self.store_hidden_state()
        self.total_time += time.time() - time_start
        for i in range(self.sequence_length):
            time_start = time.time()     
            outputs, priority_batch = self.get_train_data(state_batch[:,i,:,:,], action_batch[:,i], reward_batch[:,i], next_state_batch[:,i,:,:,], weights)
            priorities_batch.append(priority_batch)
            self.lstm.reset_states(self.hs_model_1)
            loss = self.model.train_on_batch(np.asarray(state_batch[:,i,:,:,]), np.asarray(outputs))
            self.lstm.reset_states(self.hs_model_2)
            self.total_loss += loss
            self.train_num  += 1
            self.total_time += time.time() - time_start

            # target networkの更新
            if self.train_num % self.target_model_update == 0:
                self.target_model.set_weights(self.model.get_weights())

            if self.train_num % self.print_interval == 0:
                self.print_log()
                
            self.add_network()
        
        priorities = self.assemble_priorities(np.array(priorities_batch))
        [self.memory.update(indexes[i], batchs[i], priorities[i]) for i in range(self.batch_size)]
        # 学習
        
    def burn_in(self, lstm, model, state_batch, hidden_0_batch, hidden_1_batch):
        # burn-in
        lstm.reset_states([hidden_0_batch[:,0,:], hidden_1_batch[:,0,:]])
        for i in range(self.burnin_length):
            model.predict(state_batch[:,i,:,:,], self.batch_size)
        # burn-in 後の結果を返却
        return self.get_hidden_state(lstm)

    def get_train_data(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        weights):
        self.hs_model_1   = self.get_hidden_state(self.lstm)
        outputs = self.model.predict(np.asarray(state_batch), self.batch_size)
        self.hs_model_2   = self.get_hidden_state(self.lstm)
        self.total_q_max += np.amax(outputs)
        action_q_values   = self.get_action_q_values(outputs, action_batch)
        if self.enable_double_dqn:
            # TargetNetworkとQNetworkのQ値を出す
            self.lstm.reset_states(self.hs_model_1)
            ns_model_qval_batch  = self.model.predict(np.asarray(next_state_batch), self.batch_size)
            ns_target_qval_batch = self.target_model.predict(np.asarray(next_state_batch), self.batch_size)
            td_errors = self.get_td_error_double_dqn(ns_model_qval_batch,ns_target_qval_batch,reward_batch)
        else:
            # 次の状態のQ値を取得(target_network)
            target_qvals = self.target_model.predict(np.asarray(next_state_batch), self.batch_size)
            td_errors = self.get_td_error_dqn(target_qvals,reward_batch)
        
        for i in range(self.batch_size):
            outputs[i][action_batch[i]] = td_errors[i] * weights[i]

        priorities = [self.get_priority(action_q_values[i],td_errors[i])* weights[i] for i in range(self.batch_size) ]
        return outputs, priorities

    ### hidden state method ###
    def get_hidden_state(self,lstm):
        return [K.get_value(lstm.states[0]), K.get_value(lstm.states[1])]

    def set_hidden_state(self):
        self.lstm.reset_states(self.model_hidden_buffer)
        self.target_lstm.reset_states(self.target_hidden_buffer)

    def store_hidden_state(self):
        self.model_hidden_buffer  = self.get_hidden_state(self.lstm)
        self.target_hidden_buffer = self.get_hidden_state(self.target_lstm)

    ### td error method ###
    def get_td_error_double_dqn(self, model_q_batch, target_q_batch, reward_batch):
        actions       = np.argmax(model_q_batch, axis=1)  # modelからアクションを出す 
        target_q_maxs = [target_q_batch[i][action] for i, action in enumerate(actions)] # Q値はtarget_modelを使って出す
        target_q_maxs = self.get_rescaling_q_max(target_q_maxs)
        # Q値の更新
        td_errors = [ self.get_td_error(reward,target_q_max) for reward,target_q_max in zip(reward_batch,target_q_maxs)]
        td_errors = self.get_rescaling_td_error(td_errors)
        return td_errors

    def get_td_error_dqn(self,target_q_batch,reward_batch):
        target_q_maxs = np.max(target_q_batch, axis=0)
        target_q_maxs = self.get_rescaling(target_q_maxs)
        # Q値の更新
        td_errors = [ self.get_td_error(reward,target_q_max) for reward,target_q_max in zip(reward_batch,target_q_maxs)]
        td_errors = self.self.get_rescaling_td_error(td_errors)        
        return td_errors
        
    def get_action_q_values(self, outputs, action_batch):
        return [outputs[i][action_batch[i]] for i in range(self.batch_size)]

    def get_td_error(self, reward ,target_q_max):
        return reward + self.gamma_n * target_q_max

    ### Priority method ###
    def get_priority(self, q_values, td_error):
        return abs(td_error - q_values) ** self.per_alpha 

    def assemble_priorities(self,priorities_batch):
        return [self.assemble_priority(priorities_batch[:,i]) for i in range(self.batch_size)]

    def assemble_priority(self, priorities):
        return self.priority_exponent * np.max(priorities) + (1-self.priority_exponent) * np.average(priorities)
    
    ### rescaling method ###
    def get_rescaling_q_max(self,target_q_maxs):
        return [self.rescaling_inverse(target_q_max) if self.enable_rescaling_train else target_q_max for target_q_max in target_q_maxs]
    
    def get_rescaling_td_error(self,td_errors):
        return [self.rescaling(td_error) if self.enable_rescaling_train else td_error for td_error in td_errors]

    def rescaling(self, x, epsilon=0.001):
        n = math.sqrt(abs(x)+1) - 1
        return np.sign(x)*n + epsilon*x

    def rescaling_inverse(self, x):
        return np.sign(x)*( (x+np.sign(x) ) ** 2 - 1)

    ### Log method ###
    def print_log(self):
        text_l = 'AVERAGE LOSS: {0:.5F} / AVG_MAX_Q: {1:2.4F} / LEARN PER SECOND: {2:.1F} / NUM LEARN: {3:5d} / MEMORY: {4:d}'\
            .format(self.total_loss/self.print_interval, self.total_q_max/self.print_interval, \
                self.print_interval/self.total_time, self.train_num, len(self.memory))
        print(text_l)
        with open(self.log,'a') as f:
            f.write(text_l+"\n")
        self.total_loss  = 0
        self.total_time  = 0
        self.total_q_max = 0


    
        