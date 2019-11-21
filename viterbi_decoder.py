import numpy as np


class Viterbi():

    def __init__(self, hmm_obj, smoothing=False, lambda1=0.6, lambda2=0.1, lambda3=0.3):
        self.all_states = ["*"]+list(hmm_obj.all_states)
        self.smoothing = smoothing
        self.hmm_obj = hmm_obj
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def allowable_tags(self, k):
        if k in (0,-1):
            return ["*"]
        return self.all_states[1:]


    def fill_pi_probs_table_bp_table(self,sentence):
        n = len(sentence)
        self.pi_probs_table = [{tag: {tag2: 0 for tag2 in self.all_states} for tag in self.all_states}
                               for x in range(n+1)]
        self.bp_table = [{tag: {tag2: '' for tag2 in self.all_states} for tag in self.all_states}
                         for x in range(n+1)]
        self.pi_probs_table[0]["*"]["*"] = 1
        for k in range(1,n+1):
            for u in self.allowable_tags(k-1):
                for v in self.allowable_tags(k):
                    all_w=[]
                    for w in self.allowable_tags(k-2):
                        q = self.hmm_obj.compute_transition_params(w, u, v,self.smoothing,self.lambda1,self.lambda2,
                                                                   self.lambda3)
                        e = self.hmm_obj.compute_emission_params(sentence[k-1], v) #k-1 because k starts from 1
                        all_w.append(self.pi_probs_table[k - 1][w][u] * q * e)
                    self.pi_probs_table[k][u][v] = max(all_w)
                    bp_tag = self.allowable_tags(k-2)[np.argmax(all_w)]
                    self.bp_table[k][u][v] = bp_tag


    def predict_tags_for_sentence(self,input_sentence):
        self.fill_pi_probs_table_bp_table(input_sentence)
        n = len(input_sentence)
        best_prob=0
        best_u,best_v="",""
        output_tags = ['']*n
        for u in self.allowable_tags(n - 1):
            for v in self.allowable_tags(n):
                q = self.hmm_obj.compute_transition_params(u, v, "STOP", self.smoothing,self.lambda1,self.lambda2,self.lambda3)
                prob = self.pi_probs_table[n][u][v] * q
                if prob>best_prob:
                    best_prob=prob
                    best_u=u
                    best_v=v
        output_tags[-1] = best_v
        output_tags[-2] = best_u
        for k in range(n-3,-1,-1):
            output_tags[k] = self.bp_table[k+2+1][output_tags[k+1]][output_tags[k+2]]
        return output_tags
