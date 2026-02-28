from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import time
import numpy as np


class Viterbi:

    def __init__(self, tagged_sents, tagged_words):
        self.tagged_sents = tagged_sents
        self.tagged_words = tagged_words
        self.start_prob = {}
        self.trans_prob = defaultdict(dict)
        self.tags_cnt = defaultdict(int)
        self.emit_prob = defaultdict(dict)
        self.total_words = defaultdict(int)
        self.tag_total_cnt = defaultdict(int)
        self.G = nx.DiGraph()
        self.pos = {} # node positions
        self.V = [{}]
        self.path = {}

        self.G.add_node("Start", layer=-1)
        self.pos["Start"] = (-1, 0) # (x=time, y=vertical axis)

        plt.ion()  # enable interactive mode
        self.fig = plt.figure(figsize=(12, 10))

    def init(self):
        self.clean_tags()
        self.unk_handling()
        self.start_prob_calc()
        self.trans_prob_calc()
        self.emis_prob_calc()

    # remove -NONE- tag
    def clean_tags(self):
        clean_tagged_sents = []

        for sent in self.tagged_sents:
            clean_sent = [(word, tag) for word, tag in sent if tag != '-NONE-']
            clean_tagged_sents.append(clean_sent)

        self.tagged_sents = clean_tagged_sents
    
    # replace one-time existing word with <UNK> token
    def unk_handling(self):
        word_freq = defaultdict(int)
        for sent in self.tagged_sents:
            for word, tag in sent:
                word_freq[word] += 1

        tagged_sents_with_unk = []

        for sent in self.tagged_sents:
            new_sent = []
            for word, tag in sent:
                if word_freq[word] <= 1:
                    new_word = "<UNK>"
                else:
                    new_word = word

                new_sent.append((new_word, tag))

            tagged_sents_with_unk.append(new_sent)
        
        self.tagged_sents = tagged_sents_with_unk
    
    # calculate start tag prob (stored as log)
    def start_prob_calc(self):
        init_tags = {}

        for s in self.tagged_sents:
            tag = s[0][1]
            init_tags[tag] = init_tags.get(tag, 0) + 1

        total = sum(init_tags.values())

        for k, v in init_tags.items():
            self.start_prob[k] = np.log(v / total)
    
    # calculate transition prob (stored as log)
    def trans_prob_calc(self):
        trans_cnt = defaultdict(lambda: defaultdict(int))
        
        for sent in self.tagged_sents:
            tags = [tag for word, tag in sent]

            tags = ['<s>'] + tags + ['</s>']

            for i in range(len(tags) - 1):
                current_t = tags[i]
                next_t = tags[i+1]

                trans_cnt[current_t][next_t] += 1

                self.tags_cnt[current_t] += 1

        for prev_tag, next_tags_dict in trans_cnt.items():
            total_count = self.tags_cnt[prev_tag]

            for next_tag, count in next_tags_dict.items():
                # laplace smoothing, stored as log
                self.trans_prob[prev_tag][next_tag] = np.log((count + 1) / (total_count + len(self.tags_cnt)))

    # calculate emission prob (stored as log)
    def emis_prob_calc(self):
        emit_cnt = defaultdict(lambda: defaultdict(int))

        for sent in self.tagged_sents:
            for word, tag in sent:
                emit_cnt[tag][word] += 1
                self.tag_total_cnt[tag] += 1
                self.total_words[word] += 1

        vocab_size = len(self.total_words)
        all_tags = list(emit_cnt.keys())

        for tag, words_dict in emit_cnt.items():
            total_tag_count = self.tag_total_cnt[tag]

            for word, count in words_dict.items():
                if word == "<UNK>":
                    # uniform UNK emission across all tags to prevent NNP bias
                    self.emit_prob[tag][word] = np.log(1.0 / len(all_tags))
                else:
                    # laplace smoothing, stored as log
                    self.emit_prob[tag][word] = np.log((count + 1) / (total_tag_count + vocab_size))

    # find tags of the sentence
    def find_tags(self, sentence: list[str], detailed: bool = False):
        NEG_INF = float('-inf')
        all_tags = [tag for tag in self.tags_cnt.keys() if tag not in ('<s>', '</s>')]
        best_nodes = set()  # track all winning nodes across time steps

        for t in range(len(sentence)):
            if t == 0:
                # getting start tag
                best_score_t0 = NEG_INF
                best_node_t0 = None
                surviving_nodes_t0 = []

                for i, curr_tag in enumerate(all_tags):
                    # probabilities are already in log space
                    start_log = self.start_prob.get(curr_tag, None)
                    emit_log = self.emit_prob.get(curr_tag, {}).get(sentence[0], None)

                    if start_log is None or emit_log is None:
                        continue

                    score = start_log + emit_log

                    self.V[0][curr_tag] = score
                    self.path[curr_tag] = [curr_tag]

                    node_name = f"{t}_{curr_tag}"
                    self.G.add_node(node_name)
                    self.G.add_edge("Start", node_name)
                    surviving_nodes_t0.append(node_name)

                    if score > best_score_t0:
                        best_score_t0 = score
                        best_node_t0 = node_name

                # position surviving nodes evenly (same as t>0)
                y_spacing = 2.0
                for idx, node in enumerate(surviving_nodes_t0):
                    y = (idx - len(surviving_nodes_t0) / 2) * y_spacing
                    self.pos[node] = (t, y)

                # detailed: highlight each node as it's computed
                if detailed:
                    for node in surviving_nodes_t0:
                        tag_name = node.split('_', 1)[1]
                        self.draw_graph(
                            f"Step {t}: Computing tag '{tag_name}' for '{sentence[t]}'",
                            all_tags, len(sentence),
                            highlight_node=node,
                            best_nodes=best_nodes
                        )
                        plt.pause(0.5)

                # detailed: highlight the best node in green
                if detailed and best_node_t0:
                    best_nodes.add(best_node_t0)
                    self.draw_graph(
                        f"Step {t}: Best tag for '{sentence[t]}' -> {best_node_t0.split('_', 1)[1]}",
                        all_tags, len(sentence),
                        best_nodes=best_nodes
                    )
                    plt.pause(1)

            else:
                # finding the rest
                self.V.append({}) # add another dict to collect probs of tag of current word
                new_path = {}
                surviving_nodes = []  # collect nodes that survive

                word = sentence[t]
                if word not in self.total_words:
                    word = "<UNK>"

                best_score_round = NEG_INF
                best_node_round = None

                # find probs of tags of current tag
                for i, curr_tag in enumerate(all_tags):
                    (best_prob, best_prev_tag) = (NEG_INF, None)

                    # optimization part
                    # only looks at most likely tag from previous word
                    for prev_tag in self.V[t-1]:

                        prev_score = self.V[t-1][prev_tag]

                        # probabilities are already in log space
                        trans_log = self.trans_prob.get(prev_tag, {}).get(curr_tag, None)
                        emit_log = self.emit_prob.get(curr_tag, {}).get(word, None)

                        if trans_log is None or emit_log is None:
                            continue

                        current_score = prev_score + trans_log + emit_log

                        if current_score > best_prob:
                            best_prob = current_score
                            best_prev_tag = prev_tag

                    if best_prob > NEG_INF:
                        self.V[t][curr_tag] = best_prob
                        new_path[curr_tag] = self.path[best_prev_tag] + [curr_tag]

                        prev_node = f"{t-1}_{best_prev_tag}"
                        curr_node = f"{t}_{curr_tag}"

                        self.G.add_node(curr_node)
                        self.G.add_edge(prev_node, curr_node)
                        surviving_nodes.append(curr_node)

                        if best_prob > best_score_round:
                            best_score_round = best_prob
                            best_node_round = curr_node

                # position surviving nodes evenly
                y_spacing = 2.0
                for idx, node in enumerate(surviving_nodes):
                    y = (idx - len(surviving_nodes) / 2) * y_spacing
                    self.pos[node] = (t, y)

                # detailed: highlight each surviving node
                if detailed:
                    for node in surviving_nodes:
                        tag_name = node.split('_', 1)[1]
                        self.draw_graph(
                            f"Step {t}: Computing tag '{tag_name}' for '{sentence[t]}'",
                            all_tags, len(sentence),
                            highlight_node=node,
                            best_nodes=best_nodes
                        )
                        plt.pause(0.5)

                    # highlight the best node in green
                    if best_node_round:
                        best_nodes.add(best_node_round)
                        self.draw_graph(
                            f"Step {t}: Best tag for '{sentence[t]}' -> {best_node_round.split('_', 1)[1]}",
                            all_tags, len(sentence),
                            best_nodes=best_nodes
                        )
                        plt.pause(1)

                self.path = new_path

            # non-detailed: draw once per time step
            if not detailed:
                self.draw_graph(
                    f"Step {t}: Processing word '{sentence[t]}'",
                    all_tags, len(sentence)
                )
                plt.pause(2)

        plt.ioff()  # disable interactive mode
        print("Animation Finished!")
        plt.show()  # keep the final result window open

        # return best tag sequence
        if self.V[-1]:
            best_last_tag = max(self.V[-1], key=self.V[-1].get)
            return self.path[best_last_tag], self.V[-1][best_last_tag]
        return [], 0.0

    # Get color list for all nodes in the graph.
    def get_node_colors(self, highlight_node=None, best_nodes=None):
        if best_nodes is None:
            best_nodes = set()
        colors = []
        for node in self.G.nodes():
            if node == highlight_node:
                colors.append('orange')
            elif node in best_nodes:
                colors.append('limegreen')
            elif node == 'Start':
                colors.append('lightgray')
            else:
                colors.append('lightblue')
        return colors
    
    # Draw the trellis graph with optional node highlighting."""
    def draw_graph(self, title, all_tags, sentence_len, highlight_node=None, best_nodes=None):
        self.fig.clf()

        node_colors = self.get_node_colors(highlight_node, best_nodes)

        nx.draw(self.G, self.pos,
                with_labels=True,
                node_color=node_colors,
                edge_color='gray',
                node_size=800,
                arrows=True)

        plt.title(title)
        plt.xlim(-2, sentence_len)
        y_limit = len(all_tags) + 2
        plt.ylim(-y_limit, y_limit)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

