import json
from collections import defaultdict

class VietnameseMapping:
    def __init__(self):
        self.onset = ['', 'ngh', 'tr', 'th', 'ph', 'nh', 'ng', 'kh', 'gi', 'gh', 'ch', 'đ', 'x', 'v', 't', 's', 'r', 'q', 'n', 'm', 'l', 'k', 'h', 'g', 'd', 'c', 'b']
        self.medial = ['', 'u', 'o']
        self.nucleus = ['e', 'ê', 'u', 'ư', 'ô', 'i', 'y', 'o', 'oo', 'ơ', 'â', 'a', 'a', 'o', 'ă', 'ươ', 'ưa', 'uô', 'ua', 'iê', 'yê', 'ia', 'ya']
        self.coda = ['', 'u', 'n', 'o', 'p', 'c', 'm', 'y', 'i', 't', 'ng', 'nh', 'ch']

        self.graph_onset_medial = defaultdict(list)
        self.graph_medial_nucleus = defaultdict(list)
        self.graph_nucleus_coda = defaultdict(list)

        self.load_graphs()

    def load_graphs(self):
        try:
            self.graph_onset_medial = self.load_graph_from_json('graphs/graph_onset_medial.json')
            self.graph_medial_nucleus = self.load_graph_from_json('graphs/graph_medial_nucleus.json')
            self.graph_nucleus_coda = self.load_graph_from_json('graphs/graph_nucleus_coda.json')
        except FileNotFoundError:
            print('Graph files not found. Creating new graphs.')
            

    def save_graphs(self):
        self.save_graph_to_json(self.graph_onset_medial, 'graphs/graph_onset_medial.json')
        self.save_graph_to_json(self.graph_medial_nucleus, 'graphs/graph_medial_nucleus.json')
        self.save_graph_to_json(self.graph_nucleus_coda, 'graphs/graph_nucleus_coda.json')

    def save_graph_to_json(self, graph, filename):
        graph_dict = {key: list(value) for key, value in graph.items()}
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(graph_dict, f, ensure_ascii=False, indent=4)

    def load_graph_from_json(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            graph = json.load(f)
        return defaultdict(list, graph)

    def update(self, word):
        if self._update_onset(word):
            print(f"'{word}' is a valid Vietnamese word. Graphs updated.")
        else:
            print(f"'{word}' is not a valid Vietnamese word.")

        self.save_graphs()

    def _update_onset(self, word):
        for o in self.onset:
            if word.startswith(o):
                remaining_word = word[len(o):]
                if self._update_medial(o, remaining_word):
                    return True
        return False

    def _update_medial(self, onset, word):
        for m in self.medial:
            if word.startswith(m):
                remaining_word = word[len(m):]
                if self._update_nucleus(onset, m, remaining_word):
                    return True
        return False

    def _update_nucleus(self, onset, medial, word):
        for n in self.nucleus:
            if word.startswith(n):
                remaining_word = word[len(n):]
                if self._update_coda(onset, medial, n, remaining_word):
                    return True
        return False

    def _update_coda(self, onset, medial, nucleus, word):
        for c in self.coda:
            if word == c:
                if medial not in self.graph_onset_medial[onset]:
                    self.graph_onset_medial[onset].append(medial)
                if nucleus not in self.graph_medial_nucleus[medial]:
                    self.graph_medial_nucleus[medial].append(nucleus)
                if c not in self.graph_nucleus_coda[nucleus]:
                    self.graph_nucleus_coda[nucleus].append(c)
                return True
        return False

if __name__ == '__main__':
    vm = VietnameseMapping()
    with open('vietnamese_word.txt', 'r', encoding='utf-8') as f:
        words = f.readlines()
        for word in words:
            vm.update(word.strip())