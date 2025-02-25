# For running in the console, mainly copy-passte from .ipynb
import re
from collections import defaultdict

import rich  # for beatifull output formatting
from tqdm import tqdm
import math
import argparse


def read_n_bigrams(path: str) -> tuple[dict[tuple[str] : int], set]:
    data = {}
    vocab = set()
    with open(path, "r", encoding="latin1") as file:
        for line in tqdm(file):
            line_ = line.lower().strip().split("\t")

            # padding simulation
            ln = len(line_) - 2
            tpl = tuple(["<s>"] * ln + line_[1:] + ["<s>"] * ln)
            for idx in range(len(tpl) - ln):
                data[tpl[idx : len(line_) - 1 + idx]] = int(line_[0])

            for elem in line_[1:]:
                vocab.add(elem)
    return data, vocab


class NGramModel:
    def __init__(
        self, ngram_count: dict[tuple[str], int], vocab: set, ngram_size: int = 2
    ):
        self.ngram_count = ngram_count
        self.vocab = vocab
        self.ngram_size = ngram_size

        self.prev_grams = defaultdict(int)
        for ngram, count in ngram_count.items():
            self.prev_grams[ngram[:-1]] += count

        self.probs = {}
        for ngram, count in ngram_count.items():
            self.probs[ngram] = (count + 1) / self.prev_grams[ngram[:-1]]

    def evaluate_ngram(self, sequence, verbose: bool = False):
        probability = 0
        for i in range(len(sequence) - self.ngram_size + 1):
            ngram = tuple(sequence[i : i + self.ngram_size])
            probability += math.log(self.probs.get(ngram, 1e-10))

            if verbose:
                rich.print(ngram, math.log(self.probs.get(ngram, 1e-10)), probability)

        return probability


class SpellCorrector:
    def __init__(
        self,
        vocab: set,
        ngrams: list[dict[tuple, int]],
        max_dist: int = 20,
        le: float = 0.5,
        lm_1: float = 0.5,
        lm_2: float = 0.5,
    ):
        self.vocab = vocab
        self.lm = None
        self.max_dist = max_dist

        self.ngrams = ngrams
        self.ngram_models = [
            NGramModel(ngram_dict, self.vocab, ngram_size=ngram_size)
            for ngram_dict, ngram_size in zip(self.ngrams, [2, 5])
        ]

        self.le = le
        self.lm_1 = lm_1
        self.lm_2 = lm_2

    def dameru_levenshtein_distance(self, a: str, b: str) -> tuple[int, float]:
        # source: https://en.wikipedia.org/wiki/Damerau–Levenshtein_distance
        m, n = len(a), len(b)
        max_dist = m + n

        alphabet = set(a) | set(b)
        da = [0 for _ in range(len(alphabet) + 1)]
        char2idx = {char: 0 for char in alphabet}

        d = [[0 for i in range(n + 2)] for j in range(m + 2)]
        d[0][0] = max_dist

        for i in range(0, m + 1):
            d[i + 1][0] = max_dist
            d[i + 1][1] = i

        for j in range(0, n + 1):
            d[0][j + 1] = max_dist
            d[1][j + 1] = j

        a, b = " " + a, " " + b
        for i in range(1, m + 1):
            db = 0
            min_cost = float("inf")
            for j in range(1, n + 1):
                k = da[char2idx[b[j]]]
                l = db
                if a[i] == b[j]:
                    cost = 0
                    db = j
                else:
                    cost = 2
                substitution = d[i][j] + cost
                insertion = d[i + 1][j] + 1
                deletion = d[i][j + 1] + 1
                d[i + 1][j + 1] = min(substitution, insertion, deletion)

                # trying to not doing extra calculations
                if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                    d[i + 1][j + 1] = min(d[i + 1][j + 1], d[k][l] + cost)

                min_cost = min(d[i + 1][j + 1], min_cost)

            if min_cost >= self.max_dist:
                return self.max_dist + 1

            da[char2idx[a[i]]] = i

        return d[m + 1][n + 1]

    def __generate_candidates(self, pretendant: str, top_k: int = 20):
        candidates = []
        n = len(pretendant)
        pret_set = set(pretendant)
        for word in self.vocab:
            if abs(len(word) - n) > self.max_dist:
                continue
            if len(pret_set.intersection(set(word))) < (len(pret_set) / 2):
                continue

            dist = self.dameru_levenshtein_distance(pretendant, word)
            if dist <= self.max_dist:
                candidates.append((word, dist))

        return sorted(candidates, key=lambda x: x[1])[:top_k]

    def __evaluate_candidates(
        self,
        sentence: list[str],
        candidates: tuple[list[str], int],
        candidate_idx: int,
        verbose: bool,
    ) -> str:
        best = None
        best_score = -float("inf")

        for candidate, dist in candidates:
            scores = {}
            for model in self.ngram_models:
                start_idx, end_idx = (
                    max(candidate_idx - model.ngram_size + 1, 0),
                    min(candidate_idx + model.ngram_size, len(sentence)),
                )

                sent_copy = sentence.copy()
                sent_copy[candidate_idx] = candidate

                left_slice = sent_copy[start_idx : candidate_idx + 1]
                right_slice = sent_copy[candidate_idx:end_idx]
                left = len(sent_copy[start_idx : candidate_idx + 1])

                if left != model.ngram_size:
                    left_slice = ["<s>"] * (model.ngram_size - left) + sent_copy[
                        start_idx : candidate_idx + 1
                    ]

                sent = left_slice + right_slice[1:]

                scores[model.ngram_size] = model.evaluate_ngram(
                    sequence=sent, verbose=verbose
                )

            combined_score = (
                -self.le * dist + self.lm_1 * scores[2] + self.lm_2 * scores[5]
            )
            if verbose:
                rich.print(
                    f"{candidate}\t{combined_score=}\t", dist, scores[2], scores[5]
                )

            if combined_score > best_score:
                best_score = combined_score
                best = candidate

        return best

    def spell_checker(
        self, text: str, top_k: int = 10_000, verbose: list[bool] = [False, False]
    ) -> str:
        words = re.findall(r"\w+", text.lower())
        corrected_sentence = []

        for idx, word in enumerate(words):
            corrected_word = word
            if word not in self.vocab:
                candidates = self.__generate_candidates(pretendant=word, top_k=top_k)

                best_one = self.__evaluate_candidates(
                    sentence=words,
                    candidates=candidates,
                    candidate_idx=idx,
                    verbose=verbose[0],
                )
                corrected_word = best_one
            elif verbose[1]:
                candidates = self.__generate_candidates(pretendant=word, top_k=top_k)
                ecand = self.__evaluate_candidates(
                    sentence=words,
                    candidates=candidates,
                    candidate_idx=idx,
                    verbose=verbose[1],
                )
                rich.print("non-changed:\n", ecand, "\n\n")
            words[idx] = corrected_word
            corrected_sentence.append(corrected_word)
        return " ".join(corrected_sentence)


def parse_args():
    parser = argparse.ArgumentParser(description="Spell corrector for input sentences.")
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="Path to the input file.",
        default="./data/sentences.txt",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Path to the output file.",
        default="./data/corrected_sentences.txt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    rich.print("Building vocab...")
    bigrams, vocab_bigrams = read_n_bigrams("./data/bigrams.txt")
    fivegrams, vocab_fivegrams = read_n_bigrams("./data/fivegrams.txt")

    vocab = vocab_bigrams & vocab_fivegrams
    spell_corrector = SpellCorrector(vocab=vocab, ngrams=[bigrams, fivegrams])

    sentences = []
    with open(args.input, "r") as file:
        for sent in file:
            sent = sent.lower().strip()
            sentences.append(sent)

    rich.print("Correcting senteces...")
    corrected_sentences = [
        spell_corrector.spell_checker(sent, top_k=100) for sent in sentences
    ]

    rich.print(f"Corrected sentences avaliable at {args.output}")
    rich.print(f"{corrected_sentences=}")

    with open(args.output, "w") as file:
        for sent in corrected_sentences:
            file.write(sent + "\n")
