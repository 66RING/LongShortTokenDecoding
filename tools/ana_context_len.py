from collections import Counter
import matplotlib.pyplot as plt

class Analyzer:
    def __init__(self):
        self.len_counter = Counter()

    def visit(self, len):
        self.len_counter[len] += 1

    def draw(self, save_path = None):
        print("\nToken length distribution:")
        for length, count in sorted(self.len_counter.items()):
            print(f"Length {length}: {count} samples")

        plt.figure(figsize=(10, 6))
        plt.bar(self.len_counter.keys(), self.len_counter.values(), color='skyblue')
        plt.title('Distribution of Token Lengths in Dataset')
        plt.xlabel('Token Length')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)

        plt.show()


