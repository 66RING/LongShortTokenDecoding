from collections import Counter
import matplotlib.pyplot as plt
import csv

class Analyzer:
    def __init__(self):
        self.len_counter = Counter()

    def visit(self, len):
        self.len_counter[len] += 1

    def draw(self, save_png=None):
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
        if save_png is not None:
            plt.savefig(save_png)

        plt.show()

    def save_to_csv(self, csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Length', 'Count'])
            for length, count in sorted(self.len_counter.items()):
                writer.writerow([length, count])



