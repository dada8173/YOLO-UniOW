import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ID_RE = re.compile(r'^(train|test)(\d+)$')


def parse_locount_txt(txt_path: Path):
    counts = Counter()
    for line in txt_path.read_text(encoding='utf-8', errors='ignore').splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 5:
            continue
        class_name = parts[4]
        if class_name:
            counts[class_name] += 1
    return counts


def aggregate_dir(txt_dir: Path):
    total_counts = Counter()
    per_image = {}
    for txt_path in sorted(txt_dir.glob('*.txt')):
        counts = parse_locount_txt(txt_path)
        per_image[txt_path.stem] = counts
        total_counts.update(counts)
    return total_counts, per_image


def map_imageset_id(image_id: str):
    m = ID_RE.match(image_id)
    if not m:
        return None, None
    split, num = m.group(1), m.group(2)
    return split, num


def load_imageset(path: Path):
    ids = []
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = line.strip()
        if line:
            ids.append(line)
    return ids


def counts_for_imageset(ids, per_train, per_test):
    counts = Counter()
    missing = []
    for image_id in ids:
        split, num = map_imageset_id(image_id)
        if split == 'train':
            src = per_train.get(num)
        elif split == 'test':
            src = per_test.get(num)
        else:
            src = None
        if not src:
            missing.append(image_id)
            continue
        counts.update(src)
    return counts, missing


def write_counter(path: Path, title: str, counter: Counter):
    total = sum(counter.values())
    with path.open('w', encoding='utf-8') as f:
        f.write(f"{title}\n")
        f.write(f"total_instances\t{total}\n")
        f.write(f"unique_classes\t{len(counter)}\n")
        f.write("\nclass\tcount\n")
        for cls, cnt in counter.most_common():
            f.write(f"{cls}\t{cnt}\n")


def plot_topk(counter_dict, out_path: Path, k=20, title=None):
    plt.figure(figsize=(12, 6))
    for label, counter in counter_dict.items():
        top = counter.most_common(k)
        classes = [c for c, _ in top]
        counts = [v for _, v in top]
        plt.plot(range(len(classes)), counts, marker='o', label=label)
    plt.xticks(range(len(classes)), classes, rotation=70, ha='right')
    plt.ylabel('count')
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_grouped_bar(counters, out_path: Path, classes):
    labels = list(counters.keys())
    width = 0.18
    x = range(len(classes))
    plt.figure(figsize=(14, 6))
    for idx, label in enumerate(labels):
        counts = [counters[label].get(c, 0) for c in classes]
        offset = (idx - (len(labels) - 1) / 2) * width
        plt.bar([i + offset for i in x], counts, width=width, label=label)
    plt.xticks(list(x), classes, rotation=70, ha='right')
    plt.ylabel('count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Locount class distribution and task stats')
    parser.add_argument('--train-dir', default='Locount/Locount_GtTxtsTrain')
    parser.add_argument('--test-dir', default='Locount/Locount_GtTxtsTest')
    parser.add_argument('--imageset-dir', default='data/OWOD/ImageSets/LocountOWOD')
    parser.add_argument('--out-dir', default='results/locount_stats')
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    imageset_dir = Path(args.imageset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_counts, per_train = aggregate_dir(train_dir)
    test_counts, per_test = aggregate_dir(test_dir)

    write_counter(out_dir / 'train_class_distribution.txt', 'Train (Locount_GtTxtsTrain)', train_counts)
    write_counter(out_dir / 'test_class_distribution.txt', 'Test (Locount_GtTxtsTest)', test_counts)

    task_counts = {}
    task_missing = {}
    for task_id in range(1, 5):
        train_list = imageset_dir / f't{task_id}_train.txt'
        if not train_list.exists():
            continue
        ids = load_imageset(train_list)
        counts, missing = counts_for_imageset(ids, per_train, per_test)
        task_counts[f't{task_id}'] = counts
        task_missing[f't{task_id}'] = missing
        write_counter(out_dir / f't{task_id}_class_distribution.txt', f'Task {task_id} train list', counts)

    # Differences between tasks
    diff_path = out_dir / 'task_class_differences.txt'
    with diff_path.open('w', encoding='utf-8') as f:
        prev_classes = set()
        for task_id in range(1, 5):
            key = f't{task_id}'
            if key not in task_counts:
                continue
            current_classes = {c for c, v in task_counts[key].items() if v > 0}
            added = sorted(current_classes - prev_classes)
            removed = sorted(prev_classes - current_classes)
            f.write(f"{key}\n")
            f.write(f"classes_present\t{len(current_classes)}\n")
            f.write(f"added_from_prev\t{len(added)}\n")
            if added:
                f.write("added_classes\t" + ", ".join(added) + "\n")
            f.write(f"removed_from_prev\t{len(removed)}\n")
            if removed:
                f.write("removed_classes\t" + ", ".join(removed) + "\n")
            f.write("\n")
            prev_classes = current_classes

    # Missing ids report
    missing_path = out_dir / 'missing_imageset_ids.txt'
    with missing_path.open('w', encoding='utf-8') as f:
        for key, missing in task_missing.items():
            f.write(f"{key}\tmissing_ids\t{len(missing)}\n")
            for mid in missing[:200]:
                f.write(f"{mid}\n")
            if len(missing) > 200:
                f.write(f"... {len(missing) - 200} more\n")
            f.write("\n")

    # Visualizations
    plot_topk(
        {'train': train_counts, 'test': test_counts},
        out_dir / 'train_vs_test_top20.png',
        k=20,
        title='Top-20 classes: Train vs Test',
    )

    if task_counts:
        # Top classes overall across tasks
        combined = Counter()
        for c in task_counts.values():
            combined.update(c)
        top_classes = [c for c, _ in combined.most_common(30)]
        plot_grouped_bar(task_counts, out_dir / 'tasks_top30_grouped.png', top_classes)

        # Per-task top20 line plots
        for key, counter in task_counts.items():
            plot_topk({key: counter}, out_dir / f'{key}_top20.png', k=20, title=f'Top-20 classes: {key}')

    # Write a small JSON summary
    summary = {
        'train_total_instances': sum(train_counts.values()),
        'train_unique_classes': len(train_counts),
        'test_total_instances': sum(test_counts.values()),
        'test_unique_classes': len(test_counts),
        'tasks': {k: {'instances': sum(v.values()), 'unique_classes': len(v)} for k, v in task_counts.items()},
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
