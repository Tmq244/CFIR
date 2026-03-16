#!/usr/bin/env python3

# python plot_results.py ../results/20260312_222833_combine_all_resnet152_encode_s3_b16_e150_lr0.0001_gpu1_combine_all_r152_sa3_e150.json
# --output results/my_plot.png
# --title "My Run Metrics"

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _numeric_keys(records, excluded_keys=None):
    excluded_keys = excluded_keys or set()
    keys = []
    for record in records:
        if not isinstance(record, dict):
            continue
        for key, value in record.items():
            if key in excluded_keys:
                continue
            if isinstance(value, (int, float)) and key not in keys:
                keys.append(key)
    return sorted(keys)


def _series(records, key):
    values = []
    for record in records:
        value = None
        if isinstance(record, dict):
            value = record.get(key)
        values.append(value)
    return values


def _plot_panel(ax, x, records, keys, title):
    if not keys:
        ax.text(0.5, 0.5, 'No numeric metrics', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return

    for key in keys:
        y = _series(records, key)
        ax.plot(x, y, marker='o', markersize=2, linewidth=1.2, label=key)

    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)


def _is_retrieval_metric(key, value):
    if not isinstance(value, (int, float)):
        return False
    normalized = str(key).strip().lower()
    if normalized in {'mrr', 'mean_reciprocal_rank'}:
        return True
    if normalized.startswith('r@') or normalized.startswith('recall@'):
        return True
    # Common format in this repo: R1, R5, R10, ...
    if re.fullmatch(r'r\d+', normalized):
        return True
    # Composite retrieval metric, e.g. R10R50
    return re.fullmatch(r'r\d+r\d+', normalized) is not None


def _retrieval_metric_sort_key(metric_name):
    normalized = str(metric_name).strip().lower()
    if normalized in {'mrr', 'mean_reciprocal_rank'}:
        return (1, 10 ** 9)
    if re.fullmatch(r'r\d+', normalized):
        return (0, int(normalized[1:]))
    if re.fullmatch(r'r\d+r\d+', normalized):
        parts = re.findall(r'\d+', normalized)
        left = int(parts[0]) if len(parts) > 0 else 10 ** 8
        right = int(parts[1]) if len(parts) > 1 else 10 ** 8
        return (0, left * 10000 + right)
    if '@' in normalized:
        right = normalized.split('@', 1)[1]
        try:
            return (0, int(float(right)))
        except ValueError:
            return (0, 10 ** 8)
    return (2, 10 ** 9)


def extract_last_epoch_retrieval_table(data):
    epochs_data = data.get('epochs', [])
    if not epochs_data:
        raise ValueError('Input JSON has no "epochs" data.')

    last_epoch_data = epochs_data[-1]
    last_epoch = last_epoch_data.get('epoch', len(epochs_data))
    eval_data = last_epoch_data.get('eval', {})

    scoped_records = [('overall', eval_data.get('overall', {}))]
    targets = eval_data.get('targets', {})
    for target_name in sorted(targets.keys()):
        scoped_records.append((target_name, targets.get(target_name, {})))

    metric_names = set()
    for _, record in scoped_records:
        if not isinstance(record, dict):
            continue
        for key, value in record.items():
            if _is_retrieval_metric(key, value):
                metric_names.add(key)

    metric_names = sorted(metric_names, key=_retrieval_metric_sort_key)
    if not metric_names:
        raise ValueError('No retrieval metrics (R@n/MRR) found in the last epoch eval results.')

    rows = []
    for scope_name, record in scoped_records:
        row = {'epoch': last_epoch, 'scope': scope_name}
        for metric_name in metric_names:
            value = None
            if isinstance(record, dict):
                value = record.get(metric_name)
            row[metric_name] = value
        rows.append(row)

    return rows, metric_names


def _extract_selected_args(data):
    selected_keys = [
        'backbone',
        'batch_size',
        'epochs',
        'fdims',
        'lr',
        'lr_decay_factor',
        'lr_decay_steps',
        'lrp',
        'method',
        'stack_num',
        'text_method',
    ]
    args = data.get('args', {})
    rows = []
    for key in selected_keys:
        value = args.get(key) if isinstance(args, dict) else None
        if isinstance(value, list):
            value = json.dumps(value)
        rows.append((key, value))
    return rows


def write_retrieval_table_csv(output_path, args_rows, rows, metric_names):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='') as handle:
        args_writer = csv.writer(handle)
        args_writer.writerow(['param', 'value'])
        for key, value in args_rows:
            args_writer.writerow([key, value])
        args_writer.writerow([])

        fieldnames = ['epoch', 'scope'] + metric_names
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_retrieval_table_markdown(output_path, args_rows, rows, metric_names):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as handle:
        handle.write('# Last Epoch Retrieval Metrics\n\n')
        handle.write('## Args\n\n')
        handle.write('| param | value |\n')
        handle.write('| --- | --- |\n')
        for key, value in args_rows:
            display_value = value
            if display_value is None:
                display_value = ''
            handle.write('| {} | {} |\n'.format(key, str(display_value)))

        handle.write('\n## Metrics\n\n')
        metric_headers = ['epoch', 'scope'] + metric_names
        handle.write('| {} |\n'.format(' | '.join(metric_headers)))
        handle.write('| {} |\n'.format(' | '.join(['---'] * len(metric_headers))))
        for row in rows:
            values = [row.get('epoch'), row.get('scope')] + [row.get(name) for name in metric_names]
            normalized = ['' if item is None else str(item) for item in values]
            handle.write('| {} |\n'.format(' | '.join(normalized)))


def write_retrieval_table(output_path, args_rows, rows, metric_names):
    suffix = output_path.suffix.lower()
    if suffix == '.csv':
        write_retrieval_table_csv(output_path, args_rows, rows, metric_names)
        return
    write_retrieval_table_markdown(output_path, args_rows, rows, metric_names)


def build_figure(data, title=None):
    epochs_data = data.get('epochs', [])
    if not epochs_data:
        raise ValueError('Input JSON has no "epochs" data.')

    epochs = [item.get('epoch', idx + 1) for idx, item in enumerate(epochs_data)]
    train_records = [item.get('train', {}) for item in epochs_data]
    eval_overall_records = [item.get('eval', {}).get('overall', {}) for item in epochs_data]

    first_targets = epochs_data[0].get('eval', {}).get('targets', {})
    target_names = sorted(first_targets.keys())

    # Panels: train metrics (without lr), train lr, eval overall, per-target evals.
    nrows = 3 + len(target_names)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(14, max(8, 3.2 * nrows)), sharex=True)
    if nrows == 1:
        axes = [axes]

    train_keys = _numeric_keys(train_records, excluded_keys={'num_batches', 'lr'})
    _plot_panel(axes[0], epochs, train_records, train_keys, 'Train Loss')

    _plot_panel(axes[1], epochs, train_records, ['lr'], 'Learning Rate')

    overall_keys = _numeric_keys(eval_overall_records)
    _plot_panel(axes[2], epochs, eval_overall_records, overall_keys, 'Overall Metrics')

    for idx, target in enumerate(target_names, start=3):
        target_records = [item.get('eval', {}).get('targets', {}).get(target, {}) for item in epochs_data]
        target_keys = _numeric_keys(target_records)
        _plot_panel(axes[idx], epochs, target_records, target_keys, 'Target Metrics: {}'.format(target))

    run_title = title or data.get('args', {}).get('expr_name', 'Training Metrics')
    fig.suptitle(run_title, fontsize=14)
    fig.tight_layout(rect=[0, 0.01, 1, 0.98])
    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from a results JSON file.')
    parser.add_argument('input_json', help='Path to results JSON file.')
    parser.add_argument('--output', '-o', default=None, help='Output image path (default: <input_stem>.png).')
    parser.add_argument(
        '--table-output',
        default=None,
        help='Output table path for last-epoch retrieval metrics. Use .md (default) or .csv.',
    )
    parser.add_argument('--title', default=None, help='Custom title for the figure.')
    args = parser.parse_args()

    input_path = Path(args.input_json)
    if not input_path.exists():
        raise FileNotFoundError('Input file not found: {}'.format(input_path))

    output_path = Path(args.output) if args.output else input_path.with_suffix('.png')
    table_output_path = (
        Path(args.table_output)
        if args.table_output
        else input_path.with_name('{}_last_epoch_metrics.md'.format(input_path.stem))
    )

    with input_path.open('r') as handle:
        data = json.load(handle)

    fig = build_figure(data, title=args.title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)

    rows, metric_names = extract_last_epoch_retrieval_table(data)
    args_rows = _extract_selected_args(data)
    write_retrieval_table(table_output_path, args_rows, rows, metric_names)

    print('Saved plot to {}'.format(output_path))
    print('Saved last-epoch retrieval table to {}'.format(table_output_path))


if __name__ == '__main__':
    main()
