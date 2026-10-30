#!/usr/bin/env python3
"""Create small/medium/large TFDS datasets alongside an existing `dataset`.

Each output keeps the source feature schema and selected split. The source is
scanned in order until --num-events events have been collected for EACH bin.
"""
import argparse
from pathlib import Path
import json

import numpy as np
import tensorflow_datasets as tfds

LOWER = 16000
UPPER = 20000

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True, help='TFDS root (e.g. $FAST)')
    p.add_argument('--dataset', default='dataset', help='Source TFDS dataset name')
    p.add_argument('--split', default='train')
    p.add_argument('--num-events', type=int, required=True, help='Number to save PER bin')
    p.add_argument('--shard-events', type=int, default=1000)
    p.add_argument('--log-every', type=int, default=1000)
    args = p.parse_args()
    if args.num_events < 1 or args.shard_events < 1 or args.log_every < 1:
        p.error('--num-events, --shard-events, and --log-every must be positive')

    root = args.data_dir.expanduser().resolve()
    source_builder = tfds.builder(args.dataset, data_dir=str(root))
    source = source_builder.as_data_source(split=args.split)
    print(f'Source: {source_builder.data_dir} ({len(source):,} events in {args.split})', flush=True)

    # The class names become the new TFDS dataset names. In particular, their
    # prepared directories will be $FAST/small/<version>, etc.
    def make_info(self):
        return tfds.core.DatasetInfo(builder=self, features=source_builder.info.features)

    def unused_generator(self, dl_manager):
        raise RuntimeError('This builder is populated using SequentialWriter')

    names = ('small', 'medium', 'large')
    version = tfds.core.Version(str(source_builder.info.version))
    writers = {}
    for name in names:
        output_type = type(name.title(), (tfds.core.GeneratorBasedBuilder,), {
            'VERSION': version,
            '_info': make_info,
            '_split_generators': unused_generator,
            '_generate_examples': unused_generator,
        })
        builder = output_type(data_dir=str(root))
        out_path = Path(str(builder.data_dir))
        if out_path.exists():
            p.error(f'Output already exists: {out_path} (remove it before rerunning)')
        out_path.mkdir(parents=True)
        builder.info.set_file_format("array_record", override=True)

        writer = tfds.core.SequentialWriter(
            ds_info=builder.info,
            max_examples_per_shard=args.shard_events,
            file_format="array_record",
        )
        writer.initialize_splits([args.split])
        writers[name] = writer
        print(f'{name}: {out_path}', flush=True)

    counts = {name: 0 for name in names}
    inspected_lengths = []
    buffers = {name: [] for name in names}

    def flush(name):
        if buffers[name]:
            writers[name].add_examples({args.split: buffers[name]})
            buffers[name].clear()

    try:
        for i in range(len(source)):
            example = source[i]
            x = np.asarray(example['X'])
            if x.ndim != 2 or x.shape[0] == 0:
                raise ValueError(f'Event {i}: invalid X shape {x.shape}')
            n = int(x.shape[0])
            inspected_lengths.append(n)
            name = 'small' if n < LOWER else 'medium' if n < UPPER else 'large'
            if counts[name] < args.num_events:
                buffers[name].append(example)  # Retain ALL features, not just X.
                counts[name] += 1
                if len(buffers[name]) >= 16:
                    flush(name)
            if (i + 1) % args.log_every == 0:
                print(f'Inspected {i + 1:,}: saved {counts}', flush=True)
            if all(c == args.num_events for c in counts.values()):
                break
    finally:
        for name in names:
            flush(name)
            writers[name].close_all()

    lengths = np.asarray(inspected_lengths)
    total = len(lengths)
    observed = {
        'small': int(np.sum(lengths < LOWER)),
        'medium': int(np.sum((lengths >= LOWER) & (lengths < UPPER))),
        'large': int(np.sum(lengths >= UPPER)),
    }
    print(f'\nInspected {total:,} of {len(source):,} events', flush=True)
    print(f'Input length: min={lengths.min():,}, median={np.median(lengths):,.0f}, '
          f'mean={lengths.mean():,.1f}, p95={np.percentile(lengths, 95):,.0f}, '
          f'max={lengths.max():,}', flush=True)
    for name in names:
        print(f'  {name:<6}: observed={observed[name]:>8,} '
              f'({100 * observed[name] / total:5.1f}%), saved={counts[name]:>8,}', flush=True)
    if any(v < args.num_events for v in counts.values()):
        print('WARNING: dataset ended before all bins reached --num-events', flush=True)

    stats = {
        'source': args.dataset, 'split': args.split,
        'inspected': total, 'available': len(source),
        'requested_per_bin': args.num_events,
        'observed': observed, 'saved': counts,
        'length_min': int(lengths.min()), 'length_max': int(lengths.max()),
        'length_mean': float(lengths.mean()),
        'length_median': float(np.median(lengths)),
        'length_p95': float(np.percentile(lengths, 95)),
    }
    (root / 'event_bin_statistics.json').write_text(json.dumps(stats, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(root / 'event_bin_distributions.pdf') as pdf:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(lengths, bins=70)
        ax.axvline(LOWER, linestyle='--', color='black')
        ax.axvline(UPPER, linestyle='--', color='black')
        ax.set(xlabel='Input elements per event (X.shape[0])', ylabel='Events inspected',
               title=f'Event-size distribution: {total:,} inspected events')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(names, [observed[name] for name in names])
        for b, name in zip(bars, names):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f'{100 * observed[name]/total:.1f}%', ha='center', va='bottom')
        ax.set(ylabel='Events inspected', title='Observed bin distribution')
        ax.set_ylim(0, max(observed.values()) * 1.15 if max(observed.values()) else 1)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # Validate exactly the same lookup that the benchmark will use.
    for name in names:
        result = tfds.builder(name, data_dir=str(root)).as_data_source(split=args.split)
        if len(result) != counts[name]:
            raise RuntimeError(f'{name}: wrote {counts[name]} but read back {len(result)}')
        if len(result):
            n = np.asarray(result[0]['X']).shape[0]
            if not ((name == 'small' and n < LOWER) or
                    (name == 'medium' and LOWER <= n < UPPER) or
                    (name == 'large' and n >= UPPER)):
                raise RuntimeError(f'{name}: read-back event is in wrong bin')
        print(f'Verified tfds.builder({name!r}, data_dir={str(root)!r}): {len(result):,} events')
    print(f'PDF: {root / "event_bin_distributions.pdf"}')


if __name__ == '__main__':
    main()
