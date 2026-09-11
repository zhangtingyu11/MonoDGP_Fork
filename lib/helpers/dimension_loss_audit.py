"""Read-only per-batch guard receipts and historical printed-loss comparisons."""

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import torch


class DimensionLossAudit:
    def __init__(self, output_dir, baseline_log, logger):
        self.directory = Path(output_dir) / 'dimension_loss_audit'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        raw = Path(baseline_log).read_bytes()
        self.baseline_sha256 = hashlib.sha256(raw).hexdigest()
        self.baseline_log = str(Path(baseline_log).resolve())
        pattern = re.compile(
            r'Train metrics: epoch=(\d+)/(\d+), step=(\d+)/(\d+), '
            r'image_ids=(\[[^\]]*\]), lr=(\[[^\]]*\]), '
            r'loss_detr=([^,]+), losses=\{([^\n]*)\}')
        self.baseline = {}
        for match in pattern.finditer(raw.decode(errors='replace')):
            epoch, _, step, _, ids, lr, loss, fields = match.groups()
            self.baseline[(int(epoch), int(step))] = {
                'image_ids': ast.literal_eval(ids),
                'lr': ast.literal_eval(lr), 'loss': loss,
                'fields': dict(re.findall(r'(\w+)=([^,} ]+)', fields)),
            }
        if not self.baseline:
            raise ValueError('No historical batch records available for comparison')
        self.calls = Counter()
        self.triggers = Counter()
        self.batches = 0
        self.compared = 0
        self.mismatches = 0
        self.first_mismatch = None
        self.first_trigger = None

    def observe(self, epoch, step, losses, total, image_ids, learning_rates):
        reference = self.baseline.get((epoch, step))
        guard_names = sorted(k for k in losses if 'monitor_dim_' in k)
        if not guard_names:
            raise RuntimeError('Dimension guard monitoring is missing')
        names = sorted(set(guard_names) | (
            set(reference['fields']) & set(losses) if reference else set()))
        numbers = torch.stack([
            losses[k].detach().reshape(()) for k in names]).cpu().tolist()
        values = dict(zip(names, numbers))
        flags = {k: int(values[k]) for k in guard_names}
        self.calls.update(guard_names)
        self.triggers.update(flags)
        self.batches += 1
        row = {'epoch': epoch, 'step': step, 'guards': flags}
        if any(flags.values()):
            if self.first_trigger is None:
                self.first_trigger = row.copy()
            self.logger.warning('Dimension loss guard triggered: %s', row)
        if reference:
            if torch.is_tensor(image_ids):
                image_ids = image_ids.detach().cpu().tolist()
            different = []
            if list(image_ids) != reference['image_ids']:
                different.append('image_ids')
            # Match the precision used by the historical logger.
            if [float(f'{x:.8g}') for x in learning_rates] != reference['lr']:
                different.append('lr')
            if f'{total:.6f}' != reference['loss']:
                different.append('loss_detr')
            for key, expected in reference['fields'].items():
                if key not in values or f'{values[key]:.6f}' != expected:
                    different.append(key)
            self.compared += 1
            self.mismatches += bool(different)
            row['baseline_comparison'] = {
                'match': not different, 'different_fields': different,
                'current_total': f'{total:.6f}',
                'baseline_total': reference['loss'],
            }
            if different and self.first_mismatch is None:
                self.first_mismatch = {'epoch': epoch, 'step': step,
                                       **row['baseline_comparison']}
                self.logger.warning('First Exp59 comparison mismatch: %s',
                                    self.first_mismatch)
        with (self.directory / 'batches.jsonl').open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        if self.batches == 1:
            self.save_summary(epoch, step)

    def save_summary(self, epoch, step):
        report = {
            'epoch': epoch, 'step': step, 'observed_training_batches': self.batches,
            'guard_calls_by_key': dict(self.calls),
            'guard_triggers_by_key': dict(self.triggers),
            'first_trigger': self.first_trigger,
            'baseline_log': self.baseline_log,
            'baseline_sha256': self.baseline_sha256,
            'baseline_record_count': len(self.baseline),
            'compared_batches': self.compared,
            'mismatched_batches': self.mismatches,
            'first_mismatch': self.first_mismatch,
            'scope': 'All training batches; group0 keys are diagnostic-only. '
                     'Historical comparison uses printed precision, not byte hashes. '
                     'Validation loss calls are not included.',
        }
        temporary = self.directory / 'summary.json.tmp'
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        temporary.replace(self.directory / 'summary.json')
        self.logger.info('Dimension loss audit: batches=%d triggers=%s '
                         'compared=%d mismatches=%d', self.batches,
                         dict(self.triggers), self.compared, self.mismatches)
