"""patch_model.py — Add _pool_score_fn hook to Indexer in model.py."""
import sys

path = sys.argv[1]
with open(path, 'r') as f:
    src = f.read()

# 1. Add self._pool_score_fn = None to Indexer.__init__ (after freqs_cis = None,
#    before the first def forward of Indexer)
target_init = (
    '        self.freqs_cis = None\n'
    '\n'
    '    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):'
)
replacement_init = (
    '        self.freqs_cis = None\n'
    '        self._pool_score_fn = None\n'
    '\n'
    '    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):'
)
assert target_init in src, 'init target not found'
src = src.replace(target_init, replacement_init, 1)

# 2. Add pool scoring hook in Indexer.forward before the einsum line
# The line uses double quotes inside, so we target it precisely
target_einsum = (
    '        index_score = torch.einsum("bshd,btd->bsht",'
    ' q, self.kv_cache[:bsz, :end_pos // ratio])\n'
)
replacement_einsum = (
    '        if self._pool_score_fn is not None and start_pos > 0:\n'
    '            return self._pool_score_fn(q, weights, bsz, end_pos, offset)\n'
    '        index_score = torch.einsum("bshd,btd->bsht",'
    ' q, self.kv_cache[:bsz, :end_pos // ratio])\n'
)
assert target_einsum in src, f'einsum target not found'
src = src.replace(target_einsum, replacement_einsum, 1)

with open(path, 'w') as f:
    f.write(src)
print('Patched OK')
