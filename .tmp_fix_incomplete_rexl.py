from pathlib import Path
import re

root = Path('/mnt/d/Unicamp/Mestrado/salsaarm/vendor/incomplete-rexl/incomplete-rexl-0.1.0/src')
for path in root.rglob('*.rs'):
    text = path.read_text()
    text = text.replace('#![feature(target_feature_inline_always, avx512_target_feature)]\n', '')
    text = re.sub(r'(?m)(#\[target_feature\(.*?\)\])\n#\[inline\(always\)\]', r'\1', text)
    path.write_text(text)
print('patched')
