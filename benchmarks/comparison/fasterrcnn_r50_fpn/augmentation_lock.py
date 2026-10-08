"""Verify unchanged computational class bodies despite standalone imports."""
import ast
from functools import lru_cache
from support import HERE, digest, read_json, text_hash
@lru_cache(maxsize=1)
def verify_excerpts():
    lock=read_json(HERE/'augmentation.lock.json')
    source=(HERE/'augmentation_reference.py').read_text(encoding='utf-8')
    classes={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.ClassDef)}
    for name,expected in lock['reference_pipeline']['verbatim_class_sha256'].items():
        # The reference lock hashes the class including its terminating newline.
        raw=classes[name].encode()
        if expected not in (digest(raw),digest(raw+b'\n'),digest(raw+b'\n\n')):
            raise ValueError('Reference transform body changed: '+name)
    for name,expected in lock['standalone_files'].items():
        if text_hash(HERE/name)!=expected: raise ValueError('Standalone augmentation file changed: '+name)
    return lock
