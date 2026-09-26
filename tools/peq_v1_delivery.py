"""Generate commit-addressed delivery documents outside Git; avoids self-containing SHA cycles."""
from __future__ import annotations
from peq_v1_common import *


def server_commands(sha):
    wt="/root/autodl-tmp/projects/Crack_RTDETR-peq_v1"
    py="/root/miniconda3/envs/rtdetr/bin/python"
    fence=chr(96)*3
    blocks=[f"# PEQ v1 server commands\n\nDelivered commit: {sha}\n\nEach block sets its own environment. Run sync, prepare, preflight, probe; inspect the probe before explicitly starting training.\n"]
    bootstrap=f"curl --fail --location --retry 3 'https://raw.githubusercontent.com/supershaojie/concrete-crack-rtdetr/{sha}/tools/sync_peq_v1.sh' -o /tmp/sync_peq_v1_{sha}.sh && \\\nbash /tmp/sync_peq_v1_{sha}.sh {sha}"
    blocks.append(f"## 1. Sync the exact SHA\n\n{fence}bash\n{bootstrap}\n{fence}\n")
    actions=[("Prepare","prepare"),("Bounded B16/640 AMP preflight","preflight"),("Read-only full mother val probe","probe"),
             ("Explicitly start training","start"),("Read-only status","status"),("Resume the same run from valid last","resume"),
             ("Independent best val and identity lock","val"),("Explicit test of val-locked best","test"),
             ("Pack existing evidence only","pack"),("Pack existing train/val error-analysis streams","pack --include-predictions"),
             ("Archive an eligible failed startup before a fresh start","archive-failed")]
    for title,action in actions:
        command=(f"cd {wt} && \\\n"
                 f"env PYTHONPATH={wt}/ultralytics-main:{wt}/tools PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false \\\n"
                 f"{py} {wt}/tools/peq_v1.py {action}")
        blocks.append(f"## {title}\n\n{fence}bash\n{command}\n{fence}\n")
    blocks.append(f"## Attach the training terminal\n\n{fence}bash\ntmux attach-session -t '=peq-v1-training'\n{fence}\n\nDetach without stopping training: Ctrl+B, then D.\n")
    blocks.append(f"FileZilla directory: {wt}/outputs/peq_v1/packages/\n\n"
                  f"The pack command prints exact absolute archive paths and writes {wt}/outputs/peq_v1/last_package.json. "
                  "LIGHT retains all mechanism JSONL, excludes .pt/datasets/full prediction streams, and lists omitted file hashes. "
                  "The explicit ERROR_ANALYSIS package contains existing train/val streams only; it never runs evaluation or adds test streams.\n\n"
                  "preflight is limited to 16 micro-batches or 900 seconds. Success requires at least two effective optimizer updates and nonzero upstream PEQ gradients after output-layer updates. "
                  "No action other than explicit test evaluates test. Resume requires unstripped last.pt with optimizer/scaler/EMA/epoch.\n")
    return "\n".join(blocks)


def write_delivery(sha, remote_verified=None):
    require(len(sha)==40,"Delivery needs full SHA")
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"server_commands.md").write_text(server_commands(sha),encoding="utf-8")
    validation=read_json(OUT/"validation.json",{})
    module_hashes=module_contract()
    text=(f"# PEQ v1 delivery\n\nCommit: {sha}\n\nBranch: {BRANCH}\n\nMother: {BASE}\n\n"
          f"Remote: {REMOTE}\n\nRemote verified SHA: {remote_verified or 'See sync/remote verification record'}\n\n"
          f"Worktree: {ROOT}\n\nParameters: original 20,149,765 + PEQ 17,994 = 20,167,759 (nc=1, unfused).\n\n"
          "Original LIF-Down and CBR are unchanged. PEQ reads detached pre/post-CBR local evidence and calibrates 300 final query scores. "
          "The native main/aux/DN losses retain original logits, boxes and matching. Only PEQ receives the added quality-loss gradient.\n\n"
          "Native AdamW/scaler/EMA remain; original and PEQ gradients each use max-norm 10 after one unscale. "
          "The dedicated validator fixes the historical sorted-mask mismatch. Only eager PyTorch .pt/AutoBackend is supported; enabled-PEQ export fails explicitly.\n\n"
          f"Local validation status: {validation.get('local_engineering','See validation.json')}\n\n"
          "Server B16/640 preflight: PENDING unless a server preflight.json records PASS for the current identity. "
          "Formal training, independent PEQ val and test: NOT_RUN at local delivery. Local fixtures are not formal results.\n\n"
          "Use server_commands.md. The complete mechanism/interface audit is docs/peq_v1/README.md.\n\n"
          "Original LF-normalized SHA256:\n\n"+chr(96)*3+"text\n"+
          "\n".join(f"{k}.py  {v['lf_sha256']}" for k,v in module_hashes.items())+"\n"+chr(96)*3+"\n")
    (OUT/"DELIVERY.md").write_text(text,encoding="utf-8")
    return dict(delivery=str(OUT/"DELIVERY.md"),commands=str(OUT/"server_commands.md"))
