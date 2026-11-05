"""Download official model assets and freeze source and content fingerprints."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '732d834db70ee0fc3886b4bcbcfb4ce7fb829be2'


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix + '.partial')
        print(f'Downloading {path.name}', flush=True)
        with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        partial.replace(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'url':url, 'relative_path':str(path.relative_to(ROOT)), 'sha256':sha, 'bytes':path.stat().st_size}


def main():
    records = []
    for name in ['BEATs.py','backbone.py','modules.py','README.md']:
        records.append(download(f'https://raw.githubusercontent.com/microsoft/unilm/{COMMIT}/beats/{name}', ROOT/'vendor'/'beats'/name))
    records.append(download(f'https://raw.githubusercontent.com/microsoft/unilm/{COMMIT}/LICENSE', ROOT/'vendor'/'beats'/'LICENSE'))
    records.append(download('https://download.pytorch.org/models/resnet18-f37072fd.pth',ROOT/'pretrained'/'resnet18-f37072fd.pth'))
    if not records[-1]['sha256'].startswith('f37072fd'):
        raise RuntimeError('torchvision checkpoint digest prefix mismatch')
    lock = ROOT/'locks'/'pretrained_sources.json'
    lock.parent.mkdir(exist_ok=True)
    lock.write_text(json.dumps({'beats_source_commit':COMMIT,'files':records},indent=2)+'\n',encoding='utf-8')
    print('Official sources and ResNet18 checkpoint ready',flush=True)
    # Official Iter3 pretraining checkpoint: URL from the pinned upstream README.
    # OneDrive download=1 requests original bytes rather than an HTML preview.
    official_url = 'https://1drv.ms/u/s!AqeByhGUtINrgcpxJUNDxg4eU0r-vA?download=1'
    try:
        record = download(official_url,ROOT/'pretrained'/'BEATs_iter3.pt')
    except OSError as exc:
        print(f'Official checkpoint endpoint unavailable: {type(exc).__name__}; using pinned mirror',flush=True)
        record = download('https://huggingface.co/lpepino/beats_ckpts/resolve/a2ddb6b0411c39942ae144a6414872e14e5a4329/BEATs_iter3.pt',ROOT/'pretrained'/'BEATs_iter3.pt')
        record['upstream_url'] = official_url
        record['mirror_reason'] = str(exc)
    if record['sha256'] != '8d1b234032a9ccff353612dc6c20982346dc2968b205b79d97303eb5e77bfb34':
        raise RuntimeError('BEATs checkpoint does not match pinned Iter3 SHA256')
    import torch
    checkpoint = torch.load(ROOT/'pretrained'/'BEATs_iter3.pt',map_location='cpu',weights_only=True)
    if checkpoint['cfg'].get('finetuned_model',False):
        raise RuntimeError('expected pretrained checkpoint, not AudioSet fine-tuned model')
    records.append(record)
    lock.write_text(json.dumps({'beats_source_commit':COMMIT,'files':records},indent=2)+'\n',encoding='utf-8')
    print('BEATs Iter3 checkpoint validated',flush=True)


if __name__=='__main__':
    main()
