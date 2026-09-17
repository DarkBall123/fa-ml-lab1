"""Download the pinned upstream archive and verify Git blob hashes."""
import concurrent.futures, hashlib, json, subprocess, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    manifest=json.loads((ROOT/'manifests/archive_files.json').read_text())
    archive=ROOT/'data/archives';archive.mkdir(parents=True,exist_ok=True)
    def download(item):
        path=archive/Path(item['path']).name
        def valid():
            if not path.exists() or path.stat().st_size!=item['size']:return False
            data=path.read_bytes()
            return hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==item['git_blob_sha1']
        if not valid():urllib.request.urlretrieve('https://raw.githubusercontent.com/stefyste/RIAWELC/d53d3daf28ed3901f9db89764c030d19ced7696d/'+item['path'],path)
        if not valid():raise RuntimeError(f'Integrity check failed: {path.name}')
        print(path.name,'verified',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(download,manifest))
    target=ROOT/'data/images';target.mkdir(exist_ok=True)
    subprocess.run(['unrar','x','-idq','-o+',str(archive/'RIAWELC_dataset.part01.rar'),str(target)+'/'],check=True)
    print('Image root:',target)
if __name__=='__main__':main()
