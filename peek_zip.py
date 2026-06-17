import urllib.request
import zipfile
import json

def get_download_url():
    api_url = "https://zenodo.org/api/records/5642694"
    req = urllib.request.Request(api_url, headers={'Accept': 'application/json'})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))
        for f in data.get('files', []):
            if f.get('key') == 'generated_audio.zip':
                return f.get('links', {}).get('self')
    return None

class HttpFile:
    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req) as response:
            self.size = int(response.headers['Content-Length'])
        self.pos = 0

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        if size == -1:
            size = self.size - self.pos
        if size == 0:
            return b""
        
        if size > 50 * 1024 * 1024:
            raise Exception(f"Requested too much data: {size} bytes")
            
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{self.pos+size-1}'})
        try:
            with urllib.request.urlopen(req) as response:
                data = response.read()
                self.pos += len(data)
                return data
        except Exception as e:
            print(f"Error fetching range {self.pos}-{self.pos+size-1}: {e}")
            raise e

def peek_zip():
    download_url = get_download_url()
    if not download_url:
        return
        
    print(f"Zip size: {HttpFile(download_url).size / (1024**3):.2f} GB")
    
    http_file = HttpFile(download_url)
    try:
        zf = zipfile.ZipFile(http_file)
        
        namelist = zf.namelist()
        print(f"\nTotal files in zip: {len(namelist)}")
        
        print("\nSub-directories inside generated_audio:")
        sub_dirs = set(name.split('/')[1] for name in namelist if name.startswith('generated_audio/') and '/' in name[16:])
        for d in sub_dirs:
            print(f" - {d}")
            
    except Exception as e:
        print(f"Could not parse zip directory: {e}")

if __name__ == '__main__':
    peek_zip()
