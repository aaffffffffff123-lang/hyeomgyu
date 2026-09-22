"""inbox/ 에 들어온 파일을 사이트에 반영한다.

- 그림 파일(png/jpg/jpeg/webp): 새 만화로 처리 -> img/번호.png, thumb/번호.jpg, list.csv 한 줄
- zip: 안에 list.csv 와 img/ 가 있으면 준비된 묶음으로 보고 그대로 합침,
       아니면 안에 든 그림들을 전부 새 만화로 처리 (zip 안의 폴더 하나 = 여러 장짜리 한 편)
- 여러 장짜리 만화: 이름-1.png, 이름-2.png 처럼 끝 번호만 다른 파일들을 함께 올리면 한 편으로 묶임
  (list.csv 의 episode 칸에 첫 장 번호, page 칸에 장 순서)
- 새 만화가 이미 있는 만화와 같으면(그림 해시 또는 대사 비교) 추가하지 않고
  inbox/duplicates/ 에 옮기고 중복.txt 에 적는다
- img/ 에 그림이 없어진 줄은 list.csv 에서 뺀다
처리한 inbox 파일은 지운다. GitHub Actions 가 push 마다 실행한다.
"""
import os, sys, io, csv, re, shutil, zipfile, tempfile, collections, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, 'tools')
os.environ['TESSDATA_PREFIX'] = os.path.join(TOOLS, 'tessdata')
sys.path.insert(0, TOOLS)

INBOX, IMG, THUMB, LIST = (os.path.join(ROOT, d) for d in ('inbox', 'img', 'thumb', 'list.csv'))
COLS = ['id', 'hidden', 'keywords', 'text', 'episode', 'page']
IMG_EXT = ('.png', '.jpg', '.jpeg', '.webp')


def read_list():
    if not os.path.exists(LIST):
        return []
    with open(LIST, encoding='utf-8-sig', newline='') as f:
        return [{c: (r.get(c) or '') for c in COLS} for r in csv.DictReader(f)]


def write_list(rows):
    with open(LIST, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in COLS})


def next_id(rows):
    n = max([int(r['id']) for r in rows if r['id'].isdigit()] + [0]) + 1
    return f'{n:04d}'


def to_web(src, dst_png, dst_jpg):
    """원본 -> 웹용 PNG(흑백이면 16단계 팔레트) + 썸네일 JPG"""
    from PIL import Image
    import numpy as np
    im = Image.open(src)
    if im.mode in ('RGBA', 'LA', 'P'):
        im = im.convert('RGBA')
        bg = Image.new('RGB', im.size, 'white'); bg.paste(im, mask=im.split()[3]); im = bg
    else:
        im = im.convert('RGB')
    if max(im.size) > 1600:
        im.thumbnail((1600, 1600))
    a = np.asarray(im.resize((256, 256)))
    sat = (a.max(axis=2).astype(int) - a.min(axis=2)).mean()
    if sat > 2:
        im.save(dst_png, 'PNG', optimize=True)
    else:
        im.convert('L').quantize(16).save(dst_png, 'PNG', optimize=True)
    t = im.convert('L'); t.thumbnail((360, 360)); t.save(dst_jpg, 'JPEG', quality=80, optimize=True)


def read_text(png):
    """OCR -> 대사 문자열(줄은 ' / '로 이음)"""
    try:
        from ocr_lib import ocr_image_fast
        lines, _ = ocr_image_fast(png, scale=2.5)
    except Exception:
        traceback.print_exc(); return ''
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        return ' / '.join(kiwi.space(t) for t, c, p in lines)
    except Exception:
        return ' / '.join(t for t, c, p in lines)


_vocab = None
def keywords(text):
    """대사에서 명사를 뽑아, 기존 만화들에도 나오는 낱말(vocab.txt)만 최대 20개"""
    global _vocab
    if not text:
        return ''
    if _vocab is None:
        p = os.path.join(TOOLS, 'vocab.txt')
        _vocab = set(open(p, encoding='utf-8').read().split()) if os.path.exists(p) else set()
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        cnt = collections.Counter(t.form for t in kiwi.tokenize(text.replace(' / ', ' '))
                                  if t.tag in ('NNG', 'NNP') and len(t.form) >= 2 and t.form in _vocab)
        return ' '.join(w for w, c in cnt.most_common(20))
    except Exception:
        traceback.print_exc(); return ''


HASHES = os.path.join(TOOLS, 'hashes.json')


def _norm(t):
    return re.sub(r'[^가-힣A-Za-z0-9]', '', t or '')


def _grams(t):
    return {t[i:i + 3] for i in range(len(t) - 2)}


def phash(png):
    """256비트 지각 해시 (64x64 회색조 DCT 저주파 16x16, 중앙값 기준). 4컷 테두리가 비슷해도 구분되도록 큰 해시를 쓴다."""
    from PIL import Image
    import numpy as np
    im = Image.open(png)
    if im.mode in ('RGBA', 'LA', 'P'):
        im = im.convert('RGBA'); bg = Image.new('RGB', im.size, 'white'); bg.paste(im, mask=im.split()[3]); im = bg
    im = im.convert('L').resize((64, 64), Image.LANCZOS)
    a = np.asarray(im, dtype=float)
    n = 64
    k = np.arange(n)
    D = np.cos(np.pi / n * (k[:, None] + 0.5) * k[None, :])  # DCT-II 기저
    dct = D.T @ a @ D
    low = dct[:16, :16].flatten()
    med = np.median(low)
    bits = (low > med).astype(int)
    return ''.join(str(b) for b in bits)


def load_hashes(rows):
    """기존 만화들의 해시. 없는 것만 새로 계산해서 tools/hashes.json 에 저장"""
    import json
    h = {}
    if os.path.exists(HASHES):
        try:
            h = json.load(open(HASHES, encoding='utf-8'))
        except Exception:
            h = {}
    changed = False
    ids = {r['id'] for r in rows}
    for k in list(h):  # 형식이 다른 옛 해시는 버림
        if not isinstance(h[k], str) or len(h[k]) != 256:
            del h[k]; changed = True
    for r in rows:
        p = os.path.join(IMG, r['id'] + '.png')
        if r['id'] not in h and os.path.exists(p):
            try:
                h[r['id']] = phash(p); changed = True
            except Exception:
                pass
    for k in list(h):
        if k not in ids:
            del h[k]; changed = True
    if changed:
        json.dump(h, open(HASHES, 'w', encoding='utf-8'))
    return h


def find_duplicate(png, text, rows, hashes):
    """같은 만화가 이미 있으면 (번호, 이유) 반환. 그림 해시가 가깝거나 대사가 많이 겹치면 중복."""
    try:
        hp = phash(png)
        for r in rows:
            hq = hashes.get(r['id'])
            if hq and len(hq) == len(hp) and sum(a != b for a, b in zip(hp, hq)) <= 24:
                return r['id'], '그림이 같음'
    except Exception:
        traceback.print_exc()
    n = _norm(text)
    if len(n) >= 15:
        g = _grams(n)
        for r in rows:
            m = _norm(r['text'])
            if len(m) < 15:
                continue
            gm = _grams(m)
            inter = len(g & gm)
            if not inter:
                continue
            j = inter / len(g | gm)
            if j >= 0.35 or inter / min(len(g), len(gm)) >= 0.7:
                return r['id'], f'대사가 같음 (유사도 {j:.2f})'
    return None


def add_comic(src, rows, hashes=None, episode='', page=''):
    cid = next_id(rows)
    png, jpg = os.path.join(IMG, cid + '.png'), os.path.join(THUMB, cid + '.jpg')
    to_web(src, png, jpg)
    text = read_text(png)
    dup = find_duplicate(png, text, rows, hashes or {}) if hashes is not None else None
    if dup:
        os.remove(png); os.remove(jpg)
        dupdir = os.path.join(INBOX, 'duplicates'); os.makedirs(dupdir, exist_ok=True)
        dst = os.path.join(dupdir, f'{dup[0]}과_같음__{os.path.basename(src)}')
        shutil.copy2(src, dst)
        with open(os.path.join(ROOT, '중복.txt'), 'a', encoding='utf-8') as f:
            f.write(f'{os.path.basename(src)} -> 이미 있는 {dup[0]}번과 같은 만화 ({dup[1]}). 추가하지 않고 inbox/duplicates 에 옮겨 둠\n')
        print(f'중복: {os.path.basename(src)} = {dup[0]} ({dup[1]})')
        return None
    rows.append({'id': cid, 'hidden': '', 'keywords': keywords(text), 'text': text, 'episode': episode, 'page': str(page) if page else ''})
    if hashes is not None:
        try:
            hashes[cid] = phash(png)
        except Exception:
            pass
    print(f'새 만화 {cid} <- {os.path.basename(src)} ({len(text)}자)')
    return cid


def merge_bundle(folder, rows):
    """준비된 묶음: img/, thumb/, list.csv 를 그대로 합친다"""
    n = 0
    for sub, dst in (('img', IMG), ('thumb', THUMB)):
        d = os.path.join(folder, sub)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(IMG_EXT):
                    shutil.copy2(os.path.join(d, f), os.path.join(dst, f)); n += 1
    lp = os.path.join(folder, 'list.csv')
    if os.path.exists(lp):
        byid = {r['id']: r for r in rows}
        with open(lp, encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                r = {c: (r.get(c) or '') for c in COLS}
                if not r['id']:
                    continue
                if r['id'] in byid:
                    byid[r['id']].update(r)
                else:
                    rows.append(r); byid[r['id']] = r
    print(f'묶음 합침: 그림 {n}장')
    # 썸네일이 없는 그림은 만들어 준다
    for f in os.listdir(IMG):
        cid = os.path.splitext(f)[0]
        if not os.path.exists(os.path.join(THUMB, cid + '.jpg')):
            from PIL import Image
            t = Image.open(os.path.join(IMG, f)).convert('L'); t.thumbnail((360, 360)); t.save(os.path.join(THUMB, cid + '.jpg'), 'JPEG', quality=80, optimize=True)


PAGE_RE = re.compile(r'^(.*?)[\s_\-]*[\(\[]?(\d{1,2})[\)\]]?$')


def group_pages(names):
    """같은 이름에 끝 번호만 다른 파일들(예: 이름-1.png, 이름-2.png)을 한 편으로 묶는다.
    반환: [[파일명, ...], ...] (각 묶음은 번호 순서)"""
    buckets = collections.OrderedDict()
    for n in names:
        stem, ext = os.path.splitext(n)
        m = PAGE_RE.match(stem)
        key = (m.group(1).strip().lower() if m and m.group(1).strip() else None)
        num = int(m.group(2)) if m else None
        if key is None:
            buckets.setdefault(('single', n), []).append((0, n))
        else:
            buckets.setdefault(('grp', key), []).append((num, n))
    out = []
    for k, items in buckets.items():
        items.sort()
        if k[0] == 'grp' and len(items) > 1:
            out.append([n for _, n in items])
        else:
            out.extend([[n] for _, n in items])
    return out


def add_episode(paths, rows, hashes):
    """여러 장짜리 한 편: 첫 장 번호를 episode 로, page 는 1부터"""
    first = None
    for i, pth in enumerate(paths):
        cid = add_comic(pth, rows, hashes, episode=first or '', page=i + 1 if len(paths) > 1 else '')
        if cid and first is None:
            first = cid
            rows[-1]['episode'] = cid if len(paths) > 1 else ''
            rows[-1]['page'] = '1' if len(paths) > 1 else ''
    return first


def main():
    for d in (INBOX, IMG, THUMB):
        os.makedirs(d, exist_ok=True)
    rows = read_list()
    hashes = load_hashes(rows)
    items = sorted(os.listdir(INBOX))
    loose = [n for n in items if n.lower().endswith(IMG_EXT) and not n.startswith('.')]
    for grp in group_pages(loose):
        try:
            add_episode([os.path.join(INBOX, n) for n in grp], rows, hashes)
            for n in grp:
                os.remove(os.path.join(INBOX, n))
        except Exception:
            print('처리 실패:', grp); traceback.print_exc()
    for name in items:
        p = os.path.join(INBOX, name)
        low = name.lower()
        if name in loose:
            continue
        try:
            if low.endswith('.zip'):
                with tempfile.TemporaryDirectory() as tmp:
                    with zipfile.ZipFile(p) as z:
                        z.extractall(tmp)
                    # zip 안에 폴더 하나만 있으면 그 안으로
                    inner = [x for x in os.listdir(tmp) if not x.startswith('__MACOSX')]
                    base = os.path.join(tmp, inner[0]) if len(inner) == 1 and os.path.isdir(os.path.join(tmp, inner[0])) else tmp
                    if os.path.exists(os.path.join(base, 'list.csv')) and os.path.isdir(os.path.join(base, 'img')):
                        merge_bundle(base, rows)
                    else:
                        for root, _, fs in os.walk(base):
                            if '__MACOSX' in root:
                                continue
                            imgs = sorted(f for f in fs if f.lower().endswith(IMG_EXT) and not f.startswith('.'))
                            if not imgs:
                                continue
                            if root != base and len(imgs) > 1:
                                add_episode([os.path.join(root, f) for f in imgs], rows, hashes)  # 폴더 하나 = 한 편
                            else:
                                for grp in group_pages(imgs):
                                    add_episode([os.path.join(root, f) for f in grp], rows, hashes)
                os.remove(p)
        except Exception:
            print('처리 실패:', name); traceback.print_exc()
    # episodes.txt: 이미 올라간 만화를 여러 장짜리 한 편으로 묶는 목록 (한 줄에 번호들을 순서대로)
    ep_path = os.path.join(ROOT, 'episodes.txt')
    if os.path.exists(ep_path):
        byid = {r['id']: r for r in rows}
        for line in open(ep_path, encoding='utf-8'):
            line = line.split('#')[0].strip()
            ids = [x.zfill(4) for x in re.split(r'[\s,~\-]+', line) if x.strip().isdigit()]
            ids = [x for x in ids if x in byid]
            if len(ids) < 2:
                continue
            for i, cid in enumerate(ids):
                byid[cid]['episode'] = ids[0]; byid[cid]['page'] = str(i + 1)
    # 그림이 없어진 줄 정리
    rows = [r for r in rows if os.path.exists(os.path.join(IMG, r['id'] + '.png'))]
    rows.sort(key=lambda r: int(r['id']) if r['id'].isdigit() else 10**9)
    write_list(rows)
    import json
    ids = {r['id'] for r in rows}
    json.dump({k: v for k, v in hashes.items() if k in ids}, open(HASHES, 'w', encoding='utf-8'))
    for f in os.listdir(THUMB):  # 그림이 없어진 썸네일 정리
        if f.lower().endswith('.jpg') and not os.path.exists(os.path.join(IMG, os.path.splitext(f)[0] + '.png')):
            os.remove(os.path.join(THUMB, f))
    load_hashes(rows)  # 묶음으로 들어온 그림의 해시도 채워 둠
    print(f'목록 {len(rows)}편')


if __name__ == '__main__':
    main()
