import os, re, collections
import numpy as np, cv2
from PIL import Image
import pytesseract

os.environ.setdefault('TESSDATA_PREFIX', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tessdata'))
HANGUL = re.compile(r'[가-힣]')


def load_gray(path):
    im = Image.open(path)
    if im.mode == 'RGBA':
        bg = Image.new('RGB', im.size, 'white')
        bg.paste(im, mask=im.split()[3])
        im = bg
    else:
        im = im.convert('RGB')
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)


def find_text_blocks(gray, scale_ref=1440):
    """Find blocks of typed text: many small similar-height components in horizontal runs."""
    h, w = gray.shape
    s = w / scale_ref  # size factors relative to a 1440px-wide image
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # remove long straight lines (panel borders) so text touching a border is not glued to it
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(w * 0.08)), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(h * 0.08))))
    lines_h = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    lines_v = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)
    border = cv2.dilate(cv2.bitwise_or(lines_h, lines_v), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    bw = cv2.bitwise_and(bw, cv2.bitwise_not(border))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    # character-sized components
    minh, maxh = 8 * s, 46 * s
    maxw = 70 * s
    mask = np.zeros_like(bw)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if minh <= ch <= maxh and cw <= maxw and area >= 6 * s * s:
            # discard very thin long horizontal strokes (panel borders / underlines)
            if cw > 4 * ch and ch < 10 * s:
                continue
            mask[y:y + ch, x:x + cw] = 255
    # 1) merge characters into lines (horizontal dilation)
    lines = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(26 * s)), max(1, int(3 * s)))))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    linemask = np.zeros_like(bw)
    for i in range(1, n2):
        x, y, cw, ch, area = st2[i]
        if ch < minh or ch > maxh * 1.6:
            continue
        sub = mask[y:y + ch, x:x + cw]
        nc = cv2.connectedComponentsWithStats((sub > 0).astype(np.uint8), connectivity=8)[0] - 1
        if nc < 3 or cw < 30 * s:
            continue
        linemask[y:y + ch, x:x + cw] = 255
    # 2) merge lines into blocks (vertical dilation)
    blk = cv2.dilate(linemask, cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(6 * s)), max(1, int(26 * s)))))
    n3, lab3, st3, _ = cv2.connectedComponentsWithStats(blk, connectivity=8)
    blocks = []
    for i in range(1, n3):
        x, y, cw, ch, area = st3[i]
        sub = mask[y:y + ch, x:x + cw]
        nc = cv2.connectedComponentsWithStats((sub > 0).astype(np.uint8), connectivity=8)[0] - 1
        if nc < 4:
            continue
        blocks.append({'box': [x, y, x + cw, y + ch], 'nchars': int(nc)})
    # drop boxes contained in bigger ones
    keep = []
    for i, a in enumerate(blocks):
        ax0, ay0, ax1, ay1 = a['box']
        contained = any(j != i and b['box'][0] <= ax0 and b['box'][1] <= ay0 and b['box'][2] >= ax1 and b['box'][3] >= ay1 for j, b in enumerate(blocks))
        if not contained:
            keep.append(a)
    return keep


def ocr_block(gray, box, pad=6, scale=3):
    h, w = gray.shape
    x0, y0, x1, y1 = box
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad); x1 = min(w, x1 + pad); y1 = min(h, y1 + pad)
    crop = gray[y0:y1, x0:x1]
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    crop = cv2.copyMakeBorder(crop, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    d = pytesseract.image_to_data(crop, lang='kor', config='--psm 6', output_type=pytesseract.Output.DICT)
    lines = collections.OrderedDict()
    for i in range(len(d['text'])):
        t = d['text'][i].strip()
        if not t:
            continue
        key = (d['block_num'][i], d['par_num'][i], d['line_num'][i])
        lines.setdefault(key, []).append((t, float(d['conf'][i])))
    out = []
    for k, ws in lines.items():
        txt = ' '.join(t for t, c in ws)
        confs = [c for t, c in ws if c >= 0]
        conf = sum(confs) / len(confs) if confs else 0
        out.append((txt, conf))
    return out


def clean_text(t):
    # tesseract kor often inserts spaces between syllables; collapse spaces between hangul
    t = re.sub(r'(?<=[가-힣])\s+(?=[가-힣])', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def panel_order_key(box, w, h):
    # reading order for 2x2 grid: top-left, top-right, bottom-left, bottom-right; fallback by y then x
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (int(cy > h / 2), int(cx > w / 2), y0, x0)


JUNK_TOKEN = re.compile(r'^[^가-힣A-Za-z0-9]+$')


def clean_line(txt):
    toks = [t for t in txt.split() if not JUNK_TOKEN.match(t) or t in ('?', '!', '?!', '!?', '...', '…')]
    t = ' '.join(toks)
    t = re.sub(r'(?<=[가-힣])\s+(?=[가-힣])', '', t)
    t = re.sub(r'\s*([?!…]+)\s*', r'\1 ', t)
    return re.sub(r'\s+', ' ', t).strip()


def find_panels(gray):
    """Detect comic panel rectangles (thick black borders)."""
    h, w = gray.shape
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        a = cw * ch
        if a < 0.06 * w * h or a > 0.7 * w * h:
            continue
        if not (0.35 <= cw / ch <= 3.0):
            continue
        if cv2.contourArea(c) < 0.75 * a:
            continue
        cands.append([x, y, x + cw, y + ch])
    # drop near-duplicates (inner/outer edge of the same border)
    cands.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    panels = []
    for b in cands:
        dup = False
        for p in panels:
            ix = max(0, min(b[2], p[2]) - max(b[0], p[0])); iy = max(0, min(b[3], p[3]) - max(b[1], p[1]))
            inter = ix * iy
            ua = (b[2] - b[0]) * (b[3] - b[1]) + (p[2] - p[0]) * (p[3] - p[1]) - inter
            if inter / ua > 0.6:
                dup = True; break
        if not dup:
            panels.append(b)
    # order: rows by y-center clusters, then x
    panels.sort(key=lambda b: (b[1] + b[3]) / 2)
    rows = []
    for b in panels:
        cy = (b[1] + b[3]) / 2
        if rows and abs(cy - rows[-1]['cy']) < 0.25 * (b[3] - b[1]):
            rows[-1]['items'].append(b)
        else:
            rows.append({'cy': cy, 'items': [b]})
    ordered = []
    for r in rows:
        ordered.extend(sorted(r['items'], key=lambda b: b[0]))
    return ordered


def split_wide_block(gray, box, s):
    """If a block spans two panels, split it at the widest internal white column gap."""
    x0, y0, x1, y1 = box
    sub = gray[y0:y1, x0:x1]
    col = (sub < 128).mean(axis=0)
    gaps = []
    start = None
    for i, v in enumerate(col):
        if v == 0:
            if start is None: start = i
        else:
            if start is not None: gaps.append((i - start, start, i)); start = None
    if start is not None: gaps.append((len(col) - start, start, len(col)))
    gaps = [g for g in gaps if g[1] > 20 * s and g[2] < len(col) - 20 * s and g[0] >= 30 * s]
    if not gaps:
        return [box]
    g = max(gaps)
    return [[x0, y0, x0 + g[1], y1], [x0 + g[2], y0, x1, y1]]


def ocr_image(path, min_conf=55, scale=2.5, pad=6, return_meta=False, skip_regions=None):
    gray = load_gray(path)
    h, w = gray.shape
    s = w / 1440
    panels = find_panels(gray)
    blocks = []
    for b in find_text_blocks(gray):
        for bb in split_wide_block(gray, b['box'], s):
            blocks.append(bb)

    def panel_index(box):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        for i, p in enumerate(panels):
            if p[0] <= cx <= p[2] and p[1] <= cy <= p[3]:
                return i
        return len(panels)  # outside any panel -> last

    if skip_regions:
        def inside(b):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            return any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in skip_regions)
        blocks = [b for b in blocks if not inside(b)]
    if panels:
        blocks.sort(key=lambda b: (panel_index(b), b[1], b[0]))
    else:
        blocks.sort(key=lambda b: panel_order_key(b, w, h))
    results = []
    for box in blocks:
        x0, y0, x1, y1 = box
        x0 = max(0, x0 - pad); y0 = max(0, y0 - pad); x1 = min(w, x1 + pad); y1 = min(h, y1 + pad)
        crop = gray[y0:y1, x0:x1]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        crop = cv2.copyMakeBorder(crop, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        d = pytesseract.image_to_data(crop, lang='kor', config='--psm 6', output_type=pytesseract.Output.DICT)
        lines = collections.OrderedDict()
        for i in range(len(d['text'])):
            t = d['text'][i].strip()
            if not t:
                continue
            key = (d['block_num'][i], d['par_num'][i], d['line_num'][i])
            lines.setdefault(key, []).append((t, float(d['conf'][i]), d['top'][i]))
        for k, ws in lines.items():
            confs = [c for t, c, yy in ws if c >= 0]
            conf = sum(confs) / len(confs) if confs else 0
            txt = clean_line(' '.join(t for t, c, yy in ws))
            if conf < min_conf or not (HANGUL.search(txt) or re.search(r'[A-Za-z0-9]{2,}', txt)):
                continue
            yline = y0 + (min(yy for t, c, yy in ws) - 20) / scale
            results.append((txt, round(conf), panel_index(box) + 1 if panels else 0, box, yline))
    if return_meta:
        return results, panels, blocks
    return results


def _norm(t):
    return re.sub(r'[^가-힣A-Za-z0-9]', '', t)


def strip_ocr(gray, panels, frac=0.42, scale=3, min_conf=55):
    """OCR the top strip of every panel (where dialogue usually sits)."""
    out = []
    h, w = gray.shape
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(w * 0.08)), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(h * 0.08))))
    border = cv2.dilate(cv2.bitwise_or(cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk), cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)), np.ones((5, 5), np.uint8))
    g2 = gray.copy(); g2[border > 0] = 255
    if not panels:
        panels = [[0, 0, w, h]]
    for pi, p in enumerate(panels):
        x0, y0, x1, y1 = p; ph = y1 - y0
        if ph < 40 or x1 - x0 < 40:
            continue
        crop = g2[y0 + 4:y0 + int(ph * frac), x0 + 4:x1 - 4]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        crop = cv2.copyMakeBorder(crop, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        d = pytesseract.image_to_data(crop, lang='kor', config='--psm 6', output_type=pytesseract.Output.DICT)
        lines = collections.OrderedDict()
        for i in range(len(d['text'])):
            t = d['text'][i].strip()
            if not t:
                continue
            k = (d['block_num'][i], d['par_num'][i], d['line_num'][i]); lines.setdefault(k, []).append((t, float(d['conf'][i]), d['top'][i]))
        for k, ws in lines.items():
            confs = [c for t, c, yy in ws if c >= 0]; conf = sum(confs) / len(confs) if confs else 0
            txt = clean_line(' '.join(t for t, c, yy in ws))
            if conf >= min_conf and len(re.findall(r'[가-힣]', txt)) >= 2:
                yabs = y0 + 4 + (min(yy for t, c, yy in ws) - 20) / scale
                out.append((txt, round(conf), pi + 1, yabs))
    return out


def ocr_image_full(path, scale=3):
    """Union of block-based OCR and panel-top-strip OCR, de-duplicated."""
    import difflib
    gray = load_gray(path)
    panels = find_panels(gray)
    strips = [[p[0], p[1], p[2], p[1] + int((p[3] - p[1]) * 0.42)] for p in panels]
    a = [(t, c, pi, y) for t, c, pi, box, y in ocr_image(path, scale=scale)]
    b = strip_ocr(gray, panels, scale=scale)
    merged = []
    for t, c, pi, y in sorted(a + b, key=lambda x: (-len(_norm(x[0])), -x[1])):
        n = _norm(t)
        if len(re.findall(r'[가-힣]', n)) < 2 and len(re.findall(r'[A-Za-z0-9]', n)) < 3:
            continue
        dup = False
        for j, m in enumerate(merged):
            mn = _norm(m[0])
            if n == mn or (len(n) >= 4 and (n in mn or mn in n)) or difflib.SequenceMatcher(None, n, mn).ratio() >= 0.8:
                dup = True
                if c > m[1] + 5 and abs(len(n) - len(mn)) <= 3:
                    merged[j] = (t, c, pi, y)  # near-same text, keep the more confident reading
                break
        if not dup:
            merged.append((t, c, pi, y))
    merged.sort(key=lambda x: (x[2], x[3]))
    return [(t, c, pi) for t, c, pi, y in merged], len(panels)


def ocr_blocks_stacked(path, scale=2.5, pad=6, gap=120, min_conf=55):
    """All text blocks pasted onto one tall canvas -> one tesseract call (psm 6)."""
    gray = load_gray(path); h, w = gray.shape; s = w / 1440
    panels = find_panels(gray)
    blocks = []
    for b in find_text_blocks(gray):
        for bb in split_wide_block(gray, b['box'], s):
            blocks.append(bb)

    def pidx(box):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        for i, p in enumerate(panels):
            if p[0] <= cx <= p[2] and p[1] <= cy <= p[3]:
                return i
        return len(panels)
    blocks.sort(key=lambda b: (pidx(b), b[1], b[0]))
    if not blocks:
        return [], panels
    crops = []
    for x0, y0, x1, y1 in blocks:
        x0 = max(0, x0 - pad); y0 = max(0, y0 - pad); x1 = min(w, x1 + pad); y1 = min(h, y1 + pad)
        c = gray[y0:y1, x0:x1]
        c = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        crops.append((c, y0))
    W = max(c.shape[1] for c, _ in crops) + 80
    H = sum(c.shape[0] for c, _ in crops) + gap * (len(crops) + 1)
    canvas = np.full((H, W), 255, np.uint8); y = gap; spans = []
    for (c, y0), b in zip(crops, blocks):
        canvas[y:y + c.shape[0], 40:40 + c.shape[1]] = c
        spans.append((y, y + c.shape[0], b, y0)); y += c.shape[0] + gap
    d = pytesseract.image_to_data(canvas, lang='kor', config='--psm 6', output_type=pytesseract.Output.DICT)
    lines = collections.OrderedDict()
    for i in range(len(d['text'])):
        t = d['text'][i].strip()
        if not t:
            continue
        k = (d['block_num'][i], d['par_num'][i], d['line_num'][i])
        lines.setdefault(k, {'y': d['top'][i], 'w': []}); lines[k]['w'].append((t, float(d['conf'][i])))
    out = []
    for k, v in lines.items():
        confs = [c for t, c in v['w'] if c >= 0]; conf = sum(confs) / len(confs) if confs else 0
        txt = clean_line(' '.join(t for t, c in v['w']))
        if conf < min_conf or not (HANGUL.search(txt) or re.search(r'[A-Za-z0-9]{2,}', txt)):
            continue
        sp = next((x for x in spans if x[0] - 10 <= v['y'] <= x[1] + 10), None)
        if sp is None:
            continue
        pi = pidx(sp[2]) + 1 if panels else 0
        yabs = sp[3] + (v['y'] - sp[0]) / scale
        out.append((txt, round(conf), pi, yabs))
    return out, panels


def ocr_image_fast(path, scale=2.5):
    """stacked block OCR + panel-top-strip OCR, merged (same output shape as ocr_image_full)."""
    import difflib
    gray = load_gray(path)
    a, panels = ocr_blocks_stacked(path, scale=scale)
    b = strip_ocr(gray, panels, scale=scale)
    merged = []
    for t, c, pi, y in sorted(a + b, key=lambda x: (-len(_norm(x[0])), -x[1])):
        n = _norm(t)
        if len(re.findall(r'[가-힣]', n)) < 2 and len(re.findall(r'[A-Za-z0-9]', n)) < 3:
            continue
        dup = False
        for j, m in enumerate(merged):
            mn = _norm(m[0])
            if n == mn or (len(n) >= 4 and (n in mn or mn in n)) or difflib.SequenceMatcher(None, n, mn).ratio() >= 0.8:
                dup = True
                if c > m[1] + 5 and abs(len(n) - len(mn)) <= 3:
                    merged[j] = (t, c, pi, y)
                break
        if not dup:
            merged.append((t, c, pi, y))
    merged.sort(key=lambda x: (x[2], x[3]))
    return [(t, c, pi) for t, c, pi, y in merged], len(panels)
