"""Janela de status da dublagem (so leitura: le logs e STATUS.txt, nao controla nada).
pythonw status_dub.py      (Python 3.10 do sistema, tkinter)
Atualiza a cada 15 s. Fechar a janela nao para nenhum processo.
"""
import re, csv, json, time, pathlib, datetime as dt, collections, tkinter as tk
from tkinter import ttk

V = pathlib.Path(r'C:\gta-dub\outputs\GTA_V_Dublagem_PTBR')
LOTE1 = V / 'lote_v1' / 'logs'
L = V / 'cutscenes_lote'; LOGS = L / 'logs'
VOICE = pathlib.Path(r'D:\gta_dub_work\cut\voice')
PROG = re.compile(r'^\[(\d+)/(\d+)\]', re.M)

def read(p):
    try: b = pathlib.Path(p).read_bytes()
    except OSError: return ''
    if b[:2] == b'\xff\xfe': return b.decode('utf-16-le', 'replace')
    try: return b.decode('utf-8')
    except UnicodeDecodeError: return b.decode('cp1252', 'replace')

def gen_progress(p):
    """(done, total) of a dub_batch log; None if the log does not exist yet."""
    t = read(p)
    if not t: return None
    n = len(PROG.findall(t))
    m = re.search(r'BATCH \w+: (\d+) takes', t)
    if m: total = int(m.group(1))
    else:
        m = re.search(r'STAGE \w+: (\d+) lines x (\d+) takes', t)
        total = int(m.group(1)) * int(m.group(2)) if m else 0
        last = PROG.findall(t)
        if last: n = max(n, int(last[-1][0]))
    return n, max(total, n), 'STAGE finished' in t

counts = collections.Counter(r['character'] for r in csv.DictReader(open(L / 'fila_gen.csv', encoding='utf-8-sig')))
AWCS = (L / 'awc_com_falas.txt').read_text().split()
SEC = {'MICHAEL': 8640 / max(1, counts['MICHAEL'] + counts['FRANKLIN']), 'FRANKLIN': 8640 / max(1, counts['MICHAEL'] + counts['FRANKLIN']),
       'TREVOR': 15.0}          # s per line estimates (phase B, measured rates replace them once running)
hist = collections.defaultdict(collections.deque)

def rate(key, n):
    """lines per second over the last ~15 min of samples"""
    h = hist[key]; now = time.time(); h.append((now, n))
    while h and now - h[0][0] > 900: h.popleft()
    if len(h) >= 2 and h[-1][1] > h[0][1] and h[-1][0] - h[0][0] > 50: return (h[-1][1] - h[0][1]) / (h[-1][0] - h[0][0])
    return None

def fmt(s):
    if s is None: return ''
    s = int(s); return f'{s // 3600}h{(s % 3600) // 60:02d}' if s >= 3600 else f'{s // 60} min'

def snapshot():
    st = read(L / 'STATUS.txt'); drv = read(LOTE1 / 'driver.log')
    started = lambda name: f'START {name}' in st
    finished = lambda name: f'DONE  {name}' in st
    rows = []          # (label, done, total, state, eta_seconds)
    # 1 missions Trevor
    g = gen_progress(LOTE1 / 'gen_TREVOR.log')
    if g:
        n, tot, fin = g; fin = fin or 'GEN TREVOR exit 0' in drv
        r = rate('mt', n); eta = 0 if fin else ((tot - n) / r if r else (tot - n) * 14)
        rows.append(('Missoes: Trevor (geracao)', n, tot, 'feito' if fin else 'rodando', eta))
    steps = ['QA asr final', 'QA speaker', 'QA report', 'FIT', 'DONE']
    k = sum(1 for s in steps if s in drv)
    rows.append(('Missoes: conferencia (CPU)', k, len(steps), 'feito' if 'DONE' in drv else ('rodando' if 'GEN TREVOR exit 0' in drv else 'aguardando'), None))
    # 2 separation
    nd = sum(1 for a in AWCS if (VOICE / a / 'done.txt').exists())
    r = rate('sep', nd)
    rows.append(('Cutscenes: separar vozes (CPU)', nd, len(AWCS), 'feito' if nd >= len(AWCS) else 'rodando', (len(AWCS) - nd) / r if r else None))
    # 3 qwen batch test
    tst = 'feito' if 'teste de lote do Qwen PASSOU' in st or 'nao passou' in st else ('rodando' if 'teste de lote do Qwen' in st else 'aguardando')
    tlabel = 'Teste lote Trevor: ' + ('passou' if 'PASSOU' in st else 'nao passou (1 por vez)' if 'nao passou' in st else '...')
    rows.append((tlabel, 1 if tst == 'feito' else 0, 1, tst, None if tst == 'feito' else 600))
    verdict = L / 'qwen_batch_test' / 'qwen_batch_verdict.json'
    sp = 1.0
    if verdict.exists():
        try:
            j = json.loads(read(verdict))
            if j.get('pass'): sp = j['runs'][str(j['batch_size'])]['speedup']
        except Exception: pass
    # 4 generation
    for who in ('MICHAEL', 'FRANKLIN', 'TREVOR'):
        g = gen_progress(LOGS / f'gen_{who}.log') if any(f'START gen {who}' in l for l in st.splitlines()) else None
        est = SEC[who] / (sp if who == 'TREVOR' else 1)
        if g:
            n, tot, fin = g; r = rate('g' + who, n)
            rows.append((f'Cutscenes: gerar {who.title()}', n, tot or counts[who], 'feito' if fin else 'rodando', 0 if fin else (tot - n) / r if r else (tot - n) * est))
        else:
            rows.append((f'Cutscenes: gerar {who.title()}', 0, counts[who], 'aguardando', counts[who] * est))
    # 5 rest of phase B
    later = [('pick 1', 'Escolher takes', 900), ('retake MICHAEL', 'Refazer falas ruins', 2700), ('pick 2', 'Escolher takes 2', 600),
             ('mix', 'Remixar cutscenes', 1800), ('patch', 'Gravar pacote', 600)]
    for key, label, est in later:
        if key == 'retake MICHAEL':
            done = all(any(f'DONE  retake {w}' in l for l in st.splitlines()) for w in ('MICHAEL', 'FRANKLIN', 'TREVOR'))
            run = any(f'START retake' in l for l in st.splitlines()) and not done
        else:
            done = finished(key); run = started(key) and not done
        rows.append(('Cutscenes: ' + label, 1 if done else 0, 1, 'feito' if done else ('rodando' if run else 'aguardando'), 0 if done else est))
    failed = [l for l in st.splitlines() if 'FAILED' in l]
    ready = 'CUTSCENES READY' in st
    return rows, failed, ready

root = tk.Tk(); root.title('GTA V Dublagem - status'); root.attributes('-topmost', True); root.geometry('+40+40')
style = ttk.Style(root); style.theme_use('clam')
style.configure('done.Horizontal.TProgressbar', background='#3a9a52'); style.configure('run.Horizontal.TProgressbar', background='#2f6fd0')
frm = ttk.Frame(root, padding=10); frm.pack(fill='both', expand=True)
head = ttk.Label(frm, text='', font=('Segoe UI', 11, 'bold')); head.grid(row=0, column=0, columnspan=4, sticky='w', pady=(0, 6))
widgets = []
def ensure(n):
    while len(widgets) < n:
        i = len(widgets) + 1
        a = ttk.Label(frm, width=34); b = ttk.Progressbar(frm, length=230, maximum=1.0); c = ttk.Label(frm, width=12); d = ttk.Label(frm, width=16)
        a.grid(row=i, column=0, sticky='w'); b.grid(row=i, column=1, padx=6, pady=2); c.grid(row=i, column=2, sticky='e'); d.grid(row=i, column=3, sticky='w', padx=6)
        widgets.append((a, b, c, d))
foot = ttk.Label(frm, text='', foreground='#555'); top_var = tk.BooleanVar(value=True)
def refresh():
    try:
        rows, failed, ready = snapshot()
        ensure(len(rows))
        for (a, b, c, d), (label, n, tot, state, eta) in zip(widgets, rows):
            a.config(text=label); b.config(value=n / max(1, tot), style=('done' if state == 'feito' else 'run') + '.Horizontal.TProgressbar')
            c.config(text=f'{n}/{tot}' if tot > 1 else '')
            d.config(text='feito' if state == 'feito' else (f'{state} {("~" + fmt(eta)) if eta else ""}' if state == 'rodando' else 'aguardando'))
        # critical path: missions Trevor -> test -> gen M, F, T -> pick/retake/pick -> mix -> patch (QA and separation run in parallel)
        crit = [r for r in rows if r[0].startswith(('Missoes: Trevor', 'Teste', 'Cutscenes: gerar', 'Cutscenes: Escolher', 'Cutscenes: Refazer', 'Cutscenes: Remixar', 'Cutscenes: Gravar'))]
        left = sum(r[4] or 0 for r in crit if r[3] != 'feito')
        fim = (dt.datetime.now() + dt.timedelta(seconds=left)).strftime('%H:%M')
        if ready: head.config(text='CUTSCENES PRONTAS - pacote gerado (ainda nao instalado)', foreground='#3a9a52')
        elif failed: head.config(text='ERRO: ' + failed[-1][-70:], foreground='#c0392b')
        else: head.config(text=f'Falta ~{fmt(left)}  |  previsao de fim: {fim}', foreground='#222')
        foot.config(text=f'atualizado {dt.datetime.now():%H:%M:%S}  (estimativa; recalcula com a velocidade real)')
        foot.grid(row=len(rows) + 1, column=0, columnspan=3, sticky='w', pady=(8, 0))
        ttk.Checkbutton(frm, text='sempre no topo', variable=top_var, command=lambda: root.attributes('-topmost', top_var.get())).grid(row=len(rows) + 1, column=3, sticky='e', pady=(8, 0))
    except Exception as exc:
        head.config(text='erro lendo status: ' + repr(exc)[:80], foreground='#c0392b')
    root.after(15000, refresh)
refresh(); root.mainloop()
