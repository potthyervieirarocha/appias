#!/usr/bin/env python3
"""Analisador de vídeos para o trailer da Conferência Santa Geração.

Uso:  python3 analisar_videos.py <pasta_com_videos> [--saida analise] [--metricas metricas.csv]

Para cada vídeo mede, segundo a segundo:
  - volume (LUFS momentâneo)      -> picos de louvor, plateia, clímax
  - movimento (diferença de quadros) -> energia visual, mãos levantadas, pulos
  - calor da cor (vermelho/laranja) -> combina com a paleta fogo do banner
  - brilho                            -> descarta trechos escuros/estourados
  - cortes de cena

Junta tudo numa nota por janela de 4 s e lista os melhores trechos.
Gera uma folha de contato (quadros a cada 2 s com tempo) para revisão visual.
Se houver metricas.csv (arquivo,views,mediana_conta) calcula o múltiplo de outlier.
"""
import argparse, csv, json, os, re, subprocess, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw

EXT = {'.mp4', '.mov', '.m4v', '.webm', '.mkv'}
JANELA = 4  # segundos


def loudness_por_segundo(path, dur):
    cmd = ['ffmpeg', '-hide_banner', '-nostats', '-i', path, '-vn',
           '-af', 'ebur128=metadata=1,ametadata=print:key=lavfi.r128.M', '-f', 'null', '-']
    p = subprocess.run(cmd, capture_output=True, text=True)
    t_re = re.compile(r'pts_time:([\d.]+)'); m_re = re.compile(r'lavfi\.r128\.M=(-?[\d.inf]+)')
    vals = {}; t = None
    for line in p.stderr.splitlines():
        a = t_re.search(line)
        if a: t = float(a.group(1)); continue
        b = m_re.search(line)
        if b and t is not None:
            try: v = float(b.group(1))
            except ValueError: v = -70.0
            vals.setdefault(int(t), []).append(max(v, -70.0))
    n = int(np.ceil(dur))
    return np.array([np.mean(vals[s]) if s in vals else -70.0 for s in range(n)])


def visual_por_segundo(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = total / fps if total else 0
    passo = max(1, int(round(fps / 4)))  # 4 amostras por segundo
    prev = None; prev_hist = None
    mov, calor, brilho, cortes = {}, {}, {}, []
    i = 0
    while True:
        ok = cap.grab()
        if not ok: break
        if i % passo == 0:
            ok, f = cap.retrieve()
            if not ok: break
            s = int(i / fps)
            small = cv2.resize(f, (160, int(160 * f.shape[0] / f.shape[1])))
            g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            h, sat, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            quente = (((h < 20) | (h > 165)) & (sat > 90) & (v > 60)).mean()
            calor.setdefault(s, []).append(quente)
            brilho.setdefault(s, []).append(v.mean() / 255)
            if prev is not None:
                mov.setdefault(s, []).append(cv2.absdiff(g, prev).mean() / 255)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256]); cv2.normalize(hist, hist)
            if prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > 0.45:
                cortes.append(round(i / fps, 2))
            prev, prev_hist = g, hist
        i += 1
    cap.release()
    n = int(np.ceil(dur)) or 1
    f = lambda d: np.array([np.mean(d[s]) if s in d else 0.0 for s in range(n)])
    return dur, fps, f(mov), f(calor), f(brilho), cortes


def norm(x):
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def folha_contato(path, dur, out, largura=320, colunas=6):
    ts = list(np.arange(0, max(dur - 0.1, 0.1), 2.0))[:60]
    cap = cv2.VideoCapture(path); thumbs = []
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, f = cap.read()
        if not ok: continue
        h = int(largura * f.shape[0] / f.shape[1])
        im = Image.fromarray(cv2.cvtColor(cv2.resize(f, (largura, h)), cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(im); d.rectangle([0, 0, 58, 18], fill=(0, 0, 0)); d.text((4, 3), f"{int(t//60)}:{int(t%60):02d}", fill=(255, 255, 0))
        thumbs.append(im)
    cap.release()
    if not thumbs: return
    h = thumbs[0].height; linhas = int(np.ceil(len(thumbs) / colunas))
    sheet = Image.new('RGB', (largura * colunas, h * linhas), (20, 20, 20))
    for k, im in enumerate(thumbs):
        sheet.paste(im.resize((largura, h)), ((k % colunas) * largura, (k // colunas) * h))
    sheet.save(out, quality=82)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pasta'); ap.add_argument('--saida', default='analise'); ap.add_argument('--metricas')
    a = ap.parse_args()
    os.makedirs(a.saida, exist_ok=True)
    met = {}
    if a.metricas and os.path.exists(a.metricas):
        for r in csv.DictReader(open(a.metricas, encoding='utf-8')):
            try: met[r['arquivo']] = float(r['views']) / float(r['mediana_conta'])
            except Exception: pass
    arquivos = sorted(f for f in os.listdir(a.pasta) if os.path.splitext(f)[1].lower() in EXT)
    resumo, trechos = [], []
    for nome in arquivos:
        p = os.path.join(a.pasta, nome); print('analisando', nome, file=sys.stderr)
        dur, fps, mov, calor, bri, cortes = visual_por_segundo(p)
        lou = loudness_por_segundo(p, dur)
        n = min(len(mov), len(lou))
        mov, calor, bri, lou = mov[:n], calor[:n], bri[:n], lou[:n]
        bri_ok = 1 - np.clip(np.abs(bri - 0.42) / 0.42, 0, 1)
        nota_s = 0.40 * norm(lou) + 0.30 * norm(mov) + 0.20 * np.clip(calor * 3, 0, 1) + 0.10 * bri_ok
        melhores = []
        for s in range(0, max(1, n - JANELA + 1)):
            melhores.append((float(nota_s[s:s + JANELA].mean()), s))
        melhores.sort(reverse=True); escolhidos = []
        for sc, s in melhores:
            if all(abs(s - e) >= JANELA + 2 for _, e in escolhidos): escolhidos.append((sc, s))
            if len(escolhidos) == 3: break
        for sc, s in escolhidos:
            trechos.append({'arquivo': nome, 'inicio_s': s, 'fim_s': s + JANELA, 'nota': round(sc, 3)})
        base = os.path.splitext(nome)[0]
        folha_contato(p, dur, os.path.join(a.saida, f'{base}_folha.jpg'))
        resumo.append({'arquivo': nome, 'duracao_s': round(dur, 1), 'cortes_por_10s': round(10 * len(cortes) / max(dur, 1), 2),
                       'pico_lufs': round(float(lou.max()), 1) if n else None, 'energia_media': round(float(nota_s.mean()), 3) if n else 0,
                       'calor_cor': round(float(calor.mean()), 3), 'multiplo_outlier': round(met.get(nome), 2) if nome in met else '',
                       'melhor_trecho': f"{escolhidos[0][1]}-{escolhidos[0][1]+JANELA}s" if escolhidos else ''})
        json.dump({'loudness': lou.round(1).tolist(), 'movimento': mov.round(4).tolist(), 'calor': calor.round(3).tolist(),
                   'nota': nota_s.round(3).tolist(), 'cortes': cortes}, open(os.path.join(a.saida, f'{base}_curvas.json'), 'w'))
    resumo.sort(key=lambda r: r['energia_media'], reverse=True)
    trechos.sort(key=lambda r: r['nota'], reverse=True)
    for nomearq, rows in (('resumo.csv', resumo), ('trechos.csv', trechos)):
        if rows:
            w = csv.DictWriter(open(os.path.join(a.saida, nomearq), 'w', newline='', encoding='utf-8'), fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    print(json.dumps({'videos': len(resumo), 'top_trechos': trechos[:12]}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
