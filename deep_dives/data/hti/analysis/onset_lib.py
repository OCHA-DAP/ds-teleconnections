import numpy as np, pandas as pd
def onset_dryspell(r, doy, start_doy=60, end_doy=181, acc=20.0, wet=1.0, false_window=30, false_len=10):
    """r: daily rain for one season (1D array aligned with doy). Agronomic onset: first day d>=start (1 Mar = doy 60 in non-leap)
    with 3-day total >= acc, first day wet, and no dry spell >= false_len days in the next false_window days.
    Returns onset doy (nan if none by end), max dry spell (days <wet) from onset (or start) to end, n dry spells >=10 d,
    total start..end, n false starts."""
    n=len(r); on=np.nan; fs=0
    for i in range(n):
        if doy[i]<start_doy or doy[i]>end_doy-3: continue
        if r[i]>=wet and r[i:i+3].sum()>=acc:
            w=r[i+3:i+3+false_window]; run=0; bad=False
            for v in w:
                run = run+1 if v<wet else 0
                if run>=false_len: bad=True; break
            if bad: fs+=1; continue
            on=doy[i]; break
    s = int(np.argmax(doy>=on)) if not np.isnan(on) else int(np.argmax(doy>=start_doy))
    e = int(np.argmax(doy>end_doy)) if (doy>end_doy).any() else n
    seg=r[s:e]; runs=[]; run=0
    for v in seg:
        if v<wet: run+=1
        else:
            if run: runs.append(run)
            run=0
    if run: runs.append(run)
    full=r[int(np.argmax(doy>=start_doy)):e]
    return dict(onset=on, max_dry=max(runs) if runs else 0, n_dry10=sum(1 for x in runs if x>=10), total=float(full.sum()), false_starts=fs)
