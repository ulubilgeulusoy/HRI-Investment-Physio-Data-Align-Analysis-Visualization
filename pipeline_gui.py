
import os, shutil, traceback, threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np, pandas as pd, matplotlib.pyplot as plt, pyxdf, neurokit2 as nk
from scipy.signal import butter, filtfilt, savgol_filter
import re

class App:
    def __init__(self, root):
        self.r=root; self.r.title('HRI Physio Pipeline GUI'); self.r.geometry('1160x820')
        self.repo=Path(__file__).resolve().parent
        self.raw=self.repo/'raw_xdf_acq_Data'; self.aln=self.repo/'aligned_Data'; self.cln=self.repo/'aligned_cleaned_Data'; self.fea=self.repo/'feature_extracted_Data'
        for d in [self.raw,self.aln,self.cln,self.fea]: d.mkdir(parents=True, exist_ok=True)
        self.main_thread=threading.current_thread(); self.busy=False
        self.acq=tk.StringVar(); self.xdf=tk.StringVar(); self.aligned=tk.StringVar(); self.cleaned=tk.StringVar(); self.csv=tk.StringVar(); self.status=tk.StringVar(value='Ready.')
        self.progress=tk.DoubleVar(value=0.0); self.progress_text=tk.StringVar(value='0%')
        self.flat_win=tk.DoubleVar(value=2.0); self.flat_rel=tk.DoubleVar(value=0.02); self.tol=tk.DoubleVar(value=0.05); self.include=tk.BooleanVar(value=True)
        self.scope=tk.StringVar(value='all'); self.xcol=tk.StringVar(); self.ycol=tk.StringVar(); self.mcol=tk.StringVar(); self.mstart=tk.StringVar(); self.mend=tk.StringVar()
        self.ov_csv_col=tk.StringVar(); self.ov_stream=tk.StringVar(); self.ov_xdf_col=tk.StringVar()
        self.ov_streams=[]; self.ov_stream_map={}
        self.df=None; self.df_f=None
        self.ui()

    def ui(self):
        container=ttk.Frame(self.r); container.pack(fill='both',expand=True)
        canvas=tk.Canvas(container,highlightthickness=0)
        vbar=ttk.Scrollbar(container,orient='vertical',command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side='right',fill='y'); canvas.pack(side='left',fill='both',expand=True)
        m=ttk.Frame(canvas,padding=10)
        w=canvas.create_window((0,0),window=m,anchor='nw')
        m.bind('<Configure>',lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',lambda e: canvas.itemconfigure(w,width=e.width))
        canvas.bind_all('<MouseWheel>',lambda e: canvas.yview_scroll(int(-1*(e.delta/120)),'units'))
        ttk.Label(m,text='HRI-Investment Physio Pipeline',font=('Segoe UI',14,'bold')).pack(anchor='w')
        ttk.Label(m,text=f'raw={self.raw.name}, aligned={self.aln.name}, cleaned={self.cln.name}, features={self.fea.name}',foreground='#444').pack(anchor='w',pady=(0,8))
        f1=ttk.LabelFrame(m,text='1) Raw import',padding=8); f1.pack(fill='x',pady=4)
        ttk.Label(f1,text='ACQ').grid(row=0,column=0,sticky='w'); ttk.Entry(f1,textvariable=self.acq,width=95).grid(row=0,column=1,sticky='we',padx=6); ttk.Button(f1,text='Browse',command=self.pick_acq).grid(row=0,column=2)
        ttk.Label(f1,text='XDF').grid(row=1,column=0,sticky='w'); ttk.Entry(f1,textvariable=self.xdf,width=95).grid(row=1,column=1,sticky='we',padx=6); ttk.Button(f1,text='Browse',command=self.pick_xdf).grid(row=1,column=2)
        r=ttk.Frame(f1); r.grid(row=2,column=1,sticky='w',pady=6)
        ttk.Button(r,text='Import/Copy Raw Files',command=self.import_raw).pack(side='left',padx=(0,8))
        self.btn_refresh=ttk.Button(r,text='Load Latest From Folders',command=self.refresh); self.btn_refresh.pack(side='left')
        f1.columnconfigure(1,weight=1)

        f2=ttk.LabelFrame(m,text='2) Pipeline',padding=8); f2.pack(fill='x',pady=4)
        ttk.Label(f2,text='Flat win(s)').grid(row=0,column=0,sticky='w'); ttk.Entry(f2,textvariable=self.flat_win,width=9).grid(row=0,column=1,sticky='w',padx=6)
        ttk.Label(f2,text='Flat rel').grid(row=0,column=2,sticky='w'); ttk.Entry(f2,textvariable=self.flat_rel,width=9).grid(row=0,column=3,sticky='w',padx=6)
        ttk.Label(f2,text='ASOF tol(s)').grid(row=0,column=4,sticky='w'); ttk.Entry(f2,textvariable=self.tol,width=9).grid(row=0,column=5,sticky='w',padx=6)
        ttk.Checkbutton(f2,text='Include non-physio XDF streams',variable=self.include).grid(row=1,column=0,columnspan=3,sticky='w')
        ttk.Label(f2,text='Feature extraction is marker-based in a separate window.').grid(row=2,column=0,columnspan=6,sticky='w')
        b=ttk.Frame(f2); b.grid(row=3,column=0,columnspan=6,sticky='w',pady=6)
        self.btn_align=ttk.Button(b,text='Run Alignment',command=self.run_align_btn); self.btn_align.pack(side='left',padx=(0,8))
        self.btn_clean=ttk.Button(b,text='Run Cleaning',command=self.run_clean_btn); self.btn_clean.pack(side='left',padx=(0,8))
        self.btn_feature=ttk.Button(b,text='Open Feature GUI',command=self.open_feature_gui); self.btn_feature.pack(side='left',padx=(0,8))
        self.btn_all=ttk.Button(b,text='Run Full Pipeline (Align+Clean)',command=self.run_all); self.btn_all.pack(side='left')
        ttk.Label(f2,text='Aligned CSV').grid(row=4,column=0,sticky='w'); ttk.Entry(f2,textvariable=self.aligned,width=95).grid(row=4,column=1,columnspan=5,sticky='we',padx=6)
        ttk.Label(f2,text='Cleaned CSV').grid(row=5,column=0,sticky='w'); ttk.Entry(f2,textvariable=self.cleaned,width=95).grid(row=5,column=1,columnspan=5,sticky='we',padx=6)
        f3=ttk.LabelFrame(m,text='3) Visualizers',padding=8); f3.pack(fill='x',pady=4)
        ttk.Label(f3,text='CSV').grid(row=0,column=0,sticky='w'); ttk.Entry(f3,textvariable=self.csv,width=95).grid(row=0,column=1,sticky='we',padx=6); ttk.Button(f3,text='Browse',command=self.pick_csv).grid(row=0,column=2)
        ttk.Button(f3,text='Load CSV Columns',command=self.load_cols).grid(row=1,column=1,sticky='w',pady=4)
        ttk.Label(f3,text='Scope').grid(row=2,column=0,sticky='w')
        s=ttk.Frame(f3); s.grid(row=2,column=1,sticky='w'); ttk.Radiobutton(s,text='All',value='all',variable=self.scope,command=self.scope_ui).pack(side='left',padx=(0,12)); ttk.Radiobutton(s,text='Between markers',value='markers',variable=self.scope,command=self.scope_ui).pack(side='left')
        self.ml=ttk.Label(f3,text='Marker col'); self.mc=ttk.Combobox(f3,textvariable=self.mcol,state='readonly'); self.mc.bind('<<ComboboxSelected>>',lambda e:self.marker_vals())
        self.sl=ttk.Label(f3,text='Start marker'); self.sc=ttk.Combobox(f3,textvariable=self.mstart,state='readonly')
        self.el=ttk.Label(f3,text='End marker'); self.ec=ttk.Combobox(f3,textvariable=self.mend,state='readonly')
        ttk.Label(f3,text='X').grid(row=6,column=0,sticky='w'); self.xc=ttk.Combobox(f3,textvariable=self.xcol,state='readonly'); self.xc.grid(row=6,column=1,sticky='we',padx=6)
        ttk.Label(f3,text='Y').grid(row=7,column=0,sticky='w'); self.yc=ttk.Combobox(f3,textvariable=self.ycol,state='readonly'); self.yc.grid(row=7,column=1,sticky='we',padx=6)
        q=ttk.Frame(f3); q.grid(row=8,column=1,sticky='w',pady=6)
        ttk.Button(q,text='Plot X-Y',command=lambda:self.plot_xy('line')).pack(side='left',padx=(0,8)); ttk.Button(q,text='Scatter',command=lambda:self.plot_xy('scatter')).pack(side='left',padx=(0,8)); ttk.Button(q,text='Export Selected Scope CSV',command=self.export_scope).pack(side='left',padx=(0,8))

        ov=ttk.LabelFrame(f3,text='Overlay (Configurable)',padding=6); ov.grid(row=9,column=0,columnspan=3,sticky='we',pady=(8,2))
        ttk.Button(ov,text='Load Overlay Options',command=self.load_overlay_options).grid(row=0,column=0,sticky='w',padx=(0,12))
        ttk.Label(ov,text='CSV signal').grid(row=0,column=1,sticky='w'); self.ov_csv=ttk.Combobox(ov,textvariable=self.ov_csv_col,state='readonly',width=38); self.ov_csv.grid(row=0,column=2,sticky='we',padx=6)
        ttk.Label(ov,text='XDF stream').grid(row=1,column=1,sticky='w'); self.ov_st=ttk.Combobox(ov,textvariable=self.ov_stream,state='readonly',width=38); self.ov_st.grid(row=1,column=2,sticky='we',padx=6); self.ov_st.bind('<<ComboboxSelected>>',lambda e:self.overlay_stream_changed())
        ttk.Label(ov,text='XDF signal').grid(row=2,column=1,sticky='w'); self.ov_xdf=ttk.Combobox(ov,textvariable=self.ov_xdf_col,state='readonly',width=38); self.ov_xdf.grid(row=2,column=2,sticky='we',padx=6)
        ttk.Button(ov,text='Plot Selected Overlay',command=self.overlay_selected).grid(row=2,column=0,sticky='w',padx=(0,12))
        ov.columnconfigure(2,weight=1)
        f3.columnconfigure(1,weight=1); self.scope_ui()
        pr=ttk.Frame(m); pr.pack(fill='x',pady=(6,2))
        ttk.Label(pr,text='Progress').pack(side='left',padx=(0,8))
        self.pbar=ttk.Progressbar(pr,mode='determinate',maximum=100,variable=self.progress); self.pbar.pack(side='left',fill='x',expand=True)
        ttk.Label(pr,textvariable=self.progress_text,width=8).pack(side='left',padx=(8,0))
        ttk.Label(m,textvariable=self.status,foreground='#1f4f8a').pack(anchor='w',pady=3)

    def set(self,t):
        if threading.current_thread() is not self.main_thread:
            self.r.after(0, lambda: self.set(t)); return
        self.status.set(t); self.r.update_idletasks()

    def set_progress(self,p,msg=None):
        if threading.current_thread() is not self.main_thread:
            self.r.after(0, lambda: self.set_progress(p,msg)); return
        p=max(0.0,min(100.0,float(p))); self.progress.set(p); self.progress_text.set(f'{p:.0f}%')
        if msg is not None: self.status.set(msg)
        self.r.update_idletasks()

    def set_busy(self,is_busy):
        if threading.current_thread() is not self.main_thread:
            self.r.after(0, lambda: self.set_busy(is_busy)); return
        self.busy=bool(is_busy)
        state='disabled' if self.busy else 'normal'
        for b in [self.btn_align,self.btn_clean,self.btn_feature,self.btn_all]:
            b.configure(state=state)

    def _bg(self, task, on_ok, label):
        if self.busy:
            messagebox.showwarning('Busy', 'A pipeline task is already running.'); return
        self.set_busy(True); self.set_progress(0, f'{label} started...')
        def worker():
            try:
                result=task()
                self.r.after(0, lambda: on_ok(result))
            except Exception as e:
                err=f'{e}\n\n{traceback.format_exc()}'
                self.r.after(0, lambda: messagebox.showerror(f'{label} Error', err))
                self.r.after(0, lambda: self.set_progress(0, f'{label} failed.'))
            finally:
                self.r.after(0, lambda: self.set_busy(False))
        threading.Thread(target=worker, daemon=True).start()
    def pick_acq(self):
        p=filedialog.askopenfilename(initialdir=str(self.raw),filetypes=[('ACQ','*.acq'),('All','*.*')])
        self.acq.set(p) if p else None
    def pick_xdf(self):
        p=filedialog.askopenfilename(initialdir=str(self.raw),filetypes=[('XDF','*.xdf'),('All','*.*')])
        self.xdf.set(p) if p else None
    def pick_csv(self):
        base=self.cln if self.cln.exists() else self.aln
        p=filedialog.askopenfilename(initialdir=str(base),filetypes=[('CSV','*.csv'),('All','*.*')])
        self.csv.set(p) if p else None

    def refresh(self):
        for lst,var in [(sorted(self.raw.glob('*.acq'),key=lambda p:p.stat().st_mtime,reverse=True),self.acq),(sorted(self.raw.glob('*.xdf'),key=lambda p:p.stat().st_mtime,reverse=True),self.xdf),(sorted(self.aln.glob('*.csv'),key=lambda p:p.stat().st_mtime,reverse=True),self.aligned),(sorted(self.cln.glob('*.csv'),key=lambda p:p.stat().st_mtime,reverse=True),self.cleaned)]:
            if lst: var.set(str(lst[0]))
        if self.cleaned.get(): self.csv.set(self.cleaned.get())
        elif self.aligned.get(): self.csv.set(self.aligned.get())

    def import_raw(self):
        a=self.acq.get().strip(); x=self.xdf.get().strip()
        if not (os.path.isfile(a) and os.path.isfile(x)): return messagebox.showerror('Error','Select valid ACQ and XDF files.')
        ad=self.raw/Path(a).name; xd=self.raw/Path(x).name; shutil.copy2(a,ad); shutil.copy2(x,xd); self.acq.set(str(ad)); self.xdf.set(str(xd)); self.set('Raw files copied.'); messagebox.showinfo('Imported',f'Copied:\n{ad}\n{xd}')

    def _flat_idx(self, sig, sr):
        x=np.asarray(sig,float); w=max(1,int(round(self.flat_win.get()*sr)))
        if len(x)<w+2: return None
        thr=max(1e-4,self.flat_rel.get()*float(np.std(x))); c1=np.cumsum(x); c2=np.cumsum(x*x)
        m=(c1[w:]-c1[:-w])/w; v=(c2[w:]-c2[:-w])/w-m*m; v[v<0]=0; s=np.sqrt(v); idx=np.where(s<thr)[0]
        return int(idx[0]) if len(idx) else None

    def _clean(self,s):
        s=str(s).strip().replace(' ','_'); s=''.join(ch for ch in s if ch.isalnum() or ch in ['_','-']); return s or 'ch'

    def _phys(self, meta):
        for i,n,t,c,r in meta:
            z=(n+' '+t).lower()
            if any(k in z for k in ['biopac','phys','ecg','rsp','eda','gsr']): return i
        return max(meta,key=lambda x:x[3])[0] if meta else None

    def run_align_btn(self):
        def ok(p):
            self.aligned.set(str(p)); self.csv.set(str(p)); self.set_progress(100,'Alignment done.'); messagebox.showinfo('Alignment',f'Wrote:\n{p}')
        self._bg(self.run_align, ok, 'Alignment')
    def run_align(self):
        a=self.acq.get().strip(); x=self.xdf.get().strip()
        if not (os.path.isfile(a) and os.path.isfile(x)): raise RuntimeError('Select valid ACQ and XDF first.')
        self.set_progress(5,'Alignment: loading ACQ...'); adf, asr = nk.read_acqknowledge(a); asr=float(asr)
        adf=adf.rename(columns={'RSP, X, RSPEC-R':'RSP','RSP, Y, RSPEC-R':'RSP','EDA, X, PPGED-R':'EDA','EDA, Y, PPGED-R':'EDA','ECG, X, RSPEC-R':'ECG','ECG, Y, RSPEC-R':'ECG','DTU100 - Trigger View, AMI / HLT - A11':'TRIG'})
        self.set_progress(12,'Alignment: loading XDF...'); streams,_=pyxdf.load_xdf(x, dejitter_timestamps=False)
        meta=[]
        for i,st in enumerate(streams): info=st['info']; meta.append((i,info.get('name',[''])[0],info.get('type',[''])[0],int(info.get('channel_count',['0'])[0]),float(info.get('nominal_srate',['0'])[0] or 0.0)))
        if not meta: raise RuntimeError('No streams in XDF.')
        pi=self._phys(meta); ps=streams[pi]; xr=pd.DataFrame(ps['time_series']); xt=np.asarray(ps['time_stamps'],float); xt0=xt-xt[0]
        dt=np.diff(xt0); dt=dt[(dt>0)&(dt<1.0)]; xsr=float(1.0/np.median(dt)) if len(dt) else None
        if xsr is None: raise RuntimeError('Could not estimate XDF SR.')
        if xr.shape[1]==2: xr.columns=['ECG','RSP']
        elif xr.shape[1]>=3: xr.columns=['RSP','EDA','ECG']+[f'XDF_{i}' for i in range(3,xr.shape[1])]
        else: xr.columns=['ECG']
        xa=None; xc=None
        for c in [k for k in ['ECG','RSP','EDA'] if k in xr.columns]+[k for k in xr.columns if k not in ['ECG','RSP','EDA']]:
            i=self._flat_idx(xr[c].values, xsr)
            if i is not None: xa=float(xt0[i]); xc=c; break
        if xa is None: raise RuntimeError('No flatline found in XDF physio stream.')
        aa=None
        order=[xc] if xc in adf.columns else []; order += [c for c in ['ECG','RSP','EDA'] if c in adf.columns and c not in order]; order += [c for c in adf.columns if c not in order]
        for c in order:
            i=self._flat_idx(adf[c].values, asr)
            if i is not None: aa=float(i/asr); break
        if aa is None: raise RuntimeError('No matching flatline in ACQ.')
        t_opt=aa-xa; total=float(xt0[-1]); at=np.arange(len(adf),dtype=float)/asr; ash=at-t_opt; si=int(np.sum(ash<0)); ei=int(np.sum(ash<total))
        if ei<=si+2: raise RuntimeError('After trim ACQ too short.')
        trim=adf.iloc[si:ei].copy(); at_abs=ash[si:ei]+float(xt[0]); out=pd.DataFrame({'time':at_abs})
        for c in trim.columns: out[f'ACQ_{self._clean(c)}']=trim[c].values
        if self.include.get():
            base=out['time'].values.astype(float); t0=float(base[0]); t1=float(base[-1]); tol=self.tol.get()
            total_streams=max(1,len(meta))
            for idx_meta,(i,n,t,c,sr) in enumerate(meta):
                self.set_progress(30 + 60 * ((idx_meta + 1) / total_streams), f'Alignment: merging stream {idx_meta+1}/{total_streams}...')
                st=streams[i]; ts=np.asarray(st['time_stamps'],float)
                if len(ts)<2: continue
                m=(ts>=t0)&(ts<=t1)
                if np.sum(m)<1:
                    ys_all=np.asarray(st['time_series'])
                    n_ch=int(ys_all.shape[1]) if ys_all.ndim>1 else 1
                    labs=None
                    try:
                        ch=st['info']['desc'][0].get('channels',[{}])[0].get('channel',[])
                        labs=[(z.get('label',z.get('name',['']))[0] if isinstance(z.get('label',z.get('name',[''])),list) else z.get('label',z.get('name',''))) for z in ch]
                    except Exception:
                        labs=None
                    cols=[f'XDF_{self._clean(n)}_{self._clean(l)}' for l in labs] if labs is not None and len(labs)==n_ch else [f'XDF_{self._clean(n)}_ch{j}' for j in range(n_ch)]
                    for cn in cols: out[cn]=np.nan
                    continue
                ts=ts[m]; ys=np.asarray(st['time_series'])[m]
                labs=None
                try:
                    ch=st['info']['desc'][0].get('channels',[{}])[0].get('channel',[]); labs=[(z.get('label',z.get('name',['']))[0] if isinstance(z.get('label',z.get('name',[''])),list) else z.get('label',z.get('name',''))) for z in ch]
                except Exception: labs=None
                cols=[f'XDF_{self._clean(n)}_{self._clean(l)}' for l in labs] if labs is not None and len(labs)==ys.shape[1] else [f'XDF_{self._clean(n)}_ch{j}' for j in range(ys.shape[1])]
                if float(sr)>0 and ys.shape[0]>=3:
                    o=np.argsort(ts); ts2=ts[o]; ys2=ys[o]; u=np.concatenate([[True],np.diff(ts2)>0]); ts2=ts2[u]; ys2=ys2[u]
                    try:
                        ys2=ys2.astype(float); inter=np.column_stack([np.interp(base,ts2,ys2[:,j]) for j in range(ys2.shape[1])]);
                        for j,cn in enumerate(cols): out[cn]=inter[:,j]
                    except Exception:
                        d=pd.DataFrame({'time':ts2});
                        for j,cn in enumerate(cols): d[cn]=ys2[:,j]
                        out=pd.merge_asof(out.sort_values('time'),d.sort_values('time'),on='time',direction='nearest',tolerance=tol)
                else:
                    d=pd.DataFrame({'time':ts});
                    for j,cn in enumerate(cols): d[cn]=ys[:,j]
                    out=pd.merge_asof(out.sort_values('time'),d.sort_values('time'),on='time',direction='nearest',tolerance=tol)
            out=out.sort_values('time').reset_index(drop=True)
        self.set_progress(95,'Alignment: finalizing output...')
        out['timestamp']=out['time']; out['time_sec']=out['time']-float(out['time'].iloc[0])
        op=self.aln/f"{self._clean(Path(a).stem+'__'+Path(x).stem)}_aligned_dropout_merged.csv"; out.to_csv(op,index=False); return op

    def _bf(self,x,fs,kind,freq,order=4):
        x=np.asarray(x,float); ny=0.5*fs
        if kind in ['highpass','lowpass']:
            wn=min(max(float(freq)/ny,1e-6),0.999999); b,a=butter(order,wn,btype=kind); return filtfilt(b,a,x)
        lo,hi=freq; lo=min(max(lo/ny,1e-6),0.999999); hi=min(max(hi/ny,1e-6),0.999999)
        if hi<=lo: return x
        b,a=butter(order,[lo,hi],btype=('bandpass' if kind=='bandpass' else 'bandstop')); return filtfilt(b,a,x)
    def run_clean_btn(self):
        def ok(p):
            self.cleaned.set(str(p)); self.csv.set(str(p)); self.set_progress(100,'Cleaning done.'); messagebox.showinfo('Cleaning',f'Wrote:\n{p}')
        self._bg(self.run_clean, ok, 'Cleaning')

    def run_clean(self):
        p=self.aligned.get().strip() or (str(sorted(self.aln.glob('*.csv'),key=lambda z:z.stat().st_mtime,reverse=True)[0]) if list(self.aln.glob('*.csv')) else '')
        if not p: raise RuntimeError('No aligned CSV found.')
        self.set_progress(10,'Cleaning: loading aligned CSV...')
        df=pd.read_csv(p); tcol=next((c for c in ['time','Time','timestamp','Timestamp'] if c in df.columns),None)
        if not tcol: raise RuntimeError('No time column in aligned CSV.')
        t=np.asarray(df[tcol],float); fs=float(len(t)/max(t[-1]-t[0],1e-9))
        ecg=next((c for c in ['ECG','ACQ_ECG'] if c in df.columns),None); rsp=next((c for c in ['RSP','ACQ_RSP'] if c in df.columns),None); eda=next((c for c in ['EDA','ACQ_EDA'] if c in df.columns),None)
        if ecg:
            self.set_progress(35,'Cleaning: ECG...')
            y=self._bf(df[ecg].to_numpy(float),fs,'highpass',1.0,4); y=self._bf(y,fs,'lowpass',100.0,4); df['ECG_clean']=self._bf(y,fs,'bandstop',(59.0,61.0),2)
        if rsp:
            self.set_progress(60,'Cleaning: RSP...')
            df['RSP_clean']=self._bf(df[rsp].to_numpy(float),fs,'bandpass',(0.05,3.0),2)
        if eda:
            self.set_progress(80,'Cleaning: EDA...')
            x=df[eda].to_numpy(float).copy(); x[(x>40)|(x<5)]=np.nan; d1=np.gradient(x); d2=np.gradient(d1); x[np.abs(d1)>0.5]=np.nan; x[np.abs(d2)>0.5]=np.nan
            f=pd.Series(x).interpolate(method='linear',limit_direction='both').to_numpy(float); n=len(f); w=min(2001,n if n%2==1 else n-1); df['EDA_clean']=f if w<5 else savgol_filter(f,window_length=w,polyorder=3)
        self.set_progress(95,'Cleaning: writing output...')
        op=self.cln/Path(p).name; df.to_csv(op,index=False); return op

    def open_feature_gui(self):
        cpath=self.cleaned.get().strip()
        if not cpath or not Path(cpath).is_file():
            cands=sorted(self.cln.glob('*.csv'),key=lambda p:p.stat().st_mtime,reverse=True)
            if cands:
                cpath=str(cands[0])
            else:
                cpath=self.csv.get().strip()
        if not cpath or not Path(cpath).is_file():
            return messagebox.showerror('Feature GUI', 'No valid cleaned CSV found. Run cleaning first or load latest folders.')

        ld=tk.Toplevel(self.r)
        ld.title('Loading Feature GUI')
        ld.geometry('420x120')
        ld.transient(self.r)
        ld.grab_set()
        ttk.Label(ld,text='Loading markers and preparing Feature GUI...').pack(anchor='w',padx=12,pady=(12,8))
        pb=ttk.Progressbar(ld,mode='indeterminate'); pb.pack(fill='x',padx=12,pady=(0,6)); pb.start(10)
        status=tk.StringVar(value='Reading cleaned CSV...')
        ttk.Label(ld,textvariable=status,foreground='#1f4f8a').pack(anchor='w',padx=12)
        ld.update_idletasks()

        def worker():
            try:
                df=pd.read_csv(cpath)
                marker_cols=[c for c in df.columns if 'marker' in c.lower() or 'event' in c.lower()]
                def done():
                    try:
                        pb.stop()
                        ld.grab_release()
                        ld.destroy()
                    except Exception:
                        pass
                    fw=FeatureWindow(self.r, self.cln, self.fea, cleaned_default=cpath, preload_df=df, preload_marker_cols=marker_cols)
                    if not marker_cols:
                        messagebox.showwarning('Feature GUI', 'Loaded CSV, but no marker/event columns were found.', parent=fw.top)
                self.r.after(0, done)
            except Exception as e:
                def fail():
                    try:
                        pb.stop()
                        ld.grab_release()
                        ld.destroy()
                    except Exception:
                        pass
                    messagebox.showerror('Feature GUI Load Error', f'{e}\n\n{traceback.format_exc()}')
                self.r.after(0, fail)

        threading.Thread(target=worker, daemon=True).start()

    def run_all(self):
        def task():
            self.set_progress(2,'Full pipeline: alignment...')
            a=self.run_align(); self.r.after(0, lambda: self.aligned.set(str(a)))
            self.set_progress(45,'Full pipeline: cleaning...')
            c=self.run_clean(); self.r.after(0, lambda: self.cleaned.set(str(c))); self.r.after(0, lambda: self.csv.set(str(c)))
            return (a,c)
        def ok(res):
            a,c=res; self.set_progress(100,'Full pipeline complete (align+clean).'); messagebox.showinfo('Done',f'Alignment:\n{a}\n\nCleaning:\n{c}\n\nNext: click Open Feature GUI for marker-based extraction.')
        self._bg(task, ok, 'Full Pipeline')
    def scope_ui(self):
        for w in [self.ml,self.mc,self.sl,self.sc,self.el,self.ec]: w.grid_remove()
        if self.scope.get()=='markers':
            self.ml.grid(row=3,column=0,sticky='w'); self.mc.grid(row=3,column=1,sticky='we',padx=6)
            self.sl.grid(row=4,column=0,sticky='w'); self.sc.grid(row=4,column=1,sticky='we',padx=6)
            self.el.grid(row=5,column=0,sticky='w'); self.ec.grid(row=5,column=1,sticky='we',padx=6)

    def load_cols(self):
        p=self.csv.get().strip()
        if not os.path.isfile(p): return messagebox.showerror('Error','Select valid CSV.')
        def task():
            self.set_progress(10,'Loading CSV...')
            df=pd.read_csv(p)
            self.set_progress(70,'Parsing columns...')
            if 'time_sec' not in df.columns and 'timestamp' in df.columns:
                ts=pd.to_numeric(df['timestamp'],errors='coerce'); first=ts.dropna()
                if not first.empty: df['time_sec']=ts-first.iloc[0]
            cols=list(df.columns); nums=[c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
            mk=[c for c in cols if 'marker' in c.lower() or 'event' in c.lower()]
            return (df,cols,nums,mk)
        def ok(res):
            df,cols,nums,mk=res
            self.df=df
            self.xc['values']=cols; self.yc['values']=nums if nums else cols
            self.xcol.set('time_sec' if 'time_sec' in cols else ('time' if 'time' in cols else cols[0]))
            self.ycol.set(([c for c in nums if c!=self.xcol.get()] or nums or cols)[0])
            self.mc['values']=mk
            if mk: self.mcol.set(mk[0]); self.marker_vals()
            self.df_f=None; self.set_progress(100,f'Loaded {len(cols)} columns.')
        self._bg(task, ok, 'Load CSV')

    def marker_vals(self):
        if self.df is None: return
        c=self.mcol.get().strip()
        if c not in self.df.columns: return
        v=self.df[c].dropna().astype(str); v=v[v.str.strip()!='']; u=list(dict.fromkeys(v.tolist()))
        self.sc['values']=u; self.ec['values']=u
        if u and not self.mstart.get(): self.mstart.set(u[0])
        if u and not self.mend.get(): self.mend.set(u[min(1,len(u)-1)])

    def active_df(self):
        if self.df is None: raise RuntimeError('Load CSV first.')
        if self.scope.get()=='all': return self.df
        c=self.mcol.get().strip(); s=self.mstart.get().strip(); e=self.mend.get().strip()
        if c not in self.df.columns or not s or not e: raise RuntimeError('Select marker column/start/end.')
        m=self.df[c].fillna('').astype(str).str.strip(); sh=self.df.index[m==s].tolist()
        if not sh: raise RuntimeError(f'Start marker not found: {s}')
        i0=sh[0]; eh=self.df.index[(m==e)&(self.df.index>i0)].tolist()
        if not eh: raise RuntimeError(f"No end marker '{e}' after '{s}'")
        self.df_f=self.df.loc[i0:eh[0]].copy(); return self.df_f

    def plot_xy(self, style='line'):
        try:
            if self.df is None: self.load_cols()
            d=self.active_df(); x=pd.to_numeric(d[self.xcol.get().strip()],errors='coerce'); y=pd.to_numeric(d[self.ycol.get().strip()],errors='coerce'); mask=x.notna()&y.notna()
            if not mask.any(): raise RuntimeError('No numeric points for X/Y.')
            plt.figure(figsize=(10,5)); plt.scatter(x[mask],y[mask],s=12,alpha=0.75) if style=='scatter' else plt.plot(x[mask],y[mask],linewidth=1.2)
            plt.xlabel(self.xcol.get()); plt.ylabel(self.ycol.get()); plt.title(f"{Path(self.csv.get()).name}: {self.ycol.get()} vs {self.xcol.get()}"); plt.grid(True,alpha=0.35); plt.tight_layout(); plt.show(); self.set('Plot displayed.')
        except Exception as e: messagebox.showerror('Plot Error',f'{e}\n\n{traceback.format_exc()}')

    def export_scope(self):
        try:
            d=self.active_df(); stem=Path(self.csv.get().strip() or 'selected').stem
            name=f"{stem}_{self._clean(self.mstart.get() or 'start')}_to_{self._clean(self.mend.get() or 'end')}.csv" if self.scope.get()=='markers' else f"{stem}_all_data.csv"
            p=self.cln/name; d.to_csv(p,index=False); self.set(f'Exported {name}'); messagebox.showinfo('Exported',f'Saved:\n{p}')
        except Exception as e: messagebox.showerror('Export Error',str(e))
    def load_overlay_options(self):
        try:
            cp=self.csv.get().strip() or self.aligned.get().strip(); xp=self.xdf.get().strip()
            if not (os.path.isfile(cp) and os.path.isfile(xp)): raise RuntimeError('Select valid CSV and XDF for overlay.')
            def task():
                self.set_progress(10,'Overlay: loading CSV...')
                c=pd.read_csv(cp)
                biopac_names={'ACQ_ECG','ACQ_RSP','ACQ_EDA','ECG','RSP','EDA','ECG_clean','RSP_clean','EDA_clean'}
                csv_cols=[k for k in c.columns if k in biopac_names and pd.api.types.is_numeric_dtype(c[k])]
                self.set_progress(45,'Overlay: loading XDF...')
                streams,_=pyxdf.load_xdf(xp, dejitter_timestamps=False)
                stream_opts=[]; stream_map={}
                for i,st in enumerate(streams):
                    info=st['info']; name=info.get('name',[''])[0]; typ=info.get('type',[''])[0]
                    t=(str(name)+' '+str(typ)).lower()
                    if not any(z in t for z in ['biopac','phys','ecg','rsp','eda','gsr']):
                        continue
                    key=f"{i}:{name} ({typ})"; stream_opts.append(key)
                    data=np.asarray(st['time_series']); d2=pd.DataFrame(data)
                    if d2.shape[1]==2: d2.columns=['ECG','RSP']
                    elif d2.shape[1]>=3: d2.columns=['RSP','EDA','ECG']+[f'XDF_{j}' for j in range(3,d2.shape[1])]
                    else: d2.columns=['ECG']
                    bio_xdf=[k for k in d2.columns if k in ['ECG','RSP','EDA']]
                    stream_map[key]=bio_xdf
                return (csv_cols,stream_opts,stream_map)
            def ok(res):
                csv_cols,stream_opts,stream_map=res
                if not csv_cols:
                    self.set_progress(0,'Overlay options failed.')
                    messagebox.showerror('Overlay Error','No BIOPAC CSV signals found (expected ECG/RSP/EDA style columns).')
                    return
                if not stream_opts:
                    self.set_progress(0,'Overlay options failed.')
                    messagebox.showerror('Overlay Error','No BIOPAC-like XDF stream found.')
                    return
                self.ov_streams=stream_opts; self.ov_stream_map=stream_map
                self.ov_csv['values']=csv_cols; self.ov_st['values']=stream_opts
                if csv_cols and not self.ov_csv_col.get(): self.ov_csv_col.set(csv_cols[0])
                if stream_opts and not self.ov_stream.get():
                    self.ov_stream.set(stream_opts[0]); self.overlay_stream_changed()
                self.set_progress(100,'Overlay options loaded.')
            self._bg(task, ok, 'Load Overlay Options')
        except Exception as e:
            messagebox.showerror('Overlay Error',f'{e}\n\n{traceback.format_exc()}')

    def overlay_stream_changed(self):
        k=self.ov_stream.get().strip()
        cols=self.ov_stream_map.get(k,[])
        self.ov_xdf['values']=cols
        if cols and not self.ov_xdf_col.get():
            self.ov_xdf_col.set(cols[0])

    def overlay_selected(self):
        try:
            cp=self.csv.get().strip() or self.aligned.get().strip(); xp=self.xdf.get().strip()
            if not (os.path.isfile(cp) and os.path.isfile(xp)): raise RuntimeError('Select valid CSV and XDF.')
            ccol=self.ov_csv_col.get().strip(); scol=self.ov_xdf_col.get().strip(); skey=self.ov_stream.get().strip()
            if not ccol or not scol or not skey: raise RuntimeError('Load options and select CSV signal, XDF stream, and XDF signal.')
            c=pd.read_csv(cp)
            if 'time' not in c.columns: raise RuntimeError("Overlay CSV needs 'time' column.")
            sidx=int(skey.split(':',1)[0])
            streams,_=pyxdf.load_xdf(xp, dejitter_timestamps=False); st=streams[sidx]
            xd=pd.DataFrame(np.asarray(st['time_series']))
            if xd.shape[1]==2: xd.columns=['ECG','RSP']
            elif xd.shape[1]>=3: xd.columns=['RSP','EDA','ECG']+[f'XDF_{j}' for j in range(3,xd.shape[1])]
            else: xd.columns=['ECG']
            if ccol not in c.columns or scol not in xd.columns: raise RuntimeError('Selected signal columns not found.')
            tc=c['time'].to_numpy(float); xt=np.asarray(st['time_stamps'],float); t0=float(tc[0]); tt=tc-t0; xt=xt-t0
            o=np.argsort(xt); yi=np.interp(tt,xt[o],pd.to_numeric(xd[scol],errors='coerce').to_numpy(float)[o])
            plt.figure(figsize=(10,4)); plt.plot(tt,pd.to_numeric(c[ccol],errors='coerce').to_numpy(float),label=f'CSV ({ccol})'); plt.plot(tt,yi,label=f'XDF ({scol}) interpolated',alpha=0.85)
            plt.title('Selected overlay'); plt.xlabel('Time (s)'); plt.legend(); plt.tight_layout(); plt.show()
        except Exception as e:
            messagebox.showerror('Overlay Error',f'{e}\n\n{traceback.format_exc()}')


class FeatureWindow:
    EMO_COLS=[
        'XDF_OpenFaceRealtime_emo_Neutral_pct','XDF_OpenFaceRealtime_emo_Happy_pct','XDF_OpenFaceRealtime_emo_Sad_pct','XDF_OpenFaceRealtime_emo_Surprise_pct',
        'XDF_OpenFaceRealtime_emo_Fear_pct','XDF_OpenFaceRealtime_emo_Disgust_pct','XDF_OpenFaceRealtime_emo_Anger_pct','XDF_OpenFaceRealtime_emo_Contempt_pct'
    ]
    def __init__(self, parent, cleaned_dir: Path, features_dir: Path, cleaned_default='', preload_df=None, preload_marker_cols=None):
        self.top=tk.Toplevel(parent); self.top.title('Marker-Based Feature Extraction'); self.top.geometry('980x560')
        self.cleaned_dir=Path(cleaned_dir); self.features_dir=Path(features_dir); self.features_dir.mkdir(parents=True, exist_ok=True)
        self.main_thread=threading.current_thread()
        self.busy=False
        self.csv=tk.StringVar(value=cleaned_default if cleaned_default and Path(cleaned_default).is_file() else '')
        self.marker_col=tk.StringVar(); self.start_marker=tk.StringVar(); self.end_marker=tk.StringVar(); self.segment_label=tk.StringVar(value='segment')
        self.status=tk.StringVar(value='Ready.'); self.progress=tk.DoubleVar(value=0.0); self.progress_text=tk.StringVar(value='0%')
        self.df=None
        self._ui()
        if preload_df is not None:
            self.df=preload_df
            marker_cols=preload_marker_cols if preload_marker_cols is not None else [c for c in self.df.columns if 'marker' in c.lower() or 'event' in c.lower()]
            self.c_marker['values']=marker_cols
            if marker_cols:
                self.marker_col.set(marker_cols[0])
                self.marker_values()

    def _ui(self):
        container=ttk.Frame(self.top); container.pack(fill='both',expand=True)
        canvas=tk.Canvas(container,highlightthickness=0)
        vbar=ttk.Scrollbar(container,orient='vertical',command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side='right',fill='y'); canvas.pack(side='left',fill='both',expand=True)
        m=ttk.Frame(canvas,padding=10)
        w=canvas.create_window((0,0),window=m,anchor='nw')
        m.bind('<Configure>',lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',lambda e: canvas.itemconfigure(w,width=e.width))
        canvas.bind_all('<MouseWheel>',lambda e: canvas.yview_scroll(int(-1*(e.delta/120)),'units'))
        ttk.Label(m,text='Marker-Based Feature Extraction',font=('Segoe UI',12,'bold')).pack(anchor='w')
        f=ttk.LabelFrame(m,text='Input',padding=8); f.pack(fill='x',pady=6)
        ttk.Label(f,text='Cleaned CSV').grid(row=0,column=0,sticky='w'); ttk.Entry(f,textvariable=self.csv,width=90).grid(row=0,column=1,sticky='we',padx=6)
        ttk.Button(f,text='Browse',command=self.pick_csv).grid(row=0,column=2); ttk.Button(f,text='Load Markers',command=self.load_csv).grid(row=1,column=1,sticky='w',pady=5)
        f.columnconfigure(1,weight=1)

        s=ttk.LabelFrame(m,text='Segment Selection',padding=8); s.pack(fill='x',pady=6)
        ttk.Label(s,text='Marker column').grid(row=0,column=0,sticky='w'); self.c_marker=ttk.Combobox(s,textvariable=self.marker_col,state='readonly',width=35); self.c_marker.grid(row=0,column=1,sticky='w',padx=6); self.c_marker.bind('<<ComboboxSelected>>',lambda e:self.marker_values())
        ttk.Label(s,text='Start marker').grid(row=1,column=0,sticky='w'); self.c_start=ttk.Combobox(s,textvariable=self.start_marker,state='readonly',width=35); self.c_start.grid(row=1,column=1,sticky='w',padx=6)
        ttk.Label(s,text='End marker').grid(row=2,column=0,sticky='w'); self.c_end=ttk.Combobox(s,textvariable=self.end_marker,state='readonly',width=35); self.c_end.grid(row=2,column=1,sticky='w',padx=6)
        ttk.Label(s,text='Segment label').grid(row=3,column=0,sticky='w'); ttk.Entry(s,textvariable=self.segment_label,width=36).grid(row=3,column=1,sticky='w',padx=6)
        act=ttk.Frame(s); act.grid(row=4,column=1,sticky='w',pady=8)
        self.btn_compute=ttk.Button(act,text='Compute Selected Segment Features',command=self.compute_segment)
        self.btn_compute.pack(side='left',padx=(0,8))
        self.btn_auto=ttk.Button(act,text='Compute Features Automatically for Experiment Phases',command=self.compute_auto_phases)
        self.btn_auto.pack(side='left')

        rv=ttk.LabelFrame(m,text='Computed Feature Rows',padding=8); rv.pack(fill='both',expand=True,pady=6)
        rr=ttk.Frame(rv); rr.pack(fill='x',pady=(0,6))
        ttk.Button(rr,text='Load Previously Calculated CSV',command=self.load_saved_features).pack(side='left')
        self.result_cols=('segment_label','start_marker','end_marker','mean_hr_bpm','hrv_rmssd_ms','hrv_sdnn_ms','mean_resp_bpm','n_breaths')
        self.tree=ttk.Treeview(rv,columns=self.result_cols,show='headings',height=8)
        for c in self.result_cols:
            self.tree.heading(c,text=c)
            self.tree.column(c,width=120,anchor='w')
        ys=ttk.Scrollbar(rv,orient='vertical',command=self.tree.yview); self.tree.configure(yscrollcommand=ys.set)
        self.tree.pack(side='left',fill='both',expand=True); ys.pack(side='left',fill='y')

        pr=ttk.Frame(m); pr.pack(fill='x',pady=(6,2))
        ttk.Label(pr,text='Progress').pack(side='left',padx=(0,8))
        ttk.Progressbar(pr,mode='determinate',maximum=100,variable=self.progress).pack(side='left',fill='x',expand=True)
        ttk.Label(pr,textvariable=self.progress_text,width=8).pack(side='left',padx=(8,0))
        ttk.Label(m,textvariable=self.status,foreground='#1f4f8a').pack(anchor='w')

    def _fmt(self,v):
        try:
            if isinstance(v,(float,np.floating)):
                if np.isnan(v): return ''
                return f'{float(v):.4f}'
        except Exception:
            pass
        return str(v)

    def _append_tree_row(self, row: dict):
        vals=[self._fmt(row.get(c,'')) for c in self.result_cols]
        self.tree.insert('', 'end', values=vals)

    def load_saved_features(self):
        p=self.features_dir/f"{Path(self.csv.get().strip()).stem}_marker_features.csv"
        if not p.exists():
            return messagebox.showwarning('Not Found',f'No saved feature CSV found yet:\n{p}',parent=self.top)
        try:
            df=pd.read_csv(p)
            for i in self.tree.get_children():
                self.tree.delete(i)
            for _,r in df.iterrows():
                self._append_tree_row(r.to_dict())
            self.set_progress(100,f'Loaded {len(df)} saved feature rows.')
        except Exception as e:
            messagebox.showerror('Load Error',f'{e}\n\n{traceback.format_exc()}',parent=self.top)

    @staticmethod
    def _marker_candidates(df: pd.DataFrame):
        cands=[c for c in df.columns if 'marker' in c.lower() or 'event' in c.lower()]
        cands=sorted(cands, key=lambda c: int(df[c].notna().sum()), reverse=True)
        return cands

    @staticmethod
    def _event_sequence(df: pd.DataFrame, marker_col: str):
        ms=df[marker_col].fillna('').astype(str).str.strip().tolist()
        events=[]
        prev=''
        for i,v in enumerate(ms):
            if not v:
                continue
            if v==prev:
                continue
            events.append((i,v))
            prev=v
        return events

    @staticmethod
    def _norm_marker(v):
        return str(v).strip().lower()

    @staticmethod
    def _is_start_marker(v: str):
        x=FeatureWindow._norm_marker(v)
        return any(k in x for k in ['_start','_begin','_reset','onset'])

    @staticmethod
    def _is_end_marker(v: str):
        x=FeatureWindow._norm_marker(v)
        return any(k in x for k in ['_end','_end_auto','_stop','offset'])

    @staticmethod
    def _phase_key(v: str):
        x=FeatureWindow._norm_marker(v)
        x=x.replace(' ', '_')
        x=re.sub(r'(_end_auto|_end|_stop|_offset|_start|_begin|_reset)$','',x)
        return x

    @staticmethod
    def _phase_token(v: str):
        x=FeatureWindow._norm_marker(v)
        m=re.search(r'_t(\d+)', x)
        if m:
            return f"t{m.group(1)}"
        m2=re.search(r'\bt(\d+)\b', x)
        if m2:
            return f"t{m2.group(1)}"
        return ''

    @staticmethod
    def _phase_type(v: str):
        x=FeatureWindow._norm_marker(v)
        if 'baseline' in x: return 'baseline'
        if 'trial' in x: return 'trial'
        if 'briefing' in x: return 'briefing'
        return ''

    def _detect_phase_pairs(self, df: pd.DataFrame, marker_col: str):
        idx_vals=self._event_sequence(df, marker_col)
        pairs=[]; used_ends=set()
        for pos,(i,mv) in enumerate(idx_vals):
            if not self._is_start_marker(mv):
                continue
            ptype=self._phase_type(mv)
            if ptype not in {'baseline','trial','briefing'}:
                continue
            key=self._phase_key(mv)
            tok=self._phase_token(mv)
            end_pick=None
            for j,m2 in idx_vals[pos+1:]:
                if j in used_ends:
                    continue
                if not self._is_end_marker(m2):
                    continue
                if self._phase_type(m2)!=ptype:
                    continue
                m2_tok=self._phase_token(m2)
                if tok and m2_tok and tok==m2_tok:
                    end_pick=(j,m2); break
                if (not tok) and self._phase_key(m2)==key:
                    end_pick=(j,m2); break
            if end_pick is None:
                continue
            used_ends.add(end_pick[0])
            label=key if key else f'segment_{len(pairs)+1}'
            pairs.append((label, i, mv, end_pick[0], end_pick[1]))
        return pairs

    def compute_auto_phases(self):
        p=self.csv.get().strip()
        if not p or not Path(p).is_file():
            cands=sorted(self.cleaned_dir.glob('*.csv'),key=lambda z:z.stat().st_mtime,reverse=True)
            if cands:
                p=str(cands[0]); self.csv.set(p)
            else:
                self.set_progress(0,'Auto phase setup skipped: no cleaned CSV found.')
                messagebox.showwarning('Auto Phase Features','No cleaned CSV found. Run cleaning first or choose a cleaned CSV in Feature GUI.',parent=self.top)
                return
        def task():
            self.set_progress(5,'Auto feature setup: loading cleaned CSV...')
            df=pd.read_csv(p); self.df=df
            marker_cols=self._marker_candidates(df)
            if not marker_cols:
                raise RuntimeError('No marker/event column found for auto phase extraction.')
            marker_col=marker_cols[0]
            pairs=self._detect_phase_pairs(df, marker_col)
            if not pairs:
                raise RuntimeError('No start/end phase marker pairs detected automatically.')
            n_base=sum(1 for x in pairs if 'baseline' in x[0].lower())
            n_trial=sum(1 for x in pairs if 'trial' in x[0].lower())
            n_brief=sum(1 for x in pairs if 'briefing' in x[0].lower())
            self.set_progress(20,f'Auto feature setup: {len(pairs)} phases (briefing={n_brief}, baseline={n_base}, trial={n_trial})...')
            rows=[]
            n=max(1,len(pairs))
            for k,(label,i0,m0,i1,m1) in enumerate(pairs):
                self.set_progress(20 + 70*((k+1)/n), f'Computing phase {k+1}/{n}: {label}')
                seg=df.loc[i0:i1].copy()
                feats=self._seg_features(seg)
                row={'file':Path(p).name,'segment_label':label,'marker_col':marker_col,'start_marker':m0,'end_marker':m1,'start_idx':int(i0),'end_idx':int(i1)}
                row.update(feats); rows.append(row)
            out=pd.DataFrame(rows)
            out_path=self.features_dir/f"{Path(p).stem}_marker_features.csv"
            self.set_progress(95,'Saving auto phase features...')
            out.to_csv(out_path,index=False)
            return (df, marker_col, out_path, rows)
        def ok(res):
            df, marker_col, out_path, rows=res
            self.df=df
            mcols=self._marker_candidates(df)
            self.c_marker['values']=mcols
            self.marker_col.set(marker_col)
            self.marker_values()
            for i in self.tree.get_children():
                self.tree.delete(i)
            for r in rows:
                self._append_tree_row(r)
            self.set_progress(100, f'Auto phase features ready ({len(rows)} rows).')
        self._bg(task, ok, 'Auto Phase Features')

    def set_progress(self,p,msg=None):
        if threading.current_thread() is not self.main_thread:
            self.top.after(0, lambda: self.set_progress(p,msg)); return
        p=max(0.0,min(100.0,float(p))); self.progress.set(p); self.progress_text.set(f'{p:.0f}%')
        if msg is not None: self.status.set(msg)
        self.top.update_idletasks()

    def set_busy(self,is_busy):
        if threading.current_thread() is not self.main_thread:
            self.top.after(0, lambda: self.set_busy(is_busy)); return
        self.busy=bool(is_busy)
        st=('disabled' if self.busy else 'normal')
        self.btn_compute.configure(state=st)
        self.btn_auto.configure(state=st)

    def _bg(self, task, on_ok, label):
        if self.busy:
            messagebox.showwarning('Busy','Feature computation already running.',parent=self.top); return
        self.set_busy(True); self.set_progress(0,f'{label} started...')
        def worker():
            try:
                result=task()
                self.top.after(0, lambda: on_ok(result))
            except Exception as e:
                self.top.after(0, lambda: messagebox.showerror(f'{label} Error',f'{e}\n\n{traceback.format_exc()}',parent=self.top))
                self.top.after(0, lambda: self.set_progress(0,f'{label} failed.'))
            finally:
                self.top.after(0, lambda: self.set_busy(False))
        threading.Thread(target=worker,daemon=True).start()

    def pick_csv(self):
        p=filedialog.askopenfilename(parent=self.top,initialdir=str(self.cleaned_dir),filetypes=[('CSV','*.csv'),('All','*.*')])
        if p: self.csv.set(p)

    def load_csv(self):
        p=self.csv.get().strip()
        if not Path(p).is_file(): return messagebox.showerror('Error','Select valid cleaned CSV.',parent=self.top)
        def worker():
            try:
                self.set_progress(10,'Loading cleaned CSV...')
                df=pd.read_csv(p); self.df=df
                self.set_progress(60,'Scanning marker columns...')
                marker_cols=[c for c in df.columns if 'marker' in c.lower() or 'event' in c.lower()]
                def on_ok():
                    self.c_marker['values']=marker_cols
                    if marker_cols: self.marker_col.set(marker_cols[0]); self.marker_values()
                    self.set_progress(100,f'Loaded markers from {Path(p).name}.')
                self.top.after(0,on_ok)
            except Exception as e:
                self.top.after(0,lambda: messagebox.showerror('Load Error',f'{e}\n\n{traceback.format_exc()}',parent=self.top))
                self.top.after(0,lambda: self.set_progress(0,'Load failed.'))
        threading.Thread(target=worker,daemon=True).start()

    def marker_values(self):
        if self.df is None: return
        c=self.marker_col.get().strip()
        if c not in self.df.columns: return
        vals=self.df[c].dropna().astype(str); vals=vals[vals.str.strip()!='']; uniq=list(dict.fromkeys(vals.tolist()))
        self.c_start['values']=uniq; self.c_end['values']=uniq
        if uniq and not self.start_marker.get(): self.start_marker.set(uniq[0])
        if uniq and not self.end_marker.get(): self.end_marker.set(uniq[min(1,len(uniq)-1)])

    @staticmethod
    def _seg_features(df_seg: pd.DataFrame):
        if 'time' not in df_seg.columns or len(df_seg)<3: raise RuntimeError("Segment must include 'time' and enough rows.")
        t=df_seg['time'].to_numpy(float); tr=t-t[0]; dt=np.diff(tr); dt=dt[(dt>0)&np.isfinite(dt)]; fs=float(1.0/np.median(dt)) if len(dt) else np.nan
        ecg='ECG_clean' if 'ECG_clean' in df_seg.columns else None
        rsp='RSP_clean' if 'RSP_clean' in df_seg.columns else None
        row={'fs_hz_est':fs,'n_rows':len(df_seg),'seg_start_time':float(t[0]),'seg_end_time':float(t[-1])}
        if ecg:
            x=pd.to_numeric(df_seg[ecg],errors='coerce').to_numpy(float)
            try:
                sig,inf=nk.ecg_process(x,sampling_rate=fs,method='neurokit'); rp=inf.get('ECG_R_Peaks',None)
                if rp is None or len(rp)<2: row.update({'mean_hr_bpm':np.nan,'hrv_rmssd_ms':np.nan,'hrv_sdnn_ms':np.nan,'n_rpeaks':0})
                else:
                    hrv=nk.hrv_time(rp,sampling_rate=fs,show=False); row.update({'mean_hr_bpm':float(np.nanmean(sig['ECG_Rate'].to_numpy(float))),'hrv_rmssd_ms':float(hrv.get('HRV_RMSSD',[np.nan])[0]),'hrv_sdnn_ms':float(hrv.get('HRV_SDNN',[np.nan])[0]),'n_rpeaks':int(len(rp))})
            except Exception: row.update({'mean_hr_bpm':np.nan,'hrv_rmssd_ms':np.nan,'hrv_sdnn_ms':np.nan,'n_rpeaks':0})
        else: row.update({'mean_hr_bpm':np.nan,'hrv_rmssd_ms':np.nan,'hrv_sdnn_ms':np.nan,'n_rpeaks':0})
        if rsp:
            x=pd.to_numeric(df_seg[rsp],errors='coerce').to_numpy(float)
            try:
                rsig,rinf=nk.rsp_process(x,sampling_rate=fs); pk=rinf.get('RSP_Peaks',None); row.update({'mean_resp_bpm':float(np.nanmean(rsig['RSP_Rate'].to_numpy(float))),'n_breaths':int(len(pk) if pk is not None else 0)})
            except Exception: row.update({'mean_resp_bpm':np.nan,'n_breaths':0})
        else: row.update({'mean_resp_bpm':np.nan,'n_breaths':0})
        for c in FeatureWindow.EMO_COLS:
            x=pd.to_numeric(df_seg[c],errors='coerce').to_numpy(float) if c in df_seg.columns else np.array([])
            row['mean_'+c.replace('XDF_OpenFaceRealtime_','')]=float(np.nanmean(x)) if len(x) and np.isfinite(x).any() else np.nan
        return row

    def compute_segment(self):
        if self.df is None: return messagebox.showerror('Error','Load a cleaned CSV first.',parent=self.top)
        c=self.marker_col.get().strip(); s=self.start_marker.get().strip(); e=self.end_marker.get().strip(); label=(self.segment_label.get().strip() or 'segment')
        if c not in self.df.columns or not s or not e: return messagebox.showerror('Error','Select marker column, start marker, and end marker.',parent=self.top)
        m=self.df[c].fillna('').astype(str).str.strip(); sh=self.df.index[m==s].tolist()
        if not sh: return messagebox.showerror('Error',f"Start marker '{s}' not found.",parent=self.top)
        i0=sh[0]; eh=self.df.index[(m==e)&(self.df.index>i0)].tolist()
        if not eh: return messagebox.showerror('Error',f"No end marker '{e}' found after '{s}'.",parent=self.top)
        i1=eh[0]
        def task():
            self.set_progress(12,'Preparing selected segment...')
            seg=self.df.loc[i0:i1].copy()
            self.set_progress(38,'Computing HR/HRV/RSP features...')
            feats=self._seg_features(seg)
            self.set_progress(80,'Saving feature row...')
            row={'file':Path(self.csv.get().strip()).name,'segment_label':label,'marker_col':c,'start_marker':s,'end_marker':e,'start_idx':int(i0),'end_idx':int(i1)}
            row.update(feats); out_row=pd.DataFrame([row])
            out_path=self.features_dir/f"{Path(self.csv.get().strip()).stem}_marker_features.csv"
            if out_path.exists():
                prev=pd.read_csv(out_path); out=pd.concat([prev,out_row],ignore_index=True)
            else:
                out=out_row
            out.to_csv(out_path,index=False)
            return (row,out_path)
        def ok(res):
            row,out_path=res
            self._append_tree_row(row)
            self.set_progress(100,'Feature row saved.')
            messagebox.showinfo('Done',f'Saved marker features to:\n{out_path}',parent=self.top)
        self._bg(task, ok, 'Feature Compute')

def main():
    root=tk.Tk(); App(root); root.mainloop()

if __name__=='__main__':
    main()
