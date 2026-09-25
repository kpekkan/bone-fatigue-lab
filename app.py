"""Run from this directory: python -m shiny run --reload app.py"""
from dataclasses import asdict, replace
from io import StringIO
from math import isfinite, sqrt, pi
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shiny import App, reactive, render, ui
from fatigue_model import (Parameters, solve, trajectory, stress_sweep,
                           growth_rate, format_number)

TEAL, CORAL, INK, GRAY = '#087e83', '#d4613b', '#182b40', '#8996a4'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.labelcolor': INK,
                     'text.color': INK, 'axes.edgecolor': '#b6c1ca',
                     'grid.color': '#e5eaef', 'figure.facecolor': 'white'})

CSS = """
:root { --bs-primary:#087e83; --bs-body-color:#182b40; --bs-body-bg:#f4f7f9; }
body { font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif; }
.bslib-sidebar-layout { border:0; }
.bslib-sidebar-layout>.sidebar { background:#fff; border-right:1px solid #dce5eb; }
.bslib-sidebar-layout>.main { padding:28px; }
.hero { margin-bottom:20px; }
.eyebrow { color:#087e83; text-transform:uppercase; letter-spacing:.15em;
 font-size:11px; font-weight:750; margin-bottom:10px; }
h1 { font-size:clamp(27px,3.2vw,44px); letter-spacing:-.045em; font-weight:760; }
.subtitle { color:#627487; font-size:15px; max-width:760px; }
.card { border:1px solid #dfe7eb; border-radius:14px; box-shadow:none; }
.card-header { background:#fff; padding:15px 18px; font-weight:680; }
.metric { padding:20px; background:white; border:1px solid #dfe7eb; border-radius:14px;
 min-height:126px; border-top:3px solid #087e83; }
.metric.coral { border-top-color:#d4613b; }
.metric-label { color:#627487; font-size:12px; font-weight:650; text-transform:uppercase; letter-spacing:.07em; }
.metric-value { font-size:clamp(23px,2.5vw,34px); font-weight:750; letter-spacing:-.025em; margin:4px 0; }
.metric-detail { font-size:12px; color:#627487; }
.status { border-left:4px solid #087e83; background:#eaf5f4; padding:13px 16px; border-radius:6px; margin:18px 0; }
.status.alerted { border-left-color:#d4613b; background:#fff1e9; }
.status small { display:block; padding-top:5px; }
.control-title { font-size:11px; letter-spacing:.12em; font-weight:750; color:#087e83; margin:20px 0 12px; }
.hint { font-size:12px; color:#627487; line-height:1.5; }
.btn { border-radius:7px; font-size:13px; }
.btn-default { border:1px solid #b8cbd3; color:#087e83; background:white; }
.preset-row { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:14px; }
.irs--shiny .irs-bar { border-color:#087e83; background:#087e83; }
.irs--shiny .irs-single { background:#087e83; }
.form-label { font-size:13px; font-weight:600; }
.bslib-sidebar-layout .sidebar .shiny-input-container { width:100%; }
.equation { font-family:Georgia,serif; font-size:23px; padding:17px; background:#f4f7f9; border-radius:8px; }
.lesson { max-width:920px; font-size:15px; line-height:1.65; }
.lesson li { margin:10px 0; }
.footer { color:#718293; font-size:12px; margin-top:18px; }
.download-row { display:flex; gap:10px; flex-wrap:wrap; margin-top:15px; }
@media(max-width:767px) { .bslib-sidebar-layout>.main { padding:16px; } }
"""

def metric(label, value, detail, extra=''):
    return ui.div(ui.div(label, class_='metric-label'),
                  ui.div(value, class_='metric-value'),
                  ui.div(detail, class_='metric-detail'), class_=f'metric {extra}')


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.div(ui.input_action_button('reset', 'Tibia example'),
               ui.input_action_button('tiny', 'Tiny load'), class_='preset-row'),
        ui.div('01 / LOADING', class_='control-title'),
        ui.input_slider('delta_sigma', 'Stress range Δσ (MPa)', 0, 40, 11, step=0.01),
        ui.input_checkbox('link_peak', 'Set tensile peak σmax = Δσ', True),
        ui.panel_conditional('!input.link_peak',
            ui.input_numeric('sigma_max', 'Tensile peak σmax (MPa)', 11, min=0.001, step=0.5)),
        ui.p('Δσ = σmax − σmin, the full range, not the half-range amplitude. '
             'Linked peaks reproduce the book’s conservative approximation.', class_='hint'),
        ui.div('02 / CRACK & MATERIAL', class_='control-title'),
        ui.input_slider('log_a0', 'Initial crack: log₁₀(a₀ / mm)', -4, 1, -2, step=0.05),
        ui.output_text('a0_readout'),
        ui.input_slider('Kc', 'Fracture toughness Kc (MPa√m)', 0.2, 8, 2.2, step=0.1),
        ui.input_slider('m', 'Paris exponent m', 1, 5, 2.5, step=0.05),
        ui.input_slider('log_rate', 'Growth-rate multiplier: log₁₀(factor)', -2, 2, 0, step=0.05),
        ui.p('Factor 1 gives C = 2.5 × 10⁻⁶ at m = 2.5. Varying m keeps the '
             'growth rate fixed at ΔK = 1 MPa√m; these are sensitivity experiments.', class_='hint'),
        ui.input_slider('Y', 'Geometry factor Y', 0.5, 2, 1, step=0.01),
        ui.div('03 / MODEL ASSUMPTION', class_='control-title'),
        ui.input_checkbox('threshold_on', 'Include illustrative crack-growth threshold', False),
        ui.input_numeric('delta_K_th', 'Threshold ΔKth (MPa√m)', 0.1, min=0, step=0.01),
        ui.p('The threshold value is a teaching assumption, not a measured tibial property. '
             'No growth below this cutoff does not mean proven infinite life.', class_='hint'),
        ui.tags.details(ui.tags.summary('Stride & calendar conversion'),
            ui.input_numeric('stride_m', 'Stride length (m, same foot to same foot)', 2, min=0.01, step=0.1),
            ui.input_numeric('cycles_per_day', 'Loading cycles per day', 5000, min=1, step=500),
            ui.p('Rest, repair and frequency effects are not modeled.', class_='hint')),
        width=315, open='desktop', title='Experiment controls', gap='7px', padding='18px',
    ),
    ui.tags.style(CSS),
    ui.div(ui.div('INTRO TO BIOMECHANICS · INTERACTIVE LAB', class_='eyebrow'),
           ui.h1('One small crack. Thousands of cycles.'),
           ui.p('Explore how cyclic loading drives a pre-existing crack toward fast fracture. '
                'Start with the textbook tibia, then change one parameter at a time.', class_='subtitle'),
           class_='hero'),
    ui.output_ui('metrics'),
    ui.output_ui('status'),
    ui.navset_card_tab(
        ui.nav_panel('Explore',
            ui.input_slider('progress', 'Follow the crack through the predicted propagation life (%)',
                            0, 100, 0, step=1, animate=ui.AnimationOptions(interval=120, loop=False)),
            ui.layout_columns(
                ui.card(ui.card_header('Crack length → cycles'), ui.output_plot('crack_plot', height='345px')),
                ui.card(ui.card_header('Cycles to fast fracture → stress range'), ui.output_plot('life_plot', height='345px')),
                col_widths=[6,6]),
            ui.output_ui('progress_readout'),
            ui.p('Gray dashed curve: original tibia. Colored curves: current experiment. '
                 'The stress sweep uses linked peaks when the checkbox is selected; '
                 'otherwise the tensile peak stays fixed. Sweep results with a critical crack greater '
                 'than 15 mm extend beyond the textbook section scale.', class_='hint')),
        ui.nav_panel('Growth law',
            ui.output_plot('law_plot', height='430px'),
            ui.p('Paris law represents the intermediate crack-growth regime. '
                 'Its extension toward zero driving force is a mathematical extrapolation. '
                 'The hard threshold is an intentionally simple comparison, not a fitted low-growth model.', class_='hint')),
        ui.nav_panel('Teach the model',
            ui.div(
                ui.h4('Fatigue propagation is not immediate fracture'),
                ui.p('The initial Kmax can be much smaller than Kc. Repeated cycles grow the crack; '
                     'the same load then produces a larger stress intensity.'),
                ui.div('da/dN = C(ΔK)ᵐ  ·  ΔK = YΔσ√(πa)', class_='equation'),
                ui.p('Integrate from a₀ to af, where af = (Kc / Yσmax)² / π. All crack lengths '
                     'are converted to metres inside the calculation. K uses MPa√m.'),
                ui.div('Nf = [af^(1−m/2) − a₀^(1−m/2)] / [(1−m/2) C(YΔσ)ᵐ π^(m/2)]', class_='equation'),
                ui.p('At m = 2 the numerator and exponent form is replaced by ln(af/a₀): '
                     'Nf = ln(af/a₀) / [C(YΔσ)²π].'),
                ui.h4('A 10-minute classroom sequence'),
                ui.tags.ol(
                    ui.tags.li('Start with the tibia. Why is fast fracture absent initially, yet predicted after repeated cycles?'),
                    ui.tags.li('Play the crack animation. Why does the crack grow faster near the endpoint?'),
                    ui.tags.li('Unlink the peak, keep σmax = 11 MPa, and halve Δσ to 5.5 MPa. '
                               'Predict the life ratio before moving the slider: 2²⋅⁵ ≈ 5.66.'),
                    ui.tags.li('Reset and increase a₀ tenfold. Then double the growth-rate factor. '
                               'Which change halves life exactly?'),
                    ui.tags.li('Select Tiny load. Raw Paris law predicts an enormous but finite life. '
                               'Switch on the illustrative threshold. What changed: the load, or the model?')),
                ui.h4('The statement students should leave with'),
                ui.p(ui.strong('Loads below the immediate-fracture load can cause failure after repeated cycling. '
                               'Whether an arbitrarily tiny load eventually causes failure depends on the material, '
                               'crack, environment and model.')),
                ui.p('This calculation starts with an existing crack and estimates propagation cycles, not '
                     'initiation life. Constant Y, constant-amplitude loading and no biological repair are assumed. '
                     'Compression and crack closure, microstructure, load sequence, anisotropy and the near-critical '
                     'acceleration beyond Paris law are omitted. A 10 μm bone defect challenges continuum/long-crack assumptions.'),
                ui.p('The threshold switch means da/dN = 0 for ΔK ≤ ΔKth, and the unchanged Paris law above it. '
                     'It is neither an endurance-limit model nor evidence that a living bone will never fail.'),
                ui.h4('Sources'),
                ui.p('Baseline: supplied textbook pages, Chapter 9, Eqs. 9.19–9.22 and Figs. 9.17–9.19. '
                     'The printed 12.7 mm endpoint yields 14,111.7 cycles; calculating 12.7324 mm from Kc yields 14,113.5.'),
                ui.p(ui.a('Akkus & Rimnac (2001): observed bone microcrack deceleration and arrest',
                         href='https://pubmed.ncbi.nlm.nih.gov/11470113/', target='_blank')),
                ui.p(ui.a('ASTM E647: limitations of long-crack and threshold data for small cracks',
                         href='https://store.astm.org/e0647-24.html', target='_blank')),
                class_='lesson')),
        ui.nav_panel('Export',
            ui.p('Download the current parameters and trajectory to compare classroom experiments. '
                 'For an arrested crack, the CSV contains a flat trace over a reference window; it does not contain a failure event.'),
            ui.div(ui.download_button('download_csv', 'Crack trajectory · CSV'),
                   ui.download_button('download_json', 'Parameters & results · JSON'), class_='download-row'))),
    ui.div('Teaching simulation · Constant-amplitude crack propagation · No remodeling or healing · '
           'Distance is a model conversion, not a safe running distance.', class_='footer'),
    title=None, window_title='Bone Fatigue Lab', fillable=False,
)


def server(input, output, session):
    @reactive.calc
    def params():
        values = [input.delta_sigma(), input.Kc(), input.m(), input.log_rate(),
                  input.Y(), input.delta_K_th(), input.stride_m(), input.cycles_per_day()]
        if any(v is None for v in values):
            raise ValueError('Enter a number in every field.')
        peak = input.delta_sigma() if input.link_peak() else input.sigma_max()
        p = Parameters(delta_sigma=input.delta_sigma(), sigma_max=peak,
                       a0_mm=10**input.log_a0(), Kc=input.Kc(), m=input.m(),
                       growth_ref=2.5e-6*10**input.log_rate(), Y=input.Y(),
                       threshold_on=input.threshold_on(), delta_K_th=input.delta_K_th(),
                       stride_m=input.stride_m(), cycles_per_day=input.cycles_per_day())
        solve(p)  # validates values before every dependent output
        return p

    @reactive.calc
    def result():
        return solve(params())

    def reset_values():
        for key, val in {'delta_sigma':11,'log_a0':-2,'Kc':2.2,'m':2.5,
                         'log_rate':0,'Y':1,'progress':0}.items():
            ui.update_slider(key,value=val)
        for key,val in {'sigma_max':11,'delta_K_th':0.1,'stride_m':2,'cycles_per_day':5000}.items():
            ui.update_numeric(key,value=val)
        ui.update_checkbox('link_peak',value=True)
        ui.update_checkbox('threshold_on',value=False)

    @reactive.effect
    @reactive.event(input.reset)
    def reset():
        reset_values()

    @reactive.effect
    @reactive.event(input.tiny)
    def tiny():
        reset_values()
        ui.update_slider('delta_sigma',value=0.05)

    @render.text
    def a0_readout():
        return f'a₀ = {10**input.log_a0():.4g} mm = {1000*10**input.log_a0():.4g} μm'

    @render.ui
    def metrics():
        r, p = result(), params()
        nlabel = 'Propagation cycles to criterion'
        n = format_number(r['N'])
        detail = r['endpoint'] if isfinite(r['N']) else r['status']
        return ui.layout_columns(
            metric(nlabel, n, detail),
            metric('Equivalent running distance', format_number(r['distance_km'],1) + (' km' if isfinite(r['N']) else ''),
                   f'{format_number(r["days"],1)} days at {p.cycles_per_day:g} cycles/day' if isfinite(r['N']) else 'No finite distance in this model'),
            metric('Critical crack length', format_number(r['ac_mm'],3)+' mm' if isfinite(r['ac_mm']) else 'No tensile criterion',
                   f'Initial Kmax / Kc = {r["K0"]/p.Kc:.3f}', 'coral'),col_widths=[4,4,4])

    @render.ui
    def status():
        r,p = result(),params()
        text = f'{r["status"]}. Initial ΔK = {r["dK0"]:.4g} MPa√m; σmin = {p.sigma_max-p.delta_sigma:.4g} MPa.'
        if r['N'] == 0:
            note='Kmax already reaches Kc at the initial crack. There is no stable propagation phase in this model.'
        elif not isfinite(r['N']):
            note='The crack stays fixed in this idealization. This is not a guarantee of infinite material life.'
        elif r['outside_geometry']:
            note='The critical crack exceeds the 15 mm textbook section scale. The displayed life is a mathematical extrapolation; the constant-Y geometry model must be revised.'
        else:
            note='Textbook baseline ≈ 14,112 cycles and 28.2 km. The small rounding difference comes from recalculating the critical crack length.'
        return ui.div(ui.strong(text),ui.tags.small(note),class_='status'+(' alerted' if r['N']==0 or r['outside_geometry'] else ''))

    @render.plot
    def crack_plot():
        p,r=params(),result()
        fig,ax=plt.subplots(figsize=(5.7,3.6),layout='constrained')
        nb,ab=trajectory(Parameters())
        n,a=trajectory(p)
        if r['N'] > 0:
            horizon=n[-1]
            valid=nb<=horizon
            ax.plot(nb[valid],ab[valid],'--',color=GRAY,lw=1.6,label='Original tibia')
            ax.plot(n,a,color=TEAL,lw=2.6,label='Current model')
            if isfinite(r['N']):
                now=r['N']*input.progress()/100
                length=np.interp(now,n,a)
                ax.scatter([now],[length],s=55,color=CORAL,zorder=5)
                ax.axvline(now,color=CORAL,lw=1,alpha=.5)
                ax.axhline(r['af_mm'],color=CORAL,lw=1,ls=':')
            else:
                ax.text(.05,.92,'Flat trace: no modeled growth',transform=ax.transAxes,color=TEAL)
            ax.set_xlim(0,max(horizon,1e-6))
        else:
            ax.scatter([0],[p.a0_mm],color=CORAL,s=70)
            ax.text(.08,.8,'Immediate fast fracture\nKmax(a₀) ≥ Kc',transform=ax.transAxes,color=CORAL)
            ax.set_xlim(-.05,1)
        ax.set_xlabel('Cycles, N' if isfinite(r['N']) else 'Cycles, N (reference window)')
        ax.set_ylabel('Crack length a (mm)')
        ax.grid(alpha=.7)
        handles,_=ax.get_legend_handles_labels()
        if handles: ax.legend(frameon=False,fontsize=9,loc='upper left')
        return fig

    @render.plot
    def life_plot():
        p,r=params(),result()
        x,raw,limited=stress_sweep(p,input.link_peak())
        fig,ax=plt.subplots(figsize=(5.7,3.6),layout='constrained')
        raw=np.where(raw>0,raw,np.nan)
        ax.loglog(x,raw,color=TEAL,lw=2.2,label='Paris extrapolation')
        if p.threshold_on:
            y=np.where(np.isfinite(limited)&(limited>0),limited,np.nan)
            ax.loglog(x,y,color=CORAL,lw=2,ls='--',label='With hard threshold')
            cut=p.delta_K_th/(p.Y*sqrt(pi*p.a0_mm*1e-3))
            if cut>x[0]:
                ax.axvspan(x[0],min(cut,x[-1]),color=CORAL,alpha=.10)
                ax.text(.03,.08,'Shaded: no growth with threshold',transform=ax.transAxes,fontsize=8,color=CORAL)
        if r['N']>0 and isfinite(r['N']) and p.delta_sigma>0:
            ax.scatter([p.delta_sigma],[r['N']],color=INK,s=45,zorder=5,label='Current setting')
        ax.set_xlabel('Stress range Δσ (MPa)')
        ax.set_ylabel('Cycles to fast-fracture criterion')
        ax.grid(which='major',alpha=.7)
        ax.legend(frameon=False,fontsize=8)
        return fig

    @render.ui
    def progress_readout():
        p,r=params(),result()
        if not isfinite(r['N']) or r['N']==0:
            return ui.p(r['status']+' — the propagation animation is inactive.',class_='hint')
        n,a=trajectory(p)
        now=r['N']*input.progress()/100
        length=np.interp(now,n,a)
        return ui.p(ui.strong(f'Cycle {format_number(now)} · crack {length:.4g} mm · '
                             f'Kmax = {p.Y*p.sigma_max*sqrt(pi*length*1e-3):.3g} MPa√m'),
                    f'  ({input.progress():g}% of modeled life)',class_='hint')

    @render.plot
    def law_plot():
        p=params()
        k=np.geomspace(1e-4,20,500)
        fig,ax=plt.subplots(figsize=(10,4.2),layout='constrained')
        ax.loglog(k,growth_rate(replace(p,threshold_on=False),k),color=TEAL,lw=2.5,label='Paris extrapolation')
        if p.threshold_on and p.delta_K_th>0:
            y=growth_rate(p,k)
            ax.loglog(k,np.where(y>0,y,np.nan),color=CORAL,ls='--',lw=2,label='Hard threshold model')
            ax.axvline(p.delta_K_th,color=CORAL,ls=':')
            ax.axvspan(k[0],min(p.delta_K_th,k[-1]),color=CORAL,alpha=.10)
            ax.text(.02,.87,'Below cutoff: da/dN = 0\n(zero cannot be plotted on a log axis)',transform=ax.transAxes,fontsize=10)
        r=result()
        rate=growth_rate(p,r['dK0']).item()
        if rate>0 and r['dK0']>0:
            ax.scatter([r['dK0']],[rate],color=INK,s=50,zorder=5,label='At the initial crack')
        ax.scatter([p.K_ref],[p.growth_ref],marker='x',s=55,color=GRAY,label='Fixed pivot when m changes')
        ax.set_xlabel('Stress-intensity range ΔK (MPa√m)')
        ax.set_ylabel('Growth per cycle da/dN (m/cycle)')
        ax.grid(which='major',alpha=.7)
        ax.legend(frameon=False,loc='lower right')
        return fig

    @render.download(filename='fatigue_trajectory.csv')
    def download_csv():
        p,r=params(),result()
        n,a=trajectory(p)
        buf=StringIO()
        w=csv.writer(buf)
        w.writerow(['cycles','crack_mm','distance_km','Kmax_MPa_sqrt_m','delta_K_MPa_sqrt_m','status'])
        for N,A in zip(n,a):
            w.writerow([N,A,N*p.stride_m/1000,p.Y*p.sigma_max*sqrt(pi*A*1e-3),
                        p.Y*p.delta_sigma*sqrt(pi*A*1e-3),r['status']])
        yield buf.getvalue()

    @render.download(filename='fatigue_parameters.json')
    def download_json():
        r={k:('Infinity (no finite endpoint)' if isinstance(v,float) and not isfinite(v) else v)
           for k,v in result().items()}
        yield json.dumps({'parameters':asdict(params()),'results':r},indent=2,allow_nan=False)


app=App(app_ui,server)
