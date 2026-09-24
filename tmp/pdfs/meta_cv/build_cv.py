from pathlib import Path
from xml.sax.saxutils import escape
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, KeepTogether, Flowable

ROOT = Path(r'C:\Users\BQ6757\chronos2_v1')
OUT = ROOT / 'output' / 'pdf'
OUT.mkdir(parents=True, exist_ok=True)
PDF = OUT / 'Yoan_Kesraoui_CV_Meta_FAIR_PhD_v2.pdf'

for name, filename in [('Arial', 'arial.ttf'), ('Arial-Bold', 'arialbd.ttf'), ('Arial-Italic', 'ariali.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(Path(r'C:\Windows\Fonts') / filename)))
pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='Arial-Bold', italic='Arial-Italic', boldItalic='Arial-Bold')

BLACK = colors.HexColor('#15191E')
MUTED = colors.HexColor('#4B525A')
ST = {
    'name': ParagraphStyle('Name', fontName='Arial-Bold', fontSize=23, leading=27, alignment=TA_CENTER, textColor=BLACK),
    'tagline': ParagraphStyle('Tagline', fontName='Arial-Bold', fontSize=10.3, leading=14, alignment=TA_CENTER, textColor=BLACK),
    'contact': ParagraphStyle('Contact', fontName='Arial', fontSize=9.2, leading=12, alignment=TA_CENTER, textColor=MUTED),
    'section': ParagraphStyle('Section', fontName='Arial-Bold', fontSize=10.5, leading=13, textColor=BLACK, spaceBefore=10, spaceAfter=5, keepWithNext=True),
    'body': ParagraphStyle('Body', fontName='Arial', fontSize=10, leading=13.1, textColor=BLACK),
    'role': ParagraphStyle('Role', fontName='Arial-Bold', fontSize=10.1, leading=13, textColor=BLACK),
    'meta': ParagraphStyle('Meta', fontName='Arial-Italic', fontSize=9.4, leading=12.2, textColor=MUTED),
    'bullet': ParagraphStyle('Bullet', fontName='Arial', fontSize=10, leading=13.1, textColor=BLACK, leftIndent=10, firstLineIndent=-9, spaceAfter=3),
    'skills': ParagraphStyle('Skills', fontName='Arial', fontSize=9.5, leading=12.6, textColor=BLACK, spaceAfter=2),
}

profile = (
    'Quantitative researcher trained in data science, econometrics and applied mathematics, with experience '
    'applying machine learning to forecasting, designing rigorous evaluation protocols and analysing large-scale '
    'transaction data. Seeking to pursue doctoral research in machine learning and computational statistics at Meta FAIR.'
)

education = [
    ('Université Paris 1 Panthéon-Sorbonne', 'M.Sc. Data Science', '2025-2026',
     'Machine learning, deep learning, statistics, optimisation and time series'),
    ('Université Paris Cité', 'M.Sc. Applied Economics - Econometrics', '2023-2025',
     'Econometrics, statistical inference, financial modelling and panel data'),
    ('Université Paris Cité', 'B.Sc. Economics & Mathematics', '2020-2023', None),
]

experience = [
    {
        'org': 'ENGIE Global Markets', 'role': 'Quant Research Intern - Energy Markets',
        'dates': 'Jun 2026-Present', 'team': 'Global Market Analysis, Paris',
        'bullets': [
            'Developed a <b>probabilistic day-ahead forecasting framework for five European power markets</b>, combining Chronos-2, residual models and adaptive Kalman filtering.',
            'Fine-tuned <b>Chronos-2 using LoRA (ranks 8 and 16)</b> across four markets; trained a weather-conditioned adapter using archived NOAA GFS forecasts.',
            'Implemented and benchmarked a <b>sparse mixture-of-experts residual model in PyTorch</b> to study regime-dependent forecast corrections.',
            'Implemented 365-day rolling evaluations and time-ordered out-of-fold calibration with fold-specific LoRA refits; tracked data provenance, publication cutoffs and probabilistic metrics.',
        ],
    },
    {
        'org': 'Société Générale', 'role': 'Quant Market Risk Intern',
        'dates': 'Sep 2025-Mar 2026', 'team': 'Front Office Market Risk, Paris',
        'bullets': [
            'Developed <b>GARCH, EGARCH and VAR</b> models to study financial time series and portfolio-risk dynamics.',
            'Built <b>Python/SQL/API</b> pipelines for large risk datasets; applied clustering and k-nearest neighbours to anomaly detection and risk monitoring.',
        ],
    },
    {
        'org': 'Autorité des marchés financiers (AMF)', 'role': 'Data Scientist / Economist',
        'dates': 'Jan-Jun 2025', 'team': 'Financial Stability & Risk Studies, Paris',
        'bullets': [
            'Analysed <b>millions of MiFID II transactions</b> using PySpark, Python, R and panel econometrics to study investor behaviour and market dynamics.',
            'Engineered investor-level turnover, return and trading-intensity indicators; contributed empirical analysis to the <i>AMF Market and Risk Mapping 2025</i>.',
        ],
    },
    {
        'org': 'Generali', 'role': 'Quantitative Analyst Intern',
        'dates': 'Apr-Sep 2024', 'team': 'ALM & Strategic Asset Allocation, Paris',
        'bullets': [
            'Built <b>XGBoost surrogate models</b> for computationally intensive full-revaluation pricing; investigated non-linear relationships between risk factors.',
            'Calibrated Vasicek/CIR interest-rate models and automated analytical workflows with Python, SQL and Power BI.',
        ],
    },
]

skills = [
    ('Programming and data', 'Python, SQL, PySpark/Spark, R, Git'),
    ('ML libraries', 'PyTorch, TensorFlow, scikit-learn, CatBoost, XGBoost, pandas, NumPy, statsmodels'),
    ('Methods', 'Time-series forecasting, deep learning, econometrics, statistical inference, probabilistic modelling, optimisation, anomaly detection'),
    ('Languages', 'French (native), English (C1), Spanish (beginner)'),
]


class DatedHeading(Flowable):
    def __init__(self, text, dates, size=10.1):
        super().__init__()
        style = ParagraphStyle('Dated', parent=ST['role'], fontSize=size)
        self.p = Paragraph(text, style)
        self.dates = dates
        self.date_size = 9.2

    def wrap(self, availWidth, availHeight):
        self.width = availWidth
        self.date_width = pdfmetrics.stringWidth(self.dates, 'Arial-Bold', self.date_size)
        _, self.height = self.p.wrap(availWidth - self.date_width - 16, availHeight)
        return self.width, self.height

    def draw(self):
        self.p.drawOn(self.canv, 0, 0)
        self.canv.setFont('Arial-Bold', self.date_size)
        self.canv.setFillColor(BLACK)
        self.canv.drawRightString(self.width, self.height - 10.1, self.dates)


story = []
story.append(Paragraph('YOAN KESRAOUI', ST['name']))
story.append(Spacer(1, 2))
story.append(Paragraph('MACHINE LEARNING AND QUANTITATIVE RESEARCH', ST['tagline']))
story.append(Spacer(1, 3))
story.append(Paragraph(
    'Paris, France &nbsp;|&nbsp; +33 7 81 83 13 88 &nbsp;|&nbsp; '
    '<link href="mailto:kesraoui.yoan@gmail.com" color="#4B525A">kesraoui.yoan@gmail.com</link> &nbsp;|&nbsp; '
    '<link href="https://www.linkedin.com/in/yoan-kesraoui-20b983304" color="#4B525A">LinkedIn</link>', ST['contact']))

story.append(Paragraph('RESEARCH PROFILE', ST['section']))
story.append(Paragraph(profile, ST['body']))
story.append(Paragraph('EDUCATION', ST['section']))
for i, (school, degree, dates, courses) in enumerate(education):
    block = [DatedHeading(escape(school) + ' - ' + escape(degree), dates, size=9.9)]
    if courses:
        block.append(Paragraph(escape(courses), ST['meta']))
    if i < len(education) - 1:
        block.append(Spacer(1, 4))
    story.append(KeepTogether(block))

story.append(Paragraph('RESEARCH AND PROFESSIONAL EXPERIENCE', ST['section']))
for i, exp in enumerate(experience):
    title = escape(exp['org']) + ' - ' + escape(exp['role'])
    block = [DatedHeading(title, exp['dates']), Paragraph(escape(exp['team']), ST['meta']), Spacer(1, 2)]
    for b in exp['bullets']:
        block.append(Paragraph('&#8226; ' + b, ST['bullet']))
    if i < len(experience) - 1:
        block.append(Spacer(1, 4))
    story.append(KeepTogether(block))

story.append(Paragraph('TECHNICAL SKILLS', ST['section']))
for label, value in skills:
    story.append(Paragraph('<b>' + escape(label) + ':</b> ' + escape(value), ST['skills']))

doc = SimpleDocTemplate(str(PDF), pagesize=A4, rightMargin=37, leftMargin=37,
                        topMargin=29, bottomMargin=27, title='Yoan Kesraoui - Meta FAIR AI Research Assistant PhD',
                        author='Yoan Kesraoui', subject='Application for AI Research Assistant (PhD), Meta FAIR, Paris',
                        keywords='machine learning, AI research, computational statistics, Python, PyTorch, time series, Meta FAIR')
doc.build(story)

def plain(text):
    return re.sub('<[^>]*>', '', text)

lines = ['YOAN KESRAOUI', 'MACHINE LEARNING AND QUANTITATIVE RESEARCH',
         'Paris, France | +33 7 81 83 13 88 | kesraoui.yoan@gmail.com',
         'https://www.linkedin.com/in/yoan-kesraoui-20b983304', '', 'RESEARCH PROFILE', profile, '', 'EDUCATION']
for school, degree, dates, courses in education:
    lines.append(f'{school} - {degree} | {dates}')
    if courses:
        lines.append(courses)
    lines.append('')
lines.append('RESEARCH AND PROFESSIONAL EXPERIENCE')
for exp in experience:
    lines.extend([f"{exp['org']} - {exp['role']} | {exp['dates']}", exp['team']])
    lines.extend('- ' + plain(b) for b in exp['bullets'])
    lines.append('')
lines.append('TECHNICAL SKILLS')
lines.extend(f'{label}: {value}' for label, value in skills)
(OUT / 'Yoan_Kesraoui_CV_Meta_FAIR_PhD_v2.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(PDF)
