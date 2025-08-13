import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
from matplotlib.widgets import Cursor
import matplotlib.dates as mdates
import datetime

# ===================== 中文显示配置 =====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
sns.set_theme(style="whitegrid", font='Microsoft YaHei')  # 专业金融图表风格

# ===================== 获取基金数据 =====================
fund_open_fund_info_em_df = ak.fund_open_fund_info_em(symbol="017436", indicator="单位净值走势")

# ===================== 数据预处理 =====================
df = fund_open_fund_info_em_df.copy()
df['净值日期'] = pd.to_datetime(df['净值日期'])  # 转换为日期格式
df = df.sort_values('净值日期')  # 按日期排序

# ===================== 创建画布 =====================
fig, ax = plt.subplots(figsize=(14, 8), dpi=100)

# ===================== 绘制净值曲线 =====================
line, = ax.plot(df['净值日期'], df['单位净值'], 
               color='#1f77b4', linewidth=2.5, 
               marker='o', markersize=4, 
               markevery=len(df)//20,  # 每20个点标记一个
               label='单位净值')

# ===================== 添加标题和标签 =====================
ax.set_title('基金历史净值走势 (代码: 017436)', fontsize=16, pad=20)
ax.set_xlabel('日期', fontsize=12, labelpad=10)
ax.set_ylabel('单位净值', fontsize=12, labelpad=10)

# ===================== 设置日期格式 =====================
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate()  # 自动旋转日期标签

# ===================== 设置Y轴格式 =====================
def format_y(y, _):
    return f'{y:.4f}'  # 保留4位小数
ax.yaxis.set_major_formatter(FuncFormatter(format_y))

# ===================== 添加网格 =====================
ax.grid(True, linestyle='--', alpha=0.7)

# ===================== 添加关键点标注 =====================
max_idx = df['单位净值'].idxmax()
min_idx = df['单位净值'].idxmin()
latest_idx = df.index[-1]

ax.annotate(f'最高: {df.loc[max_idx, "单位净值"]:.4f}',
            xy=(df.loc[max_idx, "净值日期"], df.loc[max_idx, "单位净值"]),
            xytext=(-50, 30), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10)

ax.annotate(f'最新: {df.loc[latest_idx, "单位净值"]:.4f}',
            xy=(df.loc[latest_idx, "净值日期"], df.loc[latest_idx, "单位净值"]),
            xytext=(-50, -40), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='green'),
            fontsize=10)

# ===================== 添加统计信息框 =====================
stats_text = f"""基金统计:
起始日期: {df['净值日期'].min().strftime('%Y-%m-%d')}
最新日期: {df['净值日期'].max().strftime('%Y-%m-%d')}
净值点数: {len(df)}
最高净值: {df['单位净值'].max():.4f}
最低净值: {df['单位净值'].min():.4f}
当前净值: {df['单位净值'].iloc[-1]:.4f}
累计涨幅: {((df['单位净值'].iloc[-1] - df['单位净值'].iloc[0])/df['单位净值'].iloc[0])*100:.2f}%"""
fig.text(0.76, 0.75, stats_text, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
         fontsize=10)

# ===================== 添加图例和布局调整 =====================
ax.legend(loc='upper left', frameon=True)
plt.tight_layout()

# ===================== 鼠标悬浮事件处理 =====================
# 创建日期到数值的映射
date_to_value = dict(zip(df['净值日期'], df['单位净值']))

# 创建注释框
annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w", alpha=0.9),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def closest_date(xdate):
    """找到最接近鼠标位置的日期"""
    # 将日期转换为数值进行比较
    xdate_num = mdates.date2num(xdate)
    dates_num = mdates.date2num(df['净值日期'])
    
    # 找到最近日期的索引
    idx = np.abs(dates_num - xdate_num).argmin()
    return df.iloc[idx]['净值日期']

def update_annot(date):
    """更新注释框内容"""
    value = date_to_value[date]
    date_str = date.strftime('%Y-%m-%d')
    
    # 查找增长率（如果存在）
    growth_rate = ""
    if '日增长率' in df.columns:
        growth_row = df[df['净值日期'] == date]
        if not growth_row.empty:
            growth_rate = f"{growth_row['日增长率'].values[0]:.2f}%"
    
    annot.xy = (date, value)
    text = f"日期: {date_str}\n单位净值: {value:.4f}"
    if growth_rate:
        text += f"\n日增长率: {growth_rate}"
    annot.set_text(text)
    
    # 自动调整位置避免出界
    x, y = annot.xy
    if x < df['净值日期'].median():
        annot.set_position((40, 30))
    else:
        annot.set_position((-120, 30))

def hover(event):
    """鼠标悬浮事件处理"""
    if event.inaxes == ax:
        if event.xdata is not None:
            # 将x坐标转换为日期
            xdate = mdates.num2date(event.xdata)
            # 找到最近的实际日期
            closest_dt = closest_date(xdate)
            
            # 更新注释框
            update_annot(closest_dt)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

# 连接事件
fig.canvas.mpl_connect("motion_notify_event", hover)

# ===================== 保存和显示图像 =====================
date_now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
plt.savefig(f'fund_net_value_chart_{date_now}.png', bbox_inches='tight')
plt.show()